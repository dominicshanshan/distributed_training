import torch.multiprocessing as mp
import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")

import horovod.torch as hvd

class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        xs = self.data[index, :-1, :]
        ys = self.data[index, -1, -1]
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

    def __len__(self):
        return len(self.data)


class GAT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, base_model="LSTM"):
        super(GAT, self).__init__()

        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.hidden_size = hidden_size
        self.d_feat = input_size
        self.bn1 = nn.BatchNorm1d(input_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def cal_attention(self, x):
        x = self.transformation(x)

        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        x = self.bn1(x)
        x = torch.transpose(x, 1, 2)
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        hidden = self.bn2(hidden)
        att_weight = self.cal_attention(hidden)
        hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        hidden = self.fc_out(hidden)
        hidden = self.sigmoid(hidden)
        return hidden.squeeze()


def train():
    sample_num = 100000
    feature_num = 10
    data_len = 5

    torch.cuda.nvtx.range_push('MyDataset')
    data = np.random.random((sample_num, feature_num + 1, data_len))
    train_dataset = MyDataset(data)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push('GAT')
    model = GAT(input_size=10, hidden_size=64)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push('h2d')
    model.cuda()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push('DataLoader')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=hvd.rank(), num_replicas=hvd.size())
    train_dataloader = DataLoader(train_dataset, shuffle=False, drop_last=True, batch_size=256, sampler=train_sampler, num_workers=0)
    torch.cuda.nvtx.range_pop()

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    model.train()

    loss_meter, it_count = 0, 0
    tq = tqdm(range(len(train_dataloader)))
    for i, (inputs, target) in enumerate(train_dataloader):
        torch.cuda.nvtx.range_push("iteration{}".format(i))
        torch.cuda.nvtx.range_push("input")
        inputs = inputs.cuda()
        target = target.cuda()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("zero_grad")
        optimizer.zero_grad()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("foward")
        output = model(inputs)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("backward")
        loss = criterion(output, target)
        loss.backward()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("step")
        optimizer.step()
        torch.cuda.nvtx.range_pop()
        loss_meter += loss.item()
        it_count += 1
        tq.set_description('batch: %d, loss: %.3f' % (i, loss.item()))
        tq.update(1)
        #if i>0:
        #    break
        torch.cuda.nvtx.range_pop()
        tq.close()



if __name__=="__main__":
    t0= time.time()
    rdseed = 318
    torch.manual_seed(rdseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(rdseed)
    # Horovod: initialize library.
    hvd.init()
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(rdseed)

    train()

    print('time taken overall',time.time()-t0)
