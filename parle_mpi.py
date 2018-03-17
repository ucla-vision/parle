from mpi4py import MPI

import torch as th
import numpy as np

import torch.nn as nn
import torchnet as tnt
from torchvision.datasets.mnist import MNIST
from torch.autograd import Variable

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse, random, pdb
from copy import deepcopy

p = argparse.ArgumentParser('Parle',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--data', type=str, default='/local2/pratikac/mnist', help='dataset')
p.add_argument('--lr', type=float, default=0.1, help='learning rate')
p.add_argument('-b', type=int, default=128, help='batch size')
p.add_argument('-L', type=int, default=25, help='prox. eval steps')
p.add_argument('--gamma', type=float, default=0.01, help='gamma')
p.add_argument('--rho', type=float, default=0.01, help='rho')
p.add_argument('-n', type=int, default=1, help='replicas')
opt = vars(p.parse_args())

comm = MPI.COMM_WORLD

opt['B'], opt['l2'] = 5, -1.0
opt['r'] = comm.Get_rank()
opt['n'] = comm.Get_size()
opt['rho'] = opt['rho']*opt['L']*opt['n']

opt['s'] = 42 + opt['r']
random.seed(opt['s'])
np.random.seed(opt['s'])
th.manual_seed(opt['s'])

opt['cuda'] = th.cuda.is_available()
if opt['cuda']:
    ngpus = th.cuda.device_count()
    opt['g'] = int(opt['r'] % ngpus)
    th.cuda.set_device(opt['g'])
    th.cuda.manual_seed(opt['s'])
print(opt)

def get_iterator(mode):
    ds = MNIST(root=opt['data'], download=True, train=mode)
    data = getattr(ds, 'train_data' if mode else 'test_data')
    labels = getattr(ds, 'train_labels' if mode else 'test_labels')
    tds = tnt.dataset.TensorDataset([data, labels])
    return tds.parallel(batch_size=opt['b'],
            num_workers=0, shuffle=mode, pin_memory=True)

class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o
    def forward(self,x):
        return x.view(-1, self.o)

def convbn(ci,co,ksz,psz,p):
    return nn.Sequential(
        nn.Conv2d(ci,co,ksz),
        nn.BatchNorm2d(co),
        nn.ReLU(True),
        nn.MaxPool2d(psz,stride=psz),
        nn.Dropout(p))

model = nn.Sequential(
    convbn(1,20,5,3,0.25),
    convbn(20,50,5,2,0.25),
    View(50*2*2),
    nn.Linear(50*2*2, 500),
    nn.BatchNorm1d(500),
    nn.ReLU(True),
    nn.Dropout(0.25),
    nn.Linear(500,10),
    )
criterion = nn.CrossEntropyLoss()

if opt['cuda']:
    model = model.cuda()
    criterion = criterion.cuda()

def parle_step(sync=False):
    eps = 1e-3

    mom, alpha = 0.9, 0.75
    lr = opt['lr']
    r = opt['r']
    nb = opt['nb']

    if not 'state' in opt:
        opt['state'] = {}
        s = opt['state']
        s['t'] = 0

        for k in ['za', 'muy', 'mux', 'xa', 'x', 'cache']:
            s[k] = {}

        for p in model.parameters():
            for k in ['za', 'muy', 'mux', 'xa']:
                s[k][p] = p.data.clone()

            s['muy'][p].zero_()
            s['mux'][p].zero_()

            s['x'][p] = p.data.cpu().numpy()
            s['cache'][p] = p.data.cpu().numpy()

    s = opt['state']
    t = s['t']

    za, muy, mux, xa, x, cache = s['za'], s['muy'], s['mux'], \
        s['xa'], s['x'], s['cache']

    gamma = opt['gamma']*(1 + 0.5/nb)**(t // opt['L'])
    rho = opt['rho']*(1 + 0.5/nb)**(t // opt['L'])
    gamma, rho = min(gamma, 1), min(rho, 10)

    def sync_with_master(xa, x):
        for p in model.parameters():
            xa[p] = xa[p].cpu().numpy()
            comm.Reduce(xa[p], s['cache'][p], op=MPI.SUM, root=0)
            xa[p] = th.from_numpy(xa[p])
            if opt['cuda']:
                xa[p] = xa[p].cuda()

        comm.Barrier()

        if r == 0:
            for p in model.parameters():
                x[p] = s['cache'][p]/float(opt['n'])

        for p in model.parameters():
            comm.Bcast(x[p], root=0)
        comm.Barrier()

    if sync:
        # add another sync, helps with large L
        sync_with_master(za, x)

        for p in model.parameters():
            tmp = th.from_numpy(x[p])
            if opt['cuda']:
                tmp = tmp.cuda()

            # elastic-sgd term
            p.grad.data.zero_()
            p.grad.data.add_(1, xa[p] - za[p]).add_(rho, xa[p] - tmp)

            mux[p].mul_(mom).add_(p.grad.data)
            p.grad.data.add_(mux[p])
            p.data.add_(-lr, p.grad.data)

            xa[p].copy_(p.data)
        sync_with_master(xa, x)
    else:
        # entropy-sgd iterations
        for p in model.parameters():
            p.grad.data.add_(gamma, p.data - xa[p])

            muy[p].mul_(mom).add_(p.grad.data)
            p.grad.data.add_(muy[p])
            p.data.add_(-lr, p.grad.data)

            za[p].mul_(alpha).add_(1-alpha, p.data)

            s['t'] += 1

def train(e):
    model.train()

    train_ds = get_iterator(True)
    train_iter = train_ds.__iter__()
    opt['nb'] = len(train_iter)

    loss = tnt.meter.AverageValueMeter()
    top1 = tnt.meter.ClassErrorMeter()

    for b in range(opt['nb']):
        for l in range(opt['L']):
            try:
                x,y = next(train_iter)
            except StopIteration:
                train_iter = train_ds.__iter__()
                x,y = next(train_iter)

            x = Variable(x.view(-1,1,28,28).float() / 255.0)
            y = Variable(th.LongTensor(y))
            if opt['cuda']:
                x, y = x.cuda(async=True), y.cuda(async=True)

            model.zero_grad()
            yh = model(x)
            f = criterion(yh, y)
            f.backward()

            if opt['l2'] > 0:
                for p in model.parameters():
                    p.grad.data.add_(opt['l2'], p.data)

            if l == 0:
                top1.add(yh.data, y.data)
                loss.add(f.data[0])

                if b % 100 == 0 and b > 0:
                    print('[%03d][%03d/%03d] %.3f %.3f%%'%(e, b, opt['nb'], \
                            loss.value()[0], top1.value()[0]))

            parle_step()

        # setup value for sync
        #opt['state']['f'][0] = loss.value()[0]
        parle_step(sync=True)

    r = dict(f=loss.value()[0], top1=top1.value()[0])
    print('+[%02d] %.3f %.3f%%', e, r['f'], r['top1'])
    return r

def dry_feed(m):
    def set_dropout(cache = None, p=0):
        if cache is None:
            cache = []
            for l in m.modules():
                if 'Dropout' in str(type(l)):
                    cache.append(l.p)
                    l.p = p
            return cache
        else:
            for l in m.modules():
                if 'Dropout' in str(type(l)):
                    assert len(cache) > 0, 'cache is empty'
                    l.p = cache.pop(0)

    m.train()
    cache = set_dropout()
    train_iter = get_iterator(True)
    for _, (x,y) in enumerate(train_iter):
        x = Variable(x.view(-1,1,28,28).float() / 255.0, volatile=True)
        if opt['cuda']:
            x = x.cuda(async=True)
        m(x)
    set_dropout(cache)

def validate(e):
    m = deepcopy(model)
    for p,q in zip(m.parameters(), model.parameters()):
        tmp = th.from_numpy(opt['state']['x'][q])
        if opt['cuda']:
            tmp = tmp.cuda()
        p.data.copy_(tmp)

    dry_feed(m)
    m.eval()

    val_iter = get_iterator(False)

    loss = tnt.meter.AverageValueMeter()
    top1 = tnt.meter.ClassErrorMeter()

    for b, (x,y) in enumerate(val_iter):
        x = Variable(x.view(-1,1,28,28).float() / 255.0)
        y = Variable(th.LongTensor(y))
        if opt['cuda']:
            x, y = x.cuda(async=True), y.cuda(async=True)

        yh = m(x)
        f = criterion(yh, y)

        top1.add(yh.data, y.data)
        loss.add(f.data[0])

    r = dict(f=loss.value()[0], top1=top1.value()[0])
    print('*[%02d] %.3f %.3f%%'%(e, r['f'], r['top1']))
    return r

for e in range(opt['B']):
    if opt['r'] == 0:
        print()

    r = train(e)
    comm.Barrier()

    if opt['r'] == 0:
        validate(e)
    comm.Barrier()

    opt['lr'] /= 10.0
