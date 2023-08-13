
import pickle
import glob
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import data_aug.loader
import torch
import math
import time
import shutil
import torch.nn as nn
from datasets import ClusterDataset
from utils_cluster import matmul_cluster_similarity, final_cluster_similarity
import os
from models import cluster_projector
import csv
import numpy as np
import torch.nn.functional as F

import torch.cuda.amp as amp


normalize = transforms.Normalize(mean=[0.7183, 0.5119, 0.6454],
                                 std=[0.2064, 0.2634, 0.2073]) #TCGA-LUNG

normalize = transforms.Normalize(mean=[0.6684, 0.5115, 0.6791],
                                 std=[0.2521, 0.2875, 0.2100]) #Camelyon16_20x

#normalize = transforms.Normalize(mean=[0.7878, 0.6257, 0.7435],
#                                 std=[0.1300, 0.1761, 0.1292]) #BACH_20x

#normalize = transforms.Normalize(mean=[0.7177, 0.5095, 0.6424],
#                                 std=[0.1965, 0.2577, 0.2026]) #TCGA_5x

#normalize = transforms.Normalize(mean=[0.7823, 0.5993, 0.7094],
#                                 std=[0.1904, 0.2577, 0.2124]) #PC_20x

augmentation = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([data_aug.loader.GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
]

#mag_level = '10x'
mag_level = '20x'

#train_dir = Path(f'/pool3/users/weiyi/data/PC_{mag_level}patch')
train_dir = Path(f'/tank/local/cgo5577/Camelyon_{mag_level}patch/train')
#train_dir = Path(f'/pool3/users/weiyi/data/PC_{mag_level}patch')
train_dir_temp = train_dir.joinpath('*', '*', '*.png')
patch_list = glob.glob(str(train_dir_temp)) # /class_name/bag_name/*.jpeg
train_dataset = ClusterDataset.PathData(patch_list,data_aug.loader.TwoCropsTransform(transforms.Compose(augmentation)))


distributed = False
train_sampler = None
gpu = 0

num_clusters = 8
eps = 1e-8 #eps for mat multiply/ hard softmax and L1 normalization
feat_size = 2048
batch_size = 512
num_worker = 8
start_epoch = 0
epochs = 50
fix_pred_lr = True
lr = 0.05
#lr = 1e-4
weight_decay = 1e-4
#weight_decay = 1e-5
momentum = 0.9
fp16 = False
#resume = '../Camelyon_ckpt/c8_b512_njsimsiam_20x_checkpoint_39.pth.tar'
resume = False
print_freq = 10
save_freq = 10
base_model = 'resnet18'
save_path = f'./Camelyon_ckpt/c{num_clusters}_b{batch_size}_Camelyon_{mag_level}_external_gumbel'
log_dst = f'./Camelyon_ckpt/c{num_clusters}_b{batch_size}_Camelyon_{mag_level}_external_gumbel'


model = cluster_projector.SimSiam_Cluster_Projector(base_model,out_dim=feat_size,target_dim=num_clusters)
#model.cluster_projector = nn.Identity()
model.cuda()



init_lr = lr * batch_size / 256

# define loss function (criterion) and optimizer
criterion = nn.CosineSimilarity(dim=1).cuda()

if fix_pred_lr:
    optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                    {'params': model.cluster_projector.parameters(), 'fix_lr': False},
                    {'params': model.cluster_assigner.parameters(), 'fix_lr': False},
                    {'params': model.predictor.parameters(), 'fix_lr': True}]
else:
    optim_params = model.parameters()

optimizer = torch.optim.SGD(optim_params, init_lr, momentum=momentum, 
                            weight_decay=weight_decay)




if resume:
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        if gpu is None:
            checkpoint = torch.load(resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(resume, map_location=loc)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, log_dst=None, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_dst = log_dst

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def write_log(self, batch):
        with open(self.log_dst+'.csv','a',newline='') as csvfile:
            writer = csv.writer(csvfile)
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [meter.avg for meter in self.meters]
            writer.writerow(entries)
            csvfile.close()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'






train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
    num_workers=num_worker, pin_memory=True, drop_last=True)

def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    contras_losses = AverageMeter('Contras',':.4f')
    cluster_losses = AverageMeter('Cluster',':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, contras_losses, cluster_losses],
        prefix="Epoch: [{}]".format(epoch), log_dst=log_dst)

    # switch to train mode
    model.train()

    end = time.time()
    for i, images in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if gpu is not None:
            images[0] = images[0].cuda(gpu, non_blocking=True)
            images[1] = images[1].cuda(gpu, non_blocking=True)

        # compute output and loss
        if False:
            with amp.autocast():
                z, p ,c , a = model(x1=images[0], x2=images[1])
                contrastive_loss = -(criterion(p[0], z[1].detach()).mean() + criterion(p[1], z[0].detach()).mean()) * 0.5  #SimSiam
                inner_similarity, outer_similarity = final_cluster_similarity(torch.cat((z[0].detach(),z[1].detach()),0),torch.cat((a[0],a[1]),0),num_clusters,eps=eps)
                #inner_similarity, outer_similarity = matmul_cluster_similarity(torch.cat((z[0],z[1]),0),torch.cat((a[0],a[1]),0),num_clusters,eps=eps)
                #print(f'{inner_similarity} {outer_similarity}')
                cluster_loss = -(inner_similarity - outer_similarity) * 0.5 #so range from [-1,1]
                #avg_inner, avg_outer = similarity_inter(torch.cat((z[0].detach(),z[1].detach()),0),torch.cat((a[0],a[1]),0),num_clusters,eps=eps)
                #print(f'center: {avg_inner[0]} {avg_outer[0]} every: {avg_inner[1]} {avg_outer[1]} prop: {avg_inner[0]-avg_outer[0]} {avg_inner[1]-avg_outer[1]} {avg_inner[0]-avg_outer[1]}')
                #cluster_loss.register_hook(lambda grad: grad.clamp(min=1e-7))
                #cluster_loss.register_hook(lambda grad: grad.clamp(max=1e7))
                loss =  contrastive_loss + cluster_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            z, p ,c , a = model(x1=images[0], x2=images[1])
            contrastive_loss = -(criterion(p[0], z[1].detach()).mean() + criterion(p[1], z[0].detach()).mean()) * 0.5  #SimSiam
            inner_similarity, outer_similarity = final_cluster_similarity(torch.cat((z[0].detach(),z[1].detach()),0),torch.cat((a[0],a[1]),0),num_clusters,eps=eps)
            cluster_loss = -(-outer_similarity)*0.5
            norm = torch.linalg.norm(torch.cat((a[0],a[1]),0))
            loss = cluster_loss + contrastive_loss# + norm_loss
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            #print(torch.linalg.norm(model.cluster_assigner[-1].weight.grad))
            #print(torch.linalg.norm(model.predictor[-1].weight.grad),torch.linalg.norm(model.cluster_assigner[-1].weight.grad))
            optimizer.step()


        contras_losses.update(contrastive_loss.item(),images[0].size(0))
        cluster_losses.update(cluster_loss.item(),images[0].size(0))
        losses.update(loss.item(), images[0].size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            progress.display(i)
            online_counter = Counter(np.argmax(torch.cat((a[0],a[1]),0).detach().cpu().numpy(),axis=1))
            gumbel_valids = len(Counter(np.argmax(F.gumbel_softmax(torch.cat((a[0],a[1]),0), hard=False).detach().cpu().numpy(),axis=1)))
            
            if len(online_counter) > 12:
                print(len(online_counter),gumbel_valids)
            else:
                print(len(online_counter), gumbel_valids, online_counter)
            print(f'inner:{inner_similarity}, outer:{outer_similarity} norm:{norm}')
    progress.write_log(i)

#Mixed precision
scaler = amp.GradScaler()
torch.autograd.set_detect_anomaly(True)
for epoch in range(start_epoch, epochs):
    if distributed:
        train_sampler.set_epoch(epoch)
    adjust_learning_rate(optimizer, init_lr, epoch, epochs)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch)

    if (epoch+1) % save_freq == 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best=False, filename='{}_{}.pth.tar'.format(save_path,epoch))

save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict(),
}, is_best=False, filename='{}_last.pth.tar'.format(save_path))


