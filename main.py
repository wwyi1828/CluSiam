import argparse
import glob
from pathlib import Path
from collections import Counter
import torchvision.transforms as transforms
import data_aug.loader
import torch
import math
import time
import shutil
import torch.nn as nn
from datasets import ClusterDataset
from utils import *
import os
from models import cluster_projector
import csv
import numpy as np
import torch.nn.functional as F


parser = argparse.ArgumentParser(description="PyTorch Pre-training")

# Model and training settings
parser.add_argument("--model_type", default="clusiam", choices=["clusiam", "clubyol"], help="Model type to train")
parser.add_argument("--num_clusters", default=100, type=int, help="Number of clusters")
parser.add_argument("--feat_size", default=2048, type=int, help="Feature size")
parser.add_argument("--start_epoch", default=0, type=int, help="Starting epoch for training")
parser.add_argument("--epochs", default=50, type=int, help="Total number of epochs for training")
parser.add_argument("--alpha", default=0.5, type=float, help="Alpha value for loss calculation")

# Optimizer settings
parser.add_argument("--lr", default=0.05, type=float, help="Learning rate")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay for optimizer")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for optimizer")
parser.add_argument("--fix_pred_lr", action="store_true", type=bool, help="Fix predictor learning rate")

# Data loading settings
parser.add_argument("--batch_size", default=512, type=int, help="Batch size for training")
parser.add_argument("--num_worker", default=8, type=int, help="Number of workers for data loading")
parser.add_argument("--eps", default=1e-8, type=float, help="Epsilon value for calculations")

# Logging and saving
parser.add_argument("--print_freq", default=10, type=int, help="Print frequency")
parser.add_argument("--save_freq", default=10, type=int, help="Save frequency")
parser.add_argument("--save_path", default=None, type=str, help="Path to save the model")
parser.add_argument("--log_dst", default=None, type=str, help="Destination for logs")
parser.add_argument("--resume", default=False, type=bool, help="Resume training from checkpoint")

parser.add_argument("train_path", metavar="DIR", help="path to dataset")



def main():


    args = parser.parse_args()
    train_dir = Path(args.train_path)

    # Constants
    CAMELYON_NORMALIZATION_MEAN = [0.6684, 0.5115, 0.6791]
    CAMELYON_NORMALIZATION_STD = [0.2521, 0.2875, 0.2100]

    # Data augmentation and normalization
    normalize = transforms.Normalize(mean=CAMELYON_NORMALIZATION_MEAN, std=CAMELYON_NORMALIZATION_STD)
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([data_aug.loader.GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    # Dataset preparation
    train_dir_temp = train_dir.joinpath("*", "*", "*.png")
    patch_list = glob.glob(str(train_dir_temp))
    train_dataset = ClusterDataset.PathData(patch_list, data_aug.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    # Model initialization
    if args.model_type == "clusiam":
        model = cluster_projector.Cluster_SimSiam(out_dim=args.feat_size, target_dim=args.num_clusters)
    elif args.model_type == "clubyol":
        model = cluster_projector.Cluster_BYOL(out_dim=args.feat_size, hidden_dim=4096, target_dim=args.num_clusters)
        args.fix_pred_lr = False

    model.cuda()

    # Optimizer settings
    init_lr = args.lr * args.batch_size / 256
    criterion = nn.CosineSimilarity(dim=1).cuda()


    def save_checkpoint(state, filename="checkpoint.pth.tar"):
        torch.save(state, filename)

    class AverageMeter(object):
        """Computes and stores the average and current value"""

        def __init__(self, name, fmt=":f"):
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
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
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
            print("\t".join(entries))

        def write_log(self, batch):
            with open(self.log_dst + ".csv", "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                entries = [self.prefix + self.batch_fmtstr.format(batch)]
                entries += [meter.avg for meter in self.meters]
                writer.writerow(entries)
                csvfile.close()

        def _get_batch_fmtstr(self, num_batches):
            num_digits = len(str(num_batches // 1))
            fmt = "{:" + str(num_digits) + "d}"
            return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    # Optimizer parameters based on fix_pred_lr setting
    if args.fix_pred_lr:
        optim_params = [
            {"params": model.encoder.parameters(), "fix_lr": False},
            {"params": model.cluster_projector.parameters(), "fix_lr": False},
            {"params": model.cluster_assigner.parameters(), "fix_lr": False},
            {"params": model.predictor.parameters(), "fix_lr": True},
        ]
    else:
        optim_params = model.parameters()

    # Initialize optimizer
    optimizer = torch.optim.SGD(
        optim_params, init_lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # Checkpoint loading logic
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # Initialize data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=True,
        drop_last=True,
    )


    def train(train_loader, model, criterion, optimizer, epoch):
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4f")
        contras_losses = AverageMeter("Contras", ":.4f")
        cluster_losses = AverageMeter("Cluster", ":.4f")
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, contras_losses, cluster_losses],
            prefix="Epoch: [{}]".format(epoch),
            log_dst=args.log_dst,
        )

        # switch to train mode
        model.train()

        end = time.time()
        for i, images in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            images[0] = images[0].cuda(non_blocking=True)
            images[1] = images[1].cuda(non_blocking=True)

            if args.model_type == "clusiam":
                z, p, c, a = model(x1=images[0], x2=images[1])
                contrastive_loss = -(criterion(p[0], z[1].detach()).mean() + criterion(p[1], z[0].detach()).mean())# SimSiam
                contrastive_loss *= (1 - args.alpha)
                outer_similarity = final_cluster_similarity(torch.cat((z[0].detach(), z[1].detach()), 0), torch.cat((a[0], a[1]), 0), args.num_clusters, eps=args.eps)
                cluster_loss = -(-outer_similarity) * args.alpha

            elif args.model_type == "clubyol":
                p, t, z, a = model(x1=images[0], x2=images[1])
                p1, p2 = p
                t1, t2 = t
                contrastive_loss = (1- 1 * (F.cosine_similarity(p1, t2.detach(), dim=-1)).mean() + 1 - 1 * (F.cosine_similarity(p2, t1.detach(), dim=-1)).mean())# BYOL
                contrastive_loss *= (1 - args.alpha)
                outer_similarity = final_cluster_similarity(torch.cat((z[0].detach(), z[1].detach()), 0), torch.cat((a[0], a[1]), 0), args.num_clusters, eps=args.eps)
                cluster_loss = -(-outer_similarity) * args.alpha

            loss = cluster_loss + contrastive_loss
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.model_type == "clubyol":
                model.momentum_update()

            contras_losses.update(contrastive_loss.item(), images[0].size(0))
            cluster_losses.update(cluster_loss.item(), images[0].size(0))
            losses.update(loss.item(), images[0].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
                
                # Calculate online counter
                concatenated_assignments = torch.cat((a[0], a[1]), 0).detach().cpu().numpy()
                online_counter = Counter(np.argmax(concatenated_assignments, axis=1))
                
                # Calculate gumbel valids
                gumbel_softmax_values = F.gumbel_softmax(concatenated_assignments, hard=False).detach().cpu().numpy()
                gumbel_valids = len(Counter(np.argmax(gumbel_softmax_values, axis=1)))
                
                # Print statistics
                if len(online_counter) > 12:
                    print(len(online_counter), gumbel_valids)
                else:
                    print(len(online_counter), gumbel_valids, online_counter)
        progress.write_log(i)

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args.epochs)

        train(train_loader, model, criterion, optimizer, epoch)

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                filename="{}_{}.pth.tar".format(args.save_path, epoch),
            )

    save_checkpoint(
        {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        filename="{}_last.pth.tar".format(args.save_path),
    )


if __name__ == "__main__":
    main()
