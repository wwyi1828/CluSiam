import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
import glob
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from datasets.ClusterDataset import RawImageDataset

class DatasetStatsComputer:
    def __init__(self, train_path, batch_size=1024, num_workers=32):
        self.train_path = train_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([transforms.ToTensor()])

    def compute_mean_std(self, loader):
        mean = 0.
        sum_of_squared_error = 0.
        nb_samples = 0.

        for data in tqdm(loader):
            data = data.to(self.device)
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples

        for data in tqdm(loader):
            data = data.to(self.device)
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            sum_of_squared_error += ((data - mean.unsqueeze(1))**2).sum([0, 2])

        std = torch.sqrt(sum_of_squared_error / (nb_samples * loader.dataset[0].size(1) * loader.dataset[0].size(2)))
        return mean.cpu(), std.cpu()

    def compute(self):
        train_dir = Path(self.train_path)
        train_dir_temp = train_dir.joinpath("*", "*.png")
        patch_list = glob.glob(str(train_dir_temp))

        dataset = RawImageDataset(patch_list, self.transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        mean, std = self.compute_mean_std(loader)
        return mean, std


def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

def final_cluster_similarity(representations, assigns, num_clusters=100, eps=1e-8):

    representations = F.normalize(representations, dim=1, eps=eps)
    soft_assigns = F.gumbel_softmax(assigns, hard=False)
    #soft_assigns = F.softmax(assigns)

    max_index = soft_assigns.max(dim=-1, keepdim=True)[1]
    hard_assigns = torch.zeros_like(soft_assigns, memory_format=torch.legacy_contiguous_format).scatter_(-1, max_index, 1.0)
    hard_assigns = hard_assigns - soft_assigns.detach() + soft_assigns

    center_feats = torch.matmul(hard_assigns.T,representations)
    center_feats /= torch.sum(hard_assigns,dim=0).unsqueeze(1).maximum(torch.tensor(1))
    center_feats = F.normalize(center_feats, dim=1, eps=eps)

    outer_simi = torch.matmul(center_feats,center_feats.T)
    outer_simi = torch.triu(outer_simi,diagonal=1) # 1: zero diagonal elements
    avg_outer = outer_simi.flatten()[torch.nonzero(outer_simi.flatten())].mean()
    avg_outer = torch.nan_to_num(avg_outer,1)

    return avg_outer
