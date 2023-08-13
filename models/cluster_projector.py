import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy

class Cluster_SimSiam(nn.Module):
    def __init__(self, out_dim=2048, hidden_dim=512, target_dim=100) -> None:
        super(Cluster_SimSiam, self).__init__()

        self.encoder = models.resnet18(pretrained=False, zero_init_residual=True)
        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Linear(prev_dim,out_dim)
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(out_dim, affine=False)) # output layer  ********
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(out_dim, hidden_dim, bias=False),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(hidden_dim, out_dim)) # output layer

        # build a cluster projector
        self.cluster_projector = nn.Sequential(nn.Linear(out_dim, hidden_dim),
                                               nn.BatchNorm1d(hidden_dim),
                                               nn.ReLU(inplace=True))
        
        # build a cluster assigner
        self.cluster_assigner = nn.Sequential(nn.Linear(hidden_dim,target_dim))


    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        """

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        c1 = self.cluster_projector(z1)
        c2 = self.cluster_projector(z2)

        a1 = self.cluster_assigner(c1)
        a2 = self.cluster_assigner(c2)

        return (z1, z2), (p1, p2), (c1, c2), (a1, a2)



class Cluster_BYOL(nn.Module):
    def __init__(self, out_dim=256, hidden_dim=4096, target_dim=100) -> None:
        super(Cluster_BYOL, self).__init__()
        self.encoder = models.resnet18(pretrained=False, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]

        self.encoder.fc = nn.Linear(prev_dim,out_dim)
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(out_dim, affine=False)) # output layer  ********
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(out_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(hidden_dim, out_dim)) # output layer

        self.target_encoder = copy.deepcopy(self.encoder)
        self.momentum = 0.99
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # build a cluster projector
        self.cluster_projector = nn.Sequential(nn.Linear(out_dim, out_dim),
                                               nn.BatchNorm1d(out_dim),
                                               nn.ReLU(inplace=True))
        
        # build a cluster assigner
        self.cluster_assigner = nn.Sequential(nn.Linear(out_dim,target_dim))


    def momentum_update(self):
        # Perform a momentum update
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)


    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        """

        online_proj_1 = self.encoder(x1)
        online_proj_2 = self.encoder(x2)

        online_pred_1 = self.predictor(online_proj_1)
        online_pred_2 = self.predictor(online_proj_2)

        with torch.no_grad():
            target_proj_1 = self.target_encoder(x1)
            target_proj_2 = self.target_encoder(x2)

        assign_1 = self.cluster_assigner(self.cluster_projector(online_proj_1))
        assign_2 = self.cluster_assigner(self.cluster_projector(online_proj_2))

        return (online_pred_1, online_pred_2), (target_proj_1.detach(), target_proj_2.detach()), (online_proj_1.detach(), online_proj_2.detach()), (assign_1, assign_2) 