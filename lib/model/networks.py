import torch.nn as nn
import torchvision.models as models
import torch
import pdb




class Basemodel(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet18
    """
    def __init__(self, backbone):
        super(Basemodel, self).__init__()
        if backbone == 'x50':
            self.oct_branch = models.resnext50_32x4d(pretrained=True)
        if backbone == '34':
            self.oct_branch = models.resnet34(pretrained=True)

        self.decision_branch = nn.Linear(2048 + 192, 52) # ResNet34 use basic block, expansion = 1

        # replace first conv layer in oct_branch
        self.oct_branch.conv1 = nn.Conv2d(256, 64,
                                          kernel_size=7,
                                          stride=2,
                                          padding=3,
                                          bias=False)  # bias_attr

        self.oct_branch.fc = nn.Sequential()  # remove fc

        self.embeddings = nn.Embedding(8, 64)
        
    def forward(self, oct_img, info):
        oct_embed = self.oct_branch(oct_img)  # ([bs, 512])
        info_embed = self.embeddings(info).reshape(-1, 192) # [bs, 3, 64]
        logit = self.decision_branch(torch.cat([oct_embed, info_embed], 1))

        return logit
