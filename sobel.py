import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x) 
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
        if out.size(0) > 1:
            x = out[1, :]
            save_image(x[0], '/data/out/img1.png')
            save_image(x[1], '/data/out/img2.png')
        else:
            print("Warning: 'out' has only one channel. Skipping save_image.")
        #i1 = x[0].cpu().detach().numpy()
        #i2 = x[1].cpu().detach().numpy()


        
        return out

