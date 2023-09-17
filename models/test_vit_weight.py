import torch
import numpy as np
import torch.nn as nn

img_size=224
x = torch.randn(1, img_size,200)

probs = torch.softmax(x, dim=-1)
# (b,class)
sum_probs1 = torch.softmax(probs.mean(1), dim=-1)
sum_probs2 = torch.softmax(x.mean(1), dim=-1)
thresholds=sum_probs2.mean(1)

for bi in range(x.size(0)):
    indices = torch.where(sum_probs2[bi]>thresholds[bi])[0]
    print(indices.shape[0])
