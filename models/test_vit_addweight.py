import timm
from typing import Union
import math
from scipy import ndimage
from torchvision.models.feature_extraction import create_feature_extractor
import torch
import torch.nn as nn
import numpy as np
from pim_module.SICE import SICE
def load_model_weights_vit(model, model_path):
    ### reference https://github.com/TACJu/TransFG
    ### thanks a lot.
    state = torch.load(model_path, map_location='cpu')
    print("测试:",state.keys())
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state.keys():
            ip = state[key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model
pretrained="./vit_base_patch16_224_in21k_miil.pth"
img_size=448
return_nodes = {
            # 'blocks.8': 'layer1',
            # 'blocks.9': 'layer2',
            # 'blocks.10': 'layer3',
            'blocks.11': 'layer4'
        }

backbone = timm.create_model('vit_base_patch16_224_miil_in21k', pretrained=pretrained)
    ### original pretrained path "./models/vit_base_patch16_224_miil_21k.pth"
    # 如果提供预训练权重的路径，则使用load_model_weights加载权重
if pretrained != "":
    backbone = load_model_weights_vit(backbone, pretrained)

backbone.train()
# 从主干模型中获取位置嵌入
# posemb_tok得到每个图片中cls的位置嵌入 即(B,1,embed_size)
# posemb_grid得到图片的位置嵌入，输入图片大小一致，故均相同 (H*W,embed_size)
posemb_tok, posemb_grid = backbone.pos_embed[:, :1], backbone.pos_embed[0, 1:]
# 将posemb_grid转为numpy数组
posemb_grid = posemb_grid.detach().numpy()
# 计算posemb_grid的尺寸，即图片的patch数,即H或W
gs_old = int(math.sqrt(len(posemb_grid)))
#根据现有的图片大小除以patch得到新的网格尺寸
gs_new = img_size//16
# 将H*W拆分为(H,W,-1)
posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
# 计算缩放因子，用于将网格缩放到新的尺寸
zoom = (gs_new / gs_old, gs_new / gs_old, 1)
# 使用 ndimage.zoom 函数按指定的缩放因子对网格进行缩放
posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
# 重新调整 posemb_grid 的形状（1,H*W,embed_size）
posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
# 转换为 PyTorch 张量
posemb_grid = torch.from_numpy(posemb_grid)
# 将 posemb_tok 和 posemb_grid 沿指定维度进行拼接，得到完整的位置嵌入
posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
# print(posemb.shape)
# 将位置嵌入作为模型的参数，并赋值给 backbone.pos_embed
backbone.pos_embed = torch.nn.Parameter(posemb)
rand_in = torch.randn(1, 3, img_size, img_size)
weights=[]
outs,weight = backbone(rand_in)
weights=weight[-3:]
feature_map=create_feature_extractor(backbone, return_nodes=return_nodes)
outs = feature_map(rand_in)
x=0
for i in range(len(weights)):
    print(x)
    x+=1

# class Mlp(nn.Module):
#     def __init__(self, hidden_size,output_size):
#         super(Mlp, self).__init__()
#         #设置第一个全连接层，维度从config.hidden_size变换为config.transformer["mlp_dim"]
#         self.fc1 = nn.Linear(hidden_size, output_size)
#         # 设置第二个全连接层，维度从config.transformer["mlp_dim"]变换回config.hidden_size
#         # 使用了GELU激活函数
#         self.act_fn = nn.GELU()
#         # 按照指定的config.transformer["dropout_rate"]概率将输入的部分元素置为零
#         self.dropout = nn.Dropout(p=0.5)
#         # 对全连接层的参数进行初始化
#         self._init_weights()
#
#     def _init_weights(self):
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.normal_(self.fc1.bias, std=1e-6)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act_fn(x)
#         x = self.dropout(x)
#         return x
# mlp=Mlp(768,256)
# for name in outs:
#     print(mlp(outs[name]).shape)

# class Part_Attention(nn.Module):
#     def __init__(self):
#         super(Part_Attention, self).__init__()
#
#     # 输入对应的注意力权重张量 (batch_size, num_attention_heads, S, S)数组
#     # 即第9层以后该Transformer的每层的注意力权重张量数组
#     def forward(self, x):
#         # 得到注意力权重张量的数量,即第9层以后的层数
#         # length=4
#         length = len(x)
#         # 得到第十层的注意力权重张量
#         last_map = x[0]
#         print("last_map1的形状:",last_map.shape)
#         # 十层之后的每一层的注意力权重张量均和上一层的注意力权重相乘
#         # (batch_size, num_attention_heads, S, S)
#         for i in range(1, length):
#             last_map = torch.matmul(x[i], last_map)
#         # (batch_size, num_attention_heads,S-1)
#         last_map = last_map[:,:,0,1:]
#         print("last_map2的形状:", last_map.shape)
#         #得到last_map 在第二个维度上的最大值和对应的索引
#         # (batch_size, num_attention_heads)
#         max_value, max_inx = last_map.max(2)
#         print("max_value的形状:", max_value.shape)
#         print("max_inx的形状:", max_inx.shape)
#
#         # 得到B,C
#         B,C = last_map.size(0),last_map.size(1)
#         # 得到patch数
#         patch_num = last_map.size(-1)
#         # 根据patch数得到高
#         H = patch_num ** 0.5
#         H = int(H)
#         # C=注意力头数
#         attention_map = last_map.view(B,C,H,H)
#         print("attention_map的形状:", attention_map.shape)
#
#         # last_map(batch_size, num_attention_heads, S-1)
#         # 最大值索引(batch_size, num_attention_heads)
#         # 最大值(batch_size, num_attention_heads)
#         # 注意力特征图(batch_size,num_attention_heads,H,H)
#         return last_map, max_inx, max_value, attention_map
# class RelativeCoordPredictor(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         N, C, H, W = x.shape
#         mask = torch.sum(x, dim=1)
#         size = H
#         mask = mask.view(N, H * W)
#         thresholds = torch.mean(mask, dim=1, keepdim=True)
#         binary_mask = (mask > thresholds).float()
#         binary_mask = binary_mask.view(N, H, W)
#         masked_x = x * binary_mask.view(N, 1, H, W)
#         masked_x = masked_x.view(N, C, H * W).transpose(1, 2).contiguous()  # (N, S, C)
#         _, reduced_x_max_index = torch.max(torch.mean(masked_x, dim=-1), dim=-1)
#         basic_index = torch.from_numpy(np.array([i for i in range(N)])).cuda().long()
#         basic_label = torch.from_numpy(self.build_basic_label(size)).float()
#         label = basic_label.cuda()
#         label = label.unsqueeze(0).expand((N, H, W, 2)).view(N, H * W, 2)  # (N, S, 2)
#         basic_anchor = label[basic_index, reduced_x_max_index, :].unsqueeze(1)  # (N, 1, 2)
#         # (N, S, 2)
#         relative_coord = label - basic_anchor
#         relative_coord = relative_coord / size
#         relative_dist = torch.sqrt(torch.sum(relative_coord ** 2, dim=-1))  # (N, S)
#         # 计算相对角度，通过调用torch.atan2函数计算相对坐标的反正切值，得到角度值范围在(-pi, pi)，(N, S)
#         relative_angle = torch.atan2(relative_coord[:, :, 1], relative_coord[:, :, 0])  # (N, S) in (-pi, pi)
#         # 将相对角度的值转换到0到1的范围内，通过将角度值除以np.pi，加1，再除以2
#         relative_angle = (relative_angle / np.pi + 1) / 2  # (N, S) in (0, 1)
#         binary_relative_mask = binary_mask.view(N, H * W).cuda()
#         relative_dist = relative_dist * binary_relative_mask
#         relative_angle = relative_angle * binary_relative_mask
#         basic_anchor = basic_anchor.squeeze(1)  # (N, 2)
#         relative_coord_total = torch.cat((relative_dist.unsqueeze(2), relative_angle.unsqueeze(2)), dim=-1)
#         position_weight = torch.mean(masked_x, dim=-1)
#         position_weight = position_weight.unsqueeze(2).cuda()
#         position_weight = torch.matmul(position_weight, position_weight.transpose(1, 2))
#         print("relative_coord_total",relative_coord_total.shape)
#         return relative_coord_total, position_weight
#     def build_basic_label(self, size):
#         basic_label = np.array([[(i, j) for j in range(size)] for i in range(size)])
#         return basic_label
#
# relative=RelativeCoordPredictor()
# part_select = Part_Attention()
# _, part_inx, part_value, a_map = part_select(weights)
# relative(a_map)
