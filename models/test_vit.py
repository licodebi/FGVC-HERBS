import torch
import timm
from typing import Union
import math
from scipy import ndimage
from torchvision.models.feature_extraction import create_feature_extractor

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
img_size=384
return_nodes = {
            'blocks.8': 'layer1',
            'blocks.9': 'layer2',
            'blocks.10': 'layer3',
            'blocks.11': 'layer4',
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
print("位置编码最后的大小:",posemb.shape)
# 将位置嵌入作为模型的参数，并赋值给 backbone.pos_embed
backbone.pos_embed = torch.nn.Parameter(posemb)
backbone=create_feature_extractor(backbone, return_nodes=return_nodes)
rand_in = torch.randn(1, 3, img_size, img_size)
outs = backbone(rand_in)
# torch.Size([1, 577, 768])
# 其中图片torch.Size([1, 576, 768])
for name in outs:
    print(outs[name].size())
name=name