import torch
import numpy as np
import torch.nn as nn

import torch
import torch.nn as nn

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

        # 初始化权重和偏置
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x

# 创建输入数据
x = torch.randn(4, 768, 448)
input_dim=768 * 448
# 创建 MLP 模型实例
mlp_model = MLP(input_dim=input_dim, hidden_dim=256, num_classes=200)

# 运行输入数据通过 MLP 模型
output = mlp_model(x)

print(output.shape)
