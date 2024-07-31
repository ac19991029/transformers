import torch.nn as nn
from model.config import *

class FeedForwardNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    # 定义一个dropout层，用于在训练过程中随机关闭一部分神经元，防止过拟合
    self.dropout_layer = nn.Dropout(dropout)
    # 定义一个前馈神经网络，包含两个线性层和一个 ReLU 激活函数
    self.ffn = nn.Sequential(
      # 第一个线性层，输入特征数为d_model，输出特征数为d_model * 4
      nn.Linear(in_features=d_model, out_features=d_model * 4, bias=False),
      # ReLU激活函数，将所有负值置 0
      nn.ReLU(),
      # 第二个线性层，输入特征数为 d_model * 4，输出特征数为 d_model
      nn.Linear(in_features=d_model * 4, out_features=d_model, bias=False),
      # 再次使用 dropout 防止过拟合
      nn.Dropout(dropout),
    )

  # 定义前向传播函数
  def forward(self, x):
    # 将输入x通过前馈神经网络，并返回结果
    return self.ffn(x);
