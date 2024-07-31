import math
import torch.nn as nn
from torch.nn import functional as F
from model.config import *

class Attention(nn.Module):
  def __init__(self):
    super().__init__()
    # 定义三个线性层，用于计算query、key和value
    self.wq = nn.Linear(d_model, head_size, bias=False)
    self.wk = nn.Linear(d_model, head_size, bias=False)
    self.wv = nn.Linear(d_model, head_size, bias=False)
    # 定义dropout层，用于防止过拟合
    self.dropout_layer = nn.Dropout(dropout)
    # 创建一个下三角矩阵，用于在后面的mask操作中使用
    self.register_buffer('tril', torch.tril(torch.ones((context_length, context_length))))
  
  def forward(self, x):
    # 获取输入的第二维度的大小
    T = x.shape[1]
    # 计算query、key和value
    q = self.wq(x)
    k = self.wk(x)
    v = self.wv(x)
    # 计算权重，这里使用了缩放点积注意力机制
    weights = q @ k.transpose(-2, -1) / math.sqrt(head_size)
    # 使用mask操作，将不需要的部分设为负无穷
    weights.masked_fill(self.tril[:T, :T] == 0, -float('inf'))
    # 对权重进行softmax操作，使其在最后一个维度上的和为1
    weights = F.softmax(weights, dim=-1)
    # 对权重使用dropout
    weights = self.dropout_layer(weights)
    # 计算输出
    out = weights @ v
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self):
    super().__init__()
    # 定义一个线性层，用于在多头注意力后对结果进行处理
    self.wo = nn.Linear(d_model, d_model, bias=False)
    # 定义一个模块列表，其中包含了num_heads个Attention模块
    self.attentions = nn.ModuleList([Attention() for _ in range(num_heads)])
    # 定义dropout层，用于防止过拟合
    self.dropout_layer = nn.Dropout(dropout)
  
  def forward(self, x):
    # 对每个Attention模块的输出进行拼接
    out = torch.cat([h(x) for h in self.attentions], -1)
    # 通过线性层进行处理
    out = self.wo(out)
    # 使用dropout
    out = self.dropout_layer(out)
    return out
