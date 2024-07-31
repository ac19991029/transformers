import torch.nn as nn 
from model.attention import MultiHeadAttention
from model.config import *
from model.ffn import FeedForwardNetwork 

class TransformBlock(nn.Module):
    def __init__(self):
      """
      Transformer Block 层需要包含以下几个部分:
      1. Layer normalization
      2. Multi-head attention
      3. Residual connection  
      4. Layer normalization
      5. Feed forward network
      6. Residual connection
      7. Layer normalization
      8. Dropout
      9. Linear layer
      10. Softmax
      """
      super().__init__()
      self.layer_norm_1 = nn.LayerNorm(d_model)  # 第一个 Layer Norm
      self.layer_norm_2 = nn.LayerNorm(d_model)  # 第二个 Layer Norm
      self.multi_attention_layer = MultiHeadAttention()  # 多头注意力层
      self.feed_forward_layer = FeedForwardNetwork()  # 前馈神经网络层

    def forward(self, x):
      """
      定义前向传播过程
      """
      x = x + self.multi_attention_layer(self.layer_norm_1(x))  # 注意力层后的残差连接
      x = x + self.feed_forward_layer(self.layer_norm_2(x))  # 前馈网络后的残差连接
      return x
