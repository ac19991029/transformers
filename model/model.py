import math 
import torch.nn as nn 
from torch.nn import functional as F  
from model.config import * 
from model.transformer_block import TransformBlock  # 导入前馈神经网络模块

class GPTModel(nn.Module):
  def __init__(self, max_token_value):
    super().__init__()
    # 📢注意，这里的 num_embeddings 一般使用 tokenizer 的词汇表大小，这里为了方便训练，直接使用了一个比较小的值，即训练集文本的词汇表大小
    self.token_embedding_lookup_table = nn.Embedding(num_embeddings=max_token_value, embedding_dim=d_model)  # 初始化词嵌入查找表
    self.transform_blocks = nn.Sequential(*([TransformBlock() for _ in range(num_blocks)] + [nn.LayerNorm(d_model)]))  # 初始化transformer block层
    self.linear_layer = nn.Linear(d_model, max_token_value, 长度不超过上下文长度
    position_encoding_lookup_table = torch.zebias=False)  # 初始化线性层
    
  def forward(self, source, target = None):
    """
    定义前向传播过程
    """
    B, T = source.shape  # 获取输入的batch size和序列长度
    assert T <= context_length  # 确保输入的序列ros(context_length, d_model)  # 初始化位置编码查找表
    position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)  # 生成位置序列
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 计算位置编码的分母项
    position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)  # 计算偶数位置的位置编码
    position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)  # 计算奇数位置的位置编码
    position_embedding = position_encoding_lookup_table[:T, :].to(device)  # 获取位置嵌入
    x = self.token_embedding_lookup_table(source) + position_embedding  # 将词嵌入和位置嵌入相加
    x = self.transform_blocks(x)  # 通过 Transformer Block层
    logits = self.linear_layer(x)  # 通过线性层

    if target is not None:  # 如果有目标值，则计算交叉熵损失
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      target = target.view(B * T)
      loss = F.cross_entropy(logits, target)
    else:
      loss = None
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    """
    定义生成新token的方法
    """
    for _ in range(max_new_tokens):  # 对于每一个新的 token
      idx_ctx = idx[:, -context_length:]  # 获取上下文
      logits, loss = self(idx_ctx)  # 通过模型获取 logits
      logits = logits[:, -1, :]  # 取最后一个时间步的 logits
      probs = F.softmax(logits, dim=-1)  # 计算 softmax 概率
      idx_next = torch.multinomial(probs, num_samples=1)  # 根据概率分布采样下一个 token
      idx = torch.cat([idx, idx_next], dim=1)  # 将新的 token 添加到序列中
    return idx
