from model.config import *

# 用于加载数据
def load_data():
    with open('data/example.txt', 'r', encoding='utf-8') as f:
        # 读取文件内容
        text = f.read()
    return text

# 用于将数据分割为训练集和验证集
def split_data(tokenized_text):
    # 计算分割点的索引，这里将前 90% 的数据作为训练集，后 10% 的数据作为验证集
    split_idx = int(len(tokenized_text) * 0.9)
    # 获取训练集数据
    train_data = tokenized_text[:split_idx]
    # 获取验证集数据
    val_data = tokenized_text[split_idx:]
    return train_data, val_data
