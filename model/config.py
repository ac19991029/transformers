import torch

# 设置一些超参数
batch_size = 8  # 批处理大小，即每次训练模型时输入的数据量
context_length = 20  # 上下文长度，即模型需要考虑的前后文本的长度
d_model = 128  # 模型的语义维度，即模型内部向量的大小
num_blocks = 2 # Transformer Block 的层数
num_heads = 2  # 多头注意力机制中的头的数量
learning_rate = 3e-4  # 学习率
dropout = 0.1  # Dropout 比率，用于防止过拟合
max_iters = 100  # 最大迭代次数，即模型训练的总次数
eval_iters = 20  # 评估迭代次数，即每训练多少次后进行一次模型效果的评估
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备选择，如果有可用的CUDA设备则使用，否则使用CPU
TORCH_SEED = 1337  # PyTorch的随机种子，用于确保实验的可重复性
head_size = d_model // num_heads  # 计算每个头的大小，即模型维度除以头的数量

torch.manual_seed(TORCH_SEED)  # 设置PyTorch的随机种子
