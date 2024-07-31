import torch
from model.config import *
from model.model import GPTModel
from model.tokenizer import tokenize
from data.data import load_data, split_data

# 定义获取批量数据的函数
def get_batch(split: str, train_data, val_data):
    # 根据split参数选择训练数据或验证数据
    data = train_data if split == 'train' else val_data
    # 随机生成索引
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    # 获取输入数据x和目标数据y
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y

# 定义处理损失的函数，使用装饰器使其不需要梯度，可以防止部分不必要的梯度计算
@torch.no_grad()
def handle_loss(model, eval_iters, train_data, val_data):
    # 定义场景列表
    scene = ['train', 'valid']
    # 将模型设置为评估模式
    model.eval()
    output = {}
    # 对于每个场景，计算损失
    for split in scene:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split, train_data, val_data)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        output[split] = losses.mean()
    # 将模型设置回训练模式
    model.train()
    return output

# 定义训练函数
def train():
    # 加载数据
    text = load_data()
    # 对文本进行分词
    tokenized_text, max_token_value = tokenize(text)
    # 划分训练数据和验证数据
    train_data, val_data = split_data(tokenized_text)

    # 初始化模型
    model = GPTModel(max_token_value)
    model = model.to(device)

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 进行训练
    for i in range(max_iters):
        xb, yb = get_batch('train', train_data, val_data)
        logits, loss = model(xb, yb)

        # 重要！每次训练前必须清空梯度缓存，然后反向传播计算当前轮的梯度
        optimizer.zero_grad(set_to_none=True)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        # 每隔一定步数，计算并打印损失
        if (i + 1) % eval_iters == 0 or i == max_iters - 1:
            out = handle_loss(model, eval_iters, train_data, val_data)
            print(f"Step {i+1}: train loss {out['train']:.4f}, val loss {out['valid']:.4f}")
    # 保存模型的状态字典
    torch.save(model.state_dict(), 'model-ckpt.pt')

# 如果直接运行此脚本，则调用train()函数
if __name__ == '__main__':
    train()
