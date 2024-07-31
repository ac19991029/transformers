import argparse
from model.config import *
from model.model import GPTModel
from data.data import load_data
from model.tokenizer import tokenize, encoding

def infer(start = "hello"):
    text = load_data()  # 加载数据
    _, max_token_value = tokenize(text)  # 对文本进行分词，并获取最大的token值

    model = GPTModel(max_token_value)  # 初始化模型
    model.load_state_dict(torch.load('model-ckpt.pt'))  # 加载模型状态字典
    model = model.to(device)  # 将模型移动到设备上
    model.eval()  # 将模型设置为评估模式

    x = (torch.tensor(encoding.encode(start), dtype=torch.long, device=device)[None, ...])  # 对输入进行编码并移动到设备上
    res = model.generate(x, max_new_tokens=200)  # 使用模型生成新的tokens
    print(encoding.decode(res[0].tolist()))  # 解码生成的tokens并打印结果
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input string to start the text generation')
    args = parser.parse_args()

    infer(args.input)  # 调用推理函数
