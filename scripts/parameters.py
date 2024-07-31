"""
这个文件用来打印模型的总参数数量和每个参数的详细信息，不属于模型的主要功能
"""
from model.model import GPTModel

# 初始化模型
model = GPTModel(max_token_value=50257)

def print_model_summary(model):
    param_summary = {}
    total_params = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_type = '.'.join(name.split('.')[:-1])  # 获取参数所属层的层级路径
        if param_type not in param_summary:
            param_summary[param_type] = {'count': 0, 'params': 0, 'shapes': set()}
        param_summary[param_type]['count'] += 1
        param_summary[param_type]['params'] += param.numel()
        param_summary[param_type]['shapes'].add(tuple(param.shape))
        total_params += param.numel()

    print(f"{'Layer Path':<70} {'Count':<10} {'Total Params':<15} {'Shapes'}")
    print("="*120)
    for param_type, summary in param_summary.items():
        shapes = ', '.join([str(shape) for shape in summary['shapes']])
        print(f"{param_type:<70} {summary['count']:<10} {summary['params']:<15} {shapes}")
    print("="*90)
    print(f"Total trainable parameters: {total_params}")

# 打印模型摘要
print_model_summary(model)
