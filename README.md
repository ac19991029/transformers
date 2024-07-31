# transformers
sample code

# Transformer 语言模型实现

## 简介
基于 Pytorch 实现类似 GPT2 架构的 Decoder-only Transformer 语言模型。

## 目录结构
```
.
├── config.py  # 主配置文件，用于设置一些超参数
├── data  # 数据相关的目录
│   ├── data.py  # 数据处理脚本，包括数据加载、预处理等功能
│   └── example.txt  # 训练数据文件
├── model  # 模型相关的目录
│   ├── attention.py  # 注意力机制模块
│   ├── transformer_block.py  # Transformer Block 模块
│   ├── ffn.py  # 前馈神经网络模块
│   ├── config.py  # 模型配置文件，用于设置模型参数
│   ├── tokenizer.py  # 分词器模块
│   └── model.py  # 主模型文件，定义了模型的架构
├── scripts
│   ├── train.py  # 训练脚本，用于模型的训练
│   ├── parameters.py  # 参数查看脚本，用于查看模型参数组成
│   └── inference.py  # 推理脚本，用于模型的推理
├── README.md
└── requirements.txt  # 项目依赖文件，列出了项目需要的所有 Python 库
```


## 开始

### 前置环境

Python 3.7+

### 安装依赖
```
pip install -r requirements.txt
```

## 模型训练
你可以运行如下的命令来训练模型：
```
python -m scripts.train
```
如果你想要调整模型训练的一些超参数，可以去 model/config.py 进行修改。

## 模型推理
```
python -m scripts.inference --input="hello"
```
你可以通过 input 参数来指定输入文本。

## 查看模型参数组成
如果你想要查看模型参数组成，你可以运行以下命令：
```
python -m scripts.parameters
```
这个命令会打印模型的全部参数数量，以及各个权重矩阵的大小信息。