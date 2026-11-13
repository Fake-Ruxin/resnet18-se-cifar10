# 基于 SE 注意力机制的 ResNet-18 图像分类

本项目使用 PyTorch 和 CIFAR-10，对比从头训练的 ResNet-18 与加入 SE 通道注意力的 ResNet-18。输入为 32×32 图像；原始训练集以随机种子 42 划分为 45,000 张训练图像与 5,000 张验证图像，原始测试集有 10,000 张图像。两组均按验证集准确率选择最佳权重。

## 实验结果

| 指标 | ResNet-18 | ResNet-18 + SE |
| --- | ---: | ---: |
| 最佳验证集准确率 | 94.18% | 94.40% |
| 测试集准确率 | 93.40% | 93.56% |
| 参数量 | 11,173,962 | 11,263,042 |
| Conv2d 和 Linear 的 MACs/张 | 555,422,720 | 555,509,760 |

上述为单个随机种子的实测结果。`results/` 中保存了原始训练配置、逐轮指标、测试指标、混淆矩阵和训练曲线。MACs 只包括卷积和全连接层，不包含归一化、激活、池化和逐元素运算。SE 的效果及推理耗时会随实验条件变化。

## 环境配置与数据准备

实验环境为 Python 3.12、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128 和 NVIDIA RTX 4060 Ti。使用 [uv](https://docs.astral.sh/uv/) 安装依赖；需要支持 CUDA 的 NVIDIA 显卡和驱动。

```powershell
uv sync
uv run python data.py
```

首次运行数据脚本时将下载 CIFAR-10。训练数据使用随机裁剪及水平翻转，验证集和测试集不使用随机增强。

## 训练

```powershell
uv run python train.py --model resnet18 --output runs/resnet18
uv run python train.py --model resnet18-se --output runs/resnet18-se
```

默认每组训练 40 轮，批量大小 128，使用 SGD 与余弦学习率调度。程序在各自运行目录保存 `config.json`、`history.csv` 和最佳权重 `best.pt`。为防止覆盖，输出目录非空时程序会报错；可通过 `uv run python train.py --help` 查看参数。

## 测试

需要先运行训练获得 `best.pt`，再执行：

```powershell
uv run python evaluate.py --run runs/resnet18
uv run python evaluate.py --run runs/resnet18-se
```

评估将指标、混淆矩阵和图表保存至对应的 `evaluation/` 目录。目录非空时程序不会覆盖。推理时间以 GPU 单张图片、50 次预热及 300 次测量统计，不含数据读取及 CPU 到 GPU 的传输。

本仓库不包含数据集、虚拟环境或模型权重；复现测试需要先下载数据并重新训练。`results/` 是已完成实验的记录，不会被上述训练命令覆盖。
