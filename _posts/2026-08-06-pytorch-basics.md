---
layout: post
title: "Pytorch学习基础"
date: 2026-08-06 17:20:00 +0800
categories: [Stanford CS336]
tags: [llm, pytorch, cs336, assignment-01]
description: "Stanford CS336 Assignment 01 的 PyTorch 基础学习笔记。"
math: true
---

# Pytorch学习基础

这篇笔记用于补齐 Stanford CS336 Assignment 01 之前需要掌握的 PyTorch 基础。Assignment 01 会从 tokenizer、Transformer、优化器到训练循环逐步实现一个语言模型，因此我们需要先熟悉张量操作、自动求导、模块定义和训练流程。

## 1 张量基础

PyTorch 的核心数据结构是 `torch.Tensor`。它既可以表示普通数组，也可以表示模型参数、输入 batch、loss 和梯度。

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.zeros((2, 3))
z = torch.randn((4, 8))

print(x.shape)
print(y.dtype)
print(z.device)
```

常用操作包括：

- `reshape` / `view`：改变张量形状
- `transpose` / `permute`：交换维度
- `matmul` / `@`：矩阵乘法
- `sum` / `mean`：聚合
- `softmax`：把 logits 转成概率分布

## 2 自动求导

如果一个张量设置了 `requires_grad=True`，PyTorch 会记录它参与的计算图。调用 `backward()` 后，相关参数的梯度会保存在 `.grad` 中。

```python
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)

loss = (w * x - 10) ** 2
loss.backward()

print(w.grad)
```

在训练语言模型时，典型流程是：

1. 前向传播得到 logits。
2. 计算 cross entropy loss。
3. 调用 `loss.backward()` 计算梯度。
4. 使用 optimizer 更新参数。
5. 清空梯度，进入下一个 batch。

## 3 定义模型

PyTorch 模型通常继承 `torch.nn.Module`，在 `__init__` 中定义层，在 `forward` 中描述计算过程。

```python
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

对 CS336 Assignment 01 来说，后面会把这个模式扩展到 embedding、attention、MLP、layer norm 和完整 Transformer language model。

## 4 训练循环

一个最小训练循环通常长这样：

```python
model = TinyMLP(d_in=8, d_hidden=32, d_out=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for step in range(100):
    x = torch.randn(16, 8)
    target = torch.randint(0, 4, (16,))

    logits = model(x)
    loss = loss_fn(logits, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(step, loss.item())
```

这里最重要的是理解 `zero_grad -> backward -> step` 的顺序。梯度默认会累积，因此每次更新前都需要清空上一次的梯度。

## 5 和 Assignment 01 的对应关系

在 Assignment 01 中，PyTorch 会主要出现在以下位置：

- 用 `torch.Tensor` 表示 token ids、embedding 和 attention scores。
- 用 `nn.Module` 组织 Transformer 的各个组件。
- 用 `autograd` 自动计算模型参数的梯度。
- 用 `AdamW` 或自己实现的 optimizer 更新参数。
- 用训练循环把 tokenizer、dataset、model、loss 和 optimizer 串起来。

掌握这些基础之后，再去实现 Transformer language model 会顺很多。
