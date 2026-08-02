---
layout: post
title: "Assignment 05: 作业概览"
date: 2026-08-02 16:20:00 +0800
categories: [Stanford CS336]
tags: [llm, reinforcement-learning, grpo, prompting, cs336]
description: "Assignment 05 作业概览中文翻译。"
math: true
---

# 1 作业概览

在这次作业中，你将获得一些实践经验：训练一个语言模型，使其能够进行推理，并解决下游任务。

## 你将实现什么

你将实现以下内容：

1. 零样本、少样本以及思维链提示。
2. 组相对策略优化（Group Relative Policy Optimization, GRPO）：一种强化学习算法，用于在给定外部奖励的情况下提升模型性能。
3. 策略梯度估计的不同变体，用于探索方差降低以及重要性权重裁剪策略。

## 你将运行什么

你将运行以下实验：

1. 在 GSM8K 上测量 OLMo-2-0425-1B 的提示性能。
2. 在 OLMo-2-0425-1B 上运行 on-policy GRPO，以提升其在 GSM8K 上的性能。
3. 运行强化学习变体，包括 RFT、Dr. GRPO 和 MaxRL，以探索强化学习中的算法选择。
4. 运行 off-policy GRPO，以加快训练速度，并探索不同的裁剪策略。

## 2.2 模型和数据集

在本次作业中，我们将使用 OLMo-2-0425-1B，这是一个基础模型，先在 OLMo-mix-1124 上进行了预训练，又在 Dolmino-mix-1124 上进行了中期训练，总共使用了 4 万亿个 token。OLMo-mix-1124 主要由 DCLM-Baseline（J. Li et al., 2024）组成，这是我们在课堂上介绍过的数据集；而 Dolmino-mix-1124 是一个更加聚焦的数据混合体，大约包含 50% 的 DCLM 数据，以及 50% 的指令跟随、数学、代码、STEM 论文和 wiki 数据。关于 OLMo-2-0425-1B 的其他细节，可以在 OLMo 2 技术报告（T. OLMo et al., 2024）中找到。到课程的这个阶段，你应该已经具备理解它们每一个设计决策所需的全部背景知识。

作为下游任务，我们将使用 GSM8K 数据集（K. Cobbe et al., 2021）。该数据集可以在作业仓库中的 `data/gsm8k/train.jsonl` 和 `data/gsm8k/test.jsonl` 找到，也可以在线获取。这个数据集包含一组相对简单的小学数学推理文字题。下面是一个示例：

```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
```

在强化学习期间，我们的模型将学习为这类问题生成推理链，从而提升其解决数学问题的能力。
