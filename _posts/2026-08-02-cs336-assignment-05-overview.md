---
layout: post
title: "Assignment 05: Chapter 1, 2, 3"
date: 2026-08-02 16:20:00 +0800
categories: [Stanford CS336]
tags: [llm, reinforcement-learning, grpo, prompting, cs336]
description: "Assignment 05 Chapter 1, 2, 3 中文笔记。"
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

## 2.3 符号说明

本次作业会涉及一些数学内容，因此在下面的表格中，我们列出了之后会用到的一些符号。语言建模和强化学习经常会用不同术语来指代同一个对象，所以表格中同时包含了这两类术语。如果你忘记某个符号指的是什么，可以随时回到这张表查看。

| 符号 | 语言模型术语 | 强化学习术语 | 含义 |
| --- | --- | --- | --- |
| $\rho$ | 提示分布或数据集 | 初始状态分布 | 提示/问题上的分布。 |
| $x$ | 提示；问题 | 初始状态 | 从 $\rho$ 中采样得到的一个提示/问题。 |
| $y$ | 回复；补全；生成；样本 | rollout；轨迹；采样得到的动作序列 | 针对提示 $x$ 采样得到的一个答案。 |
| $y_t$ | token | 动作 | $y$ 中第 $t$ 个生成的 token。 |
| $y_{<t}$ | 前缀 |  | 位置 $t$ 之前的所有已生成 token，即 $y_1, \ldots, y_{t-1}$。 |
| $\pi_\theta$ | 模型 | 策略 | 参数为 $\theta$ 的模型；给定提示 $x$ 时，它会为回复 $y$ 分配概率 $\pi_\theta(y \mid x)$。 |
| $\pi_\theta(y_t \mid x, y_{<t})$ | 下一 token 分布 | 时间步 $t$ 的策略 | 在给定提示和之前已生成 token 的条件下，token $y_t$ 的条件概率。 |
| $r(y \mid x)$ |  | 奖励 | 一个标量分数，用来表示采样回复的正确性（在本次作业中为 0 或 1）。 |
| $B$ | 每个 batch 中的提示数量 |  | 每个推理 batch 中的提示数量。 |
| $G$ | 每个提示的生成数量 | 组大小 | 对每个提示采样得到的回复数量。 |
| $\mathrm{len}(y)$ 或 $L$ | 回复长度 | horizon | 一个回复中生成 token 的数量。 |
| $A^{(i,j)}$ |  | advantage | 在经过 baseline 和归一化之后，分配给第 $i$ 个提示下第 $j$ 个回复的权重。 |
