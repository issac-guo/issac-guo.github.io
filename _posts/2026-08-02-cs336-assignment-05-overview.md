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
