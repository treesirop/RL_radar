# 基于多模型集成的雷达回波预测系统

## 项目概述

本项目实现了一个基于强化学习的多模型集成预测系统，用于雷达回波的时空预测。系统包含4个不同损失函数训练的预测模型和一个智能体调度器，能够在35dBz阈值下优化CSI、POD和FAR指标。

## 系统设计

1. **多模型训练**：使用4个不同的损失函数训练4个模型
   - RegLoss30: 侧重30分钟的预测
   - RegLoss60: 侧重60分钟的预测
   - RegLoss90: 侧重90分钟的预测
   - RegLoss120: 侧重120分钟的预测

2. **智能体训练**：训练一个强化学习智能体，学习如何为每个时间步选择最佳模型

3. **模型评估**：使用以下指标评估系统性能
   - CSI (Critical Success Index): 评估预测的整体准确性
   - FAR (False Alarm Rate): 评估误报率
   - POD (Probability of Detection): 评估检测率

## 使用方法

1. **训练模型**
```python
python main_new.py --train
```

2. **评估系统**
```python
python main_new.py --eval
```

3. **训练智能体**
```python
python main_new.py --train_agent
```

## 文件说明

- `model.py`: 基础预测模型实现
- `loss.py`: 不同时间尺度的损失函数
- `agent.py`: 强化学习智能体实现
- `train_utils.py`: 训练和评估工具函数
- `radar_environment.py`: 强化学习环境
- `dataset.py`: 数据加载和处理
- `config.py`: 配置文件

## 评估结果

评估结果会输出到`result.txt`文件，包含:
- Agent调度性能的CSI/POD/FAR指标
- 各单独模型的CSI/POD/FAR指标

所有评估均在35dBz阈值下进行。