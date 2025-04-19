# 天气预报项目

本项目专注于使用深度学习模型进行天气预报。它包含了多种模型的实现，包括 ConvLSTM、CuboidTransformer、ForcastNet、RainFormer 和 SimVP，并使用强化学习智能体来融合这些模型的预测结果。

## 主要文件:

- `CSI.py`: 包含临界成功指数 (CSI) 、检测概率 (POD) 和误报率 (FAR) 的计算逻辑。
- `agent.py`: 定义用于融合模型预测的时空注意力强化学习智能体 (`SpatioTemporalAttentionFusionAgent`)。
- `attention.py`: 实现某些模型中使用的注意力机制。
- `config.py`: 包含项目的配置设置（如数据路径、批处理大小、模型参数等）。
- `dataset.py`: 处理天气雷达数据的加载和预处理。
- `environment.py`: 定义强化学习智能体运行的优化环境 (`OptimizedRadarEnvironment`)。
- `loss.py`: 定义用于训练基础预测模型的损失函数。
- `main.py`: **项目的主入口脚本**。负责协调数据加载、模型训练、智能体训练和系统评估。详见下文。
- `train_utils.py`: 包含用于训练强化学习智能体的实用函数 (`train_rl_agent_optimized`)。

## 模型:

`models` 目录包含以下基础预测模型的实现：

- `ConvLSTM`: 基于卷积 LSTM 的模型。
- `CuboidTransformer`: 基于 Transformer 的模型，处理时空数据块。
- `ForcastNet` (AFNONet): 基于傅里叶神经算子的预报网络。
- `RainFormer`: (代码库中存在，但未在 `main.py` 中直接使用)
- `SimVP`: 简单的基于视频预测的模型。

## `main.py` 详解:

`main.py` 脚本是整个项目的核心，协调了以下主要功能：

1.  **环境设置**: 
    -   `setup_seed()`: 设置随机种子以确保实验的可复现性。
2.  **数据加载**: 
    -   `load_data()`: 加载训练和验证数据集。
    -   `load_test_data()`: 加载测试数据集。
3.  **基础模型训练**: 
    -   `train_models_with_different_batch()`: 训练四个基础预测模型（SimVP, CuboidTransformer, ConvLSTM, AFNONet），每个模型针对不同的预测时间范围（30, 60, 90, 120分钟），并使用不同的批处理大小。训练过程中会使用验证集进行评估，并保存性能最佳的模型权重到 `./model_path` 目录。
4.  **强化学习智能体训练**: 
    -   `train_agent()`: 加载预训练好的基础模型，并训练 `SpatioTemporalAttentionFusionAgent`。该智能体学习如何根据输入数据动态地为四个基础模型的预测结果分配权重，以生成更准确的融合预测。训练过程中使用 `OptimizedRadarEnvironment` 作为环境，并保存训练好的智能体权重到 `./rl_model` 目录以及训练结果（奖励、指标）到 `result.txt`。
5.  **系统评估**: 
    -   `evaluate_system_with_metrics()`: 使用测试数据集评估整个系统的性能。它加载训练好的基础模型和智能体，进行预测融合，并计算详细的性能指标（CSI, POD, FAR）。
    -   **指标计算**: 分别计算智能体融合预测和每个基础模型预测在不同时间步长（30, 60, 90, 120分钟）以及所有时间步平均的 CSI, POD, FAR。
    -   **结果保存**: 将详细的指标结果保存到 `result/detailed_metrics.txt`。
    -   **可视化**: 生成并保存对比图表：
        -   `result/comparison_batch{batch_idx}_time.png`: 对比特定批次的真实雷达图、智能体预测图以及各基础模型的预测图。
        -   `result/metrics_by_timestep.png`: 按时间步对比智能体和各基础模型的 CSI, POD, FAR 分布（箱线图）。
        -   `result/overall_metrics_comparison.png`: 对比智能体和各基础模型的整体平均 CSI, POD, FAR 分布（箱线图）。
6.  **反归一化**: 
    -   `denormalize()`: 将模型输出的归一化数据（通常在0-1范围）转换回原始的雷达强度值（dBZ），使用的公式是 `image * 85 - 10`。
7.  **命令行接口**: 
    -   `main()`: 解析命令行参数，允许用户选择执行特定任务：
        -   `--train_models`: 只训练基础模型。
        -   `--train_agent`: 只训练强化学习智能体。
        -   `--eval`: 只进行系统评估。
        -   `--all`: 按顺序执行基础模型训练、智能体训练和评估。
        -   `--train_eval`: 按顺序执行智能体训练和评估。

## 使用方法:

1.  **配置**: 根据需要修改 `config.py` 中的数据路径和超参数。
2.  **训练基础模型**: 
    ```bash
    python main.py --train_models
    ```
3.  **训练智能体**: (确保基础模型已训练)
    ```bash
    python main.py --train_agent
    ```
4.  **评估系统**: (确保基础模型和智能体已训练)
    ```bash
    python main.py --eval
    ```
5.  **运行完整流程**: 
    ```bash
    python main.py --all
    ```

## 评估:

项目使用临界成功指数 (CSI)、检测概率 (POD) 和误报率 (FAR) 来评估模型和智能体的性能。评估脚本 (`evaluate_system_with_metrics`) 会在 `result/` 目录下生成详细的指标报告 (`detailed_metrics.txt`) 和性能对比图表。
