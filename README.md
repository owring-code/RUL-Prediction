# ⚙️ 航空发动机剩余使用寿命预测 
*(Remaining Useful Life Prediction - RUL)*

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)
![Algorithm](https://img.shields.io/badge/Algorithm-GA%20%2B%20Neural%20Network-orange.svg)
![Dataset](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Paper](https://img.shields.io/badge/Paper-Measurement%20Science%20and%20Technology-purple.svg)

## 📖 项目概述

本项目致力于解决工业预测性维护（Predictive Maintenance）中的核心问题：**剩余使用寿命 (Remaining Useful Life, RUL) 预测**。

基于经典的 **NASA C-MAPSS (FD001)** 涡扇发动机退化数据集，本项目构建了一个融合了 **遗传算法 (Genetic Algorithm, GA)** 与 **神经网络 (Neural Network)** 的混合预测模型。通过遗传算法对神经网络的关键结构参数进行全局寻优，有效提升模型在复杂传感器时间序列数据上的预测精度与泛化能力。

该项目相关研究已进一步整理并发表为论文 **Evo-NAS: An Evolutionary Neural Architecture Search Framework for Industrial Time Series Forecasting**。在论文版本中，方法被扩展为面向工业时间序列预测的进化式神经架构搜索框架，并在 NASA C-MAPSS 多个子数据集以及真实航空发动机润滑油消耗预测任务上进行了验证。

---

## 📝 论文信息

**Evo-NAS: An Evolutionary Neural Architecture Search Framework for Industrial Time Series Forecasting**

**Authors:** Wenrui Ouyang, Jing Wang, Qi Xi  
**Journal:** *Measurement Science and Technology*, 2026  
**DOI:** [10.1088/1361-6501/ae646a](https://doi.org/10.1088/1361-6501/ae646a)

论文提出的 **Evo-NAS** 框架通过遗传算法在受控搜索空间中自动组合和筛选时序建模模块，包括：

- Temporal Convolutional Network (TCN)
- BiLSTM
- BiGRU
- Attention mechanism

该框架能够针对不同工业数据集和不同退化工况，自适应搜索更合适的神经网络结构，从而减少人工调参和手工设计模型结构的成本。


## 📂 项目文件结构

```text
RUL-Prediction/
├── GA_Net_FD001.py      # 🚀 核心算法入口：包含遗传算法优化逻辑与 PyTorch 神经网络训练模型
├── Data_until.py        # 🛠️ 数据预处理工具类：负责数据集的读取、归一化、滑动窗口特征提取等
├── dataset.zip          # 📦 原始数据集压缩包：包含 NASA C-MAPSS FD001 训练集与测试集
├── fitness_log.txt      # 📄 训练日志：记录了遗传算法迭代过程中的适应度 (Fitness) 变化
└── README.md            # 📖 项目说明文档
```

---

## 🧠 核心技术与算法

1. **数据预处理 (Data Preprocessing)**:
   * 针对多维度传感器数据进行特征筛选与异常值处理。
   * 采用 `MinMaxScaler` 进行归一化，消除量纲差异带来的影响。
   * 采用**滑动时间窗口 (Sliding Window)** 技术，结合 `DataLoader` 将多元时间序列转化为适用于神经网络监督学习的批量样本对。

2. **神经网络模型 (Neural Network)**:
   * 基于 **PyTorch** 构建深度学习网络，提取多传感器时间序列中的空间与时序退化特征。
   * 引入 `thop` 模块进行模型的算力 (FLOPs) 与参数量分析，确保预测模型的轻量化与高效性。
   * 输出层直接拟合当前工况下的连续 RUL 数值。

3. **遗传算法优化 (Genetic Algorithm Optimization)**:
   * **种群初始化**：将神经网络的关键结构参数编码为染色体。
   * **适应度评估**：以模型在验证集上的预测误差倒数作为适应度函数。
   * **交叉与变异**：通过 GA 的进化机制跳出局部最优解，寻找全局最优的网络配置。

4. **Evo-NAS 扩展思想**:
   * 在论文版本中，搜索空间进一步扩展为由 TCN、BiLSTM、BiGRU 和注意力机制组成的异构时序模块池。
   * 通过遗传算法自动搜索网络层类型、层数和隐藏维度等结构参数。
   * 针对不同数据集和工况条件自动搜索最优结构，提高模型对复杂工业退化模式的适应能力。

---

## 🚀 快速开始

### 1. 环境依赖

本项目基于 **PyTorch** 深度学习框架构建。请确保您的 Python 环境（建议 `Python 3.8+`）中已安装以下核心依赖库：

```bash
# 基础数据处理、机器学习与可视化库
pip install numpy pandas scikit-learn matplotlib tqdm

# PyTorch 深度学习框架
# 请根据您的显卡 CUDA 版本到 PyTorch 官网获取对应的安装命令
pip install torch torchvision torchaudio

# 模型复杂度计算工具
pip install thop
```

### 2. 数据准备

运行代码前，请务必先解压项目根目录下的数据集文件：

```bash
unzip dataset.zip
```

注：解压后请确保数据文件路径与 `Data_until.py` 中的读取路径保持一致。

### 3. 运行模型

直接运行主程序即可开始遗传算法的迭代与神经网络的训练：

```bash
python GA_Net_FD001.py
```

系统内置了 `tqdm` 进度条，可直观查看每个 Epoch 的训练进度。

### 4. 查看日志

训练过程中的迭代信息和最优适应度变化会自动保存在 `fitness_log.txt` 中，您可以随时打开该文件复盘模型的收敛情况与优化效果。

---

## 📚 引用

如果本项目或论文对您的研究有所帮助，欢迎引用我们的工作：

```bibtex
@article{ouyang2026evonas,
  title = {Evo-NAS: An Evolutionary Neural Architecture Search Framework for Industrial Time Series Forecasting},
  author = {Ouyang, Wenrui and Wang, Jing and Xi, Qi},
  journal = {Measurement Science and Technology},
  year = {2026},
  doi = {10.1088/1361-6501/ae646a}
}
```

---
