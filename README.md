# Android 恶意软件检测

## 项目背景

随着智能手机和移动互联网的普及，Android 系统已成为最广泛使用的操作系统，但也成为了恶意软件攻击的主要目标。恶意软件检测已成为信息安全领域的重要研究课题。本项目旨在通过经典的机器学习算法，利用 Android 应用的静态特征（如权限、API 调用等），构建恶意软件检测模型，并通过创新的特征选择方法，提高模型性能和检测精度。
## 一、项目执行

- **环境配置**：确保 Python 环境中安装了所需的库，包括 `pandas`、`scikit-learn`、`joblib`、`matplotlib`、`Androguard` 等。
- **数据集**：数据集 `CICMalDroid 2020` 包含多个 APK 样本，涉及广告软件、银行恶意软件、SMS 恶意软件等类别。
- **模型评估**：通过准确率、精确度、召回率和 F1 分数评估模型性能。

---


## 二、特征提取

### 1. `特征提取草稿1.py`
该脚本用于从 `Riskware` 文件夹下的 APK 文件中提取特征。通过使用 **Androguard** 库解析 APK 文件，提取应用的权限信息，并将权限信息作为特征保存用于后续模型训练。

**功能：**
- 使用 Androguard 解析 APK 文件。
- 提取并保存 APK 文件中的权限特征。

---

### 2. `特征提取草稿2.py`
该脚本用于分别统计恶意软件和良性软件的权限特征，并基于出现频率进行特征选择。

**功能：**
- 统计并选择 `Banking`、`Riskware` 和 `Benign` 文件夹中的 APK 文件权限。
- 根据权限频率选择合适的特征，用于后续的特征提取与模型训练。

---

### 3. `总结特征.py`
该脚本通过合并来自 `feature_of_all.csv` 和 `feature_of_e.csv` 中的特征，最终选定符合要求的特征，并将其写入 `feature.csv` 文件。

**功能：**
- 读取 `feature_of_all.csv` 和 `feature_of_e.csv`，合并并去除无用的特征。
- 将最终的特征集保存到 `feature.csv` 中，用于后续的 APK 文件特征提取。

---

### 4. `feature_out_final.py`
根据 `feature.csv` 文件中的特征，提取指定文件夹（`Banking`、`Riskware` 和 `Benign`）中的 APK 文件特征，并将其保存到 `feature_out.csv`。

**功能：**
- 读取 `feature.csv` 中的特征。
- 提取指定文件夹中的 APK 文件特征，生成最终的特征文件 `feature_out.csv`。

---

## 三、模型训练

### 1. `train_final.py`
该脚本用于从 `feature_out.csv` 文件中加载特征数据，并使用不同的机器学习算法进行训练，评估模型效果并进行对比。

**功能：**
- 加载 `feature_out.csv` 中的特征数据。
- 使用 **支持向量机（SVM）**、**决策树**、**随机森林** 等算法进行模型训练。
- 输出每个模型的训练和测试得分，并对模型进行调优。
- 绘制每个模型的学习曲线并进行对比展示。

---

## 四、其他

### 1. 下载 APK 文件时间过长
下载期间使用derbin的特征csv文件进行初步代码编写

### 2. `草稿.py`
该脚本通过使用 **支持向量机（SVM）** 对公开的 `drebin` 数据集进行特征训练，并生成一个初步的模型。

**功能：**
- 使用公开的 `drebin` 数据集特征文件并训练模型。
- 该模型可用于进一步的恶意软件检测。

---

### 3. `草稿2.py`
该脚本用于从数据集中的恶意软件和良性软件中统计权限特征，并选择合适的特征进行后续处理。

**功能：**
- 选择频繁出现的特征，并过滤冗余特征。
- 提升特征选择的准确性与针对性。

---
### 4. `草稿3.py`
该脚本用于对新数据集（如 `val.csv`）中的 APK 文件进行特征提取，并使用之前训练好的 **支持向量机（SVM）** 模型进行预测，输出每个 APK 文件属于良性还是恶意的概率。

**功能：**
- 从 `val.csv` 文件中加载特征数据。
- 使用加载的 **RBF 核 SVM** 模型预测每个 APK 文件属于良性还是恶意的概率。
- 输出预测结果，包括每个 APK 文件的良性与恶意概率。

---


## 五、贡献

### 1. **heizix**
- 负责数据集的准备和处理。
- 进行特征选择与构建。
- 训练和评估模型，撰写实验报告。

### 2. **hsy**
- 配置和维护实验环境。
- 协助完成模型训练与评估。
- 协助撰写实验报告。

---

## 六、参考文献

1. Mahdavifar, S., Alhadidi, D., & Ghorbani, A. A. (2020). Using Semi-supervised Deep Learning for Dynamic Android Malware Classification. IEEE International Conference on Reliable, Autonomous and Secure Computing.
2. Mahdavifar, S., Alhadidi, D., & Ghorbani, A. A. (2022). Effective and Efficient Hybrid Android Malware Classification Using Pseudo-label Stacking Autoencoders. *Journal of Network and Systems Management*.
3. Z. Zhang, et al. (2018). Malware detection and classification using deep learning.
4. F. Xiao, et al. (2019). Application of machine learning models in Android malware detection.

---

通过此格式，您的 `README.md` 文件将清晰明了地展示项目的结构、目标和功能，便于用户快速理解和上手使用代码。
