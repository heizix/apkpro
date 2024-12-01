import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

# 忽略警告信息
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

# 文件路径
file_path = "drebin-215-dataset-5560malware-9476-benign.csv"

"""
    加载数据
"""
# 读取CSV文件，避免低内存模式警告
data = pd.read_csv(file_path, low_memory=False)

# 提取特征（第一行作为特征名，最后一列为类别）
features = data.columns[:-1]
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 替换非数值字符（如 `?`）为 NaN
X.replace("?", float("nan"), inplace=True)

# 将类别转换为数值（S -> 1, B -> 0）
y = y.map({"S": 1, "B": 0})

print(f"数据集形状: {data.shape}")
print(f"特征数量: {len(features)}")
print(f"类别分布: {y.value_counts()}")

"""
    数据清洗
"""
# 使用均值填充缺失值（也可选择其他填充方式）
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

"""
    数据划分
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""
    训练支持向量机模型
"""
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

"""
    评估模型
"""
y_pred = svm_model.predict(X_test)
print("\n分类报告:")
print(classification_report(y_test, y_pred))
print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")

"""
    保存特征和类别信息到同级目录
"""
output_features_path = "drebin_features.txt"
output_classes_path = "drebin_classes.txt"

with open(output_features_path, "w") as f:
    f.write("\n".join(features))
print(f"特征保存至: {output_features_path}")

with open(output_classes_path, "w") as f:
    f.write("Classes: S (1), B (0)")
print(f"类别说明保存至: {output_classes_path}")
