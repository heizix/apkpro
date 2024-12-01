# -*- coding: utf-8 -*-
# Time：2024/11/28
# Author: heizixiao

"""
    支持向量机实战-恶意软件检测
"""
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split, learning_curve
import pandas as pd

"""
    加载数据
"""
file_path = "drebin-215-dataset-5560malware-9476-benign.csv"  # 修改为实际的文件路径
data = pd.read_csv(file_path)

# 处理缺失值，将 '?' 替换为 0
data = data.replace("?", 0)

# 提取特征和标签
X = data.iloc[:, :-1].values  # 所有特征列
y = data.iloc[:, -1].map({"B": 0, "S": 1}).values  # 转换类别为数值型

print(f"Shape of X: {X.shape}; Positive samples: {np.sum(y == 1)}; Negative samples: {np.sum(y == 0)}")

"""
    拆分数据集
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
    训练模型: RBF核函数，启用概率估计
"""
clf_rbf = SVC(C=1.0, kernel="rbf", gamma=0.1, probability=True)
clf_rbf.fit(X_train, y_train)
train_score_rbf = clf_rbf.score(X_train, y_train)
test_score_rbf = clf_rbf.score(X_test, y_test)
print(f"RBF Kernel - Train Score: {train_score_rbf:.2f}; Test Score: {test_score_rbf:.2f}")

"""
    绘制学习曲线
"""
def plot_learning_curve(plt, estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(
        train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r"
    )
    plt.fill_between(
        train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g"
    )
    plt.plot(train_sizes, train_scores_mean, 'o--', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plt.figure(figsize=(10, 4), dpi=144)
plot_learning_curve(plt, SVC(C=1.0, kernel="rbf", gamma=0.1), "Learning Curve - RBF Kernel", X, y, ylim=(0.5, 1.01), cv=cv)
plt.show()

"""
    模型调优
"""
param_grid = {"gamma": np.linspace(0.001, 0.1, 20)}
grid_search = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best Params: {grid_search.best_params_}; Best Score: {grid_search.best_score_:.2f}")

"""
    训练模型: 多项式核函数
"""
clf_poly = SVC(C=1.0, kernel="poly", degree=2)
clf_poly.fit(X_train, y_train)
train_score_poly = clf_poly.score(X_train, y_train)
test_score_poly = clf_poly.score(X_test, y_test)
print(f"Polynomial Kernel - Train Score: {train_score_poly:.2f}; Test Score: {test_score_poly:.2f}")

"""
    保存模型：
    
"""
model_file_path = "svm_rbf_model.pkl"
joblib.dump(clf_rbf, model_file_path)
print(f"Model saved to {model_file_path}")

"""
    绘制学习曲线: 多项式核函数
"""
degrees = [1, 2]
plt.figure(figsize=(12, 4), dpi=144)
for i, degree in enumerate(degrees):
    plt.subplot(1, len(degrees), i + 1)
    plot_learning_curve(
        plt, SVC(C=1.0, kernel="poly", degree=degree), f"Learning Curve - Degree={degree}", X, y, ylim=(0.8, 1.01), cv=cv
    )
plt.show()
