import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split, learning_curve
import pandas as pd

# 读取 feature_out.csv 文件
file_path = "feature_out.csv"
data = pd.read_csv(file_path)

# 查看数据的一些基本信息，确保数据读取正确
print(f"Shape of data: {data.shape}")
print(data.head())

# 处理缺失值，将 '?' 替换为 0
data = data.replace("?", 0)


X = data.iloc[:, :-1].values
y = data["class"].values

print(f"Shape of X: {X.shape}; Positive samples: {np.sum(y == 1)}; Negative samples: {np.sum(y == 0)}")

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型: SVM - RBF 核函数，启用概率估计
clf_svm_rbf = SVC(C=1.0, kernel="rbf", gamma=0.1, probability=True)
clf_svm_rbf.fit(X_train, y_train)
train_score_svm_rbf = clf_svm_rbf.score(X_train, y_train)
test_score_svm_rbf = clf_svm_rbf.score(X_test, y_test)
print(f"SVM RBF Kernel - Train Score: {train_score_svm_rbf:.2f}; Test Score: {test_score_svm_rbf:.2f}")

# 训练模型: 决策树
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt.fit(X_train, y_train)
train_score_dt = clf_dt.score(X_train, y_train)
test_score_dt = clf_dt.score(X_test, y_test)
print(f"Decision Tree - Train Score: {train_score_dt:.2f}; Test Score: {test_score_dt:.2f}")

# 训练模型: 随机森林
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
train_score_rf = clf_rf.score(X_train, y_train)
test_score_rf = clf_rf.score(X_test, y_test)
print(f"Random Forest - Train Score: {train_score_rf:.2f}; Test Score: {test_score_rf:.2f}")

# 模型调优：通过GridSearchCV调整SVM模型的参数
param_grid = {"gamma": np.linspace(0.001, 0.1, 20)}
grid_search = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best Params for SVM: {grid_search.best_params_}; Best Score: {grid_search.best_score_:.2f}")

# 绘制每个模型的学习曲线
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

# 设置学习曲线的绘制
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

# 创建子图，绘制每个模型的学习曲线
plt.figure(figsize=(12, 8), dpi=144)

# 绘制SVM RBF学习曲线
plt.subplot(2, 2, 1)
plot_learning_curve(plt, SVC(C=1.0, kernel="rbf", gamma=0.1), "Learning Curve - SVM RBF", X, y, ylim=(0.5, 1.01), cv=cv)

# 绘制决策树学习曲线
plt.subplot(2, 2, 2)
plot_learning_curve(plt, clf_dt, "Learning Curve - Decision Tree", X, y, ylim=(0.5, 1.01), cv=cv)

# 绘制随机森林学习曲线
plt.subplot(2, 2, 3)
plot_learning_curve(plt, clf_rf, "Learning Curve - Random Forest", X, y, ylim=(0.5, 1.01), cv=cv)

# 绘制模型对比图
plt.subplot(2, 2, 4)
labels = ['SVM RBF', 'Decision Tree', 'Random Forest']
scores = [test_score_svm_rbf, test_score_dt, test_score_rf]
plt.bar(labels, scores, color=['r', 'g', 'b'])
plt.ylim(0, 1)
plt.title('Model Comparison')
plt.ylabel('Test Score')

plt.tight_layout()
plt.show()

# 保存模型
joblib.dump(clf_svm_rbf, "svm_rbf_model.pkl")
joblib.dump(clf_dt, "decision_tree_model.pkl")
joblib.dump(clf_rf, "random_forest_model.pkl")
print("Models saved to disk.")
