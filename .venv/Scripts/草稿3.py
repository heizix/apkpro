import pandas as pd
import joblib

# 读取val.csv文件，指定制表符作为分隔符
val_df = pd.read_csv("val.csv", sep='\t', header=0)  # 确保第一行被识别为列名

# 假设特征列在val.csv中是从第一列开始的
feature_columns = val_df.columns  # 选择所有列作为特征列

# 将特征数据转换为浮点数类型
new_apk_features = val_df[feature_columns].astype(float).values

# 模型文件路径
model_file_path = "svm_rbf_model.pkl"

# 加载之前保存的RBF核SVM模型
loaded_model = joblib.load(model_file_path)

# 确保模型已设置probability=True
if hasattr(loaded_model, "predict_proba"):
    # 使用加载的模型预测概率
    probabilities = loaded_model.predict_proba(new_apk_features)

    # 输出预测概率
    for i, probs in enumerate(probabilities):
        benign_prob, malicious_prob = probs
        print(
            f"The APK file {i + 1} has {benign_prob:.2f} probability of being benign and {malicious_prob:.2f} probability of being malicious.")
else:
    print("Model does not support probability estimation. Please ensure the model was trained with probability=True.")