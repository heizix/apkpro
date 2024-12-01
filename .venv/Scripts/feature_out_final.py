import os
import pandas as pd
from androguard.core import apk


# 定义特征提取函数
def extract_apk_features(apk_file, selected_features):
    try:

        app = apk.APK(apk_file)
        print(f"Processing {apk_file}...")


        permissions = app.get_permissions()
        features = {}


        for feature in selected_features:
            if feature in permissions:
                features[feature] = 1
            else:
                features[feature] = 0


        return features
    except Exception as e:
        print(f"Error processing {apk_file}: {e}")
        return None


# 获取所有 APK 文件
def get_apk_files(directory):
    apk_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".apk"):
                apk_files.append(os.path.join(root, file))
    return apk_files


# 执行特征提取并保存到 CSV 文件
def extract_and_save_features(directory, selected_features, class_value, output_csv):
    feature_list = []
    for apk_file in get_apk_files(directory):
        features = extract_apk_features(apk_file, selected_features)
        if features:

            features["class"] = class_value
            feature_list.append(features)

    # 将特征保存为 CSV 文件，使用追加模式（'a'）
    if feature_list:
        df = pd.DataFrame(feature_list)

        df.to_csv(output_csv, mode='a', index=False, header=not os.path.exists(output_csv))
        print(f"Feature extraction completed. Data saved to {output_csv}")
    else:
        print("No features were extracted.")


# 读取 feature.csv 中的权限特征名称
with open("feature.csv", "r") as f:
    selected_features = [line.strip() for line in f.readlines()]

# 提取并保存 Banking、Riskware 和 Benign 中的特征
extract_and_save_features("Banking", selected_features, class_value=1, output_csv="feature_out.csv")
extract_and_save_features("Riskware", selected_features, class_value=1, output_csv="feature_out.csv")
extract_and_save_features("Benign", selected_features, class_value=0, output_csv="feature_out.csv")
