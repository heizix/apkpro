import os
import pandas as pd
from collections import defaultdict, Counter
from androguard.core import apk
import json


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


        features["class"] = 1

        return features
    except Exception as e:
        print(f"Error processing {apk_file}: {e}")
        return None


# 获取所有 APK 文件
def get_apk_files(directory):
    apk_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(f"Found file: {file}")
            if file.lower().endswith(".apk") or file.isdigit():
                apk_files.append(os.path.join(root, file))

            elif file.lower().endswith(('.zip', '.rar', '.7z')):

                apk_files.append(os.path.join(root, file))
    return apk_files


# 提取特征并保存到 CSV 文件
def extract_and_save_features(directory, output_csv):
    # 收集权限数据
    map_permissions = defaultdict(list)
    for apk_file in get_apk_files(directory):
        try:
            app = apk.APK(apk_file)
            permissions = app.get_permissions()
            for perm in permissions:
                map_permissions[perm].append(apk_file)
        except Exception as e:
            print(f"Error processing {apk_file}: {e}")

    # 统计权限出现次数
    permission_counts = Counter()
    for perm, files in map_permissions.items():
        permission_counts[perm] = len(files)

    # 选择出现次数大于 100 的特征
    selected_features = {perm for perm, count in permission_counts.items() if count >= 100}
    print(f"Selected features: {selected_features}")


    feature_list = []
    for apk_file in get_apk_files(directory):
        features = extract_apk_features(apk_file, selected_features)
        if features:
            feature_list.append(features)

    # 将特征保存为 CSV 文件
    if feature_list:
        df = pd.DataFrame(feature_list)
        df.to_csv(output_csv, index=False)
        print(f"Feature extraction completed. Data saved to {output_csv}")
    else:
        print("No features were extracted.")


# 执行特征提取并保存
extract_and_save_features("Riskware", "Riskware_apk_features.csv")
