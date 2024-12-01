import os
import pandas as pd
from collections import defaultdict, Counter
from androguard.core import apk


# 定义特征提取函数
def extract_apk_features(apk_file, selected_features):
    try:

        app = apk.APK(apk_file)
        print(f"Processing {apk_file}...")  # 调试输出


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



def get_permission_counts(directory):
    map_permissions = defaultdict(list)
    for apk_file in get_apk_files(directory):
        try:
            app = apk.APK(apk_file)
            permissions = app.get_permissions()
            for perm in permissions:
                map_permissions[perm].append(apk_file)
        except Exception as e:
            print(f"Error processing {apk_file}: {e}")

    permission_counts = Counter()
    for perm, files in map_permissions.items():
        permission_counts[perm] = len(files)

    return permission_counts



def extract_and_save_features(directory, permission_counts, output_csv, threshold):

    selected_features = {perm for perm, count in permission_counts.items() if count >= threshold}
    print(f"Selected features: {selected_features}")


    feature_list = []
    for apk_file in get_apk_files(directory):
        features = extract_apk_features(apk_file, selected_features)
        if features:
            feature_list.append(features)


    if feature_list:
        df = pd.DataFrame(feature_list)
        df.to_csv(output_csv, index=False)
        print(f"Feature extraction completed. Data saved to {output_csv}")
    else:
        print("No features were extracted.")


# 主程序
def main():
    # 统计恶意软件中的权限（Banking + Riskware）
    malware_directory = ["Banking", "Riskware"]
    permission_counts_malware = Counter()
    for dir in malware_directory:
        permission_counts_malware.update(get_permission_counts(dir))

    # 统计良性软件中的权限（Benign）
    benign_directory = "Benign"
    permission_counts_benign = get_permission_counts(benign_directory)

    # 统计所有软件中的权限（Banking + Benign + Riskware）
    all_directory = ["Banking", "Benign", "Riskware"]
    permission_counts_all = Counter()
    for dir in all_directory:
        permission_counts_all.update(get_permission_counts(dir))

    # 提取并保存恶意软件特征到 feature_of_1.csv（恶意软件权限出现次数大于 50）
    extract_and_save_features("Banking", permission_counts_malware, "feature_of_1.csv", 50)
    extract_and_save_features("Riskware", permission_counts_malware, "feature_of_1.csv", 50)

    # 提取并保存良性软件特征到 feature_of_2.csv（良性软件权限出现次数大于 100）
    extract_and_save_features("Benign", permission_counts_benign, "feature_of_2.csv", 100)

    # 提取并保存所有软件中的权限特征到 feature_of_all.csv（所有软件权限出现次数大于 100）
    extract_and_save_features("Banking", permission_counts_all, "feature_of_all.csv", 100)
    extract_and_save_features("Riskware", permission_counts_all, "feature_of_all.csv", 100)
    extract_and_save_features("Benign", permission_counts_all, "feature_of_all.csv", 100)

    # 读取 feature_of_1.csv 和 feature_of_all.csv，找出 feature_of_1 中但不在 feature_of_all 中的权限
    df_feature_of_1 = pd.read_csv("feature_of_1.csv")
    df_feature_of_all = pd.read_csv("feature_of_all.csv")

    features_of_1 = set(df_feature_of_1.columns) - {"class"}  # 去掉 class 列
    features_of_all = set(df_feature_of_all.columns) - {"class"}  # 去掉 class 列

    # 计算出在 feature_of_1 中，但不在 feature_of_all 中的权限
    feature_of_e = features_of_1 - features_of_all
    print(f"Features in feature_of_1 but not in feature_of_all: {feature_of_e}")

    # 保存 feature_of_e.csv
    df_feature_of_e = df_feature_of_1[list(feature_of_e) + ["class"]]
    df_feature_of_e.to_csv("feature_of_e.csv", index=False)

    # 最终提取的特征是出现在 feature_of_all 和 feature_of_e 中的权限
    final_features = features_of_all.union(feature_of_e)
    print(f"Final selected features: {final_features}")



main()
