import pandas as pd

# 读取 feature_of_all.csv 和 feature_of_e.csv
df_all = pd.read_csv("feature_of_all.csv")
df_e = pd.read_csv("feature_of_e.csv")

# 提取所有特征列
features_all = set(df_all.columns) - {"class"}
features_e = set(df_e.columns) - {"class"}

# 合并两个特征集合
final_features = features_all.union(features_e)

# 将最终特征写入 feature.csv
with open("feature.csv", "w") as f:
    for feature in final_features:
        f.write(f"{feature}\n")

print("Feature names written to feature.csv")
