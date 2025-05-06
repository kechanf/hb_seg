import pandas as pd
import json

atlas_json_file  = "/data/kfchen/trace_ws/atlas/yale/Atlas_Dict.json"
with open(atlas_json_file, "r", encoding="utf-8") as f:
    atlas_data = json.load(f)

# 提取所有的 "Region" 值，并使用集合去重
regions = {entry["Lobe"] for entry in atlas_data}

# 输出不同的 region 数量和详细的 region 列表
print(f"不同的 region 数量：{len(regions)}")
print("所有 region 列表：")
for region in regions:
    print(region)