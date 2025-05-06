import requests
import os

# API Token
token = "fH781zPU26eu5hS5X6tvZ0Q6dNSdUUKkkMMYJZnys5OvNu0s16w6BShvZeRv"
ACCESS_TOKEN = token

# API URL
url = "https://zenodo.org/api/deposit/depositions"

# 请求头
headers = {
    "Authorization": f"Bearer {token}"
}
params = {'access_token': token}

# 获取用户的所有存储库
response = requests.get(url, headers=headers)
depositions = response.json()

print(depositions)
file_path = r"C:\Users\12626\Desktop\50k\final\patient_info.csv"  # 要上传的文件路径

deposition_id = "15189542"  # 已有的存储库 ID

# 获取文件上传的 URL
upload_url = f"https://zenodo.org/api/deposit/depositions/{deposition_id}/files"

# 请求头，包含授权信息
headers = {
    "Authorization": f"Bearer {token}"
}

ws_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/imgs_to_release"
ws_files = os.listdir(ws_dir)
for file_name in ws_files:
    if(not file_name.startswith("img_8398.z")):
        continue
    file_path = os.path.join(ws_dir, file_name)
    # 打开文件并上传
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(upload_url, headers=headers, files=files)

    print(response.json())