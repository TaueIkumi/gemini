import requests

url = "http://127.0.0.1:8000/upload/"  # FastAPIサーバーのURL

# 送信する画像ファイルのパス
file1_path = "img/osho1.jpg"
file2_path = "img/osho2.jpg"

# 画像ファイルを開いてmultipart/form-dataで送信
with open(file1_path, "rb") as file1, open(file2_path, "rb") as file2:
    files = {
        "file1": file1,
        "file2": file2
    }
    response = requests.post(url, files=files)

# レスポンスを表示
print(response.json())
