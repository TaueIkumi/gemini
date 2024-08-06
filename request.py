import requests

url = "http://127.0.0.1:8000/lpips/"  # FastAPIサーバーのURL

# 送信する画像ファイルのパス
game_master_path = "img/daigaku4.jpg"
player1_path = "img/castle1.jpg"
player2_path = "img/castle2.jpg"

# 画像ファイルを開いてmultipart/form-dataで送信
with open(game_master_path, "rb") as game_master, open(player1_path, "rb") as player1, open(player2_path, "rb") as player2:
    files = {
        "gamemaster": game_master,
        "player1": player1,
        "player2": player2
    }
    response = requests.post(url, files=files)

print(response.text)
