from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from io import BytesIO
import lpips_py

import remap

app = FastAPI()


@app.post("/lpips/")
async def lpips_images(gamemaster: UploadFile = File(...), player1: UploadFile = File(...), player2: UploadFile = File(...)):    
    player_img1 = await player1.read()
    player_img2 = await player2.read()
    game_master_img = await gamemaster.read()
    # Convert bytes to numpy arrays
    nparr1 = np.frombuffer(player_img1, np.uint8)
    nparr2 = np.frombuffer(player_img2, np.uint8)
    nparr3 = np.frombuffer(game_master_img, np.uint8)

    player_img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    player_img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
    game_master_img = cv2.imdecode(nparr3, cv2.IMREAD_COLOR)

    height, width = game_master_img.shape[:2]
    player_img1 = cv2.resize(player_img1, (width, height))
    player_img2 = cv2.resize(player_img2, (width, height))

    if player_img1 is None or player_img2 is None or game_master_img is None:
        return {"error": "Invalid image"}
    player_remapped_img = remap.main(player_img1, player_img2)
    result = lpips_py.lpips_function(player_remapped_img, game_master_img)
    return {"score" : result}




@app.get("/")
def read_root():
    return {"message": "Welcome to the image comparison API!"}