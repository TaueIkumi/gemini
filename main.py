from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import sift_py
import lpips_py
import ssim_py
import akaze_py
import imgsim_py
from io import BytesIO

app = FastAPI()

def plot_scores(results, img1, img2):
    names = [result['name'] for result in results]
    scores = [result['score'] for result in results]
    times = [result['time'] for result in results]

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize = (8,5))

    ax_1 = fig.add_subplot(2,2,1)
    ax_2 = fig.add_subplot(2,2,2)
    ax_3 = fig.add_subplot(2,2,(3,4))

    ax_1.imshow(img1)
    ax_2.imshow(img2)
    ax_3.scatter(times, scores, color='blue')
    ax_3.set_xlabel("time")
    ax_3.set_ylabel("score")
    ax_3.set_ylim(0, 100)
    ax_3.grid(False)

    for i, name in enumerate(names):
        ax_3.text(times[i], scores[i], name, fontsize=12, ha='right')

    plt.show()
    print(scores)
    return fig, scores

def main(img1, img2):
    h, w = img1.shape[:2]
    img1, img2 = cv2.resize(img1, (w//5, h//5)), cv2.resize(img2, (w//5, h//5))

    # 初期化
    sift_score, lpips_score, ssim_score, akaze_score, imgsim_score = None, None, None, None, None
    sift_time, lpips_time, ssim_time, akaze_time, imgsim_time = None, None, None, None, None

    try:
        # スコア&時間計測
        st = time.time()
        sift_score = sift_py.sift_function(img1, img2)
        sift_time = time.time() - st

        st = time.time()
        lpips_score = lpips_py.lpips_function(img1, img2)
        lpips_time = time.time() - st

        st = time.time()
        ssim_score = ssim_py.ssim_function(img1, img2)
        ssim_time = time.time() - st

        st = time.time()
        akaze_score = akaze_py.akaze_function(img1, img2)
        akaze_time = time.time() - st
        
        st = time.time()
        imgsim_score = imgsim_py.imgsim_function(img1, img2)
        imgsim_time = time.time() - st

    except cv2.error as e:
        print(f"OpenCVエラーが発生しました: {e}")
    except ZeroDivisionError: # 単色画像だとエラーが起きる
        print("ZeroDivisionError")
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")

    result = [
        {
            "name": "SIFT",
            "score": sift_score,
            "time": sift_time
        },
        {
            "name": "LPIPS",
            "score": lpips_score,
            "time": lpips_time
        },
        {
            "name": "SSIM",
            "score": ssim_score,
            "time": ssim_time
        },
        {
            "name": "AKAZE",
            "score": akaze_score,
            "time": akaze_time
        },
        {
            "name": "IMGSIM",
            "score": imgsim_score,
            "time": imgsim_time
        }
    ]
    # plot_scores(result,img1,img2)
    return result

@app.post("/upload/")
async def upload_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = await file1.read()
    img2 = await file2.read()

    # Convert bytes to numpy arrays
    nparr1 = np.frombuffer(img1, np.uint8)
    nparr2 = np.frombuffer(img2, np.uint8)
    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    result = main(img1, img2)
    return result

@app.get("/")
def read_root():
    return {"message": "Welcome to the image comparison API!"}

if __name__ == "__main__":
    main(cv2.imread("img/osho1.jpg"), cv2.imread("img/daigaku1.jpg"))