import cv2
from skimage.metrics import structural_similarity as compare_ssim

def akaze_function(img1, img2):
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    ratio = 0.7
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    akaze_percentage = (len(good) / len(matches)) * 100
    return akaze_percentage
