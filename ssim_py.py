import cv2
from skimage.metrics import structural_similarity as compare_ssim

def ssim_function(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ssim_value, ssim_image = compare_ssim(img1_gray, img2_gray, full=True)
    ssim_image = (ssim_image * 255).astype("uint8")
    ssim_percentage = ssim_value * 100

    return ssim_percentage