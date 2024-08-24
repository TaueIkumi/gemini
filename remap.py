import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov9c.pt")


def create_human_mask(image):
    results = model(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            if cls == 0:  # human class id in COCO dataset
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                mask[y1:y2, x1:x2] = 255

    return mask


def get_matcher(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.AKAZE_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("Descriptors could not be computed.")
        return None, None
    
    if len(kp1) == 0 or len(kp2) == 0:
        print("No keypoints found in one of the images.")
        return None, None

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) == 0:
        print("No good matches found.")
        return None, None

    target_position = []
    base_position = []
    for g in good:
        target_position.append([kp1[g.queryIdx].pt[0], kp1[g.queryIdx].pt[1]])
        base_position.append([kp2[g.trainIdx].pt[0], kp2[g.trainIdx].pt[1]])

    apt1 = np.array(target_position)
    apt2 = np.array(base_position)

    return apt1, apt2


def align_images(img1, img2, apt1, apt2):
    h, w = img1.shape[:2]
    if len(apt1) < 4 or len(apt2) < 4:
        print("Not enough points to compute homography.")
        return img1
    homography, _ = cv2.findHomography(apt2, apt1, cv2.RANSAC)
    if homography is None:
        print("Homography could not be computed.")
        return None
    aligned_img2 = cv2.warpPerspective(img2, homography, (w, h))
    return aligned_img2


def inpaint_image(foreground, background, mask):
    inpainted_image = np.where(mask[:, :, None] == 255, background, foreground)
    return inpainted_image


def calculate_difference(img1, img2):
    difference = np.sum(np.abs(img1.astype(int) - img2.astype(int)))
    return difference


def main(img1, img2):
    human_mask1 = create_human_mask(img1)
    human_mask2 = create_human_mask(img2)

    pt1, pt2 = get_matcher(img1, img2)

    if pt1 is None or pt2 is None:
        print("Matching points could not be computed. return img1")
        return img1  # return img1

    aligned_img2 = align_images(img1, img2, pt1, pt2)

    if aligned_img2 is None:
        print("Image alignment failed.")
        return img1

    inpainted_img1 = inpaint_image(img1, aligned_img2, human_mask1)

    aligned_img1 = align_images(img2, img1, pt2, pt1)
    if aligned_img1 is None:
        print("Image alignment failed.")
        return img2

    inpainted_img2 = inpaint_image(img2, aligned_img1, human_mask2)

    diff1 = calculate_difference(img1, inpainted_img1)
    diff2 = calculate_difference(img2, inpainted_img2)

    if diff1 < diff2:
        return inpainted_img1
    else:
        return inpainted_img2


if __name__ == "__main__":
    img1 = cv2.imread("img/castle1.jpg")
    img2 = cv2.imread("img/castle2.jpg")
    result = main(img1, img2)
    if isinstance(result, np.ndarray):
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error occurred while processing images.")
