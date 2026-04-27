import cv2
import numpy as np


def create_panorama(left_img_path, center_img_path, right_img_path):
    # Загрузка изображений
    img_l = cv2.imread(left_img_path)
    img_c = cv2.imread(center_img_path)
    img_r = cv2.imread(right_img_path)

    def get_homography(img1, img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    # Расчет гомографии для правой и левой частей относительно центральной
    H_rc = get_homography(img_r, img_c)
    H_lc = get_homography(img_l, img_c)

    # Определение размеров холста (с запасом для панорамы)
    h, w = img_c.shape[:2]
    # Примерный расчет смещения для размещения центра в середине
    canvas_w = w * 3
    canvas_h = h * 2
    offset = np.array([[1, 0, w], [0, 1, h / 2], [0, 0, 1]])

    # Варпинг изображений
    result_r = cv2.warpPerspective(img_r, offset @ H_rc, (canvas_w, canvas_h))
    result_l = cv2.warpPerspective(img_l, offset @ H_lc, (canvas_w, canvas_h))

    # Совмещение
    result = result_r.copy()
    # Наложение центрального изображения
    img_c_warped = cv2.warpPerspective(img_c, offset, (canvas_w, canvas_h))
    mask_c = (img_c_warped > 0).astype(np.uint8)
    result = result * (1 - mask_c) + img_c_warped

    # Наложение левого изображения
    mask_l = (result_l > 0).astype(np.uint8)
    result = result * (1 - mask_l) + result_l

    # Обрезка черных краев
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w_box, h_box = cv2.boundingRect(contours[0])
    final_panorama = result[y:y + h_box, x:x + w_box]

    cv2.imwrite('task4_1_panorama.jpg', final_panorama)
    print("Панорама успешно создана и сохранена.")


if __name__ == "__main__":
    create_panorama('pavilionLeft.jpg', 'pavilionCenter.jpg', 'pavilionRight.jpg')
