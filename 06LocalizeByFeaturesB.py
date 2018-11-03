# Weil die standard template matching funktion ungenügend ist, soll hier versucht werden
# vorerst ohne kamera kalibrierung etc. eine lokalisierung des gitters zu machen.
# Code Basis ist ein Beispiel von:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
# Problem: andere Lichtverhältnisse, Nässe etc --> Gitter wird überhaupt nicht gefunden !!
# Optionen: Nur Canny vom img und tmpl verwenden ===> funktioniert nicht
# und/oder Bild deutlich verkleinern.


import cv2
import numpy as np

tem = cv2.imread("data/06template1.png", cv2.IMREAD_GRAYSCALE)  # queryiamge
sbb = cv2.imread("sbb/4-OK1L.png", cv2.IMREAD_GRAYSCALE)

#canny
# tem = cv2.Canny(tem, 400, 300)
# sbb = cv2.Canny(sbb, 400, 300)


#verkleinern
scale = 0.1
tem = cv2.resize(tem, (int(tem.shape[0]*scale), int(tem.shape[1]*scale)), interpolation=cv2.INTER_LANCZOS4)
sbb = cv2.resize(sbb, (int(sbb.shape[0]*scale), int(sbb.shape[1]*scale)))

# Features
orb = cv2.ORB_create(2000, scaleFactor=1.4, nlevels=16)
kp_tem, desc_tem = orb.detectAndCompute(tem, None)
kp_sbb, desc_sbb = orb.detectAndCompute(sbb, None)


# Feature matching
# flann geht ohne weiteres nur mit SIFt und SURF
# index_params = dict(algorithm=0, trees=5)
# search_params = dict()
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# flann = cv2.FlannBasedMatcher_create()
# matches = flann.knnMatch(desc_tem, desc_sbb, k=2)


# Alternativ: Bruteforce matcher verwenden.
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc_tem, desc_sbb, k=2)

good_points = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good_points.append(m)

img3 = cv2.drawMatches(tem, kp_tem, sbb, kp_sbb, good_points, sbb)

# Homography
cv2.namedWindow('Homography', cv2.WINDOW_NORMAL)

print("good_points: " + str(len(good_points)))
if len(good_points) > 10:
    query_pts = np.float32([kp_tem[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
    train_pts = np.float32([kp_sbb[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # Perspective transform
    h, w = tem.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, matrix)

    homography = cv2.polylines(sbb, [np.int32(dst)], True, (255, 0, 0), 3)
    h = cv2.putText(homography, 'Homography', (10, 100), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Homography", h)
else:
    cv2.imshow("Homography", sbb)



# cv2.imshow("Image (template)", tem)
# cv2.imshow("grayFrame (sbb)", sbb)

cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
cv2.imshow("img3", img3)

cv2.resizeWindow('Homography', 800, 800)
cv2.resizeWindow('img3', 800, 800)

cv2.waitKey(0)
cv2.destroyAllWindows()
