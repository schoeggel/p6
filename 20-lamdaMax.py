from archive import trainfeature
import cv2

a = (0 , 104)
b = (2 , 103)
c = (4 , 102)
d = (6 , 101)

mylist  = [a,b,c,d]

mymax = max(mylist, key=lambda x: x[1])
print(mymax)


# anderer test

a = cv2.imread("sample/HappyFish.jpg", cv2.IMREAD_COLOR)
b = cv2.imread("sample/HappyFish2.jpg", cv2.IMREAD_GRAYSCALE)

trainfeature.Trainfeature.imgMergerH([a, b])
bigpic = trainfeature.Trainfeature.imgMergerV([a, b])

cv2.namedWindow("test1", cv2.WINDOW_NORMAL)
cv2.imshow("test1", bigpic)
cv2.waitKey(0)