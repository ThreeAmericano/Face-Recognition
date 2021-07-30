import cv2

img = cv2.imread("/home/pi/test.jpg") #이미지를 불러옴
print(img.shape) #제대로 불러와졌는지 이미지 사이즈를 반환.
cv2.imshow("Test",img) #이미지를 윈도우창으로 실행시킴

img_canny = cv2.Canny(img, 50, 150)
cv2.imshow("Test img Edge", img_canny)

cv2.waitKey(0)
cv2.destroyAllWindows()
