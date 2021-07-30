import cv2
stream_url = "http://211.179.42.130:9999/stream/video.mjpeg"

cap = cv2.VideoCapture(stream_url)
if cap.isOpened():
	ret, frame = cap.read() #성공적으로 이미지를 불러왔는지, 실제 이미지(프레임) 자체
	if ret :
		print(frame.shape) #해당 이미지의 해상도 print
		cv2.imshow("test",frame)
	else :
		print("failed to read frame")
else :
	print("can't open")
	
cv2.waitKey(0)
cv2.destroyAllWindows()
