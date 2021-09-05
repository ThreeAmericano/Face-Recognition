# Face-Recognition
얼굴인식 부분 (서버 처리)



**텐서플로우 설치 : https://angel-breath.tistory.com/19**



## 사용모듈

```
python3.7
tensorflow==1.7 (나는 1.14.0 사용함)
scipy
scikit-learn
opencv-python
h5py
matplotlib
Pillow
requests
psutil
```





### 수정사항

scipy 內 imresize 함수에 대한 지원이 없어짐에 따라 해당 항목을 PIL에 Image함수로 수정

비디오 카메라를 VPN망 내에 있는 카메라 기기로 변경



### 오류발생

can't initialize gtk backend in function 'cvinitsystem'

https://github.com/opencv/opencv/issues/18461

https://unix.stackexchange.com/questions/94497/org-eclipse-swt-swterror-no-more-handles-gtk-init-check-failed-while-runnin

https://discuss.luxonis.com/d/175-depthai-gen2-example-01-rgb-preview-fail-to-run-on-raspberry-pi-zero-w/22



=> 파이썬 실행버전이 맞지않아 발생한 문제

나는 `pi` 유저에게 `python3.7`로 `GTK` 버전인 opencv를 설치하였으나,  이를 다른버전에 파이썬이나 유저(root) 등 으로 실행하려고 하면 해당 에러가 발생함.. (왜냐면 그들로는 정상적으로 Xwindow 표현을 할 수 없기 때문.)
