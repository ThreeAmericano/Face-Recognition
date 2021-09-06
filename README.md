# Face-Recognition

얼굴인식 부분 (서버 처리) / 다른 백엔드 프로그램에 의해 간접 실행됨.





## 📄 Tech Spec

- Python3.7
- Tensorflow 1.14.0
- IPC with backend_process.py (`File: face.result`)
- FaceNet





## 📁Tree

```python
📁 Face-Recognition
┗ 📁 20180408-102900
  ┗ 📄 20180408-102900.pb
  ┗ 📄 model-20180408-102900.ckpt-90.index
  ┗ 📄 model-20180408-102900.meta
┗ 📁 npy
  ┗ 📄 det1.npy
  ┗ 📄 det2.npy
  ┗ 📄 det3.npy
┗ 📄 facenet.py
┗ 📄 detect_face.py
┗ 📄 face.result
┗ 📄 realtime_facenet_git.py

```





##  ✅ TODO

✅ Backend-Server의 backend_process.py와 연동 테스트

⬜ (할일작성)





## 기타

### 사용모듈 상세

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

- scipy 內 imresize 함수에 대한 지원이 없어짐에 따라 해당 항목을 PIL에 Image함수로 수정

- 비디오 카메라를 VPN망 내에 있는 카메라 기기로 변경
- 외부 모듈을 통한 시작을 위하여 절대경로 기준실행으로 변경



### 오류발생

**can't initialize gtk backend in function 'cvinitsystem'**

=> 파이썬 실행버전이 맞지않아 발생한 문제

나는 `pi` 유저에게 `python3.7`로 `GTK` 버전인 opencv를 설치하였으나,  이를 다른버전에 파이썬이나 유저(root) 등 으로 실행하려고 하면 해당 에러가 발생함.. (왜냐면 그들로는 정상적으로 Xwindow 표현을 할 수 없기 때문.)

https://github.com/opencv/opencv/issues/18461

https://unix.stackexchange.com/questions/94497/org-eclipse-swt-swterror-no-more-handles-gtk-init-check-failed-while-runnin

---

**os.system filenotfounderror errno2**

=> 파이썬 모듈내에 있는 파일 구문들이 `상대경로` 로 작성되어 있고, 이를 불러오는 경우 오류발생. (즉, 이를 불러오게되면 실행명령이 시작되는 위치를 기준으로 상대경로가 작성되게 되는데, 모듈의 경우에는 모듈위치를 기준으로 상대경로가 작성되기 때문에 둘사이에 차이가 발생하여 파일을 못찾는 오류가 발생하게 된다.)

https://velog.io/@anjaekk/python%EC%A0%88%EB%8C%80%EA%B2%BD%EB%A1%9C%EC%83%81%EB%8C%80%EA%B2%BD%EB%A1%9C-%EC%83%81%EB%8C%80%EA%B2%BD%EB%A1%9C-import-%EC%97%90%EB%9F%AC%EC%9D%B4%EC%9C%A0%EC%99%80-%ED%95%B4%EA%B2%B0

https://discuss.luxonis.com/d/175-depthai-gen2-example-01-rgb-preview-fail-to-run-on-raspberry-pi-zero-w/22os.system filenotfounderror errno2



### 참고사이트

**텐서플로우 설치 : https://angel-breath.tistory.com/19**



**파이썬 결과 return을 bash에게 : https://sujinlee.dev/python/python-return-values-to-shell-script/**
