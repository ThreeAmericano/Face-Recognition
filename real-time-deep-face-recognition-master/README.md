https://drive.google.com/file/d/1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz/view

다운로드 -> 20180408-102900 폴더에 넣기 (pre trained data)

활용 : https://github.com/bearsprogrammer/real-time-deep-face-recognition/blob/master/Make_classifier_git.py

# 가이드

가상환경 설정 : environment.yml
 > conda env create -f environment.yml

가상환경 실행
 > conda activate venv

사진 촬영 후 저장 폴더 (30장 이상)
 > ./faceData/train

얼굴 부분 사진 align
 > python Make_aligndata_git.py
 
 > ./output_dir 에 사진 저장됨 

사진 학습
 > python Make_classifier_git.py

실시간 분석
 > realtime_facenet_git.py -> line 41 학습한 사람 이름 변경 (HumanNames)

 > python realtime_facenet_git.py

![image](https://user-images.githubusercontent.com/12757811/129246692-a85308ca-6244-4efb-9cb8-f21a088a3a08.png)


# real-time-deep-face-recogniton

Real-time face recognition program using Google's facenet.
* [youtube video](https://www.youtube.com/watch?v=T6czH6DLhC4)
## Inspiration
* [OpenFace](https://github.com/cmusatyalab/openface)
* I refer to the facenet repository of [davidsandberg](https://github.com/davidsandberg/facenet).
* also, [shanren7](https://github.com/shanren7/real_time_face_recognition) repository was a great help in implementing.
## Dependencies
* Tensorflow 1.2.1 - gpu
* Python 3.5
* Same as [requirement.txt](https://github.com/davidsandberg/facenet/blob/master/requirements.txt) in [davidsandberg](https://github.com/davidsandberg/facenet) repository.
## Pre-trained models
* Inception_ResNet_v1 CASIA-WebFace-> [20170511-185253](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE/edit)
## Face alignment using MTCNN
* You need [det1.npy, det2.npy, and det3.npy](https://github.com/davidsandberg/facenet/tree/master/src/align) in the [davidsandberg](https://github.com/davidsandberg/facenet) repository.
## How to use
* First, we need align face data. So, if you run 'Make_aligndata.py' first, the face data that is aligned in the 'output_dir' folder will be saved.
* Second, we need to create our own classifier with the face data we created. <br/>(In the case of me, I had a high recognition rate when I made 30 pictures for each person.)
</br>Your own classifier is a ~.pkl file that loads the previously mentioned pre-trained model ('[20170511-185253.pb](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE/edit)') and embeds the face for each person.<br/>All of these can be obtained by running 'Make_classifier.py'.<br/>
* Finally, we load our own 'my_classifier.pkl' obtained above and then open the sensor and start recognition.
</br> (Note that, look carefully at the paths of files and folders in all .py)
## Result
<img src="https://github.com/bearsprogrammer/real-time-deep-face-recogniton/blob/master/realtime_demo_pic.jpg" width="60%">
