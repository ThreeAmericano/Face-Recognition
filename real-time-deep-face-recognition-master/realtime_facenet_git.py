# 파일 실행하는 명령어
# python Make_aligndata_git.py [촬영한 사진 자르면서 조정]
# python Make_classifier_git.py  [조정된 사진으로 학습]
# python realtime_facenet_git.py [실시간 얼굴 인식]
#####################################################################################
#
# 얼굴인식 로직
# 2021.08.29
#
#####################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import sys
import time
import copy
import math
import pickle
from sklearn.svm import SVC
#from sklearn.externals import joblib
#import sklearn.external.joblib as extjoblib
import joblib
from PIL import Image

# 사진 설정
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 44
frame_interval = 3
batch_size = 1000
image_size = 182
input_image_size = 160


def face_recognition(cam_info, detecting_time, time_limit):
    global minsize, threshold, factor, margin, frame_interval, batch_size, image_size, input_image_size
    who_is = "error"

    print('[FaceR] Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './npy')

            # 텐서플로우 모델 불러오기
            print('[FaceR] Loading feature extraction model')
            modeldir = './20180408-102900/20180408-102900.pb'   # 구글 드라이브에서 받은 미리 학습된 데이터
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # 학습한 데이터
            classifier_filename = './my_classifier/my_classifier.pkl'
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
                print('[FaceR] load classifier file-> %s' % classifier_filename_exp)

            # 캠 설정 하기 / cv2.VideoCapture(숫자) <= 연결된 캠이 1개라면 0, 2번째 캠을 설정하고 싶으면 1
            video_capture = cv2.VideoCapture(cam_info)
            c = 0

            # ----------------------------------------------------------------------------------------------------------------------------
            # --------------------------------------------------- 실시간 얼굴인식 부분 ----------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------------------------
            print('[FaceR] Start Recognition!')
            starting_time = time.time()
            prevTime = 0
            detectedPerson = [0, 0, 0, 0]  # 얼굴인식 시 배열에서 인식한 사람의 번호가 올라감
            detectingTime = detecting_time  # 감지시간 (해당시간이상 감지될 경우 인식함)
            HumanNames = ['hyeon', 'junho', 'park']   # 학습한 사람 이름 설정
            while \
                    detectedPerson[0] < detectingTime and \
                    detectedPerson[1] < detectingTime and \
                    detectedPerson[2] < detectingTime:  # 사람(detectedPerson)을 설정한 검사시간(detectingTime) 동안 인식할 때 까지 반복
                # 캠 설정
                ret, frame = video_capture.read()
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize frame (optional)

                # 시간제한에 걸렸는지 확인
                limit_interval = time.time() - starting_time
                if limit_interval > time_limit:
                    print("[FaceR] time limit, close the section")
                    break

                curTime = time.time()    # calc fps
                timeF = frame_interval
                if c % timeF == 0:
                    find_results = []

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('Face: {0}, Count: {1}, TimeLimit[{2:0.1}/{3}]'.format(nrof_faces, detectedPerson, limit_interval, time_limit)) # 디버그 구문.

                    if nrof_faces > 0:  # 1명 이상 인식하면
                        # 얼굴인식 처리 부분
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]

                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces,4), dtype=np.int32)

                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('[FaceR] face is inner of range!')
                                continue

                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[0] = facenet.flip(cropped[0], False)
                            scaled.append(np.array(Image.fromarray(cropped[0]).resize((image_size, image_size), Image.BILINEAR)).astype(np.double))  # scipy의 imresize가 지원종료되어 변경
                            scaled[0] = cv2.resize(scaled[0], (input_image_size,input_image_size), interpolation=cv2.INTER_CUBIC)
                            scaled[0] = facenet.prewhiten(scaled[0])
                            scaled_reshape.append(scaled[0].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[0], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                            # plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20

                            # 인식된 사람이 누구인지 확인하여 카운트
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]
                                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)

                                    # 인식한 사람 번호 : best_class_indices[0]
                                    detectedPerson[best_class_indices[0]] += 1
                    else:
                        print('[FaceR] Unable to align')

                # 화면 안에 적히는 글씨
                sec = curTime - prevTime
                prevTime = curTime
                fps = 1 / (sec)
                str = 'FPS: %2.1f' % fps
                text_fps_x = len(frame[0]) - 150
                text_fps_y = 20
                cv2.putText(frame, str, (text_fps_x, text_fps_y),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


            # 한 명이 설정 시간 이상 인식되면
            if detectedPerson[0] >= detectingTime :
                who_is = HumanNames[0]
            elif detectedPerson[1] >= detectingTime :
                who_is = HumanNames[1]
            elif detectedPerson[2] >= detectingTime :
                who_is = HumanNames[2]
            else:
                who_is = "none"
            print("[FaceR] detected Person is : ", who_is)

            video_capture.release()
            cv2.destroyAllWindows()

    return who_is


if __name__ == "__main__":
    print("main")

    # UV4L Streaming Path
    stream_url = "http://10.8.0.2:8090/stream/video.mjpeg"
    name = face_recognition(stream_url, 15, 60)
    if name == 'none':
        print("감지되지않음")
    elif name == 'error':
        print("잘못되엇슴")
    else:
        print("그의 이름은 %r" % name)

