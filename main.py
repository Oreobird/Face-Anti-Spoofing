import tensorflow as tf
from tensorflow.python.platform import app
import os
import numpy as np
import cv2
import sys
import argparse
import datasets
import models


def main(unused_args):
    PROJECT_DIR = FLAGS.proj_dir
    NUAA_DATA_DIR = PROJECT_DIR + 'data/NUAA/'
    
    MODEL_DIR = PROJECT_DIR + 'model/'
    LOG_DIR = PROJECT_DIR + 'log/'
    
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
        
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
        
    face_landmark_path = MODEL_DIR + 'shape_predictor_68_face_landmarks.dat'
    
    INPUT_SIZE = 128
    BATCH_SIZE = 64
    EPOCHS = 50
    
    CLASS_NAMES = ['fake', 'live']
    CLASS_NUM = len(CLASS_NAMES)
    
    LOAD_WEIGHT = True
    
    if FLAGS.train:
        LOAD_WEIGHT = False

    dataset = datasets.NUAA(NUAA_DATA_DIR, batch_size=BATCH_SIZE, input_size=INPUT_SIZE, class_num=CLASS_NUM)

    net = models.FasNet(dataset, CLASS_NUM, batch_size=BATCH_SIZE, input_size=INPUT_SIZE,
                        fine_tune_model_file=MODEL_DIR + 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    net.train(MODEL_DIR + 'fas_model.h5', MODEL_DIR, LOG_DIR, max_epoches=EPOCHS, load_weight=LOAD_WEIGHT)

    if FLAGS.train:
        net.predict()
    
    if FLAGS.online:
        import dlib
        from imutils import face_utils
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to connect to camera.")
            exit(-1)
    
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(face_landmark_path)
    
        frames = []
    
        def crop_face_loosely(shape, img, input_size):
            x = []
            y = []
            for (_x, _y) in shape:
                x.append(_x)
                y.append(_y)
        
            max_x = min(max(x), img.shape[1])
            min_x = max(min(x), 0)
            max_y = min(max(y), img.shape[0])
            min_y = max(min(y), 0)
        
            Lx = max_x - min_x
            Ly = max_y - min_y
        
            Lmax = int(max(Lx, Ly))
        
            delta = Lmax // 2
        
            center_x = (max(x) + min(x)) // 2
            center_y = (max(y) + min(y)) // 2
            start_x = int(center_x - delta)
            start_y = int(center_y - delta - 30)
            end_x = int(center_x + delta)
            end_y = int(center_y + delta)
        
            if start_y < 0:
                start_y = 0
            if start_x < 0:
                start_x = 0
            if end_x > img.shape[1]:
                end_x = img.shape[1]
            if end_y > img.shape[0]:
                end_y = img.shape[0]
        
            crop_face = img[start_y:end_y, start_x:end_x]
        
            # cv2.imshow('crop_face', crop_face)
            img_hsv = cv2.cvtColor(crop_face, cv2.COLOR_BGR2HSV)
            img_ycrcb = cv2.cvtColor(crop_face, cv2.COLOR_BGR2YCrCb)
        
            img_hsv = cv2.resize(img_hsv, (input_size, input_size)) / 255.0
            img_ycrcb = cv2.resize(img_ycrcb, (input_size, input_size)) / 255.0
        
            return img_hsv, img_ycrcb, start_y, end_y, start_x, end_x
    
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                face_rects = detector(frame, 0)
            
                if len(face_rects) > 0:
                    shape = predictor(frame, face_rects[0])
                    shape = face_utils.shape_to_np(shape)
                
                    input_img_hsv, input_img_ycrcb, start_y, end_y, start_x, end_x = crop_face_loosely(shape, frame,
                                                                                                       INPUT_SIZE)
                
                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), thickness=2)
                
                    frames.append({'hsv': input_img_hsv, 'yuv': input_img_ycrcb})
                    if len(frames) == 1:
                        # print(shape[30])
                        pred = net.test_online(frames)
                    
                        # print(pred)
                        idx = np.argmax(pred[0])
                        if pred[0][0] < 0.85:
                            idx = 1
                    
                        text = CLASS_NAMES[idx] + ":" + str(pred[0][idx])
                        print(text)
                        cv2.putText(frame, text, (start_x, start_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
                        frames = []
                
                    cv2.imshow("frame", frame)
                    # cv2.waitKey(0)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--proj_dir",
        type=str,
        default="",
        help="Project dir"
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=False,
        help="Train model"
    )
    parser.add_argument(
        "--online",
        type=bool,
        default=False,
        help="Test face input via camera"
    )
    return parser.parse_known_args()

if __name__ == "__main__":
    FLAGS, unparsed = parse_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)