import cv2
import numpy as np

import numpy as np
import cv2
import os


CUR_DIR = os.getcwd()
CUR_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
VEDIO_DIR = os.path.join(CUR_DIR, "VEDIO\\")
all_file = os.listdir(VEDIO_DIR)
for vedio in all_file:
    if os.path.splitext(vedio)[1] == '.avi':  # 分离文件名和扩展名
        Vedio_file = os.path.join(VEDIO_DIR, vedio)
        cap = cv2.VideoCapture(Vedio_file)
        fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        # fgbg = cv2.createBackgroundSubtractorKNN()
        while (cap.isOpened()):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1024, 760), interpolation=cv2.INTER_CUBIC)
            fgmask = fgbg.apply(frame)
            cv2.imshow('frame', fgmask)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

