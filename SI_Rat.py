# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import time

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
from util import tool
from util.tool import detect_box,detect_Euclidean,zSpecularFreeImage,HomomorphicFilter,Illumi_adjust

class yoloV3(YOLO):
    _defaults = {
        "model_path": 'logs/000/trained_weights.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # 创建bonding box 和 score
        out_box_score = []
        for i in range(len(out_boxes)):
            out_box_score.append((out_boxes[i][0],out_boxes[i][1],
                                 out_boxes[i][2],out_boxes[i][3],
                                 out_scores[i]))
            # out_box_score.append(((np.append(out_boxes[i], out_scores[i]))))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            # My kingdom for a good redistributable image drawing library.
            # for i in range(thickness):
            #     draw.rectangle(
            #         [left + i, top + i, right - i, bottom - i],
            #         outline=self.colors[c])
            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     fill=self.colors[c])
            # draw.text(text_origin, label, fill=(255, 0, 0), font=font)
            # del draw
        return image,out_box_score

    def close_session(self):
        self.sess.close()

def test_model(img):
    yoloV3_test = yoloV3()
    image = Image.open(img)
    r_image = yoloV3_test.detect_image(image)
    r_image.show()
    yoloV3_test.close_session()

def contrast_brightness(image_temp,c,b,blank):
    # blank=np.zeros_like(image,image.dtype)
    # 计算两个数组的加权和(dst = alpha*src1 + beta*src2 + gamma)
    #dst=cv.addWeighted(image,c,white,c,b)
    dst=cv2.addWeighted(image_temp,c,blank,1-c,b)#这样才能增加对比度
    return dst
import time as TI



def read_Vedio():
    yoloV3_test = yoloV3()
    Rat_method = tool.Rat_Deal()

    fobj = open("point_location.txt", 'w')

    Rat_State_last = []  # 用于储存上一帧老鼠的状态
    Rat_State_cur = []  # 用于储存当前帧老鼠的状态
    Rat_num = tool.Rat_num(0,0)
    """
    用于保存变量
    """
    Kalman_list = []    #卡尔曼列表
    current_pre = []
    center_new = []  # 检测到的数量少 然后进行重新排列
    STATE_List = []
    RAT_NUM = 3

    image_test = 0

    img_num = 0

    CUR_DIR = os.getcwd()
    VEDIO_DIR = os.path.join(CUR_DIR, "VEDIO\\")
    VEDIO_DIR = "D:\\Downloads\\RatSI\\RatSI_videos\\videos"
    all_file = os.listdir(VEDIO_DIR)
    for vedio in all_file:
        if os.path.splitext(vedio)[1] == '.mp4':  # 分离文件名和扩展名
            Vedio_file = os.path.join(VEDIO_DIR, vedio)
            cap = cv2.VideoCapture(Vedio_file)

        while (cap.isOpened()):

            center_new.clear()  # 清空列表
            Rat_State_cur.clear()    # 清空list数据
            current_pre.clear()      # 清空预测的目标中心值
            ret, frame = cap.read()

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # openCV转PIL

            r_image, out_boxes_score = yoloV3_test.detect_image(image)
            imggg = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)
            print(out_boxes_score)
            if len(out_boxes_score)!=0:
                for i in range(len(out_boxes_score)):
                    cv2.circle(imggg, (np.int(((out_boxes_score[i][1])+ (out_boxes_score[i][3]))/2), \
                                       np.int(((out_boxes_score[i][0])+ (out_boxes_score[i][2]))/2)),\
                                        3, (255, 0, 0), 2)


            cv2.imshow("openCV", imggg)
            k = cv2.waitKey(1)
            if (k & 0xff == ord('q')):
                break


        cap.release()
    yoloV3_test.close_session()


import cv2
import os
from PIL import Image, ImageFont, ImageDraw

def read_camera():
    my_yoloV3 = yoloV3()

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    capture.set(cv2.CAP_PROP_CONTRAST, 150)  # 设置对比度
    capture.set(cv2.CAP_PROP_SATURATION, 50)  # 设置饱和度
    capture.set(cv2.CAP_PROP_HUE, 50)  # 设置色调
    capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)  # 设置亮度
    # capture.set(cv2.CAP_PROP_FPS, 30)
    FPS = capture.get(cv2.CAP_PROP_FPS)  # 设置帧率
    capture.set(cv2.CAP_PROP_EXPOSURE, 299.8)  # 设置曝光
    # 视频的编码
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))

    if not capture.isOpened():
        raise IOError("Couldn't open webcam or video")

    # 定义输出视频
    video = cv2.VideoWriter("out.mp4", fourcc, 100, (1080, 720))
    prev_time = 0
    while (True):
        # 获取一帧
        ret, frame = capture.read()
        image = Image.fromarray(frame)
        image,out_boxes = my_yoloV3.detect_image(image)
        result = np.asarray(image)
        curr_time = time.time()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        print(exec_time)
        cv2.imshow("result", result)

        while cv2.waitKey(500) | 0xFF == ord('q'):
            cv2.destroyAllWindows()
            # break
    my_yoloV3.close_session()


if __name__ == "__main__":
    read_Vedio()
