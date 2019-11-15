# import numpy as np
# import cv2 as cv
#
# class Rat_state:
#     def __init__(self):
#         self.direction = 0
#         self.x = 0
#         self.y = 0
#     # 合并相似的框
#     def NMS(self,dets, thresh):
#
#         boxes = np.array(dets)
#         x1 = boxes[:, 0]    # x min
#         y1 = boxes[:, 1]    # y min
#         x2 = boxes[:, 2]    # x max
#         y2 = boxes[:, 3]    # y max
#         scores = boxes[:, 4]
#         print(scores)
#
#         areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#         keep = []  #最后保留的边框
#
#         for i in range(len(scores)-1):
#             # 计算窗口i与其他所有窗口的交叠部分的面积
#             print("**********")
#             xx1 = np.maximum(x1[i], x1[i+1:])
#             yy1 = np.maximum(y1[i], y1[i+1:])
#             xx2 = np.minimum(x2[i], x2[i+1:])
#             yy2 = np.minimum(y2[i], y2[i+1:])
#
#             w = np.maximum(0.0, xx2 - xx1 + 1)
#             h = np.maximum(0.0, yy2 - yy1 + 1)
#             inter = w * h
#
#             # 交/并得到iou值
#             ovr = inter / (areas[i] + areas[i+1:] - inter)
#
#             delet_box = []
#             if len(ovr) != 1:   # 表示不是最后两个方框进行IOU计算
#                 ovr_flag = 0
#                 for ii in range(len(ovr)):
#                     if ovr[ii] <= thresh:  # 代表两个不同的bounding box 取两个框
#                         if ovr_flag == len(ovr):  # 所有阈值都小于bonding box 只有一个
#                             ovr_flag += 1
#                     else:  # 代表有bounding box 可以合并
#                         delet_box.append(i+ii+1)
#                         x1[i] = np.minimum(x1[i], x1[i + ii+1])
#                         y1[i] = np.minimum(y1[i], y1[i + ii+1])
#                         x2[i] = np.maximum(x2[i], x2[i + ii+1])
#                         y2[i] = np.maximum(y2[i], y2[i + ii+1])
#                         ovr_flag += 1
#                 keep.append([x1[i], y1[i], x2[i], y2[i]])
#             else:
#                 if ovr <= thresh:  # 代表两个不同的bounding box 取两个框
#                     if i not in delet_box:
#                         keep.append([x1[i],y1[i],x2[i],y2[i]])
#                         keep.append([x1[i+1], y1[i+1], x2[i+1], y2[i+1]])
#                     else:
#                         keep.append([x1[i + 1], y1[i + 1], x2[i + 1], y2[i + 1]])
#                 else:         #代表两个相同的bounding box  取两个框的最大框
#                     keep.append([np.minimum(x1[i],x1[i+1]),np.minimum(y1[i],y1[i+1]),np.maximum(x2[i],x2[i+1]),np.maximum(y2[i],y2[i+1])])
#         print(keep)
#         return keep
#
# from util import tool
# if __name__ == '__main__':
#     image = cv.imread('1.jpg')
#
#     cur = tool.Rat_state()
#     bounding_boxes = [(187, 82, 337, 317,0.9),
#                      (150, 67, 305, 282,0.75),
#                      (246, 121, 368, 304,0.8),
#                      (0, 0, 100, 200, 0.8)]
#     box_num = cur.NMS(bounding_boxes,0.4)
#     print(box_num)
#     for i in range(len(box_num)):
#         deb = bounding_boxes[box_num[i]]
#         cv.rectangle(image,(deb[0],deb[1]),(deb[2],deb[3]),(0, 255, 255), 2)
#
#     cv.imshow('image', image)
#     cv.waitKey(0)



# python 使用类创建结构体
# class Myclass(object):
#     class Struct(object):
#         def __init__(self, name, age, job):
#             self.name = name
#             self.age = age
#             self.job = job
#
#     def make_struct(self, name, age, job):
#         return self.Struct(name, age, job)
#
# myclass = Myclass()
# test1 = myclass.make_struct('xsk', '22', 'abc')
# test2 = myclass.make_struct('mtt', '23', 'def')
# test1.name = 1
# test1.age = 0
# test1.job = 0
#
# print (test1.name)
# print (test1.age)
# print (test1.job)



import cv2
import numpy as np
from pykalman import KalmanFilter
import pykalman
# pykalman.UnscentedKalmanFilter

frame = np.zeros((800, 800, 3), np.uint8)
kf = KalmanFilter(transition_matrices=np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]),
                  observation_matrices=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
                  transition_covariance=0.03 * np.eye(4))
# transition_matrices：公式中的A
# observation_matrices：公式中的H
# transition_covariance：公式中的Q

t = 0


# 状态值为x_t=[x,y,dx,dy],其中(x,y)为鼠标当前位置，（dx,dy）指速度分量
# 直接获得的观测为位置(x,y)

def mousemove(event, x, y, s, p):
    global t, filtered_state_means0, filtered_state_covariances0, lmx, lmy, lpx, lpy
    current_measurement = np.array([np.float32(x), np.float32(y)])
    cmx, cmy = current_measurement[0], current_measurement[1]
    if t == 0:
        filtered_state_means0 = np.array([0.0, 0.0, 0.0, 0.0])
        filtered_state_covariances0 = np.eye(4)
        lmx, lmy = 0.0, 0.0
        lpx, lpy = 0.0, 0.0

    filtered_state_means, filtered_state_covariances = (
    kf.filter_update(filtered_state_means0,filtered_state_covariances0, current_measurement))
    cpx, cpy = filtered_state_means[0], filtered_state_means[1]
    # 绘制测量值轨迹（绿色）
    cv2.line(frame, (int(lmx), int(lmy)), (int(cmx), int(cmy)), (0, 100, 0))
    # 绘制预测值轨迹（红色）
    cv2.line(frame, (int(lpx), int(lpy)), (int(cpx), int(cpy)), (0, 0, 200))
    filtered_state_means0, filtered_state_covariances0 = filtered_state_means, filtered_state_covariances
    t = t + 1
    lpx, lpy = filtered_state_means[0], filtered_state_means[1]
    lmx, lmy = current_measurement[0], current_measurement[1]


cv2.namedWindow("kalman_tracker")
cv2.setMouseCallback("kalman_tracker", mousemove)
while True:
    cv2.imshow("kalman_tracker", frame)
    if (cv2.waitKey(30) & 0xff) == 27:
        break

cv2.destroyAllWindows()
