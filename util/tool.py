import numpy as np
import cv2 as cv
from filterpy.kalman import KalmanFilter
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
import math

# 创建小鼠状态结构体
class Rat_state(object):
    def __init__(self, x, y, vx, vy, w, h,id):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.w = w
        self.h = h
        self.id = id

    def STATE(self, x, y, vx, vy, w, h, id):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.w = w
        self.h = h
        self.id = id
        return self.x, self.y, self.vx, self.vy, self.w, self.h, self.id

class Rat_num(object):
    def __init__(self,last_num, cur_num):
        self.last_num = last_num
        self.cur_num = cur_num
    def num(self,last_num,cur_num):
        self.cur_num = cur_num
        self.last_num = last_num
        return self.cur_num,self.cur_num

class Rat_Deal:
    def __init__(self):
        self.direction = 0
        self.x = 0
        self.y = 0

    # 合并相似的框
    def NMS(self,dets, thresh):
        boxes = np.array(dets)
        """Pure Python NMS baseline."""
        x1 = boxes[:, 1]   #xmin
        y1 = boxes[:, 0]   #ymin
        x2 = boxes[:, 3]   #xmax
        y2 = boxes[:, 2]   #ymax
        scores = boxes[:, 4]  # bbox打分

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # 打分从大到小排列，取index
        order = scores.argsort()[::-1]
        # keep为最后保留的边框
        keep = []
        out_box = []
        while order.size > 0:
            # order[0]是当前分数最大的窗口，肯定保留
            i = order[0]
            keep.append(i)
            # 计算窗口i与其他所有窗口的交叠部分的面积
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # 交/并得到iou值
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
            inds = np.where(ovr <= thresh)[0]
            # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
            order = order[inds + 1]
        Center = []
        for i in range(len(keep)):
            out_box.append(boxes[keep[i]])
            Center.append(((boxes[keep[i]][1] + boxes[keep[i]][3]) / 2, (boxes[keep[i]][0] + boxes[keep[i]][2]) / 2))
        return keep,out_box,Center

    def Center(self, x):
        Center = []
        for i in range(len(x)):
            Center.append(((x[i][1]+x[i][3])/2,(x[i][0]+x[i][2])/2))
        return (Center)

# kalman预测Class
class KalmanPredict():

    def __init__(self,id):
        '''
        :param location_Array:
        '''
        # 创建一个kalman预测器模型
        # Location = F*Location- + Q
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)  #状态转移矩阵
        self.kf.H = np.array([[1,0,0,0],[0,1,0,0]],np.float32)  #测量矩阵
        self.kf.P *= 2.
        self.kf.Q *= 0.003  # processNoiseCov
        self.kf.R = 1.
        # self.kf.R[2:, 2:] *= 10.
        # self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        # self.kf.P *= 10.
        # self.kf.Q[-1, -1] *= 0.01
        # self.kf.Q[4:, 4:] *= 0.01

        self.id = id

    def update(self,location):
        self.kf.update(location)
    def predict(self):
        self.kf.predict()
        A =  self.kf.x
        return (A[0,0],A[1,0])
'''
# Array1:上一帧带标 坐标位置，Array2:当前帧坐标位置
# 返回一个3x3的位置矩阵 
    0
    1
    2
    3
        0 1 2 3 4   当前帧
'''
def Hungarian_Array(last_array,cur_array):
    #创建一个3*3数组
    # Distance_Array = [([0] * 3) for i in range(3)]
    dist = []
    for location in last_array:
        for curren_location in cur_array:
            dist.append(np.sqrt(np.sum(np.square(location - curren_location))))
    dist_array = np.array(dist, dtype=float)
    dist_array_reshaped = dist_array.reshape((len(last_array),len(cur_array)))
    return dist_array_reshaped
def Hungarian_sklearn(cost):
    matched_indices = linear_assignment(cost)
    return matched_indices


# 返回Hungarian sort结果
def Hungarian_Result(last_array, cur_array):
    result = Hungarian_Array(last_array, cur_array)
    result2 = Hungarian_sklearn(result)

    return result2

# 由Rat_State生成用于Hungarian的array
def Rat_State_Sort(last_rat):
    Rat_state_list = []
    for i in range(len(last_rat)):
        Rat_state_list.append((last_rat[i].x,last_rat[i].y))
    Rat_state_array = np.array(Rat_state_list)
    return Rat_state_array


# 检测识别出来的框中心是否是正确的detect_box()  九点判断法  计算的是一个中心的
#
def detect_box(img,img_gray, center):
    Num = 3
    total_num = 0
    nine_value = [[], [], []]
    return_center = [0, 0]
    for i in range(Num):
        for j in range(Num):
            if (0 < (center[0] + (i - 1) * 30) < 1024) and (0 < (center[1] + (j - 1) * 30) < 760):
                cv.circle(img_gray, (np.int(center[0] + (i - 1) * 30), np.int(center[1] + (j - 1) * 30)), 5,
                           (255, 0, 0), 2)
                print(img_gray[np.int(center[1] + (j - 1) * 30)][np.int(center[0] + (i - 1) * 30)])
                if img_gray[np.int(center[1] + (j - 1) * 30)][np.int(center[0] + (i - 1) * 30)] > 105:
                    total_num = total_num + 1
                    nine_value[i].append(1)
                else:
                    nine_value[i].append(0)
            # 避免超出边界后数组的
            else:
                nine_value[i].append(0)
    print(nine_value)
    nine_value = (np.array(nine_value).T).tolist()
    up_down = []
    left_right = []
    # 表示起码有一个点在老鼠身上 并且中心点没在老鼠身上
    if nine_value[1][1] < 100 and max(max(nine_value)) != 0:
        for i in range(3):
            temp = 0
            temp1 = 0
            for j in range(3):
                temp = temp + nine_value[i][j]
                temp1 = temp1 + nine_value[j][i]
            up_down.append(temp)
            left_right.append(temp1)
        print(up_down,left_right)
        if 0 in [up_down.index(int(max(up_down)))]:
            return_center[1] = np.int(center[1]) - 20
        elif 2 in [up_down.index(int(max(up_down)))]:
            return_center[1] = np.int(center[1]) + 20
        else:
            return_center[1] = np.int(center[1])
        if 0 in [left_right.index(int(max(left_right)))]:
            return_center[0] = np.int(center[0]) - 20
        elif 2 in [left_right.index(int(max(left_right)))]:
            return_center[0] = np.int(center[0]) + 20
        else:
            return_center[0] = np.int(center[0])
        cv.circle(img_gray, (np.int(return_center[0]), np.int(return_center[1])), 10, (255, 0, 0), 2)
    else:
        return_center = center
    return total_num,return_center

# 在同一帧上面进行三只老鼠距离检测
def Cal_distance(center_i):
    center = np.array(center_i)
    Num = 3
    distant_list = []
    for i in range(Num - 1):
        for j in range(i + 1, Num, 1):
            distant_list.append(np.sqrt(np.sum(np.square(center[i] - center[j]))))
    return distant_list

# 计算center的欧式距离  用于判断老鼠是否挨到一起
# 返回 距离较近的两个边框index  Xmin Xmax Ymin Ymax
def detect_Euclidean(center_i):
    center = np.array(center_i)
    close_list = []
    Num = 3
    for i in range(Num-1):
        for j in range(i+1,Num,1):
            # 判断距离是否小于300 如果小于300则代表两只老鼠靠近  否则则代表两只老鼠单独活动
            if np.sqrt(np.sum(np.square(center[i]- center[j]))) < 300:
                xmin = max((min(np.int(center[i][0]), np.int(center[j][0])) - 50),0)
                xmax = min((max(np.int(center[i][0]), np.int(center[j][0])) + 50),760)
                ymin = max((min(np.int(center[i][1]), np.int(center[j][1])) - 50),0)
                ymax = min((max(np.int(center[i][1]), np.int(center[j][1])) + 50),1024)
                close_list.append((i,j,xmin,xmax,ymin,ymax))
    return close_list


CAMERA_DARK = 5

def zMaxChroma(pixel):
    return max(pixel)/sum(pixel)


Lambda = 0.4
camDark = 0.4
def zSpecularFreeImage(img):
    for x in range(400,600,1):
        for y in range(400,600,1):
            img[x][y][0] = int(img[x][y][0] * 0.2)
            img[x][y][1] = int(img[x][y][1] * 0.2)
            img[x][y][2] = int(img[x][y][2] * 0.2)
    return img


class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.

    High-frequency filters implemented:
        butterworth
        gaussian
    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H
    """

    def __init__(self, a=0.5, b=1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = 1 / (1 + (Duv / filter_params[0] ** 2) ** filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = np.exp((-Duv / (2 * (filter_params[0]) ** 2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b * H) * I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H=None):
        """
        Method to apply homormophic filter on an image
        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) is not 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter == 'butterworth':
            H = self.__butterworth_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'gaussian':
            H = self.__gaussian_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'external':
            print('external')
            if len(H.shape) is not 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')

        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I=I_fft, H=H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt)) - 1
        return np.uint8(I)

def Illumi_adjust(alpha, img):
    if alpha > 0:
        img_out = img * (1 - alpha) + alpha * 255.0
    else:
        img_out = img * (1 + alpha)
    return img_out / 255.0
# def Sort_Num(hungarian_array,Array):
#     # temp_array = Array
#     temp_array = Array.copy()
#     for i in range(len(hungarian_array)):
#         Array[i] = temp_array[hungarian_array[i][1]]
#     return Array

from util import tool
# if __name__ == '__main__':
#     image = cv.imread('0.jpg')
#     cur = tool.Rat_Deal()
#     # bounding_boxes = [(317,187, 82, 337, 0.9),
#     #                  (282,150, 67, 305, 0.75),
#     #                  (304,246, 121, 368, 0.8)]
#     bounding_boxes = [(611.4058, 96.72619, 747.07983, 175.57538, 0.99297357),
#                       (629.2013, 608.4068, 744.9247, 738.6535, 0.9287137),
#                       (24.876785, 328.37402, 129.77792, 472.88812, 0.376685),
#                       (31.264349, 373.33978, 88.31443, 462.73773, 0.3113263)]
#     print((bounding_boxes))
#     box_num = cur.NMS(bounding_boxes,0.2)
#     for i in range(len(box_num)):
#         deb = bounding_boxes[box_num[i]]
#         cv.rectangle(image,(np.int(deb[1]), np.int(deb[0])), (np.int(deb[3]), np.int(deb[2])),(0, 255, 255), 2)
#
#     cv.imshow('image', image)
#     cv.waitKey(0)
# from util import tool
# if __name__ == '__main__':
#     cur = tool.KalmanPredict(3)
#     print(cur.id)
# from util import tool
# if __name__ == '__main__':
#     num = tool.Rat_num(0,0)
#     num.cur_num = 1
#     print(num.last_num,num.cur_num)

# if __name__ == "__main__":
#     list = []
#     a = np.array([[0,0],[1,1],[2,2],[3,3]])          # 原始array
#     b = np.array([[0,0],[2,2],[3,3]])                # 当前检测到的array
#     print("**************")
#     index = Hungarian_Result(a,b)
#     index_a = index[:,0]
#     current_pre = []
#     A= index_a.tolist()
#     for i in range(4):
#         if i in index_a:
#             current_pre.append(b[A.index(i)])
#         else:
#             current_pre.append([1,1])
#     print(current_pre)
#     # for i in range(len(a)):
#     #     list.append(b[index_a[i]])
#     # print(list)
#     # print(Hungarian_sklearn(Hungarian_Array(a,b)))
#     # last_location = Sort_Num(Hungarian_sklearn(Hungarian_Array(a,b)),b)

# if __name__ == "__main__":
#     A = []
#     A.append(tool.Rat_state(0, 1, 2, 3, 4, 5, 6))
#     A.append(tool.Rat_state(0, 2, 2, 3, 4, 5, 6))
#     array = Rat_State_Sort(A)
#     print("a",array)


from util.tool import detect_Euclidean
if __name__ == "__main__":
    center = np.array([(1,1),(2,2),(3,3)])
    detect_Euclidean(center)