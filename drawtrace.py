import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import _flatten

def get_maxpixel(img,pixel1,pixel2):
    pixel = []
    for w in range(pixel1[0],pixel2[0]):
        for h in range(pixel1[1],pixel2[1]):
            pixel.append(img[w][h][2])
    return min(pixel)

# 计算像素点对应的对热图中的位置

def cal_hot(pixel_position):
    x_position = pixel_position[0]//16
    y_position = pixel_position[1]//16
    if x_position < 0:
        x_position = 0
    elif x_position > 39:
        x_position = 39
    if y_position < 0:
        y_position = 0
    elif y_position >29:
        y_position = 29
    return (x_position,y_position)
# 处理
def pretreatment(pixel_position):
    a1 = pixel_position[0]
    a2 = pixel_position[1]
    if a1 < 0:
        a1 = 0
    elif a1 >= 640:
        a1 = 639
    if a2 < 0:
        a2 = 0
    elif a2 >= 480:
        a2= 479
    return (a1,a2)

def cal_velocity(pixel1,pixel2):
    return np.sqrt(((pixel2[0] - pixel1[0])**2)+((pixel2[1] - pixel1[1])**2))

# 将整张图分格子 每个点投进该格子 该格子数值加一
# 读取txt文件数据
f = open('D:\\Documents\\TestProject_multi\\1\\Tracking_0.txt')
# f = open('F:\\Project\\Machinelearning\\Vedio_libo\\point_location1.txt')
# Init hot figure array
List_location = [[0]*40 for row in range(30)]

Line_length = 0
while True:
    lines = f.readline()                         # readline
    if lines:
        Line_length = Line_length + 1
        list_str = re.findall(r"\d+\.?\d*", lines)  # 查找数字
        # x = int(float(list_str[1]))   # str转为整型  自己的数据
        # y = int(float(list_str[2]))
        x = int(float(list_str[3]))   # str转为整型
        y = int(float(list_str[4]))
        temp_hot = cal_hot((x,y))
        print(temp_hot)
        List_location[temp_hot[1]][temp_hot[0]] = List_location[temp_hot[1]][temp_hot[0]]+1
        # print(List_location)
    else:       # 读到文本最后一句跳出
        break

A = np.array(List_location)/Line_length
plt.figure(1)
sns.set()
print(len(A),len(A[0]))
ax1 = sns.heatmap(A, annot=False,cmap="coolwarm")
plt.figure(2)
B = (list(_flatten(List_location)))
plt.plot(B)
plt.show()

# # calculate the Trajectory
# first_flag = 0
# img = np.ones((760,1024,3),dtype=np.uint8)*125
# f = open('F:\\Project\\Machinelearning\\Vedio_libo\\point_location.txt')
# # f = open('D:\\Documents\\TestProject\\1\\Tracking_0.txt')
# jpeg = cv2.imread('F:\\Project\\Machinelearning\\Vedio_libo\\Vedio\\image\\0.jpg')
# temp_position = []
# lost_frame = 0
# while True:
#     lines = f.readline()                         # readline
#     if lines:
#         list_str = re.findall(r"\d+\.?\d*", lines)  # 查找数字
#         if list_str:
#             x = int(float(list_str[1]))   # str转为整型
#             y = int(float(list_str[2]))
#             if x <= 0 or y <= 0 or x > 640 or y > 480:
#                 print(x,y)
#             else:
#                 location = pretreatment((x,y))
#                 if first_flag != 0:
#                     cv2.line(jpeg, (temp_position[0], temp_position[1]),\
#                              (location[0], location[1]), (0,255,0),3)
#                     temp_position.clear()
#                     temp_position.append(location[0])
#                     temp_position.append(location[1])
#                 else:
#                     first_flag = first_flag + 1
#                     temp_position.clear()
#                     temp_position.append(location[0])
#                     temp_position.append(location[1])
#         else:
#             lost_frame = lost_frame + 1
#             print(lost_frame)
#     else:
#         break
# cv2.imwrite("trace.jpg",jpeg)
# cv2.imshow('img',jpeg)
# cv2.waitKey(0)


# 计算老鼠速度
# 计算需要保存上一步老鼠数据
temp_data = []
figure_data = []
f = open('F:\\Project\\Machinelearning\\Vedio_libo\\point_location.txt')
while True:
    lines = f.readline()
    if lines:
        list_str = re.findall(r"\d+\.?\d*", lines)
        if list_str:
            x = (float(list_str[1]))   # str转为整型
            y = (float(list_str[2]))
            if temp_data:
                EuropeanDis = cal_velocity(temp_data[0],(x,y))
                temp_data.clear()
                temp_data.append((x,y))
                figure_data.append(EuropeanDis)
                print(EuropeanDis)
            else:
                temp_data.clear()
                temp_data.append((x,y))
    else:
        break
plt.plot(figure_data)
plt.show()













# img = np.ones((760,1024,3),dtype=np.uint8)*125
# b = []
# flag = 0
# temp_position = []    # 里面数据实时更新
# f = open('point_location2.txt')
#
# while True:
#     lines = f.readline()
#     # lines.encode("utf-8")
#     list_str = re.findall(r"\d+\.?\d*", lines)
#
#     if not lines:
#         break
#     else:
#         max_pixel = get_maxpixel(img,
#                                  (round(float(list_str[1])) - 20,round(float(list_str[2])) - 20),
#                                  (round(float(list_str[1])) + 20, round(float(list_str[2])) + 20))
#
#         if flag != 0:
#             # cv2.line(img,(temp_position[0],temp_position[1]),(round(float(list_str[1])),round(float(list_str[2]))),(0,255,0),3)
#             # cv2.circle(img, (round(float(list_str[1])),round(float(list_str[2]))), 1, ( 0,255, 0), 8)
#             # cv2.circle(img, (int(list_str[4]), int(list_str[5])), 3, (255, 0, 0), 8)
#             # cv2.circle(img, (int(list_str[7]), int(list_str[8])), 3, (0, 0, 255), 8)
#             # cv2.line(img,(temp_position[0],temp_position[1]),(round(float(list_str[1])),round(float(list_str[2]))),(0,255,0),3)
#             # cv2.line(img,(temp_position[2],temp_position[3]),(int(list_str[4]),int(list_str[5])),(255,0,0),3)
#             # cv2.line(img,(temp_position[4],temp_position[5]),(int(list_str[7]),int(list_str[8])),(0,0,255),3)
#             # cv2.rectangle(img,(110,113),(200,300),(255, 0, 0), thickness=-1)
#             cv2.rectangle(img,\
#                           (round(float(list_str[1]))-10,round(float(list_str[2]))-10),\
#                           (round(float(list_str[1]))+ 10,round(float(list_str[2])) + 10),\
#                           (0, \
#                            0, \
#                            int(max_pixel-50)),thickness=-1)
#             print(img[150, 200][0])
#
#
#
#             temp_position.clear()
#             temp_position.append(round(float(list_str[1])))
#             temp_position.append(round(float(list_str[2])))
#             # temp_position.append(int(list_str[4]))
#             # temp_position.append(int(list_str[5]))
#             # temp_position.append(int(list_str[7]))
#             # temp_position.append(int(list_str[8]))
#         else:
#             flag = flag+1
#             temp_position.clear()
#             temp_position.append(round(float(list_str[1])))
#             temp_position.append(round(float(list_str[2])))
#             # temp_position.append(int(list_str[4]))
#             # temp_position.append(int(list_str[5]))
#             # temp_position.append(int(list_str[7]))
#             # temp_position.append(int(list_str[8]))
# cv2.imwrite("trace.jpg",img)
# cv2.imshow('img',img)
# cv2.waitKey(0)