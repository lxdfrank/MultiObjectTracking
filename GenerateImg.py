import cv2
import os

CUR_DIR = os.getcwd()
VEDIO_DIR = os.path.join(CUR_DIR,"VEDIO\\")
print(VEDIO_DIR)
all_file=os.listdir(VEDIO_DIR)
print(all_file)
for vedio in all_file:
    if os.path.splitext(vedio)[1] == '.avi':  # 分离文件名和扩展名
        Vedio_file = os.path.join(VEDIO_DIR, vedio)
        cap = cv2.VideoCapture(Vedio_file)
    c = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        img_new = cv2.resize(frame, (1024,760), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('image', img_new)
        cv2.imwrite('images/'+str(c)+'.jpg',img_new)

        c = c + 1
        k = cv2.waitKey(1)
        if (k & 0xff == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()

