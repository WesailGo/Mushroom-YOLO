import numpy
import cv2

def open(file_path):
    cap = cv2.VideoCapture(file_path)

    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('image', frame)
        k = cv2.waitKey(20)  # q键退出
        if (k & 0xff == ord('q')):
            cap.release()
            cv2.destroyAllWindows()
            break

# cap.release()
# cv2.destroyAllWindows()
