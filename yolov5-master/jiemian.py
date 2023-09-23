
import tkinter as tk
# from tkinter import Label, filedialog,ttk #导入文件对话框函数库
from tkinter import *
from tkinter import filedialog,ttk
from PIL import Image, ImageTk # 导入图像处理函数库


def resize(w,h,w_box,h_box,pil_image):
    w_box = 300
    h_box = 250
    f1 = 1.0*w_box/w # 1.0 forces float division in Python2
    f2 = 1.0*h_box/h
    factor = min([f1, f2])
  #print(f1, f2, factor) # test
  # use best down-sizing filter
    width = int(w*factor)
    height = int(h*factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)
