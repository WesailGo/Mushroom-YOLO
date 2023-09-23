import tkinter as tk
# from tkinter import Label, filedialog,ttk #导入文件对话框函数库
from tkinter import *
from tkinter import filedialog,ttk
from PIL import Image, ImageTk # 导入图像处理函数库
import detect
import jiemian
import video

# 创建窗口 设定大小并命名
window = tk.Tk()
window.title('Yolov5--Wesail')
window.geometry('1000x700')

global img_png           # 定义全局变量 图像的
global file_path

im1 = Image.open("WuHan.png")
w1, h1 = im1.size
im1 =jiemian.resize(w1, h1, 450, 450, im1)
img1 = ImageTk.PhotoImage(im1)

im2 = Image.open("HangZhou.png")
w2, h2 = im2.size
im2 =jiemian.resize(w2, h2, 450, 450, im2)
img2 = ImageTk.PhotoImage(im2)

label = tk.Label(window,image=img2,width=500, height=500)
label.grid(row="0", column="0",sticky=NW,pady=0)
label1 = tk.Label(window,image=img1,width=500, height=500)
label1.grid(row="0", column="1",sticky=NE,pady=0)

def Open_Img():
    global img_png
    global file_path
    OpenFile = tk.Tk() #创建新窗口
    OpenFile.withdraw()
    file_path = filedialog.askopenfilename()
    Img = Image.open(file_path)
    print(file_path)
    w,h = Img.size
    Img = jiemian.resize(w, h, 450, 450, Img)
    img_png = ImageTk.PhotoImage(Img)
    # print(file_path.split("/")[-1])
    Show_Img()
def Show_Img():
    global img_png
    global file_path
    label.configure(image=img_png)
def process():
    global file_path,label1,label,result
    detect.main(file_path)

    Imgpath="C:/Users/86150/Desktop/all/GraduateProject/YoloV5/YOLO5/YOLO5/Mushroom/output/"+file_path.split("/")[-1]
    print(Imgpath)
    siv= Image.open(Imgpath)
    w,h = siv.size
    siv = jiemian.resize(w, h, 450, 450, siv)
    result = ImageTk.PhotoImage(siv)
    label1.configure(image=result)
def clean():
    label.configure(image=img2)
    label1.configure(image=img1)

def Check_realtime():
    detect.main("0")
def Video():
    global file_path
    OpenFile = tk.Tk()  # 创建新窗口
    OpenFile.withdraw()
    file_path = filedialog.askopenfilename()
    detect.main(file_path)
def openvideo():
    OpenFile = tk.Tk()  # 创建新窗口
    OpenFile.withdraw()
    video_path = filedialog.askopenfilename()
    video.open(video_path)
# 创建打开图像按钮
btn_Open = tk.Button(window,
    text='打开图像',      # 显示在按钮上的文字
    width=15, height=2,font=(("宋体"), 12),
    command=Open_Img).grid(row="3", column="0")     # 点击按钮式执行的命令
# btn_Open.pack()    # 按钮位置


# 创建显示图像按钮
btn_process = tk.Button(window,
    text='处理图像',      # 显示在按钮上的文字
    width=15, height=2,font=(("宋体"), 12),
    command=process).grid(row="3", column="1",padx=0)      # 点击按钮式执行的命令
# btn_process.pack()    # 按钮位置

# 创建显示图像按钮
btn_check = tk.Button(window,
    text='实时检测',      # 显示在按钮上的文字
    width=15, height=2,font=(("宋体"), 12),
    command=Check_realtime).grid(row="6", column="0",padx=0)      # 点击按钮式执行的命令
btn_video = tk.Button(window,
    text='处理视频',      # 显示在按钮上的文字
    width=15, height=2,font=(("宋体"), 12),
    command=Video).grid(row="5", column="0",padx=0)      # 点击按钮式执行的命令
btn_ovideo = tk.Button(window,
    text='查看视频',      # 显示在按钮上的文字
    width=15, height=2,font=(("宋体"), 12),
    command=openvideo).grid(row="4", column="0",padx=0)      # 点击按钮式执行的命令
# btn_process.pack()    # 按钮位置

btn_clean = tk.Button(window,
    text='清除图像',      # 显示在按钮上的文字
    width=15, height=2,font=(("宋体"), 12),
    command=clean).grid(row="4", column="1",padx=0)      # 点击按钮式执行的命令
# btn_clean.pack()    # 按钮位置

b = tk.Button(window, text='退出按钮',width=15, height=2, command=window.quit,font=(("宋体"), 12)).grid(row="5",column="1",sticky=S,padx=0)
# 运行整体窗口
window.mainloop()