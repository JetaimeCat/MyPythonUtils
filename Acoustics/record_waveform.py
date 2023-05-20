# -*- coding: utf-8 -*-
# Author   : Xia Zhaoxiang
# FileName : record_waveform.py
# Software : PyCharm
# Time     : 2023/5/19 16:45
# Email    : 1206572082@qq.com

import wave
import pyaudio
import tkinter
import threading
from datetime import datetime
from tkinter.messagebox import showerror

allowRecording = False


def record_waveform():
    record = pyaudio.PyAudio()
    for i in range(record.get_device_count()):
        devices = record.get_device_info_by_index(i)
        print("index:", devices["index"], "——", devices["name"], devices["maxInputChannels"])
    print("*" * 50)
    device = int(input("选择监听设备编号(index):"))

    # wav 基本参数
    CHUNK = 512
    RATE = 48000
    CHANNELS = 1
    FORMAT = pyaudio.paInt16

    stream = record.open(
        input=True,
        rate=RATE,
        channels=CHANNELS,
        format=FORMAT,
        frames_per_buffer=CHUNK,
        input_device_index=device,
    )

    # 录制完成的声音内容保存至当前文件中
    file_record = f"{str(datetime.now())[:19].replace(':', '_')}.wav"
    writer = wave.open(file_record, "wb")
    writer.setnchannels(CHANNELS)
    writer.setsampwidth(record.get_sample_size(FORMAT))
    writer.setframerate(RATE)
    while allowRecording:
        # 从设备中读取数据，直接写入wav文件
        data = stream.read(CHUNK)
        writer.writeframes(data)
    writer.close()
    stream.stop_stream()
    stream.close()
    record.terminate()


def start():
    global allowRecording
    allowRecording = True
    lb_status["text"] = "正在录音....."
    threading.Thread(target=record_waveform).start()
    btn_start["state"] = "disabled"
    btn_stop["state"] = "normal"


def stop():
    global allowRecording
    allowRecording = False
    lb_status["text"] = "准备就绪"
    btn_start["state"] = "normal"
    btn_stop["state"] = "disabled"


def close_window():
    if allowRecording:
        showerror("正在录制音频", "请先停止录制")
        return
    root.destroy()


root = tkinter.Tk()  # 创建 tkinter 程序
root.title("采集 SDRSharp 数据")  # 设置窗口标题
root.geometry('300x80+400+300')  # 初始大小与位置
root.resizable(False, False)  # 两个方向不允许改变

btn_start = tkinter.Button(root, text="开始录音", command=start)
btn_start.place(x=30, y=20, width=100, height=20)

btn_stop = tkinter.Button(root, text="停止录音", state="disabled", command=stop)
btn_stop.place(x=140, y=20, width=100, height=20)

lb_status = tkinter.Label(root, text="准备就绪", anchor="w", fg="green")
lb_status.place(x=30, y=50, width=200, height=20)

root.protocol("WM-DELETE_WINDOW", close_window)

root.mainloop()
