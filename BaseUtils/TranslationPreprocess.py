import re
import html
import requests
import tkinter as tk
from urllib import parse
import tkinter.messagebox as messagebox

GOOGLE_TRANSLATE_URL = "https://translate.google.com/m?q=%s&tl=%s&sl=%s"
proxy_servers = {}


def save_proxy():
    address = entry.get()
    if address == "":
        messagebox.showwarning("Warning", "未输入代理地址!")
        return
    proxy_address = "http://" + address  # 将用户输入的代理地址拼接上"http://"
    proxy_servers["http"] = proxy_address
    proxy_servers["https"] = proxy_address
    # 在这里执行保存代理地址的逻辑
    messagebox.showinfo("设置成功", "代理地址已设置!")


def translate(text, to_language="zh-CN", text_language="en"):
    if len(proxy_servers.keys()) == 0:
        messagebox.showinfo("Error", "请先设置代理地址！")
        return ""
    text = parse.quote(text)
    url = GOOGLE_TRANSLATE_URL % (text, to_language, text_language)
    response = requests.get(url, proxies=proxy_servers)
    data = response.text
    expr = r'(?s)class="(?:t0|result-container)">(.*?)<'
    result = re.findall(expr, data)
    if len(result) == 0:
        return ""
    return html.unescape(result[0])


def process_text():
    input_text = text_entry.get(1.0, tk.END)  # 获取输入文本框中的内容
    if input_text == "\n":
        input_text = window.clipboard_get()
        text_entry.insert(tk.END, input_text)
    if input_text == "\n":
        messagebox.showwarning("Warning", "无内容可用于处理!")
        return
    # 在这里执行您的文本处理操作
    processed_text = input_text.replace("", "-")
    processed_text = processed_text.replace("\n", " ")
    processed_text = processed_text.replace("  ", " ")
    processed_text = processed_text.strip()

    result_text.config(state=tk.NORMAL)  # 将结果文本框的状态设置为可编辑
    result_text.delete(1.0, tk.END)  # 清空结果文本框
    result_text.insert(tk.END, translate(processed_text))  # 在结果文本框中显示处理后的文本
    result_text.config(state=tk.DISABLED)  # 将结果文本框的状态设置为不可编辑
    copy_text()


def copy_text():
    text = result_text.get(1.0, tk.END).strip()  # 获取结果文本框中的内容
    if text:
        window.clipboard_clear()  # 清空剪贴板内容
        window.clipboard_append(text)  # 将内容添加到剪贴板
        messagebox.showinfo("复制", "文本已复制到剪贴板。")
    else:
        messagebox.showwarning("复制", "没有可复制的文本。")


def clear_text():
    text_entry.delete(1.0, tk.END)
    result_text.config(state=tk.NORMAL)  # 将结果文本框的状态设置为可编辑
    result_text.delete(1.0, tk.END)
    result_text.config(state=tk.DISABLED)  # 将结果文本框的状态设置为不可编辑


# 创建主窗口
window = tk.Tk()
window.title("文本处理窗口")

# 创建Frame用于容纳label和entry
frame = tk.Frame(window)
frame.pack(padx=20, pady=10)

# 创建标签和输入框
label_text = tk.StringVar()
label_text.set("代理地址: http://")
label = tk.Label(frame, textvariable=label_text)
label.pack(side="left")

entry = tk.Entry(frame)
entry.insert(tk.END, "")  # 设置默认文本值为"http://"
entry.pack(side="left", padx=5)

# 创建保存按钮
button = tk.Button(frame, text="保存", command=save_proxy)
button.pack(padx=5, pady=5)

# 创建输入文本框
text_entry = tk.Text(window, height=10, bg="#f5f5f5", fg="black", font=("Arial", 12))
text_entry.pack(padx=10, pady=10)

# 创建处理按钮
process_button = tk.Button(window, text="处理文本", command=process_text)
process_button.pack(pady=5)

# 创建结果文本框
result_text = tk.Text(window, height=10, state=tk.DISABLED, bg="#f5f5f5", fg="black", font=("Arial", 12))
result_text.pack(padx=10, pady=10)

# 创建复制按钮
copy_button = tk.Button(window, text="复制", command=copy_text)
copy_button.pack(pady=5)

# 创建清空按钮
clear_button = tk.Button(window, text="清空文本", command=clear_text)
clear_button.pack(pady=5)

# 运行主循环
window.mainloop()
