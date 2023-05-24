import re
import html
import json
import urllib
import random
import hashlib
import requests
import tkinter as tk
from tkinter import ttk
from urllib import parse
import tkinter.messagebox as messagebox

GOOGLE_TRANSLATE_URL = "https://translate.google.com/m?q=%s&tl=%s&sl=%s"
PROXY_SERVERS = dict()


def translate(text, to_language="中文", text_language="英文", api="Google Translate"):
    if len(PROXY_SERVERS.keys()) == 0:
        messagebox.showinfo("错误", "请先设置代理地址！")
        return ""
    text = parse.quote(text)
    result = ""
    if api == "Google Translate":
        to_language = "zh-CN" if to_language == "中文" else "en"
        text_language = "zh-CN" if text_language == "中文" else "en"
        url = GOOGLE_TRANSLATE_URL % (text, to_language, text_language)
        response = requests.get(url, proxies=PROXY_SERVERS)
        data = response.text
        expr = r'(?s)class="(?:t0|result-container)">(.*?)<'
        result = re.findall(expr, data)
    elif api == "Baidu Translate":
        to_language = "zh" if to_language == "中文" else "en"
        text_language = "zh" if text_language == "中文" else "en"
        # Add code for Baidu Translate API URL
        appid = "20230524001687786"  # Replace with your Baidu Translate app ID
        appkey = "nMAazOFyCfDsFdLAo8P8"  # Replace with your Baidu Translate app key
        salt = str(random.randint(32768, 65536))
        sign = hashlib.md5((appid + text + str(salt) + appkey).encode("utf-8")).hexdigest()
        BAIDU_TRANSLATE_URL = "https://fanyi-api.baidu.com/api/trans/vip/translate"
        # Build request
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'appid': appid, 'q': text, 'from': text_language, 'to': to_language, 'salt': salt, 'sign': sign}
        # Send request
        r = requests.post(BAIDU_TRANSLATE_URL, params=payload, headers=headers, proxies=PROXY_SERVERS)
        data = eval(json.dumps(r.json(), indent=4, ensure_ascii=False))
        result = data["trans_result"]
        result = [urllib.parse.unquote(res["dst"]) for res in result]
    if len(result) == 0:
        return ""
    return html.unescape(result[0])


def process_text():
    text = input_text.get(1.0, tk.END)  # 获取输入文本框中的内容
    if text == "\n":
        text = window.clipboard_get()
        input_text.insert(tk.END, text)
    if text == "\n":
        messagebox.showwarning("警告", "无内容可用于处理!")
        return
    # 在这里执行您的文本处理操作
    processed_text = text.replace("", "-")
    processed_text = processed_text.replace("\n", " ")
    processed_text = processed_text.replace("  ", " ")
    processed_text = processed_text.strip()

    input_language = input_language_combobox.get()
    output_language = output_language_combobox.get()
    chrom_text.config(state=tk.NORMAL)  # 将结果文本框的状态设置为可编辑
    chrom_text.delete(1.0, tk.END)  # 清空结果文本框
    translated_text = translate(processed_text, output_language, input_language, "Google Translate")
    chrom_text.insert(tk.END, translated_text)  # 在结果文本框中显示处理后的文本
    chrom_text.config(state=tk.DISABLED)  # 将结果文本框的状态设置为不可编辑

    baidu_text.config(state=tk.NORMAL)  # 将结果文本框的状态设置为可编辑
    baidu_text.delete(1.0, tk.END)  # 清空结果文本框
    translated_text = translate(processed_text, output_language, input_language, "Baidu Translate")
    baidu_text.insert(tk.END, translated_text)  # 在结果文本框中显示处理后的文本
    baidu_text.config(state=tk.DISABLED)  # 将结果文本框的状态设置为不可编辑


# 更新代理服务器内容
def update_proxy_server(first: bool = False):
    PROXY_ADDRESS = "127.0.0.1" if first else address_entry.get()
    PROXY_PORT = "15732" if first else port_entry.get()
    if (PROXY_ADDRESS == "" or PROXY_PORT == "") and not first:
        messagebox.showwarning("警告", "未输入代理端口!")
        return
    PROXY_SERVERS["http"] = f"http://{PROXY_ADDRESS}:{PROXY_PORT}"
    PROXY_SERVERS["https"] = f"http://{PROXY_ADDRESS}:{PROXY_PORT}"
    if not first:
        # 在这里执行保存代理地址的逻辑
        messagebox.showinfo("设置成功", "代理地址已设置!")


def clear_text():
    input_text.delete(1.0, tk.END)
    chrom_text.config(state=tk.NORMAL)  # 将结果文本框的状态设置为可编辑
    chrom_text.delete(1.0, tk.END)
    chrom_text.config(state=tk.DISABLED)  # 将结果文本框的状态设置为不可编辑
    baidu_text.config(state=tk.NORMAL)  # 将结果文本框的状态设置为可编辑
    baidu_text.delete(1.0, tk.END)
    baidu_text.config(state=tk.DISABLED)  # 将结果文本框的状态设置为不可编辑


def copy_text_chrom():
    text = chrom_text.get(1.0, tk.END).strip()  # 获取结果文本框中的内容
    if text:
        window.clipboard_clear()  # 清空剪贴板内容
        window.clipboard_append(text)  # 将内容添加到剪贴板
        messagebox.showinfo("复制", "文本已复制到剪贴板。")
    else:
        messagebox.showwarning("复制", "没有可复制的文本!")


def copy_text_baidu():
    text = baidu_text.get(1.0, tk.END).strip()  # 获取结果文本框中的内容
    if text:
        window.clipboard_clear()  # 清空剪贴板内容
        window.clipboard_append(text)  # 将内容添加到剪贴板
        messagebox.showinfo("复制", "文本已复制到剪贴板。")
    else:
        messagebox.showwarning("复制", "没有可复制的文本!")


def set_topmost():
    is_topmost = topmost_var.get()
    window.attributes('-topmost', is_topmost)


def set_resizable():
    resizable = resizable_var.get()
    window.resizable(resizable, resizable)


update_proxy_server(first=True)

# 创建主窗口
window = tk.Tk()
window.title("文本处理窗口")

# 创建Frame用于容纳label和entry
top_frame = tk.Frame(window)
top_frame.pack()
# 设置代理部分——创建标签和输入框
label_text = tk.StringVar()
label_text.set("代理地址: http://")
label = tk.Label(top_frame, textvariable=label_text)
label.pack(side="left")
# 创建代理服务器输入框——获取代理地址
address_input = tk.StringVar()
address_entry = ttk.Entry(top_frame, textvariable=address_input, width=15)
address_entry.insert(tk.END, PROXY_SERVERS["http"].split("//")[-1].split(":")[0])
address_entry.pack(side="left", pady=5)
tk.Label(top_frame, text=":").pack(side="left")
# 创建代理服务器输入框——获取代理端口
port_input = tk.StringVar()
port_entry = ttk.Entry(top_frame, textvariable=port_input, width=4)
port_entry.insert(tk.END, PROXY_SERVERS["http"].split("//")[-1].split(":")[1])
port_entry.pack(side="left")
# 创建保存按钮
button = tk.Button(top_frame, text="保存", command=update_proxy_server)
button.pack(side="left", padx=20, pady=5)
# 设置界面置顶
topmost_var = tk.BooleanVar(value=False)
topmost_checkbox = tk.Checkbutton(top_frame, text='置顶', variable=topmost_var, command=set_topmost)
topmost_checkbox.pack(side="left", padx=10, pady=5)
# 设置界面置顶
resizable_var = tk.BooleanVar(value=True)
resizable_checkbox = tk.Checkbutton(top_frame, text='允许拉伸', variable=resizable_var, command=set_resizable)
resizable_checkbox.pack(side="left", padx=5, pady=5)

# 创建Frame用于容纳input和output选择框
select_frame = tk.Frame(window)
select_frame.pack()
# 创建"输入语言"选择框标签
input_language = tk.Label(select_frame, text="输入语言:")
input_language.pack(side="left", padx=5)
# 创建"输入语言"选择框
input_language_combobox = ttk.Combobox(select_frame, state="readonly")
input_language_combobox["values"] = ["英语", "中文"]  # 添加您想要的语言选项
input_language_combobox.current(0)  # 设置默认选项
input_language_combobox.pack(side="left", padx=5)
# 创建"输出语言"选择框标签
output_language = tk.Label(select_frame, text="输出语言:")
output_language.pack(side="left", padx=5)
# 创建"输出语言"选择框
output_language_combobox = ttk.Combobox(select_frame, state="readonly")
output_language_combobox["values"] = ["中文", "英语"]  # 添加您想要的语言选项
output_language_combobox.current(0)  # 设置默认选项
output_language_combobox.pack(side="left", padx=5)

# 创建输入文本框
input_text = tk.Text(window, width=65, height=10, bg="#f5f5f5", fg="black", font=("Consolas", 12))
input_text.pack(padx=10, pady=10)
# 创建结果文本显示框——Chrom
tk.Label(window, text="Google 翻译:").pack(anchor="w", pady=5)
chrom_text = tk.Text(window, width=65, height=10, state=tk.DISABLED, bg="#f5f5f5", fg="black", font=("Consolas", 12))
chrom_text.pack(padx=10, pady=10)
# 创建结果文本显示框——Baidu
tk.Label(window, text="Baidu 翻译:").pack(anchor="w", pady=5)
baidu_text = tk.Text(window, width=65, height=10, state=tk.DISABLED, bg="#f5f5f5", fg="black", font=("Consolas", 12))
baidu_text.pack(padx=10, pady=10)

# 创建底部
foot_frame = tk.Frame(window)
foot_frame.pack(pady=5)
# 创建处理按钮
process_button = tk.Button(foot_frame, text="处理文本", command=process_text)
process_button.pack(side="left", padx=5)
# 创建复制按钮
copy_button_chrom = tk.Button(foot_frame, text="Chrom复制", command=copy_text_chrom)
copy_button_chrom.pack(side="left", padx=5)
copy_button_baidu = tk.Button(foot_frame, text="Baidu复制", command=copy_text_baidu)
copy_button_baidu.pack(side="left", padx=5)
# 创建清空按钮
clear_button = tk.Button(foot_frame, text="清空文本", command=clear_text)
clear_button.pack(side="left", padx=5)
# 运行主循环
window.mainloop()
