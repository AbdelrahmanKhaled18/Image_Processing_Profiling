from client import *
import os
import sys
import subprocess
import signal
import threading

dir = os.path.dirname(__file__)
img_names = [
    "Screenshot_453.png",
    "Screenshot_454.png",
    "Screenshot_455.png",
    "Screenshot_456.png",
    "Screenshot_457.png",
    "Screenshot_459.png",
]

img_paths_list = [os.path.join(dir, "static", img_name) for img_name in img_names]
img_paths_string = "\n".join(img_paths_list)
selected_option = "enhance"

server_process = subprocess.Popen([sys.executable, "server.py"])
client_processes = [subprocess.Popen([sys.executable, "client.py"]) for _ in range(10)]
[p.wait() for p in client_processes]
server_process.send_signal(signal.CTRL_C_EVENT)
server_process.wait()
