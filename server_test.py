from client import *
import os

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

connect_to_server()
upload_file(img_paths_string, selected_option)
