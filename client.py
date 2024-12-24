import json
from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import showinfo
import socket
import cv2
import numpy as np

SERVER_HOST = "localhost"
SERVER_PORT = 1234

imgs = []
client_socket = None


def connect_to_server():
    global client_socket
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_HOST, SERVER_PORT))
        print("Connected to the server")
    except Exception as e:
        print(f"Error connecting to the server: {e}")
        showinfo("Error", "Failed to connect to the server.")


connect_to_server()


def create_form_elements(root):
    """
    Create form elements such as labels, buttons, and entry fields.
    """
    file_label = Label(root, text="File:", background="#0078D4", font="bold")
    file_entry = Entry(root, state="readonly")
    file_button = Button(
        root,
        text="Browse",
        command=lambda: browse_file(file_entry),
        background="white",
        highlightbackground="white",
        highlightcolor="white",
    )
    upload_button = Button(
        root,
        text="Upload File",
        command=lambda: upload_file(file_entry, selected_option),
        background="white",
        highlightbackground="white",
        highlightcolor="white",
    )
    download_button = Button(
        root,
        text="Download Image",
        command=lambda: download_images(imgs),
        background="white",
        highlightbackground="white",
        highlightcolor="white",
    )

    options = ["edge_detection", "color_inversion", "erosion", "dilation", "adaptive_threshold",
               "histogram_equalization", "sharpen", "gaussian_blur", "enhance"]
    selected_option = StringVar()
    selected_option.set(options[0])
    option_menu = OptionMenu(root, selected_option, *options)
    file_label.place(x=110, y=144)
    file_entry.place(x=160, y=150)
    file_button.place(x=287, y=147)
    upload_button.place(x=212, y=200)
    option_menu.place(x=185, y=258)
    download_button.place(x=197, y=230)


def browse_file(file_entry):
    file_paths = filedialog.askopenfilenames()
    if file_paths:
        file_entry.config(state="normal")
        file_entry.delete(0, "end")
        file_entry.insert(0, "\n".join(file_paths))
        file_entry.config(state="readonly")


def upload_file(file_entry, selected_option):
    global imgs
    file_paths = file_entry.get().strip().split("\n")

    if file_paths and any(file_paths):
        try:
            # Send the selected option
            client_socket.sendall(selected_option.get().encode())
            print(f"Sent selected option: {selected_option.get()}")

            # Send the number of images
            client_socket.sendall(len(file_paths).to_bytes(8, byteorder="big"))

            for file_path in file_paths:
                # Read the image
                img = cv2.imread(file_path)
                if img is None:
                    raise FileNotFoundError(f"Unable to load file: {file_path}")

                # Encode image to bytes
                _, img_encoded = cv2.imencode(".jpg", img)
                img_bytes = img_encoded.tobytes()

                # Send the size of the image
                client_socket.sendall(len(img_bytes).to_bytes(8, byteorder="big"))

                # Send the image bytes
                client_socket.sendall(img_bytes)
                print("Image sent to server.")

                # Receive the size of the processed image
                bytes_no = int.from_bytes(client_socket.recv(8), byteorder="big")

                # Receive the processed image bytes
                raw_image = b""
                while len(raw_image) < bytes_no:
                    bytes_remaining = bytes_no - len(raw_image)
                    raw_image += client_socket.recv(min(4096, bytes_remaining))

                print(f"Received processed image ({len(raw_image)} bytes)")

                # Decode the received image
                nparr = np.frombuffer(raw_image, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Append and display the image
                imgs.append(img)
                cv2.imshow("Processed Image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            showinfo("Success", "Files have been uploaded and displayed.")
            reconnect_to_server()

        except FileNotFoundError as e:
            showinfo("Error", str(e))
        except OSError as e:
            print(f"Connection error: {e}")
            reconnect_to_server()
        except BrokenPipeError as e:
            print(f"Broken pipe error: {e}")
            reconnect_to_server()
    else:
        showinfo("Error", "Please select one or more files to upload.")


def download_images(images):
    """
    Download the processed images.
    """
    if not images:
        showinfo("Error", "No images to download.")
        return

    valid_extensions = [".jpg", ".jpeg", ".png"]
    for i, img in enumerate(images):
        file_extension = ".jpg"  # Default file extension
        save_path = filedialog.asksaveasfilename(defaultextension=file_extension,
                                                 filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"),
                                                            ("All files", "*.*")])

        # Ensure the save_path has a valid extension
        if not any(save_path.lower().endswith(ext) for ext in valid_extensions):
            save_path += file_extension  # Default to .jpg if no valid extension is found

        try:
            cv2.imwrite(save_path, img)
            showinfo("Success", f"Image {i + 1} has been downloaded.")
        except cv2.error as e:
            showinfo("Error", f"Failed to save image {i + 1}: {e}")


def reconnect_to_server():
    print("Trying to reconnect...")
    global client_socket

    try:
        # Close the socket if it's still open
        client_socket.shutdown(socket.SHUT_RDWR)
        client_socket.close()
        print("Closed existing connection")
    except Exception as e:
        print(f"Error closing existing connection: {e}")

    try:
        # Reconnect to the server
        connect_to_server()
    except Exception as e:
        print(f"Error reconnecting to server: {e}")
