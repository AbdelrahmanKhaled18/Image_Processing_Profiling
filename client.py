import json
import base64
from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import showinfo
import socket
import cv2
import numpy as np
from recvall import recvall

SERVER_HOST = "localhost"
SERVER_PORT = 1234

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
        command=lambda: upload_file(file_entry.get().strip(), selected_option.get()),
        background="white",
        highlightbackground="white",
        highlightcolor="white",
    )

    options = [
        "edge_detection",
        "color_inversion",
        "erosion",
        "dilation",
        "adaptive_threshold",
        "histogram_equalization",
        "sharpen",
        "gaussian_blur",
        "enhance",
    ]
    selected_option = StringVar()
    selected_option.set(options[0])
    option_menu = OptionMenu(root, selected_option, *options)
    file_label.place(x=110, y=144)
    file_entry.place(x=160, y=150)
    file_button.place(x=287, y=147)
    upload_button.place(x=212, y=200)
    option_menu.place(x=185, y=258)


def browse_file(file_entry):
    file_paths = filedialog.askopenfilenames()
    if file_paths:
        file_entry.config(state="normal")
        file_entry.delete(0, "end")
        file_entry.insert(0, "\n".join(file_paths))
        file_entry.config(state="readonly")


def send_json(data):
    """
    Send JSON-encoded data to the server.
    """
    json_data = json.dumps(data).encode("utf-8")
    client_socket.sendall(len(json_data).to_bytes(8, byteorder="big"))
    client_socket.sendall(json_data)


def receive_json():
    """
    Receive JSON-encoded data from the server.
    """
    data_size = int.from_bytes(client_socket.recv(8), byteorder="big")
    json_data = recvall(client_socket, data_size).decode("utf-8")
    return json.loads(json_data)


def upload_file(file_paths, selected_option):
    file_paths = file_paths.split("\n")

    if file_paths and any(file_paths):
        try:
            # Prepare images and metadata
            images = []
            for file_path in file_paths:
                # Read the image
                img = cv2.imread(file_path)
                if img is None:
                    raise FileNotFoundError(f"Unable to load file: {file_path}")

                # Encode image to base64
                _, img_encoded = cv2.imencode(".jpg", img)
                img_base64 = base64.b64encode(img_encoded).decode("utf-8")

                # Add to image list
                images.append(img_base64)

            # Create JSON payload
            payload = {
                "selected_option": selected_option,
                "num_images": len(images),
                "images": images,
            }

            # Send JSON payload
            send_json(payload)
            print("Sent JSON payload with images.")

            # Receive processed data
            response = receive_json()
            print("Received response from server.")

            # Decode and display images
            for img_base64 in response.get("processed_images", []):
                img_data = base64.b64decode(img_base64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cv2.imshow("Processed Image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            showinfo("Success", "Files have been uploaded and displayed.")

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
        save_path = filedialog.asksaveasfilename(
            defaultextension=file_extension,
            filetypes=[
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("All files", "*.*"),
            ],
        )

        # Ensure the save_path has a valid extension
        if not any(save_path.lower().endswith(ext) for ext in valid_extensions):
            save_path += (
                file_extension  # Default to .jpg if no valid extension is found
            )

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
