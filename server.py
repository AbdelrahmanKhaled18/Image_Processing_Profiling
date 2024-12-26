import os
import socket
import cv2
import numpy as np
import logging
import time
import threading
import cProfile
import pstats
import json
import base64
import subprocess
import sys
from multiprocessing.pool import ThreadPool
from functools import partial
from recvall import recvall

dir = os.path.dirname(__file__)
SERVER_HOST = "localhost"
SERVER_PORT = 1420
THREADS_DIMENSION = 3

# Initialize logging
initTime = time.time()
logging.basicConfig(
    filename="log.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def log(step: str):
    logging.info(f"{step}: {time.time() - initTime:.4f}s")


def process_image(decoded_chunk, selected_option):
    try:
        log(f"Processing image with option {selected_option}")

        if selected_option == "edge_detection":
            processed_chunk = cv2.Canny(decoded_chunk, 100, 200)
        elif selected_option == "color_inversion":
            processed_chunk = cv2.bitwise_not(decoded_chunk)
        elif selected_option == "gaussian_blur":
            processed_chunk = cv2.GaussianBlur(decoded_chunk, (5, 5), 0)
        elif selected_option == "sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            processed_chunk = cv2.filter2D(decoded_chunk, -1, kernel)
        elif selected_option == "histogram_equalization":
            if len(decoded_chunk.shape) == 2:
                processed_chunk = cv2.equalizeHist(decoded_chunk)
            else:
                img_yuv = cv2.cvtColor(decoded_chunk, cv2.COLOR_BGR2YUV)
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                processed_chunk = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        elif selected_option == "adaptive_threshold":
            gray = cv2.cvtColor(decoded_chunk, cv2.COLOR_BGR2GRAY)
            processed_chunk = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif selected_option == "dilation":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
            processed_chunk = cv2.dilate(decoded_chunk, kernel)
        elif selected_option == "erosion":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
            processed_chunk = cv2.erode(decoded_chunk, kernel)
        elif selected_option == "enhance":
            lab = cv2.cvtColor(decoded_chunk, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
            processed_chunk = cv2.filter2D(enhanced_image, -1, kernel)
        else:
            log("Invalid processing option. Returning original image.")
            processed_chunk = decoded_chunk

        log("Processing for thread completed")
        return processed_chunk
    except Exception as e:
        log(f"Error processing image: {e}")
        return decoded_chunk


def divide_chunks(img):
    chunks = []
    for i in range(THREADS_DIMENSION):
        for j in range(THREADS_DIMENSION):
            top_bound = img.shape[0] * i // THREADS_DIMENSION
            lower_bound = min(img.shape[0] * (i + 1) // THREADS_DIMENSION, img.shape[0])
            left_bound = img.shape[1] * j // THREADS_DIMENSION
            right_bound = min(img.shape[1] * (j + 1) // THREADS_DIMENSION, img.shape[1])
            chunk = img[
                top_bound:lower_bound,
                left_bound:right_bound,
            ]
            chunks.append(chunk)
    return chunks


def combine_chunks(chunks):
    rows = []
    for i in range(THREADS_DIMENSION):
        row_chunks = chunks[i * THREADS_DIMENSION : (i + 1) * THREADS_DIMENSION]
        row = np.concatenate(row_chunks, axis=1)
        rows.append(row)
    return np.concatenate(rows, axis=0)


def handle_client(client_socket):
    try:
        log("Started handling client")

        payload_size = int.from_bytes(client_socket.recv(8), byteorder="big")
        log(f"Expecting JSON payload of size {payload_size} bytes")

        payload_data = recvall(client_socket, payload_size).decode("utf-8")
        payload = json.loads(payload_data)
        log("Received JSON payload")

        selected_option = payload["selected_option"]
        images_base64 = payload["images"]

        processed_images_base64 = []
        for img_index, img_base64 in enumerate(images_base64):
            log(f"Processing image {img_index + 1}/{len(images_base64)}")

            img_data = base64.b64decode(img_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                log("Failed to decode the received image. Skipping processing.")
                continue

            log(f"Image decoded. Dimensions: {img.shape}")

            start_time = time.time()
            divided_image = divide_chunks(img)
            with ThreadPool(processes=THREADS_DIMENSION**2) as pool:
                processed_chunks = pool.map(
                    partial(process_image, selected_option=selected_option),
                    divided_image,
                )
            processed_img = combine_chunks(processed_chunks)
            processing_time = time.time() - start_time
            log(f"Image completely processed in {processing_time:.4f}s")

            _, img_encoded = cv2.imencode(".jpg", processed_img)
            processed_base64 = base64.b64encode(img_encoded).decode("utf-8")
            processed_images_base64.append(processed_base64)

        response = {"processed_images": processed_images_base64}
        response_data = json.dumps(response).encode("utf-8")

        client_socket.sendall(len(response_data).to_bytes(8, byteorder="big"))
        client_socket.sendall(response_data)
        log("Processed images sent back to client")

    except Exception as e:
        log(f"Error handling client: {e}")
    finally:
        client_socket.close()
        log("Client connection closed")


def register_profiling(profiler):
    profiler.disable()
    profiling_path = os.path.join(dir, "profiling_results.prof")
    profiler.dump_stats(profiling_path)

    log(f"Profiling results saved to {profiling_path}")

    detailed_stats_path = os.path.join(dir, "profiling_results.txt")
    with open(detailed_stats_path, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats()
        stats.print_callers()
        stats.print_callees()
    log(f"Detailed profiling results saved to {detailed_stats_path}")

    try:
        log("Launching Snakeviz for detailed profiling visualization...")
        subprocess.run(["snakeviz", profiling_path])
    except FileNotFoundError:
        log(
            "Snakeviz is not installed. Please install it using 'pip install snakeviz'."
        )
    except Exception as e:
        log(f"Error launching Snakeviz: {e}")
    print("You pressed Ctrl+C!")
    sys.exit(0)


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(5)
    log(f"Server listening on {SERVER_HOST}:{SERVER_PORT}")

    try:
        while True:
            try:
                server_socket.settimeout(1)
                client_socket, client_address = server_socket.accept()
                log(f"Connection from {client_address}")
                client_thread = threading.Thread(
                    target=handle_client, args=(client_socket,), daemon=True
                )
                client_thread.start()
            except socket.timeout:
                pass

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log(f"Unhandled exception: {e}")
    finally:
        server_socket.close()
        log("Server socket closed.")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log(f"Unhandled exception in main: {e}")
    finally:
        register_profiling(profiler)
        log("Shutting down the server...")
