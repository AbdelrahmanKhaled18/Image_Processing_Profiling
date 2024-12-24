import socket
import cv2
import numpy as np
import logging
import time
import threading
import cProfile
import pstats
import io

SERVER_HOST = "localhost"
SERVER_PORT = 1234

# Initialize logging
initTime = time.time()
logging.basicConfig(filename="log.txt", filemode="w", datefmt="%H:%M:%S", level=logging.INFO)


def log(step: str):
    logging.info(f"{step}: {time.time() - initTime}")


def process_image(decoded_chunk, selected_option):
    """
    Process the image chunk based on the selected option.
    """
    try:
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
            processed_chunk = decoded_chunk

        return processed_chunk
    except Exception as e:
        log(f"Error processing image: {e}")
        return decoded_chunk


def handle_client(client_socket):
    """
    Handles a single client request.
    """
    try:
        # Receive the selected option
        selected_option = client_socket.recv(1024).decode()
        print(f"Received processing option: {selected_option}")

        # Receive the number of images
        num_images = int.from_bytes(client_socket.recv(8), byteorder="big")
        print(f"Number of images to process: {num_images}")

        for _ in range(num_images):
            # Receive the size of the incoming image
            img_size = int.from_bytes(client_socket.recv(8), byteorder="big")
            print(f"Expecting {img_size} bytes for the image.")

            # Receive the image bytes
            raw_image = b""
            while len(raw_image) < img_size:
                bytes_remaining = img_size - len(raw_image)
                raw_image += client_socket.recv(min(4096, bytes_remaining))

            print(f"Received image ({len(raw_image)} bytes)")

            # Decode the image
            nparr = np.frombuffer(raw_image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode the received image.")

            print(f"Image decoded. Dimensions: {img.shape}")

            # Process the image
            start_time = time.time()
            processed_img = process_image(img, selected_option)
            processing_time = time.time() - start_time
            log(f"Image processed in {processing_time:.4f}s")

            # Encode the processed image to bytes
            _, img_encoded = cv2.imencode(".jpg", processed_img)

            # Send the size of the processed image
            client_socket.sendall(len(img_encoded).to_bytes(8, byteorder="big"))

            # Send the processed image bytes
            client_socket.sendall(img_encoded.tobytes())
            print("Processed image sent back to client.")

    except Exception as e:
        print(f"Error handling client: {e}")
        log(f"Error handling client: {e}")
    finally:
        client_socket.close()


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(5)
    print(f"Server listening on {SERVER_HOST}:{SERVER_PORT}")
    try:
        while True:
            client_socket, client_address = server_socket.accept()
            print(f"Connection from {client_address}")
            client_thread = threading.Thread(target=handle_client, args=(client_socket,))
            client_thread.start()
    except KeyboardInterrupt:
        print("Shutting down the server...")
    finally:
        server_socket.close()
        print("Server socket closed.")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Exiting gracefully...")
    except Exception as e:
        print(f"Unhandled exception: {e}")
    finally:
        print("Saving profiling results...")
        profiler.disable()
        try:
            file_path = "E:/Image_Processing_Profiling/profiling_results.txt"
            with open(file_path, "w") as f:
                stats = pstats.Stats(profiler, stream=f)
                stats.strip_dirs()
                stats.sort_stats("cumtime")
                stats.print_stats()
            print(f"Profiling results saved to {file_path}")
        except Exception as e:
            print(f"Error saving profiling results: {e}")
        print("Shutting down the server...")
