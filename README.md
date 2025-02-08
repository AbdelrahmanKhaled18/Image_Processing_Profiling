# Image Processing Client-Server Application

## Overview
This project is a multithreaded client-server application for processing images using various techniques such as edge detection, sharpening, blurring, and more. The system supports both JSON-based and raw byte-based communication protocols. It includes a GUI for user-friendly interaction, along with a profiling script to analyze server performance.

## Features
- **Server**: Handles multiple clients, processes images in parallel, and returns the results.
- **Client**: Sends images to the server and retrieves processed images.
- **GUI**: Provides a graphical interface for users to select images, choose processing options, and download results.
- **Profiling**: Uses `cProfile` to analyze server performance.

## Technologies Used
- **Python**
- **OpenCV** (Image Processing)
- **Socket Programming** (Client-Server Communication)
- **Multiprocessing & Threading** (Parallel Processing)
- **Tkinter** (GUI Development)
- **cProfile** (Performance Profiling)
- **SnakeViz** (Profiling Visualization)

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/AbdelrahmanKhaled18/Image_Processing_Profiling
   cd Image_Processing_Profiling
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the server:
   ```sh
   python server.py
   ```
4. Run the client:
   ```sh
   python client.py
   ```
5. (Optional) Run the GUI:
   ```sh
   python GUI.py
   ```
6. (Optional) Profile the server:
   ```sh
   python profiling_program.py
   ```

## Usage
### GUI Usage
1. Run `GUI.py` to open the graphical interface.
2. Click **Browse** to select an image.
3. Choose an image processing option from the dropdown menu.
4. Click **Upload File** to send the image to the server.
5. Once processing is complete, click **Download Image** to save the output.

### Client Usage
1. Modify `client.py` to specify the image file paths and processing option.
2. Run `client.py` to send images to the server and receive processed results.

### Server Functionality
- Listens for client connections and processes image requests.
- Divides images into smaller chunks for parallel processing.
- Supports multiple processing techniques (e.g., edge detection, blurring, sharpening, etc.).
- Logs execution time for performance monitoring.

### Profiling
- `profiling_program.py` runs the server under `cProfile`, simulating 20 parallel clients.
- Generates profiling results in `.prof` and `.txt` formats.
- Optionally visualizes results using SnakeViz (`pip install snakeviz`).

## Image Processing Techniques Supported
- **Edge Detection**
- **Color Inversion**
- **Gaussian Blur**
- **Sharpening**
- **Histogram Equalization**
- **Adaptive Thresholding**
- **Dilation & Erosion**
- **Image Enhancement**

## Configuration
Modify `communication_helper.py` to adjust:
- `PROTOCOL` (JSON or BYTES-based communication)
- `SERVER_HOST` and `SERVER_PORT` for server binding

## Troubleshooting
- If the client fails to connect, ensure the server is running and using the correct host and port.
- If performance is slow, consider increasing `THREADS_DIMENSION` in `server.py`.
- Use `profiling_program.py` and `snakeviz` for performance analysis.

