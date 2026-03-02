
```markdown
# Real-Time Visual Assistance System for the Visually Impaired Using Qwen VL Models 

## About the Project
The Real-Time Visual Assistance System is an interactive, client-server platform designed to empower visually impaired individuals by providing context-aware, real-time descriptions of their surroundings. Developed by Adnan Rasool Sofi at the Central University of Kashmir, the system utilizes the Qwen2.5-VL-7B-Instruct vision-language model to process both static images and dynamic video feeds captured via a webcam. To ensure full accessibility, the system is entirely navigable through speech, converting audio prompts to text and vocalizing the AI's descriptive analysis.

## Key Features
* **Multimodal Visual Input:** Captures and processes single images and short 3-second video clips at 10 FPS. 
* **Assistive Prompt Formatting:** Employs a specialized template to structure AI responses into three actionable parts: Direct Answer, Reasoning, and Detailed Description.
* **Speech Interaction:** Integrates speech-to-text (`speech_recognition`) for taking user commands and text-to-speech (`pyttsx3`) for reading the results aloud at an optimal 200 words per minute.
* **Low-Latency Architecture:** Uses Flask-SocketIO over a TCP/IP network to quickly transmit base64-encoded media from the client to the processing server.

## Architecture
The system operates on a split Client-Server model:
1. **Client Application (Windows):** Handles the UI, webcam capture (`OpenCV`), audio processing, and networking.
2. **Server Application (Ubuntu):** Manages a processing queue, extracts video frames, and runs the Qwen-VL model inference via PyTorch and Transformers. 

## Hardware Requirements
Based on the testing and evaluation environment:
* **Client Side:** Windows 10/11, Intel i5/i7 processor, 8 GB+ RAM, 1080p Webcam, Microphone, and Speakers.
* **Server Side:** Ubuntu 20.04 LTS, AMD Ryzen/Intel Xeon processor, 32 GB+ RAM, and an NVIDIA GPU with at least 16 GB VRAM (tested on 47.45 GiB VRAM) to handle CUDA acceleration.

## Repository Structure
```text
visual-assistance-project/
├── .gitignore
├── README.md
├── client/
│   ├── realtime_client.py
│   └── requirements.txt
└── server/
    ├── processor.py
    ├── server_app.py
    ├── test.py
    ├── requirements.txt
    ├── test_dataset/
    │   ├── images/
    │   └── videos/
    └── test_results/
        └── .gitkeep               
```

*(Note: Empty folders are not tracked by Git. The `.gitkeep` files ensure the directory structure remains intact for the `test.py` script to function properly upon cloning.)*

## Setup & Installation

### 1. Client Setup (Local Machine)

Ensure Python 3.10 is installed.

```bash
git clone <your-repository-url>
cd visual-assistance-project/client
pip install -r requirements.txt
```

**Important:** Open `realtime_client.py` and ensure you add your server's IP address to the configuration section:

```python
SERVER_URL = '' # Add server ip address here
```

### 2. Server Setup (Remote Machine)

Ensure Python 3.10 and NVIDIA CUDA drivers (11.7 or later) are installed.

```bash
cd visual-assistance-project/server

# Install PyTorch with CUDA support first (Example for CUDA 11.8)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install remaining dependencies
pip install -r requirements.txt
```

**Important:** Open `server_app.py` and ensure you add a secure secret key to the Flask configuration:

```python
app.config['SECRET_KEY'] = '' # Add secret key here
```

## Usage

### Running the Server

Start the server to begin listening for client connections on port 5000.

```bash
cd server
python server_app.py
```

### Running the Client

Launch the client application. A window will open displaying the webcam feed.

```bash
cd client
python realtime_client.py
```

**Controls:**

* Press `c`: Capture a single image, prompt for input, and analyze.


* Press `v`: Capture a 3-second video, prompt for input, and analyze.


* Press `q` or `ESC`: Quit the application.



### Running Local Tests

To verify the Qwen model is working locally without the client, you can run the standalone test script.

1. Place `.jpg` or `.png` files in `server/test_dataset/images/`.


2. Place `.mp4` or `.avi` files in `server/test_dataset/videos/`.


3. Run the script:

```bash
cd server
python test.py
```

Results, including processing times and model outputs, will be saved as a text summary in the `server/test_results/` directory.
