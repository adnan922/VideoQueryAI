# server_app.py (runs on Ubuntu Server)

import os
import time
import base64
import uuid
import cv2 # Still needed for process_single_image
import numpy as np # Still needed for process_single_image
from pathlib import Path
import traceback
import threading
import queue
import eventlet
import shutil # Ensure shutil is imported
from typing import Optional

# Patch before importing socketio
eventlet.monkey_patch()

from flask import Flask, request
from flask_socketio import SocketIO, emit


try:
    from processor import QwenVLProcessor
    PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import QwenVLProcessor: {e}")
    print("Ensure processor.py and its dependencies are available.")
    PROCESSOR_AVAILABLE = False
except Exception as e:
    print(f"ERROR: An unexpected error occurred during QwenVLProcessor import: {e}")
    PROCESSOR_AVAILABLE = False

# Configuration
TEMP_FOLDER = Path("./server_temp_files")
TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSING_MAX_TOKENS_IMG = 256
PROCESSING_MAX_TOKENS_VID = 512 
VIDEO_PROCESSING_FPS = 5.0      
VIDEO_PROCESSING_MAX_FRAMES = 10 

# Add a worker queue for processing requests
processing_queue = queue.Queue()

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = '' # Add secret key here
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins='*')

qwen_processor = None
if PROCESSOR_AVAILABLE:
    print("Initializing QwenVLProcessor...")
    try:
        qwen_processor = QwenVLProcessor()
    except Exception as e:
        print(f"FATAL: Failed to initialize QwenVLProcessor on server: {e}")
        traceback.print_exc()
        PROCESSOR_AVAILABLE = False
else:
    print("QwenVLProcessor not available. Server will run without processing capabilities.")


@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    emit('server_status', {'message': 'Connected to processing server.'})
    if not PROCESSOR_AVAILABLE or qwen_processor is None:
        emit('server_status', {'message': 'WARNING: AI Processor is not available on the server.'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")


def save_base64_video_to_tempfile(encoded_video, client_sid) -> Optional[str]:
    """Saves base64 encoded video to a temporary file and returns the file path."""
    temp_video_path_obj = None
    try:
        video_bytes = base64.b64decode(encoded_video)
        video_filename = f"{uuid.uuid4()}.mp4" # Assuming mp4, adjust if necessary
        temp_video_path_obj = TEMP_FOLDER / video_filename

        with open(temp_video_path_obj, 'wb') as f:
            f.write(video_bytes)
        print(f"Temporary video saved to: {str(temp_video_path_obj)}")
        return str(temp_video_path_obj)
    except Exception as e:
        print(f"Error saving base64 video to temp file: {e}")
        traceback.print_exc()
        # Emit error to client directly from here if appropriate, or let the caller handle it
        socketio.emit('processing_result', {'error': f'Failed to decode/save video data: {str(e)}'}, room=client_sid)
        if temp_video_path_obj and temp_video_path_obj.exists():
            try:
                os.remove(temp_video_path_obj)
            except OSError:
                pass # Ignore cleanup error if primary error is saving
        return None


def worker_thread():
    """Background worker thread to process requests from the queue."""
    while True:
        try:
            task = processing_queue.get()
            if task is None:
                break
            
            client_sid, data_type, data, prompt = task
            
            if data_type == 'image':
                process_single_image(client_sid, data, prompt)
            elif data_type == 'video':
                process_video_request_worker(client_sid, data, prompt) # Renamed for clarity
                
            processing_queue.task_done()
        except Exception as e:
            print(f"Error in worker thread: {e}")
            traceback.print_exc()

def process_single_image(client_sid, encoded_image, prompt):
    """Process a single image."""
    temp_image_path = None
    try:
        image_bytes = base64.b64decode(encoded_image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode image data.")

        unique_filename = f"{uuid.uuid4()}.jpg"
        temp_image_path = str(TEMP_FOLDER / unique_filename)
        cv2.imwrite(temp_image_path, frame)
        print(f"Temporary image saved to: {temp_image_path}")

        print(f"Processing image {temp_image_path} with prompt: '{prompt}'")
        socketio.emit('server_status', {'message': f'Processing image...'}, room=client_sid)
        start_time = time.time()

        result = qwen_processor.process_image(
            temp_image_path,
            prompt,
            max_tokens=PROCESSING_MAX_TOKENS_IMG
        )

        duration = time.time() - start_time
        print(f"Processing finished in {duration:.2f} seconds.")

        if result.lower().startswith("error:"):
            socketio.emit('processing_result', {'error': result, 'duration': duration}, room=client_sid)
        else:
            socketio.emit('processing_result', {'result': result, 'duration': duration}, room=client_sid)

    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        socketio.emit('processing_result', {'error': f'Server error processing image: {str(e)}'}, room=client_sid)
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                print(f"Temporary image removed: {temp_image_path}")
            except OSError as e_clean:
                print(f"Error removing temporary file {temp_image_path}: {e_clean}")

# Renamed from process_video to avoid confusion with QwenVLProcessor.process_video
def process_video_request_worker(client_sid, encoded_video, prompt):
    """Handles the actual processing of a video request from the worker queue."""
    temp_video_path = None
    try:
        # 1. Save the encoded video to a temporary file path
        temp_video_path = save_base64_video_to_tempfile(encoded_video, client_sid)
        
        if not temp_video_path:
            # save_base64_video_to_tempfile should have emitted an error
            # If not, or for a more general fallback:
            # socketio.emit('processing_result', {'error': 'Failed to prepare video file for processing.'}, room=client_sid)
            return

        print(f"Processing video file '{temp_video_path}' with prompt: '{prompt}'")
        socketio.emit('server_status', {'message': f'Processing video file (may take time)...'}, room=client_sid)
        start_time = time.time()

        # 2. Call QwenVLProcessor's process_video with the video file path
        # This method will handle its own frame extraction and cleanup of those frames.
        result = qwen_processor.process_video(
            video_path=temp_video_path,  # Correct argument name
            prompt=prompt,
            fps=VIDEO_PROCESSING_FPS,      # Using configured FPS
            max_frames=VIDEO_PROCESSING_MAX_FRAMES, # Using configured max frames
            max_tokens=PROCESSING_MAX_TOKENS_VID # Using configured video token limit
        )

        duration = time.time() - start_time
        print(f"Video processing finished in {duration:.2f} seconds.")

        if result.lower().startswith("error:"):
            socketio.emit('processing_result', {'error': result, 'duration': duration}, room=client_sid)
        else:
            socketio.emit('processing_result', {'result': result, 'duration': duration}, room=client_sid)

    except Exception as e:
        print(f"Error processing video request: {e}")
        traceback.print_exc()
        socketio.emit('processing_result', {'error': f'Server error processing video: {str(e)}'}, room=client_sid)
    finally:
        # 3. Clean up the temporary video file itself
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                print(f"Temporary video file removed: {temp_video_path}")
            except OSError as e_clean:
                print(f"Error removing temporary video file {temp_video_path}: {e_clean}")

@socketio.on('process_image_request')
def handle_image_processing(data):
    client_sid = request.sid
    print(f"Received image processing request from {client_sid}")

    if not PROCESSOR_AVAILABLE or qwen_processor is None:
        print("Processor not available, cannot process request.")
        emit('processing_result', {'error': 'AI Processor not available on server.'}, room=client_sid)
        return

    prompt = data.get('prompt')
    encoded_image = data.get('image_data')

    if not prompt or not encoded_image:
        print("Missing prompt or image data.")
        emit('processing_result', {'error': 'Missing prompt or image data in request.'}, room=client_sid)
        return

    processing_queue.put((client_sid, 'image', encoded_image, prompt))
    emit('server_status', {'message': 'Image request queued for processing...'}, room=client_sid)

@socketio.on('process_video_request')
def handle_video_processing(data):
    client_sid = request.sid
    print(f"Received video processing request from {client_sid}")

    if not PROCESSOR_AVAILABLE or qwen_processor is None:
        print("Processor not available, cannot process request.")
        emit('processing_result', {'error': 'AI Processor not available on server.'}, room=client_sid)
        return

    prompt = data.get('prompt')
    encoded_video = data.get('video_data')

    if not prompt or not encoded_video:
        print("Missing prompt or video data.")
        emit('processing_result', {'error': 'Missing prompt or video data in request.'}, room=client_sid)
        return

    processing_queue.put((client_sid, 'video', encoded_video, prompt))
    emit('server_status', {'message': 'Video request queued for processing...'}, room=client_sid)


if __name__ == '__main__':
    worker = threading.Thread(target=worker_thread, daemon=True)
    worker.start()
    
    print("Starting Flask-SocketIO server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    print("Flask-SocketIO server stopped.")

    processing_queue.put(None)
    worker.join(timeout=5)