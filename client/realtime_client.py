import os
import sys
import time
import base64
import socketio
import numpy as np
import cv2
import threading
from collections import deque

# --- Configuration ---

SERVER_URL = '' # Add servr ip adress here

# Video capture settings
VIDEO_FPS = 10  # Target FPS for video capture
VIDEO_DURATION = 3  # Duration in seconds
VIDEO_FRAMES = VIDEO_FPS * VIDEO_DURATION  # Total frames to capture for video

# Performance settings
DISPLAY_SCALE = 0.75  # Scale factor for display (reduce to save CPU)
JPEG_QUALITY = 85  # JPEG quality for image/video encoding (lower = smaller size)

# --- SocketIO Client Setup ---
sio = socketio.Client()
connected_to_server = False
last_result = "No analysis yet."
last_status = "Connecting..."
processing_in_progress = False

# Circular buffer for frames
frame_buffer = deque(maxlen=VIDEO_FRAMES)

@sio.event
def connect():
    global connected_to_server, last_status
    print("Successfully connected to processing server.")
    connected_to_server = True
    last_status = "Connected. Press 'c' for image, 'v' for video capture, 'q' to quit."

@sio.event
def connect_error(data):
    global connected_to_server, last_status
    print(f"Connection failed: {data}")
    connected_to_server = False
    last_status = f"Connection Failed: {data}. Check server URL and status."

@sio.event
def disconnect():
    global connected_to_server, last_status
    print("Disconnected from server.")
    connected_to_server = False
    last_status = "Disconnected. Press 'q' to quit."

@sio.on('server_status')
def on_server_status(data):
    """Handles status messages from the server."""
    global last_status
    message = data.get('message', 'Status update received')
    print(f"[Server Status]: {message}")
    last_status = message  

@sio.on('processing_result')
def on_processing_result(data):
    """Handles the analysis result from the server."""
    global last_result, last_status, processing_in_progress
    print("\n--- Analysis Result Received ---")
    if 'error' in data:
        error_msg = data['error']
        print(f"Processing Error: {error_msg}")
        last_result = f"Error: {error_msg}"
    elif 'result' in data:
        result_text = data['result']
        duration = data.get('duration', -1)
        print(result_text)
        if duration >= 0:
            print(f"(Server processing took {duration:.2f} seconds)")
        last_result = result_text
    else:
        print("Received unknown result format.")
        last_result = "Received unknown result format."
    print("------------------------------\n")
    last_status = "Ready. Press 'c' for image, 'v' for video capture, 'q' to quit."
    processing_in_progress = False


def get_user_prompt():
    """Gets a text prompt from the user via console input."""
    print("\nWaiting for prompt input in console...")
    while True:
        try:
            prompt = input("Enter your prompt (or type 'quit_prompt' to cancel): ")
            if prompt.lower() == 'quit_prompt':
                return None
            if prompt:
                return prompt
            else:
                print("Prompt cannot be empty.")
        except (EOFError, KeyboardInterrupt):
            print("\nInput cancelled.")
            return None


def capture_video_frames(cap, frames_needed, frame_rate):
    """Captures a set of frames at the specified frame rate."""
    global last_status, frame_buffer
    
    # Clear buffer
    frame_buffer.clear()
    
    start_time = time.time()
    frame_count = 0
    target_interval = 1.0 / frame_rate

    last_status = f"Capturing video: 0/{frames_needed} frames..."
    
    while frame_count < frames_needed:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            return False
            
        # Add frame to buffer
        frame_buffer.append(frame.copy())
        frame_count += 1
        
        # Update status every 5 frames
        if frame_count % 5 == 0 or frame_count == frames_needed:
            last_status = f"Capturing video: {frame_count}/{frames_needed} frames..."
            # Show a preview of the current frame during capture
            display_frame = frame.copy()
            if DISPLAY_SCALE != 1.0:
                display_frame = cv2.resize(display_frame, 
                    (int(frame.shape[1] * DISPLAY_SCALE), int(frame.shape[0] * DISPLAY_SCALE)))
            cv2.putText(display_frame, last_status, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Webcam Feed (Capturing Video...)', display_frame)
            cv2.waitKey(1)  # Process UI events
            
        # Calculate sleep time to maintain frame rate
        elapsed = time.time() - start_time
        expected_time = frame_count * target_interval
        sleep_time = max(0, expected_time - elapsed)
        
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    total_time = time.time() - start_time
    print(f"Captured {frame_count} frames in {total_time:.2f} seconds ({frame_count/total_time:.1f} fps)")
    return True


def encode_video_to_mp4(frames, fps=10, quality=85):
    """Encode frames to an MP4 video in memory."""
    if not frames:
        return None
        
    # Create a temporary file for the video
    temp_file = f"temp_video_{time.time()}.mp4"
    
    # Get dimensions from the first frame
    height, width = frames[0].shape[:2]
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(temp_file, fourcc, fps, (width, height))
    
    # Write frames
    for frame in frames:
        out.write(frame)
    
    # Release the VideoWriter
    out.release()
    
    # Read the file back into memory
    with open(temp_file, 'rb') as f:
        video_bytes = f.read()
        
    # Delete the temporary file
    try:
        os.remove(temp_file)
    except:
        pass
        
    # Encode to base64
    encoded_video = base64.b64encode(video_bytes).decode('utf-8')
    
    return encoded_video


def process_frames_thread(mode, prompt):
    """Thread function to process frames to avoid blocking the UI."""
    global processing_in_progress, last_status, frame_buffer
    
    processing_in_progress = True
    
    try:
        if mode == 'image' and frame_buffer:
            # Get the last frame from buffer
            frame = frame_buffer[-1]
            
            # Encode frame to JPEG
            retval, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not retval:
                raise ValueError("Failed to encode frame to JPEG")
                
            # Encode to base64
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            
            # Send to server
            print(f"Sending image with prompt: '{prompt}'")
            last_status = "Sending image to server..."
            sio.emit('process_image_request', {
                'prompt': prompt,
                'image_data': encoded_image
            })
            
        elif mode == 'video':
            # Encode video from buffer
            print("Encoding video...")
            last_status = "Encoding video..."
            encoded_video = encode_video_to_mp4(list(frame_buffer), fps=VIDEO_FPS, quality=JPEG_QUALITY)
            
            if not encoded_video:
                print("Failed to encode video.")
                last_status = "Failed to encode video."
                processing_in_progress = False
                return
                
            # Send to server
            print(f"Sending video with prompt: '{prompt}'")
            last_status = "Sending video to server..."
            sio.emit('process_video_request', {
                'prompt': prompt,
                'video_data': encoded_video
            })
            
    except Exception as e:
        print(f"Error in processing thread: {e}")
        last_status = f"Error: {str(e)}"
        processing_in_progress = False


def main_client_loop():
    global last_status, last_result, processing_in_progress, frame_buffer

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Try to set webcam resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened: {frame_width}x{frame_height}")

    # Calculate display size
    display_width = int(frame_width * DISPLAY_SCALE)
    display_height = int(frame_height * DISPLAY_SCALE)

    print("\n--- Controls ---")
    print(" 'c' - Capture current frame, get prompt, send for analysis")
    print(" 'v' - Capture short video, get prompt, send for analysis")
    print(" 'q' - Quit")
    print("----------------\n")

    window_name = 'Webcam Feed (Press c=image, v=video, q=quit)'
    cv2.namedWindow(window_name)
    
    # Flag to indicate window is created and key handling is active
    window_active = True

    # Main loop
    try:
        while window_active:
            # Capture frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                last_status = "Error: Webcam frame capture failed."
                break

            # Add to frame buffer
            frame_buffer.append(frame.copy())
                
            # Create a copy for display
            display_frame = frame.copy()
            
            # Resize for display if needed
            if DISPLAY_SCALE != 1.0:
                display_frame = cv2.resize(display_frame, (display_width, display_height))

            # Add status text
            status_text = f"Status: {last_status}"
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Add result text (truncated)
            result_text = f"Last Result: {last_result[:100]}{'...' if len(last_result)>100 else ''}"
            cv2.putText(display_frame, result_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Display frame
            cv2.imshow(window_name, display_frame)

            # Process keypress (waitKey returns -1 if no key is pressed)
            key = cv2.waitKey(1) & 0xFF
            
            # Debug key press
            if key != 255:  # 255 is when no key is pressed
                print(f"Key pressed: {key} (ASCII: {chr(key) if key < 128 else 'non-ASCII'})")

            # Quit - handle both 'q' and ESC key
            if key == ord('q') or key == 27:  # 27 is ESC key
                print("Quitting...")
                window_active = False
                break

            # Process single image
            elif key == ord('c') and not processing_in_progress:
                if not connected_to_server:
                    print("Not connected to server. Cannot process.")
                    last_status = "Cannot process: Not connected to server."
                    continue

                print("\nImage capture requested...")
                
                # Get prompt from user
                prompt = get_user_prompt()
                if not prompt:
                    print("Processing cancelled (no prompt provided).")
                    last_status = "Ready. Press 'c' for image, 'v' for video capture, 'q' to quit."
                    continue
                
                # Process in background thread
                threading.Thread(target=process_frames_thread, args=('image', prompt)).start()

            # Process video
            elif key == ord('v') and not processing_in_progress:
                if not connected_to_server:
                    print("Not connected to server. Cannot process.")
                    last_status = "Cannot process: Not connected to server."
                    continue

                print(f"\nVideo capture requested ({VIDEO_DURATION}s @ {VIDEO_FPS}fps)...")
                
                # Capture video frames
                if not capture_video_frames(cap, VIDEO_FRAMES, VIDEO_FPS):
                    print("Video capture failed.")
                    last_status = "Video capture failed."
                    continue
                
                # Get prompt from user
                prompt = get_user_prompt()
                if not prompt:
                    print("Processing cancelled (no prompt provided).")
                    last_status = "Ready. Press 'c' for image, 'v' for video capture, 'q' to quit."
                    continue
                
                # Process in background thread
                threading.Thread(target=process_frames_thread, args=('video', prompt)).start()

    finally:  # Cleanup
        print("\nReleasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        if connected_to_server:
            sio.disconnect()


if __name__ == "__main__":
    try:
        print(f"Attempting to connect to server at {SERVER_URL}...")
        sio.connect(SERVER_URL, wait_timeout=10)
        
        if connected_to_server:
            main_client_loop()
        else:
            print("Could not connect to the server. Please check the URL and ensure the server is running.")
            print("Exiting.")

    except socketio.exceptions.ConnectionError as e:
        print(f"\nFatal Connection Error: {e}")
        print("Please ensure the server is running and accessible at the specified URL.")
    except KeyboardInterrupt:
        print("\nClient interrupted. Exiting.")
    finally:
        if sio.connected:
            sio.disconnect()