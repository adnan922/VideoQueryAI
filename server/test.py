# test.py
# Standalone script to test the QwenVLProcessor class with a user-provided dataset.
#
# How to use:
# 1. Place this file in the same directory as 'processor.py' on your server.
# 2. Create a directory named 'test_dataset' in the same location.
# 3. Inside 'test_dataset', create two more directories: 'images' and 'videos'.
# 4. Populate 'test_dataset/images' with your image files (e.g., .jpg, .png).
# 5. Populate 'test_dataset/videos' with your video files (e.g., .mp4).
# 6. Run from the terminal: python test_processor.py

import os
import time
from pathlib import Path
import shutil
import torch # Import torch to use for CUDA cache clearing

# Make sure to handle potential import errors if the processor is not available
try:
    from processor import QwenVLProcessor
    PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"FATAL: Could not import QwenVLProcessor from processor.py: {e}")
    print("Please ensure 'processor.py' is in the same directory and all dependencies are installed.")
    PROCESSOR_AVAILABLE = False
except Exception as e:
    print(f"FATAL: An unexpected error occurred during import: {e}")
    PROCESSOR_AVAILABLE = False

# --- Test Configuration ---
TEST_DATA_DIR = Path("./test_dataset")
IMAGE_DIR = TEST_DATA_DIR / "images"
VIDEO_DIR = TEST_DATA_DIR / "videos"
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']

def load_tests_from_dataset():
    """
    Loads test cases by discovering image and video files in the user-provided dataset directory.
    """
    print("--- Loading test cases from user-provided dataset ---")
    
    # Verify that the main directories exist
    if not TEST_DATA_DIR.is_dir() or not IMAGE_DIR.is_dir() or not VIDEO_DIR.is_dir():
        print(f"Error: Required directory structure not found.")
        print(f"Please create the following directories and add your files:")
        print(f"  - {TEST_DATA_DIR}/")
        print(f"    - {IMAGE_DIR}/")
        print(f"    - {VIDEO_DIR}/")
        return None

    test_cases = []
    
    # Generic prompt for any image
    image_prompt = "You are an advanced visual assistant. Describe the contents of this image in detail, including all objects, people, text, and the overall scene."
    
    # Load images
    image_files = []
    for ext in SUPPORTED_IMAGE_FORMATS:
        image_files.extend(list(IMAGE_DIR.glob(f'*{ext}')))
    
    print(f"Found {len(image_files)} images in '{IMAGE_DIR}'.")
    for img_path in image_files:
        test_cases.append({
            "type": "image",
            "path": str(img_path),
            "prompt": image_prompt
        })

    # Generic prompt for any video
    video_prompt = "You are an advanced visual assistant. Describe the events in this video clip from beginning to end. Detail the actions, objects, and any changes in the scene."

    # Load videos
    video_files = []
    for ext in SUPPORTED_VIDEO_FORMATS:
        video_files.extend(list(VIDEO_DIR.glob(f'*{ext}')))

    print(f"Found {len(video_files)} videos in '{VIDEO_DIR}'.")
    for vid_path in video_files:
        test_cases.append({
            "type": "video",
            "path": str(vid_path),
            "prompt": video_prompt
        })
        
    if not test_cases:
        print("\nWarning: No image or video files were found in the dataset directories.")
        return None

    print(f"--- Loaded a total of {len(test_cases)} test cases ---\n")
    return test_cases

def run_tests(processor, test_cases):
    """
    Iterates through the test cases and runs them against the QwenVLProcessor.
    """
    if not test_cases:
        print("No test cases to run.")
        return

    print("--- Starting Model Inference Tests ---")
    total_start_time = time.time()
    
    # Create a directory to store text results
    results_dir = Path("./test_results")
    results_dir.mkdir(exist_ok=True)
    results_summary_file = results_dir / f"summary_{time.strftime('%Y%m%d-%H%M%S')}.txt"

    with open(results_summary_file, 'w', encoding='utf-8') as summary_f:
        for i, test in enumerate(test_cases):
            print(f"\n--- Running Test Case {i+1}/{len(test_cases)} ---")
            print(f"Type:   {test['type'].upper()}")
            print(f"File:   {test['path']}")
            print("-" * 20)
            
            summary_f.write(f"--- Test Case {i+1}/{len(test_cases)} ---\n")
            summary_f.write(f"File: {test['path']}\n")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            start_time = time.time()
            result = ""
            try:
                if test['type'] == 'image':
                    result = processor.process_image(
                        image_path=test['path'],
                        prompt=test['prompt'],
                        max_tokens=256
                    )
                elif test['type'] == 'video':
                    result = processor.process_video(
                        video_path=test['path'],
                        prompt=test['prompt'],
                        fps=5.0,
                        max_frames=10,
                        max_tokens=512
                    )
            except Exception as e:
                result = f"TEST FAILED WITH AN EXCEPTION: {e}"
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            duration = time.time() - start_time
            
            print(f"\n[MODEL OUTPUT]")
            # Print a snippet of the result to the console
            print(result[:400] + ('...' if len(result) > 400 else ''))
            print(f"\nTest case finished in {duration:.2f} seconds.")
            print("-" * 40)
            
            # Write full result to the summary file
            summary_f.write(f"Duration: {duration:.2f}s\n")
            summary_f.write(f"Prompt: \"{test['prompt']}\"\n")
            summary_f.write("[MODEL OUTPUT]\n")
            summary_f.write(result + "\n\n")

    total_duration = time.time() - total_start_time
    print(f"\n--- All tests completed in {total_duration:.2f} seconds. ---")
    print(f"Full results have been saved to: {results_summary_file.resolve()}")


if __name__ == "__main__":
    if not PROCESSOR_AVAILABLE:
        print("Exiting because QwenVLProcessor could not be loaded.")
    else:
        # 1. Load test cases from the user-provided dataset
        test_cases = load_tests_from_dataset()
        
        if test_cases:
            # 2. Initialize the model processor
            qwen_processor = None
            try:
                print("Initializing QwenVLProcessor... (This may take a while on first run)")
                qwen_processor = QwenVLProcessor()
            except Exception as e:
                print(f"FATAL: Failed to initialize QwenVLProcessor: {e}")
            
            # 3. Run the tests if processor is ready
            if qwen_processor:
                run_tests(qwen_processor, test_cases)
