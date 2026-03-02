# test Video/processor.py (Updated for Visual Assistance)

import os
import torch
import cv2
import time
from typing import List, Dict, Union, Optional, Tuple # Keep typing imports
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info # Ensure this import works

# Import gc for potential explicit collection if needed later, though often unnecessary here
import gc

class QwenVLProcessor:
    """Class for processing images and videos with Qwen VL models."""

    # --- NEW: Detailed prompt template for visual assistance ---
    ASSISTIVE_PROMPT_TEMPLATE = """You are an advanced AI assistant for a person who is blind or has low vision. Your primary goal is to follow the user's instructions precisely and provide clear, actionable answers.

**RESPONSE PROTOCOL:** You absolutely MUST structure your response in the following three parts:

1.  **DIRECT ANSWER:**
    - Immediately and directly answer the user's specific question (e.g., "Yes, this is a can of soda," or "The text on the sign reads 'DANGER'.").
    - If the user only asks for a description, this section should be a brief one-sentence summary (e.g., "You are looking at your bedroom desk.").

2.  **REASONING:**
    - Briefly explain *why* you gave the direct answer, based on the visual evidence.
    - If the user provides personal context (e.g., "I am Muslim," "I am allergic to nuts"), you MUST use that context in your reasoning. For example: "My reasoning is that the bottle is clearly labeled as 'Whiskey,' which contains alcohol. Therefore, for a Muslim, it is not permissible to drink."

3.  **DETAILED DESCRIPTION:**
    - After answering the question, provide a full, objective description of the entire scene.
    - Describe objects, their colors, and their positions relative to each other (e.g., "a red book is to the left of a black laptop").
    - **Important:** If the main object is on a phone or computer screen, you must focus your analysis on the **content shown on that screen**, not the screen itself.

---
**User's Request:** "{user_prompt}"
"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """Initialize the processor with the specified model."""
        print(f"Loading model: {model_name}")
        init_start_time = time.time()

        # Determine device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model and processor
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.process_vision_info = process_vision_info

        load_time = time.time() - init_start_time
        print(f"\n\nModel loaded in {load_time:.2f} seconds")

    def run_inference(self, messages: List[Dict], max_tokens: int = 512) -> str:
        """Run inference with the model."""
        inference_total_start_time = time.time()
        try:
            # Prepare input prompt
            prep_start_time = time.time()
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            image_inputs, video_inputs = self.process_vision_info(messages)

            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            prep_time = time.time() - prep_start_time
            print(f"  [Processor] Input preparation time: {prep_time:.2f}s")


            # Generate output
            print("  [Processor] Generating response...")
            gen_start_time = time.time()
            output_text = "" # Initialize output_text

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)

                # Handle potential padding differences if input length varies
                generated_ids = []
                input_lengths = [len(inputs.input_ids[i]) for i in range(len(output_ids))]
                for i in range(len(output_ids)):
                    # Ensure slicing doesn't go out of bounds if generation is short
                    start_index = input_lengths[i]
                    if start_index < len(output_ids[i]):
                         generated_ids.append(output_ids[i][start_index:])
                    else:
                         # Handle case where no new tokens were generated (or only pad tokens)
                         generated_ids.append(torch.tensor([], dtype=torch.long, device=output_ids.device))


                # Decode output
                # Check if generated_ids actually contains tokens before decoding
                if any(len(ids) > 0 for ids in generated_ids):
                     output_text = self.processor.batch_decode(
                         generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                     )[0] # Assuming batch size is 1
                else:
                     output_text = "[No new tokens generated]" # Or some other placeholder

            gen_time = time.time() - gen_start_time
            print(f"  [Processor] Raw generation time: {gen_time:.2f}s")

            return output_text

        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error during inference: {str(e)}"
        finally:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            inference_total_time = time.time() - inference_total_start_time
            print(f"  [Processor] Total run_inference time (incl. prep, gen, cleanup): {inference_total_time:.2f}s")

    def process_image(self, image_path: str, prompt: str, max_tokens: int = 256) -> str:
        """Process a single image with an assistive prompt."""
        if not os.path.exists(image_path):
            return f"Error: Image file not found: {image_path}"

        print(f"Processing image with assistive context: {image_path}")

        full_prompt = self.ASSISTIVE_PROMPT_TEMPLATE.format(user_prompt=prompt)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]
        return self.run_inference(messages, max_tokens)

    def extract_frames(self, video_path: str, output_dir: str = "frames", fps: float = 1.0, max_frames: Optional[int] = None) -> List[str]:
        """Extract frames from a video file."""
        extract_total_start_time = time.time()
        frame_paths = []
        try:
            output_directory = Path(output_dir)
            output_directory.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                 print(f"Warning: Could not read video FPS for {video_path}. Assuming 30.")
                 video_fps = 30
            frame_interval = max(1, int(video_fps / fps))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Extracting frames from {video_path}")
            print(f"Video FPS: {video_fps:.2f}, Target Extraction FPS: {fps:.2f}")
            print(f"Total frames: {total_frames}, Frame interval: {frame_interval}")
            frame_count = 0
            saved_count = 0
            loop_start_time = time.time()
            pbar_desc = "Extracting frames"
            pbar_total = total_frames if max_frames is None else min(max_frames * frame_interval, total_frames)
            pbar_total = max(1, pbar_total)
            with tqdm(total=pbar_total, desc=pbar_desc) as pbar:
                while True:
                    if frame_interval > 1 and frame_count > 0 :
                        next_frame_to_read = frame_count + frame_interval - (frame_count % frame_interval)
                        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        if next_frame_to_read > current_pos + 1:
                             cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_to_read)
                             actual_pos_after_set = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                             pbar.update(actual_pos_after_set - current_pos)
                             frame_count = actual_pos_after_set
                    ret, frame = cap.read()
                    if not ret:
                        pbar.update(pbar.total - pbar.n)
                        break
                    current_pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    pbar.update(current_pos_frame - frame_count)
                    frame_count = current_pos_frame
                    frame_index_to_check = frame_count - 1
                    if frame_index_to_check % frame_interval == 0:
                        timestamp_sec = frame_index_to_check / video_fps
                        filename = f"frame_{timestamp_sec:08.3f}s.jpg"
                        output_path = str(output_directory / filename)
                        cv2.imwrite(output_path, frame)
                        frame_paths.append(output_path)
                        saved_count += 1
                        if max_frames is not None and saved_count >= max_frames:
                            pbar.update(pbar.total - pbar.n)
                            break
            cap.release()
            loop_time = time.time() - loop_start_time
            print(f"  [Processor] Frame reading/saving loop time: {loop_time:.2f}s")
            print(f"Extracted {len(frame_paths)} frames to '{output_directory.resolve()}'")
            return frame_paths
        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            extract_total_time = time.time() - extract_total_start_time
            print(f"  [Processor] Total extract_frames time: {extract_total_time:.2f}s")

    def process_video(self, video_path: str, prompt: str, fps: float = 1.0, max_frames: int = 10, max_tokens: int = 512) -> str:
        """Process a video by extracting frames and analyzing them with an assistive prompt."""
        if not os.path.exists(video_path):
            return f"Error: Video file not found: {video_path}"

        print(f"Processing video with assistive context: {video_path}")
        frame_output_dir = Path("frames") / Path(video_path).stem
        frames = self.extract_frames(video_path, output_dir=str(frame_output_dir), fps=fps, max_frames=max_frames)

        if not frames:
            if frame_output_dir.exists():
                 try:
                     import shutil
                     shutil.rmtree(frame_output_dir)
                     print(f"Cleaned up empty/failed frame directory: {frame_output_dir}")
                 except Exception as e_clean:
                     print(f"Could not cleanup frame directory {frame_output_dir}: {e_clean}")
            return "Error: Could not extract frames from video, or no frames were selected based on FPS/max_frames."

        result = "Error: Processing step skipped due to frame extraction issues."
        try:
            full_prompt = self.ASSISTIVE_PROMPT_TEMPLATE.format(user_prompt=prompt)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": full_prompt}
                    ],
                }
            ]
            result = self.run_inference(messages, max_tokens)
        finally:
            try:
                import shutil
                if frame_output_dir.exists():
                    shutil.rmtree(frame_output_dir)
                    print(f"Cleaned up temporary frames in {frame_output_dir}")
            except Exception as e:
                print(f"Could not clean up frames directory {frame_output_dir}: {e}")

        return result