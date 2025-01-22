import sys
import os
import json
import time
import subprocess
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import argparse
import traceback
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import sounddevice as sd
from vosk import Model as STTModel, KaldiRecognizer
from pydub import AudioSegment
import simpleaudio as sa


import aria.sdk as aria
from projectaria_tools.core.sensor_data import (
    ImageDataRecord,
    AudioData,
    AudioDataRecord,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
caption_queue = queue.Queue(
    maxsize=1
)  # Limit the queue to 1 element to always keep the newest caption
audio_enabled = False
language = "en"  # Default language is English

samplerate = 48000  # TODO adapt to config
channels = 1

listening = False


def update_iptables() -> None:
    """
    Update firewall to permit incoming UDP connections for DDS
    """
    update_iptables_cmd = [
        "sudo",
        "iptables",
        "-A",
        "INPUT",
        "-p",
        "udp",
        "-m",
        "udp",
        "--dport",
        "7000:8000",
        "-j",
        "ACCEPT",
    ]
    print("Running the following command to update iptables:")
    print(update_iptables_cmd)
    subprocess.run(update_iptables_cmd)


def quit_keypress():
    key = cv2.waitKey(1)
    # Press ESC, 'q'
    return key == 27 or key == ord("q")


# Function to initialize TTS and start threading
def init_tts_engine():
    # Start the caption worker thread
    worker_thread = threading.Thread(target=speak_caption_worker, daemon=True)
    worker_thread.start()
    return None  # No need for further initialization


# Function to run macOS 'say' command in a separate thread and speak captions sequentially
def speak_caption_worker():
    while True:
        # Get the next caption from the queue
        caption = caption_queue.get()
        if caption is None:
            break  # Exit the worker thread

        # Speak the caption using 'say' command only if audio is enabled
        if audio_enabled:
            command = ["say"]
            if language == "de":
                command.extend(["-v", "Anna"])
            if caption:
                command.extend([caption])
                subprocess.run(command)

        # Mark the task as done
        caption_queue.task_done()


# Generate the response of the model based on the frame and prompt
def generate_caption(model, frame, prompt):
    # Convert OpenCV frame to PIL Image for the VLM
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Generate caption
    result = model.generate_caption(pil_image, prompt=prompt)
    return result


def replace_umlaute(text: str) -> str:
    """
    Replaces German special characters and others that can't
    be displayed by OpenCV with substitute characters.
    """
    return (
        text.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
        .replace("\n", " ")
    )


# Function to split the text into multiple lines that fit within the image width
def wrap_text(text, font, font_scale, thickness, img_width):
    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        # Calculate the width of the current line if we add this word
        text_size, _ = cv2.getTextSize(current_line + word, font, font_scale, thickness)
        line_width = text_size[0]

        # If the current line width exceeds the image width, start a new line
        if line_width > img_width - 20:  # 20 is the padding
            lines.append(current_line)
            current_line = word + " "  # Start a new line with the current word
        else:
            current_line += word + " "

    # Append the last line
    if current_line:
        lines.append(current_line.strip())

    return lines


# Function to add transparent rectangle with multiline text at the bottom of the frame
def add_caption_to_frame(
    frame,
    text="Placeholder caption",
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.7,
    thickness=1,
):
    img_height, img_width, _ = frame.shape
    text = replace_umlaute(text)
    lines = wrap_text(text, font, font_scale, thickness, img_width)

    text_height = cv2.getTextSize("Test", font, font_scale, thickness)[0][1]
    line_spacing = 5
    total_text_height = len(lines) * (text_height + line_spacing)

    rect_x1, rect_y1 = 0, img_height - total_text_height - 50
    rect_x2, rect_y2 = img_width, img_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    y_offset = img_height - total_text_height - 15
    for line in lines:
        text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_width = text_size[0]
        text_x = (img_width - text_width) // 2
        cv2.putText(
            frame,
            line,
            (text_x, y_offset),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        y_offset += text_height + line_spacing

    return frame


# Function to add a new caption to the queue (replace the old one if the queue is full)
def add_caption_to_queue(caption):
    if not caption_queue.empty():
        try:
            caption_queue.get_nowait()  # Remove the oldest caption
        except queue.Empty:
            pass  # In case there is no caption, do nothing
    caption_queue.put(caption)


def transcribe_audio(recognizer, audio_buffer):
    """
    Transcribes the recorded audio into text (STT).
    """
    print("Transcribing...")
    audio_data = (audio_buffer * 32767).astype(np.int16)
    audio_data = audio_data.tobytes()
    if not recognizer.AcceptWaveform(audio_data):
        print("Transcription error due to not accepted waveform.")
    result = json.loads(recognizer.Result()).get("text", "")
    print("Transcription complete. You said:\n", result)
    return result


# Function to display available key commands
def display_help():
    print("\nAvailable Key Commands:")
    print("'q' : Quit the application.")
    print("'x' : Activate camera only mode.")
    print("'c' : Activate captioning mode.")
    print("'v' : Activate vision assistant mode.")
    print("'o' : While in vision assistant mode, start listening.")
    print("'p' : While in vision assistant mode, stop listening.")
    print("'l' : Toggle language.")
    print("'a' : Toggle audio on/off.")
    print("'1' : Switch the camera.")
    print("'h' : Display this help message.\n")


# Class for the stream observer
class StreamingClientObserver:
    def __init__(self):
        self.images = {}
        self.audio = []
        self.audio_timestamps_ns = []

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.images[record.camera_id] = image

    def on_audio_received(
        self,
        audio_data: AudioData,
        record: AudioDataRecord,
    ):
        self.audio += audio_data.data
        self.audio_timestamps_ns += record.capture_timestamps_ns


def main():
    parser = argparse.ArgumentParser(
        description="Vision assistant that helps interacting with the environment using the Aria glasses."
    )
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=False,
        default="wifi",
        choices=["usb", "wifi"],
        help="Type of interface to use for streaming. Options are usb or wifi.",
    )
    parser.add_argument(
        "--device-ip", help="IP address to connect to the device over wifi."
    )
    parser.add_argument(
        "-c",
        "--camera",
        dest="camera_index",
        type=int,
        required=False,
        default=0,
        choices=[0, 1, 2],
        help="0: RGB Camera, 1: SLAM1 Camera, 2: SLAM2 Camera.",
    )
    parser.add_argument(
        "-p",
        "--profile",
        dest="profile_name",
        type=str,
        required=False,
        default="profile18",
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--update-iptables",
        dest="update_iptables",
        action="store_true",
        required=False,
        default=False,
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        type=str,
        required=False,
        default="llava",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--mlx",
        dest="mlx",
        action="store_true",
        required=False,
        default=False,
        help="Use the mlx version of LLava.",
    )
    parser.add_argument(
        "--caption-interval",
        dest="caption_interval",
        type=int,
        required=False,
        default=0,
        help="Interval in seconds between caption updates.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        default=False,
        help="Get Aria debug information on the console.",
    )
    args = parser.parse_args()

    camera_index = args.camera_index
    model_name = args.model
    caption_interval = args.caption_interval

    cameras = {
        0: aria.CameraId.Rgb,
        1: aria.CameraId.Slam1,
        2: aria.CameraId.Slam2,
    }

    global listening
    stt_model_en = STTModel("./vosk-model-small-en-us-0.15")
    stt_model_de = STTModel("./vosk-model-small-de-0.15")
    recognizer_en = KaldiRecognizer(stt_model_en, samplerate)
    recognizer_de = KaldiRecognizer(stt_model_de, samplerate)
    recognizer = recognizer_en  # Default language is English

    # Update the ip tables on Linux to ensure streaming from the Aria glasses to work properly
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    # Set Aria log level to debug if verbosity is desired
    if args.verbose:
        aria.set_log_level(aria.Level.Debug)
    else:
        aria.set_log_level(aria.Level.Info)

    # Create DeviceClient instance, setting the IP address if specified
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)

    # Connect to the Aria glasses
    device = device_client.connect()

    # Retrieve the streaming_manager and streaming_client
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    # Set custom config for streaming
    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name

    # Note: by default streaming uses Wifi
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb

    # Use ephemeral streaming certificates
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    # Start streaming
    streaming_manager.start_streaming()

    # Get streaming state
    streaming_state = streaming_manager.streaming_state
    print(f"Streaming state: {streaming_state}")

    # Create and attach observer
    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    if args.verbose:
        print(f"Aria Streaming profile: {streaming_config.profile_name}")

    # Initialize TTS engine (macOS 'say' does not need initialization)
    init_tts_engine()

    # Load in the VLM model
    if args.mlx:
        # Load LLava model and processor once, after threading is initialized
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.getcwd(), "../mlx-examples/llava"))
        )
        model_name = "llava-hf/llava-1.5-7b-hf"
        import llava_ifc

        model = llava_ifc.LLavaMLX(model_name, {})
    else:
        # Use the Ollama interface
        import ollama_ifc

        model = ollama_ifc.OllamaVLM(model_name)
        print(f"Using model: {model_name}")

    mode = "watching"
    global audio_enabled
    global language

    latest_frame = None
    frozen_image = None

    captioning_prompt_en = "Describe this image in a short single sentence. Please do not exceed 15 words in total."
    captioning_prompt_de = "Beschreibe dieses Bild in einem einzigen kurzen Satz. Verwende auf keinen Fall mehr als insgesamt 15 Worte in deiner Antwort."
    assistant_prompt_en = "You are assisting a visually impaired person. Their question and the image they 'see' are provided. Answer concisely to directly address their question based on the visual and contextual input. Please do not exceed 25 words in total. Here comes the user's request: "
    assistant_prompt_de = "Du unterstützt eine sehbehinderte Person. Ihre Frage und das Bild, das sie 'sieht', werden bereitgestellt. Antworte präzise und gehe direkt auf die Frage ein, basierend auf den visuellen und kontextuellen Eingaben. Nutze auf keinen Fall mehr als 25 Worte. Hier ist die Frage des Nutzers: "
    captioning_prompt = captioning_prompt_en
    assistant_prompt = assistant_prompt_en

    latest_instruction = ""
    latest_caption = "No caption available"
    executor = ThreadPoolExecutor(max_workers=1)
    caption_future = None
    waiting_for_response = False

    def update_caption_in_background(model, frame, prompt):
        nonlocal latest_caption
        prompt = prompt
        response = generate_caption(model, frame, prompt)
        latest_caption = response
        add_caption_to_queue(response)

    last_caption_time = time.time()  # Track the last time a caption was updated

    images = {}
    audio = []

    # Start the program loop
    try:
        while not quit_keypress():
            # Retrieve the current RGB image
            if aria.CameraId.Rgb in observer.images:
                rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                images[aria.CameraId.Rgb] = rgb_image
                del observer.images[aria.CameraId.Rgb]

            # Retrieve the current SLAM1 image
            if aria.CameraId.Slam1 in observer.images:
                slam1_image = np.rot90(observer.images[aria.CameraId.Slam1], -1)
                slam1_image = cv2.cvtColor(slam1_image, cv2.COLOR_BGR2RGB)
                images[aria.CameraId.Slam1] = slam1_image
                del observer.images[aria.CameraId.Slam1]

            # Retrieve the current SLAM2 image
            if aria.CameraId.Slam2 in observer.images:
                slam2_image = np.rot90(observer.images[aria.CameraId.Slam2], -1)
                slam2_image = cv2.cvtColor(slam2_image, cv2.COLOR_BGR2RGB)
                images[aria.CameraId.Slam2] = slam2_image
                del observer.images[aria.CameraId.Slam2]

            # Retrieve recorded audio
            if not listening:
                observer.audio = []  # Clear audio data when not in use
                observer.audio_timestamps_ns = []
            elif observer.audio:
                audio = observer.audio

            # Choose the image to put into the model
            try:
                latest_frame = images[cameras[camera_index]]
            except:
                print(
                    f"No {cameras[camera_index]} image detected. {len(observer.images) = }"
                )
                continue

            # For the assisting mode, get the image at the moment the user starts speaking
            if frozen_image is None:
                frozen_image = latest_frame

            # Update caption based on mode
            current_time = time.time()
            if caption_future is None or caption_future.done():

                # Captioning mode
                if mode == "captioning" and (
                    current_time - last_caption_time >= caption_interval
                ):
                    caption_future = executor.submit(
                        update_caption_in_background,
                        model,
                        latest_frame,
                        captioning_prompt,
                    )
                    print(
                        f"Caption done in {(current_time - last_caption_time):4.2} seconds."
                    )
                    last_caption_time = current_time

                # Vision assistant mode
                elif mode == "assisting" and waiting_for_response:
                    image = frozen_image if frozen_image is not None else latest_frame

                    mono_audio = audio[::7]  # TODO adapt to number of channels
                    max_sample_value = max(abs(min(mono_audio)), max(mono_audio))
                    normalized_audio = (
                        np.array(mono_audio, dtype=np.float32) / max_sample_value
                    )
                    print(
                        f"Listened for {round(time.time() - start_listening_time, 1)}s."
                    )
                    audio = []  # Clear audio data after usage
                    if args.verbose:
                        sd.play(normalized_audio, samplerate)

                    latest_instruction = transcribe_audio(
                        recognizer, normalized_audio
                    )  # TODO Fehlermeldung dass abgebrochen wird, aber führt trotzdem aus
                    prompt = assistant_prompt + latest_instruction

                    caption_future = executor.submit(
                        update_caption_in_background,
                        model,
                        image,
                        prompt,
                    )
                    waiting_for_response = False

            # Add caption to frame if captioning is enabled
            frame_with_caption = (
                add_caption_to_frame(latest_frame, text=latest_caption)
                if mode == "captioning" or mode == "assisting"
                else latest_frame
            )

            # Display the stream
            cv2.imshow("Aria Glasses View", frame_with_caption)

            # Handle input
            key = cv2.waitKey(5) & 0xFF
            # Quit
            if key == ord("q"):
                print("Exiting the loop.")
                break
            elif key == ord("x"):
                mode = "watching"
                listening = False
                print("Watching mode is active.")

            # Activate caption mode
            elif key == ord("c"):
                mode = "captioning"
                listening = False

                # Empty the queue
                while not caption_queue.empty():
                    caption_queue.get_nowait()

                latest_caption = "No caption available"
                print(f"Captioning mode active. Interval: {caption_interval}s.")
            elif key == ord("v"):
                mode = "assisting"
                listening = False

                # Empty the queue
                while not caption_queue.empty():
                    caption_queue.get_nowait()

                latest_caption = "Press 'o' and ask a question. Press 'p' to stop."
                print("Assisting mode is active")

            # Toggle audio on/off
            elif key == ord("a"):
                audio_enabled = not audio_enabled
                print(f"Audio {'enabled' if audio_enabled else 'disabled'}.")

            # Switch camera
            elif key == ord("1"):
                camera_index = (
                    camera_index + 1 if camera_index < len(cameras) - 1 else 0
                )
                print(f"Switching to camera: {cameras[camera_index]}")

            # Listen to user
            elif (
                key == ord("o") and not listening and mode == "assisting"
            ):  # Press down the key
                print("Listening...")
                start_listening_time = time.time()
                listening = True
                frozen_image = None
            elif listening and (
                key == ord("p") or (time.time() - start_listening_time) >= 30
            ):  # Release the key or exceed the timer
                listening = False
                waiting_for_response = True

            # Toggle language between German and English
            elif key == ord("l"):
                if language == "en":
                    language = "de"
                    recognizer = recognizer_de
                    captioning_prompt = captioning_prompt_de
                    assistant_prompt = assistant_prompt_de
                else:
                    language = "en"
                    recognizer = recognizer_en
                    captioning_prompt = captioning_prompt_en
                    assistant_prompt = assistant_prompt_en
                print("Changed language to", language.upper())

            # Print help
            elif key == ord("h"):
                display_help()

    except Exception as e:
        print("Oops! Something went wrong. Shutting down the stream...")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        executor.shutdown()

        # Stop streaming and disconnect the glasses
        print("Stop listening to image data")
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)

        # Clean up the model after use
        del model

        # Stop the worker thread
        add_caption_to_queue(None)


if __name__ == "__main__":
    main()
