from dataclasses import dataclass
import sys
import os
import json
import time
import subprocess
import logger
import queue
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
from vosk import Model as STTModel, KaldiRecognizer
import argparse
import traceback
import pyaudio
#import sounddevice as sd
import numpy as np
from PIL import Image
import cv2
import soundfile as sf
from TTS.api import TTS
import multiprocessing
from tts_process import tts_worker


import aria.sdk as aria
from projectaria_tools.core.sensor_data import (
    ImageDataRecord,
    AudioData,
    AudioDataRecord,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
caption_queue = queue.Queue(maxsize=1)  # Limit the queue to 1 element to always keep the newest caption
caption_lock = False
audio_stopped = False

audio_enabled = False

audio_queue = deque(maxlen=10)  # keep latest 10 chunks
assistant_queue = queue.Queue()





def assistant_worker():
    print("Assistant worker thread started")
    while True:
        item = assistant_queue.get()
        print("Assistant worker received item type:", type(item))
        if item is None:
            break  # Stop signal
        
        image, response_processor, assistant_executor, recognizer_manager, observer, args, samplerate, channels = item
        
        # Run VisionAssistantInteraction synchronously here
        vision_assistant_interaction = VisionAssistantInteraction(
            response_processor=response_processor,
            assistant_executor=assistant_executor,
            recognizer_manager=recognizer_manager,
            observer=observer,
            args=args,
            samplerate=samplerate,
            channels=channels
        )
        
        vision_assistant_interaction.run(image)
        
        assistant_queue.task_done()




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


'''
def tts_speaker():
    print("TTS speaker thread started")
    p = pyaudio.PyAudio()

    while True:
        item = tts_queue.get()
        if item is None:
            break
        text, lang = item
        engine = tts_engine_en if lang == "en" else tts_engine_de
        audio = engine.tts(text, split_sentences=False)

        # Play using pyaudio
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=24000, output=True)
        stream.write(np.array(audio, dtype=np.float32).tobytes())
        stream.close()
        tts_queue.task_done()

    p.terminate()


# Function to initialize TTS and start threading
def init_tts_engine():
    # Start the caption worker thread
    worker_thread = threading.Thread(target=speak_caption_worker, daemon=True)
    worker_thread.start()
    return None  # No need for further initialization


def tts_speaker(): ##if thread starts for the very first time, the downloading of the model starts mid programme loop, which is wacky
    print("Starting TTS speaker thread...")
    

    try:
       tts_engine_en = TTS(model_name="tts_models/en/ljspeech/fast_pitch", progress_bar=True, gpu=False)
       tts_engine_de = TTS(model_name="tts_models/de/css10/vits-neon", progress_bar=True, gpu=False)
       dummy_audio = tts_engine_en.tts("This is a dummy audio to warm up the TTS engine.", split_sentences=False)

    except Exception as e:
        print(f"❌ Failed to initialize TTS engines: {e}")
        import traceback
        traceback.print_exc()  
        return

    while True:
        try:
            item = tts_queue.get()
            if item is None:
                break  # Graceful shutdown

            text, text_language = item
            tts_engine = tts_engine_en if text_language == "en" else tts_engine_de

            speak(text, tts_engine)

        except Exception as e:
            print(f"❌ TTS speaker thread encountered an error: {e}")
        
        finally:
            tts_queue.task_done()

'''
            
# Function to run macOS 'say' command in a separate thread and speak captions sequentially
def speak(spoken_text, tts_engine):
    print("Speaking:")

    print("Speaking:")
    print("Speaking:")
    print("Speaking:")
    print("Speaking:")

    # Generate speech audio
    audio = tts_engine.tts(spoken_text, split_sentences=False)

    # Ensure float32
    audio = np.array(audio, dtype=np.float32)

    print("about to play audio:")
    print("about to play audio:")
    # Initialize PyAudio
    p = pyaudio.PyAudio() #pyaudio should not be initialized at each call
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=24000,
                    output=True)
    
    print("audio played by pyaudi")

    stream.write(audio.tobytes())

    stream.stop_stream()
    stream.close()
    p.terminate()



'''
def speak_caption_worker():
    global audio_stopped

    tts_engine_en = TTS(
        model_name="tts_models/en/ljspeech/glow-tts",
        progress_bar=True,
        gpu=False,
    )

    tts_engine_de = TTS(
        model_name="tts_models/de/thorsten/vits",
        progress_bar=True,
        gpu=False,
    )

    while True:
        # Get the next caption from the queue
        caption = caption_queue.get()
        if caption is None:
            break  # Exit the worker thread
        audio_stopped = False

        # Speak the caption using 'say' command only if audio is enabled
        if audio_enabled:
            if language == "de":
                audio = tts_engine_de.tts(caption)
            elif language == "en":
                audio = tts_engine_en.tts(caption)
            else:
                audio = np.array((0), dtype=np.float32)

            # Play the audio as long as it is not stopped elsewhere
            if not audio_stopped:  #change to pyaudio later
                sd.play(audio, samplerate=22050)
                sd.wait()

        # Mark the task as done
        caption_queue.task_done()
'''





# Generate the response of the model based on the frame and prompt



def replace_umlaute(text: str) -> str:
    """
    Replaces German special characters and others that can't
    be displayed by OpenCV with substitute characters.
    """
    return (
        text.replace("ä", "ae")
        .replace("Ä", "Ae")
        .replace("ö", "oe")
        .replace("Ö", "Oe")
        .replace("ü", "ue")
        .replace("Ü", "Ue")
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
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.7,
    thickness=1,
):
    try:
        caption = caption_queue.queue[-1]  # Peek without removing
    except IndexError:
        return frame  # No caption to add

    img_height, img_width, _ = frame.shape
    caption = replace_umlaute(caption)
    lines = wrap_text(caption, font, font_scale, thickness, img_width)

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
    try:
        caption_queue.get_nowait()  # Clear previous caption if any
    except queue.Empty:
        pass
    caption_queue.put(caption)



def empty_and_lock_queue():
    global caption_lock
    caption_lock = True
    while not caption_queue.empty():
        caption_queue.get_nowait()


'''
def stop_audio():
    global audio_stopped
    audio_stopped = True
    sd.stop()
'''


def transcribe_audio(recognizer, audio_buffer):
    """
    Transcribes the recorded audio into text (STT).
    """
    print("Transcribing...")
    audio_data = (audio_buffer * 32767).astype(np.int16)
    audio_data = audio_data.tobytes()
    if not recognizer.AcceptWaveform(audio_data):
        print(
            "Transcription error due to not accepted waveform."
        )  # TODO check why this is always the case
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
    #print("'o' : While in vision assistant mode, start listening.")
    #print("'p' : While in vision assistant mode, stop listening.")
    print("'l' : Toggle language.")
    print("'a' : Toggle audio on/off.")
    print("'1' : Switch the camera.")
    print("'h' : Display this help message.\n")

def play_prompt_audio(data, samplerate):
    sf.write("debug_prompt_audio.wav", data, samplerate)  #for debugging

    if data.dtype != np.float32:
        data = data.astype(np.float32)
    p = pyaudio.PyAudio()                   # TODO: 2 instances of pyaudio -> work with 1 instance instead

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1 if data.ndim == 1 else data.shape[1],
                    rate=samplerate,
                    output=True)

    stream.write(data.tobytes())

    stream.stop_stream()
    stream.close()
    p.terminate()
    

def play_keyword_sound(frequency=600, duration=0.3, samplerate=48000, volume=0.3):
    p = pyaudio.PyAudio()

    t = np.linspace(0, duration, int(samplerate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)

    # Apply fade-in and fade-out envelope
    fade_length = int(0.02 * samplerate)  # 20 ms fade
    envelope = np.ones_like(tone)
    envelope[:fade_length] = np.linspace(0, 1, fade_length)
    envelope[-fade_length:] = np.linspace(1, 0, fade_length)
    soft_tone = (tone * envelope * volume).astype(np.float32)

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=samplerate,
                    output=True)
    
    stream.write(soft_tone.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()


def start_tts_process():
    queue = multiprocessing.Queue()
    keyword_listening_flag = multiprocessing.Value('b', True)
    tts_enabled_flag = multiprocessing.Value('b', True)   
    process = multiprocessing.Process(target=tts_worker, args=(queue, keyword_listening_flag, tts_enabled_flag))
    process.start()
    return queue, process, keyword_listening_flag, tts_enabled_flag




def normalize_audio_buffer(buffer):
    max_sample = max(abs(min(buffer)), max(buffer))
    if max_sample == 0:
        return None  # Avoid division by zero
    normalized = (np.array(buffer, dtype=np.float32) / max_sample)
    return (normalized * 32767).astype(np.int16)



def create_kws_model():
    grammar = '["vision", "assistant", "computer", "caption", "observer", "language", "please", "obligation", "misunderstanding", "actually", "basically", "literally", "seriously", "honestly", "definitely", "probably", "anyway", "certainly", "absolutely", "ultimately", "eventually", "genuinely", "ostensibly", "apparently", "evidently", "naturally", "obviously", "remarkably", "specifically", "especially", "importantly", "consequently", "subsequently", "furthermore", "meanwhile", "nonetheless", "regardless", "wherever", "whenever", "however", "therefore", "although", "whereas", "unless", "besides", "indeed", "merely", "simply", "barely", "hardly", "seldom", "rarely", "always", "usually", "often", "hardly", "merely", "nearly", "quite", "rather", "pretty", "truly", "really", "fully", "partly", "mostly", "merely", "solely", "chiefly", "largely", "mainly", "namely", "broadly", "roughly", "mostly", "usually", "often", "seldom", "rarely", "always", "sometimes", "anyhow", "somehow", "anywhere", "somewhere", "everywhere", "nowhere", "anytime", "sometime", "everytime", "never", "forever", "today", "tomorrow", "yesterday", "tonight", "indeed", "rather", "pretty", "quite", "just", "then", "soon", "early", "late", "next", "last", "first", "final", "briefly", "suddenly", "slowly", "quickly", "hardly", "softly", "loudly", "clearly", "fairly"]'
    model = STTModel("./vosk-model-small-en-us-0.15")
    recognizer = KaldiRecognizer(model, 48000, grammar)
    return recognizer



def keyword_listener(response_processor): # TODO: refactor further
    
    keywords = ['computer', 'caption', 'observer', 'language']
    keyword_audio = []
    
    kws_recognizer = create_kws_model()
    last_switch_time = 0

    print("[KWS-LISTENER-THREAD] Starting keyword listener...")

    last_switch_time = 0
    cooldown_seconds = 2

    while True:
        if not response_processor.is_keyword_listening():
            time.sleep(0.1)
            continue
        
        try:
            if audio_queue:
                keyword_audio.extend(audio_queue.popleft())

            #skip data if no data yet
            if not keyword_audio:
                continue

            # Trim buffer, > 48000 was the value before
            if len(keyword_audio) > 24000:
                keyword_audio = keyword_audio[-24000:]

            audio_data = normalize_audio_buffer(keyword_audio)

            if kws_recognizer.AcceptWaveform(audio_data.tobytes()):
                final_result = json.loads(kws_recognizer.Result())
                text = final_result.get("text", "").lower()
                print("[KWS-LISTENER-THREAD] Keyword recognition result as non-json:", text)
            

                matched = [kw for kw in keywords if kw in text]
                if not matched:
                    continue

                current_time = time.time()
                if current_time - last_switch_time < cooldown_seconds:
                    print("[KWS-LISTENER-THREAD] Skipped trigger due to cooldown")
                    continue
                    
                
                if "computer" in matched:
                    print("[KWS-LISTENER-THREAD] 🟢 Keyword 'computer' detected! Switching to assisting mode...")
                    play_keyword_sound()
                    response_processor.set_mode("assisting")
                                    
                        
                elif "caption" in matched:
                    print("[KWS-LISTENER-THREAD] 🟢 Keyword 'caption' detected! Switching to captioning mode...")
                    response_processor.set_mode("captioning")

                elif "language" in text:
                    new_lang = "de" if response_processor.language == "en" else "en"
                    response_processor.set_language(new_lang)
                    print(f"[KWS-LISTENER-THREAD] 🌐 Language switched to: {response_processor.language}")

                elif "observer" in text:
                    print("[KWS-LISTENER-THREAD] Watching mode is active.")
                    response_processor.set_mode("watching")

            else:
                time.sleep(0.3)

        except Exception:
            print("[KWS-LISTENER-THREAD] 🔴 Exception in keyword_listener:")
            traceback.print_exc()







class RecognizerManager:
    def __init__(self, samplerate):
        self.samplerate = samplerate
        self.models = {
            "en": STTModel("./vosk-model-en-us-0.22"),
            "de": STTModel("./vosk-model-de-0.21"),
        }
        self.recognizers = {
            lang: KaldiRecognizer(model, samplerate)
            for lang, model in self.models.items()
        }

    def get_recognizer(self, language_code: str = "en"):
        return self.recognizers.get(language_code, self.recognizers["en"])


class ResponseStateProcessor:
    
    captioning_prompt = {
        "en": "Describe this image in a short single sentence. Please do not exceed 15 words in total.",
        "de": "Beschreibe dieses Bild in einem einzigen kurzen Satz. Verwende auf keinen Fall mehr als insgesamt 15 Worte in deiner Antwort."
    }
    assistant_prompt = {
        "en": "I am a visually impaired person and need assistance. I am wearing glasses which capture the image that is being provided. Please answer concisely to directly address my question based on the visual and contextual input. Do not exceed 25 words in total. Do not mention my visual impairment or the camera's fisheye lens. This is my question:\n",
        "de": "Ich bin eine sehbehinderte Person und benötige Hilfe. Ich trage eine Brille, die das bereitgestellte Bild einfängt. Bitte antworte präzise, um meine Frage anhand der visuellen und textuellen Eingaben direkt zu beantworten. Bitte nutze nicht mehr als 25 Worte für deine Antwort. Erwähne unter keinen Umständen meine Sehbehinderung. Meine Frage lautet:\n"
    }

    def __init__(self, model, mode, tts_queue, language, kws_flag, tts_enabled_flag, add_caption_func):
        self.model = model
        self.mode = mode
        self.tts_queue = tts_queue
        self.language = language
        self.add_caption_func = add_caption_func
        self.keyword_listening_flag = kws_flag
        self.tts_enabled_flag = tts_enabled_flag
        self.assistant_processing = False


        

    def set_mode(self, new_mode):
        print(f"🔁 Switching mode to: {new_mode}")
        self.mode = new_mode

    def set_language(self, new_language):
        print(f"🌐 Language set to: {new_language}")
        self.language = new_language

    def ask_model(self, frame, prompt, mode):
        lang = (self.language or "en").strip().lower()
        if lang not in self.captioning_prompt:
            lang = "en"
        if mode == "captioning":
            full_prompt = self.captioning_prompt[lang]
        elif mode == "assisting":
            full_prompt = self.assistant_prompt[lang] + prompt

        if mode == "watching":
            print("🟢 Skipping model call in 'watching' mode.")
            return None  #
        

        print(f"DEBUG: language={lang}")
        print("DEBUG: full_prompt =", repr(full_prompt))

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.model.ask(pil_image, prompt=full_prompt)
        return result

    def enqueue_speech(self, text, language):
        print(f"📥 Enqueuing speech: {text} (lang={language})")
        self.tts_queue.put((text, language))


    def set_tts_enabled(self, enabled: bool):
        print(f"{'🔇 Pausing' if not enabled else '🔊 Resuming'} TTS playback")
        self.tts_enabled_flag.value = enabled


    def is_keyword_listening(self):
        return self.keyword_listening_flag.value
    
    def disable_keyword_listening(self):
        print("🛑 Disabling keyword listening...")
        self.keyword_listening_flag.value = False

    def enable_keyword_listening(self):
        print("🟢 Enabling keyword listening...")
        self.keyword_listening_flag.value = True


    def process_frame(self, frame, language, prompt, mode):
        """Main method that calls the model and handles the response"""
        print("🟡 Processing frame started")
        print("🟡 Processing frame started")
        print("With prompt: and language:", prompt, language)

        try:
            response = self.ask_model(frame, prompt, mode)  # Your core model call
            print("🟢 Models response:", response)
            self.handle_response(response, language, mode)  # Process based on mode
            return response  # Return response for caller
        except Exception as e:
            print("🔴 Exception in process_frame:", e)
            traceback.print_exc()
            return None

    def handle_response(self, response, language, mode):
        self.enqueue_speech(response, language)  # Every llm response is spoken
        if mode == "captioning":
            self.add_caption_func("Loading caption...")
            self.handle_caption_display(response)  # Only caption mode gets visual display

    def handle_caption_display(self, response):
        try:
            current_caption = caption_queue.queue[-1]
            if current_caption == "Loading caption...":
                caption_queue.get_nowait()
        except queue.Empty:
            pass

        self.add_caption_func(response)

        






def init_aria(args):

    # Set Aria log level to debug if verbosity is desired
    if args.verbose:
        aria.set_log_level(aria.Level.Debug)
    else:
        aria.set_log_level(aria.Level.Info)

    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)

    device = device_client.connect()

    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name

    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb

    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    streaming_manager.start_streaming()

    streaming_state = streaming_manager.streaming_state
    print(f"Streaming state: {streaming_state}")

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    if args.verbose:
        print(f"Aria Streaming profile: {streaming_config.profile_name}")

    return streaming_client, device_client, observer, device, streaming_manager

def parse_args():
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
    return parser.parse_args()


class VisionAssistantInteraction:
    def __init__(self, response_processor, assistant_executor, recognizer_manager, observer, args, samplerate, channels):
        self.response_processor = response_processor
        self.assistant_executor = assistant_executor
        self.recognizer_manager = recognizer_manager
        self.observer = observer
        self.args = args
        self.samplerate = samplerate
        self.channels = channels
    
    def run(self, image):
        print("Keyword COMPUTER detected, please speak your prompt")
        self.response_processor.disable_keyword_listening()
        
        audio = self.record_audio_with_silence_detection()
        
        if audio:
            print(f"Recorded {len(audio)} samples of audio.")
            norm_audio, latest_instruction = self.transcribe_audio(audio)
            #self.response_processor.enable_keyword_listening()
            
            if self.args.verbose:
                print("Audio data received, playing back prompt audio...")
                play_prompt_audio(norm_audio, self.samplerate)
            
            print("latest_instruction result:", latest_instruction)
        
        

        # Submit to VLM
        #time.sleep(0.5)
        self.submit_vlm_query(image, latest_instruction)
        self.response_processor.enable_keyword_listening()

        self.response_processor.assistant_processing = False
        self.response_processor.set_mode("watching")

    
    def record_audio_with_silence_detection(self):
        self.observer.audio.clear()
        audio = []
        start_time = time.time()
        duration = 20
        silence_threshold = 5
        silence_timeout = 4
        chunk_check_interval = 0.1
        last_chunk_time = start_time
        last_audio_activity = start_time
        
        while time.time() - start_time < duration:
            if self.observer.audio:
                audio.extend(self.observer.audio)
                self.observer.audio.clear()
            
            current_time = time.time()
            if current_time - last_chunk_time >= chunk_check_interval:
                if audio:
                    recent_samples = int(self.samplerate * chunk_check_interval)
                    recent_chunk = audio[-recent_samples:] if len(audio) >= recent_samples else audio
                    
                    if recent_chunk:
                        mono_chunk = recent_chunk[::self.channels]
                        max_amplitude = max(abs(sample) for sample in mono_chunk)
                        max_amplitude = max_amplitude >> 20
                    
                    if max_amplitude > silence_threshold:
                        last_audio_activity = current_time
                    last_chunk_time = current_time
            
            time.sleep(0.01)
            
            if current_time - last_audio_activity > silence_timeout:
                print(f"Silence detected after {round(time.time() - start_time, 1)}s, stopping query early")
                break
        

        return audio
    
    def transcribe_audio(self, audio):
        mono_audio = audio[::self.channels]
        max_sample_value = max(abs(min(mono_audio)), max(mono_audio))
        
        if max_sample_value > 0:
            norm_audio = np.array(mono_audio, dtype=np.float32) / max_sample_value
        else:
            print("⚠️ Warning: Audio normalization by zero. Using fallback normalization.")
            norm_audio = np.array(mono_audio, dtype=np.float32) / 1e-6
        
        print(f"Listened for {round(len(audio) / self.samplerate, 1)}s.")  # accurate duration
        
        recognizer = self.recognizer_manager.get_recognizer(self.response_processor.language)
        latest_instruction = transcribe_audio(recognizer, norm_audio)
        
        return norm_audio, latest_instruction
    
    def submit_vlm_query(self, image, instruction):
        response_future = self.assistant_executor.submit(
            self.response_processor.process_frame,
            frame=image,
            language=self.response_processor.language,
            prompt=instruction,
            mode = self.response_processor.mode
            
        )
        print("Future object:", response_future)
        
        if self.response_processor.language == "en":
            self.response_processor.enqueue_speech("Processing", language="en")
        elif self.response_processor.language == "de":
            self.response_processor.enqueue_speech("In Bearbeitung", language="de")



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

    #tts seperate process
    multiprocessing.set_start_method("spawn") 
    tts_queue, tts_proc, keyword_listening_flag, tts_enabled_flag = start_tts_process()

    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()
    
    streaming_client, device_client, observer, device, streaming_manager  = init_aria(args)

    
    camera_index = args.camera_index
    model_name = args.model
    caption_interval = args.caption_interval
    global caption_lock

    # profile18 is the only supported streaming profile with audio
    samplerate = 48000
    global channels
    channels = 7

    cameras = {
        0: aria.CameraId.Rgb,
        1: aria.CameraId.Slam1,
        2: aria.CameraId.Slam2,
    }


   

    
    

    # Load in the VLM model
    if args.mlx: #test mlx later
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

    
    #initial mode is watching
    response_processor = ResponseStateProcessor(model = model, mode="watching", tts_queue=tts_queue, language="en", kws_flag=keyword_listening_flag, tts_enabled_flag=tts_enabled_flag, add_caption_func=add_caption_to_queue)
    kws_thread = threading.Thread(target=keyword_listener, args=(response_processor,), daemon=True)
    kws_thread.start()
    recognizer_manager = RecognizerManager(samplerate)

    assistant_worker_thread = threading.Thread(target=assistant_worker, daemon=True)
    assistant_worker_thread.start()
    


    latest_frame = None
    frozen_image = None

    captioning_prompt_en = "Describe this image in a short single sentence. Please do not exceed 15 words in total."
    captioning_prompt_de = "Beschreibe dieses Bild in einem einzigen kurzen Satz. Verwende auf keinen Fall mehr als insgesamt 15 Worte in deiner Antwort."
    assistant_prompt_en = "I am a visually impaired person and need assistance. I am wearing glasses which capture the image that is being provided. Please answer concisely to directly address my question based on the visual and contextual input. Do not exceed 25 words in total. Do not mention my visual impairment or the camera's fisheye lens. This is my question:\n"
    assistant_prompt_de = "Ich bin eine sehbehinderte Person und benötige Hilfe. Ich trage eine Brille, die das bereitgestellte Bild einfängt. Bitte antworte präzise, um meine Frage anhand der visuellen und textuellen Eingaben direkt zu beantworten. Bitte nutze nicht mehr als 25 Worte für deine Antwort. Erwähne unter keinen Umständen meine Sehbehinderung. \n"
    captioning_prompt = captioning_prompt_en
    assistant_prompt = assistant_prompt_en

    #latest_instruction = ""
    latest_caption = "No caption available"
    caption_executor = ThreadPoolExecutor(max_workers=1)
    assistant_executor = ThreadPoolExecutor(max_workers=1)

    caption_future = None

   
    
    def process_camera_image(camera_id, observer, images_dict):
        """Extracts and processes a camera image from the observer and stores it."""
        if camera_id in observer.images:
            image = np.rot90(observer.images[camera_id], -1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_dict[camera_id] = image
            del observer.images[camera_id]

    
    
  


    last_caption_time = time.time()  # Track the last time a caption was updated

    images = {}
    audio = []

    # Start the program loop
    try:
        while not quit_keypress():
            
            # process 3 camera images
            process_camera_image(aria.CameraId.Rgb, observer, images)
            process_camera_image(aria.CameraId.Slam1, observer, images)
            process_camera_image(aria.CameraId.Slam2, observer, images)

            if observer.audio:
                        
                new_audio = observer.audio
                mono_audio = new_audio[::channels]
                audio_queue.append(mono_audio)

            

            
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
            if response_processor.mode == "assisting" and not response_processor.assistant_processing:
                    
                assistant_queue.put((
                frozen_image,
                response_processor,
                assistant_executor,
                recognizer_manager,
                observer,
                args,
                samplerate,
                channels
                ))
                print("✅ Assistant query enqueued")
                response_processor.assistant_processing = True

            #continue

            elif response_processor.mode == "watching":
                pass
            

                # Captioning mode
            elif response_processor.mode == "captioning" and not response_processor.assistant_processing:
                    print("KWS JUMPS HERE FIRST")
                    print("KWS JUMPS HERE FIRST")
                    print("KWS JUMPS HERE FIRST")
                    
                    #response_processor.model.conversation = []  # New chat
                    #model.conversation = []  # Delete chat history
                    caption_lock = False
                    
                    
                    caption_future = caption_executor.submit(response_processor.process_frame, frame=latest_frame, language=response_processor.language, prompt=response_processor.captioning_prompt(response_processor.language)) #processor should know if captioning_prompt or assistant_prompt is used
                    print(
                        f"Caption done in {(current_time - last_caption_time):.3f} seconds."
                    )
                    last_caption_time = current_time


                


                # Vision assistant mode
                
                    #vision assistant mode ends here


    
            
            frame_with_caption = (add_caption_to_frame(latest_frame)
            if response_processor.mode == "captioning"
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

            

            # Activate caption mode
            elif response_processor.mode == "captioning":
                listening = False
                #empty_and_lock_queue()
                #stop_audio()
                model.conversation = []  # New chat

                last_caption_time = time.time()
                latest_caption = "No caption available"
               # print(f"Captioning mode active. Interval: {caption_interval}s.")

            # Activate assisting mode
            elif key == ord("v"): # if mode == "assisting"
                #response_processor.mode == "captioning"
                listening = False
                #empty_and_lock_queue()
                #stop_audio()
                model.conversation = []  # New chat
                model.conversation.append(
                    {"role": "user", "content": assistant_prompt}
                )  # Define role

                #latest_caption = "Press 'o' and ask a question. Press 'p' to stop."
                latest_caption = "Speak keyword 'computer' to activate listening mode."

                print("Assisting mode is active")

            # Toggle audio on/off
            elif key == ord("a"):
                audio_enabled = not audio_enabled
                #stop_audio()
                print(f"Audio {'enabled' if audio_enabled else 'disabled'}.")

            # Switch camera
            elif key == ord("1"):
                camera_index = (
                    camera_index + 1 if camera_index < len(cameras) - 1 else 0
                )
                print(f"Switching to camera: {cameras[camera_index]}")

         
            


            # Print help
            elif key == ord("h"):
                display_help()

    except Exception as e:
        print("Oops! Something went wrong. Shutting down the stream...")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

        assistant_queue.put(None)
        assistant_worker_thread.join()

        caption_executor.shutdown()
        assistant_executor.shutdown()
        # Stop the keyword listener thread
        if kws_thread.is_alive():
            print("Stopping keyword listener thread...")
            response_processor.disable_keyword_listening()
            kws_thread.join(timeout=1)
            if kws_thread.is_alive():
                print("Keyword listener thread did not stop in time, forcefully terminating.")
                kws_thread.join()

        # Stop streaming and disconnect the glasses
        print("Stop listening to image data")
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)

        # Clean up the model after use
        del model  

        # Stop the worker thread
        add_caption_to_queue(None)

        tts_queue.put(None)
        tts_proc.join()

        del tts_queue
        print("Stream stopped and device disconnected. Goodbye!")


if __name__ == "__main__":
    main()


