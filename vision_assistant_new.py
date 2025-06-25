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
cv2.getBuildInformation()
import soundfile as sf
from TTS.api import TTS
import multiprocessing
import gc


from tts_process import tts_worker

import ollama_ifc


import aria.sdk as aria
from projectaria_tools.core.sensor_data import (
    ImageDataRecord,
    AudioData,
    AudioDataRecord,
)





class StreamingClientObserver:
    def __init__(self):
        print("Init StreamingClientObserver!!")
        self.images = {}
        self.audio = deque(maxlen=500000)
        self.audio_timestamps_ns = [] #makes sense to also use a deque here

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.images[record.camera_id] = image

    def on_audio_received(
        self,
        audio_data: AudioData,
        record: AudioDataRecord,
    ):
        self.audio.extend(audio_data.data)
        #self.audio_timestamps_ns += record.capture_timestamps_ns



def init_aria(args):
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

#plays a beep sound
def play_keyword_sound(frequency=600, duration=0.3, samplerate=48000, volume=0.3): #TODO: store the tone np.array instead of calculating it each time
    p = pyaudio.PyAudio()

    t = np.linspace(0, duration, int(samplerate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)

    fade_length = int(0.02 * samplerate)  
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

def keyword_listener(event_queue, keyword_listening_flag, global_running_flag, cooldown_seconds=3):
    """
    Listens for keywords and pushes Events onto event_queue.
    - event_queue: queue.Queue[Event]
    - audio_queue: deque of raw audio chunks from ARIA
    - keyword_listening_flag: multiprocessing.Value('b', True/False)
    """
    keywords = ['computer', 'caption', 'guiding', 'watching', 'language']
    recognizer = create_kws_model()
    last_switch = 0
    keyword_audio = []

    print("[KWS] Starting keyword listener…")
    while global_running_flag.value == True:
        # honor the shared flag to pause KWS when, e.g., recording for assistant
        if not keyword_listening_flag.value:
            #print("[KWS] SLEEPING DUE TO DISABLED keyword_listening_flag.value")
            time.sleep(0.1)
            continue
        
        

        try:

            if audio_queue:
                buffer = audio_queue.popleft()
            else:
            # sleep longer so we give the collector time to refill
                time.sleep(0.1)
                continue
    

            buffer = audio_queue.popleft()
            keyword_audio.extend(buffer)
        except IndexError:
            print("KWS] EMPTY AUDIO QUEUE")
            time.sleep(0.05)
            continue

        if len(keyword_audio) > 24000:
            keyword_audio = keyword_audio[-24000:]

        # 3) Normalize and run KWS
        try:
            audio_buffer = normalize_audio_buffer(keyword_audio)
        except Exception:
            print("[KWS] 🔴 Error normalizing audio buffer")
            traceback.print_exc()
            continue  # skip this chunk
            

        # feed into recognizer
        if not recognizer.AcceptWaveform(audio_buffer.tobytes()):
            #print("Sleeping due to recognizer not AcceptWaveForm")
            time.sleep(0.02)
            continue

        # got a result
        result = json.loads(recognizer.Result())
        text = result.get("text", "").lower()
        print("KWS RECOGNITION RESULTS: ", text)
        matched = [kw for kw in keywords if kw in text]
        if not matched:
            continue

        now = time.time()
        if now - last_switch < cooldown_seconds:
            continue
        last_switch = now

        # fire exactly one event per iteration
        kw = matched[0]
        

        if kw == "computer":
            print("[KWS] → COMPUTER")
            print("[KWS] → COMPUTER")
            print("[KWS] → COMPUTER")
            play_keyword_sound()

            print("[KWS] about to queue event KW_COMPUTER…")
            try:
                event_queue.put(Event(EventType.KW_COMPUTER))
                print(f"[KWS] queued event, queue size now {event_queue.qsize()}")
            except Exception as e:
                print(f"[KWS] ⚠️ event_queue.put() threw: {e}")

        elif kw == "caption":
            keyword_audio = []
            print("[KWS] → CAPTION")
            event_queue.put(Event(EventType.KW_CAPTION))

        elif kw == "guiding":
            keyword_audio = []
            print("[KWS] → GUIDANCE")
            event_queue.put(Event(EventType.KW_GUIDANCE))

        elif kw == "watching":
            keyword_audio = []
            print("[KWS] → WATCHING")
            event_queue.put(Event(EventType.WATCHING))

        elif kw == "language":
            keyword_audio = []
            print("[KWS] → LANGUAGE_SWITCH")
            # payload can be omitted; FSM will flip context.language
            event_queue.put(Event(EventType.LANGUAGE_SWITCH))

        # drop any partial results before continuing
        recognizer.Reset()

        # tiny sleep to avoid tight loop
        time.sleep(0.1)



def start_tts_process(global_running_flag):
    queue = multiprocessing.Queue()
    keyword_listening_flag = multiprocessing.Value('b', True)
    tts_enabled_flag = multiprocessing.Value('b', True)   
    process = multiprocessing.Process(target=tts_worker, args=(queue, keyword_listening_flag, tts_enabled_flag, global_running_flag))
    process.start()
    return queue, process, keyword_listening_flag, tts_enabled_flag

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


class TTSService:
    def __init__(self, tts_queue, tts_enabled_flag):
        print("Init TTSService!!")
        print("Init TTSService!!")
        self._queue = tts_queue
        self._enabled = tts_enabled_flag

    def enqueue_speech(self, text: str, lang: str):
        """Enqueue text if TTS isn’t muted."""
        if not self._enabled.value:
            return
        print(f"[TTSService] → {text!r} ({lang})")
        self._queue.put((text, lang))



from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Dict

from abc import ABC, abstractmethod





class RepeatedTimer:
    def __init__(self, interval: float, callback, *args, **kwargs):
        self.interval = interval
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
        self._timer = None
        self._running = False
        self.start()

    def _run(self):
        self._running = False
        self.start()
        self.callback(*self.args, **self.kwargs)

    def start(self):
        if not self._running:
            self._timer = threading.Timer(self.interval, self._run)
            self._timer.daemon = True
            self._timer.start()
            self._running = True

    def cancel(self):
        if self._timer:
            self._timer.cancel()
        self._running = False


class ModeHandler(ABC):
    @abstractmethod
    def on_enter(self):
        """Called once when the FSM transitions *into* this mode."""
        raise NotImplementedError

    @abstractmethod
    def on_exit(self):
        """Called once when the FSM transitions *out of* this mode."""
        raise NotImplementedError

    @abstractmethod
    def on_frame(self, frame):
        """
        Called on each new camera frame *if* this mode cares about frames.
        frame is your raw image data (e.g. numpy array).
        """
        raise NotImplementedError

    @abstractmethod
    def on_tts_done(self):
        """
        Called when the TTSService emits an ASSISTANT_DONE event,
        so you can clean up or transition back to “watching.”
        """
        raise NotImplementedError
        
    def on_language_switch(self):
        """Called whenever the user says 'language' in *any* mode."""
        pass
    

class VLMService:
    def __init__(self, model):
        print("Init VLMSeervice!!")
        print("Init VLMSeervice!!")
        self.model = model
        self.captioning_prompt = {
        "en": "Describe this image in a short single sentence. Please do not exceed 15 words in total.",

        "de": "Beschreibe dieses Bild in einem einzigen kurzen Satz. Verwende auf keinen Fall mehr als insgesamt 15 Worte in deiner Antwort."
        }
        self.assistant_prompt = {
        "en": "I am a visually impaired person and need assistance. I am wearing glasses which capture the image that is being provided. Please answer concisely to directly address my question based on the visual and contextual input. Do not exceed 25 words in total. Do not mention my visual impairment or the camera's fisheye lens. My question is:\n",
        "de": "Ich bin eine sehbehinderte Person und benötige Hilfe. Ich trage eine Brille, die das bereitgestellte Bild einfängt. Bitte antworte präzise, um meine Frage anhand der visuellen und textuellen Eingaben direkt zu beantworten. Bitte nutze nicht mehr als 25 Worte für deine Antwort. Erwähne unter keinen Umständen meine Sehbehinderung. Meine Frage lautet:\n"
    }
        self.guiding_prompt = {
            "en": "IMPORTANT: Your response must be no more than 25 words. Do not exceed this limit. I am a visually impaired person and need assistance navigating my environment. I am wearing glasses that capture this image from my perspective. Please provide detailed spatial guidance including: \n - distances to objects,\n - potential obstacles or hazards,\n - directional instructions (left/right/forward),\n - and step-by-step navigation advice.\n Be specific about what I should do next. Do not mention my visual impairment or camera details",
            #rewrite the prompts as you are [ROLE] ...
            "de": "WICHTIG: Deine Antwort darf maximal 25 Wörter haben. Ich benötige Hilfe bei der Navigation. Ich trage eine Brille mit Kamera. Gib mir räumliche Orientierung: Entfernungen, Hindernisse, Richtungsangaben (links/rechts/vorwärts) und konkrete nächste Schritte. Erwähne nicht meine Sehbehinderung. Meine Frage:\n"
        }

    def _get_prefix(self, mode: str, lang: str) -> str:
        d = {
            "captioning": self.captioning_prompt,
            "assisting":  self.assistant_prompt,
            "guiding":    self.guiding_prompt
        }[mode]
        return d.get(lang, d["en"])

    def ask(self, frame: np.ndarray, text_prompt: str, mode: str, lang: str) -> str:
        """
        mode in {"captioning","assisting","guiding"}.
        text_prompt is extra text (e.g. the transcript in 'assisting').
        """
        prefix = self._get_prefix(mode, lang)
        full_prompt = prefix + (text_prompt or "")
        print("[VLM SERVICE] - passing full prompt into vlm: ", full_prompt)
        # convert OpenCV frame to PIL if your model needs
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


        # ─── DEBUG: dump PIL to disk ───
        os.makedirs("debug_pil", exist_ok=True)
        debug_path = f"debug_pil/frame_for_vlm_{int(time.time()*1000)}.jpg"
        pil.save(debug_path, format="JPEG")
        print(f"[DEBUG] Saved PIL image to {debug_path}")
        
        return self.model.ask(pil, prompt=full_prompt)



class WatchingHandler(ModeHandler):
    def __init__(self, lang):
        print("Init WatchingHandler!!")
        print("Init WatchingHandler!!")
        self.lang_state = lang

    
    def on_language_switch(self):
        old = self.lang_state.lang
        self.lang_state.toggle()
        new = self.lang_state.lang
        print(f"[WatchingHandler] 🌐 Language: {old} → {new}")

    def on_enter(self):
        print("▶ Now watching (idle).")

    def on_exit(self):
        print("◀ Leaving watch mode.")

    def on_frame(self, frame):
        pass

    def on_tts_done(self):
        pass

class CaptioningHandler(ModeHandler):
    def __init__(self, vlm_service: VLMService, tts_service: TTSService, annotation_queue, lang: str, fsm_queue):
        print("Init CaptioningHandler!!")
        print("Init CaptioningHandler!!")
        self.vlm = vlm_service
        self.tts = tts_service
        self.lang_state = lang
        self._timer = None
        self._latest_frame = None  # ← buffer here
        self.annotation_queue = annotation_queue
        self.fsm_queue = fsm_queue


        self._busy       = False
        self._stopped    = threading.Event()
        self._thread     = None
        self._interval   = 1

    def on_language_switch(self):
        old = self.lang_state.lang
        self.lang_state.toggle()
        new = self.lang_state.lang
        print(f"[CaptioningHandler] 🌐 Language: {old} → {new}")

    def on_enter(self):
        print("CAPTIONING MODE ENTERED")
        self._stopped.clear()
        self._busy = False
        # Kick off the first one immediately
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def on_exit(self):
        # Signal the background loop to stop and wait for it

        self._stopped.set()
        if self._thread:
            self._thread.join(timeout=1)

    def _loop(self):
        """
        Background loop running in CAPTIONING mode.
        Ensures at most one VLM call at a time, with interval delay.
        """
        while not self._stopped.is_set():
            if not self._busy:
                self._busy = True
                try:
                    # Grab the very latest frame from the FSM
                    frame = self.fsm_queue_latest_frame()
                    if frame is not None:
                        prompt = ""  # caption mode uses fixed prompt inside VLMService
                        caption = self.vlm.ask(frame, prompt, "captioning", self.lang_state.lang)
                        self.tts.enqueue_speech(caption, self.lang_state.lang)
                        self.annotation_queue.append(caption)
                except Exception as e:
                    print(f"[CaptioningHandler] ❌ Error during caption: {e}")
                finally:
                    self._busy = False

            # Wait for the configured interval, or until we’re told to stop
            self._stopped.wait(self._interval)
    
    def fsm_queue_latest_frame(self):
        """
        Helper: pull out the most recent FRAME_CAPTURED event from the FSM queue
        so we caption the freshest image.
        """
        latest = None
        try:
            while True:
                event = self.fsm_queue.get_nowait()
                if event.type == EventType.FRAME_CAPTURED:
                    latest = event.payload
                else:
                    # Put back any non-frame events for the FSM to handle
                    self.fsm_queue.put(event)
                    break
        except queue.Empty:
            pass
        return latest

    def on_frame(self, frame):
        # called by the FSM on every FRAME_CAPTURED event
        self._latest_frame = frame

    def on_tts_done(self):
        # not needed in captioning mode
        pass

    def _do_caption(self):
        start = time.perf_counter()
        frame = self._latest_frame
        if frame is None:
            return

        caption = self.vlm.ask(frame, "", "captioning", self.lang_state.lang)
        elapsed = time.perf_counter() - start
        print(f"[Caption] took {elapsed:.2f}s")

        self.tts.enqueue_speech(caption, self.lang_state.lang)
        self.annotation_queue.append(caption)

        



class GuidingHandler(ModeHandler):
    def __init__(self, vlm_service: VLMService, tts_service: TTSService, annotation_queue, lang: str):
        print("Init GuidingHandler!!")
        print("Init GuidingHandler!!")

        self.vlm = vlm_service
        self.tts = tts_service
        self.lang_state = lang
        self._timer = None
        self._latest_frame = None  # ← buffer here
        self.annotation_queue = annotation_queue

    
    def on_language_switch(self):
        old = self.lang_state.lang
        self.lang_state.toggle()
        new = self.lang_state.lang
        print(f"[GuidingHandler] 🌐 Language: {old} → {new}")
    
    def on_enter(self):
        # every 3s grab the latest frame
        self._timer = RepeatedTimer(3.0, self._do_caption)

    def on_exit(self):
        if self._timer:
            self._timer.cancel()

    def on_tts_done(self):
        # guidance mode doesn’t use ASSISTANT_DONE events
        pass

    def on_frame(self, frame):
    # called by the FSM on every FRAME_CAPTURED event
        self._latest_frame = frame

    def _do_guidance(self):
        if self._latest_frame is None:
            return
        
        guidance_text = self.vlm.ask(self._latest_frame, text_prompt="", mode="guiding", lang=self.lang_state.lang)
        self.tts.enqueue_speech(guidance_text, self.lang_state.lang)
        self.annotation_queue.append(guidance_text)



    

class AssistantHandler(ModeHandler):
    def __init__(self, vlm_service, tts_service, recognizers, annotation_queue, observer, lang, keyword_listening_flag, event_queue, pause_frame_flag, pause_audio_flag):
        print("Init AssistantHandler!!")
        print("Init AssistantHandler!!")
        self.vlm = vlm_service
        self.tts = tts_service
        self.observer = observer
        self.lang_state = lang
        self.event_queue = event_queue
        self._latest_frame = None
        self.kws_flag = keyword_listening_flag
        self.annotation_queue = annotation_queue
        self.samplerate = 48000
        self.channels = 7
        self.recognizers = recognizers
        self.pause_frame_flag = pause_frame_flag
        self.pause_audio_flag = pause_audio_flag

        
    def on_language_switch(self):
        old = self.lang_state.lang
        self.lang_state.toggle()
        new = self.lang_state.lang
        print(f"[AssistantHandler] 🌐 Language: {old} → {new}")
    
    def on_enter(self):
        # run in a thread so you don't block the FSM loop
        threading.Thread(target=self._run_assistant_flow, daemon=True).start()

    def on_exit(self):
        pass

    def on_frame(self, frame):
        # buffer the latest frame for when we run the assistant flow
        self._latest_frame = frame

    def on_tts_done(self):
        return super().on_tts_done()

    def _run_assistant_flow(self):
        print("STARTING ASSISTANT FLOW")
        self.pause_audio_flag.value = True

        



        # pause KWS
        self.kws_flag.value = False
        # pause frame captured events
        self.pause_frame_flag.value = True
        # 1) record user speech & transcribe
        print("self.lang_state.lang is: ", self.lang_state.lang)
        recognizer = self.recognizers.get(self.lang_state.lang, self.recognizers["en"])
        transcript = self._record_and_transcribe(recognizer)
        
        self.pause_audio_flag.value = False
        self.kws_flag.value = True

        # 0) Wait up to 0.5s for at least one on_frame() callback
        timeout = time.time() + 1
        while self._latest_frame is None and time.time() < timeout:
            time.sleep(0.01)

        # 2) get the latest frame we saw
        if self._latest_frame is None:
            print("[AssistantHandler] ⚠️  No frame available for assistance.")
            self.event_queue.put(Event(EventType.ASSISTANT_DONE))
            self.pause_frame_flag.value = False
            return

        frame = self._latest_frame


        # ─── DEBUG: write out the frame ───
        debug_dir = "debug_frames"
        os.makedirs(debug_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        debug_path = os.path.join(debug_dir, f"assistant_frame_{ts}.jpg")
        # frame is your numpy array (whatever channel order it currently has)
        if cv2.imwrite(debug_path, frame):
            print(f"[DEBUG] Saved frame to {debug_path}")
        else:
            print(f"[DEBUG] ❌ Failed to save frame")
        

        # 3) call VLM
        print("[ASSISTANT] Asking VLM:", transcript)
        reply = self.vlm.ask(frame, transcript, "assisting", self.lang_state.lang)

        # 4) speak the reply
        self.tts.enqueue_speech(reply, self.lang_state.lang)
        self.annotation_queue.append(reply)

        # 5) notify FSM that we're done
        self.event_queue.put(Event(EventType.ASSISTANT_DONE))
        self.pause_frame_flag.value = False


    #detects silence
    def record_audio(self):
        self.observer.audio.clear()
        
        audio = []
        start_time = time.time()
        duration = 20
        silence_threshold = 5
        silence_timeout = 4
        chunk_check_interval = 0.1
        last_chunk_time = start_time
        last_audio_activity = start_time

        print("Audio recording - Please speak your query")
        print("Audio recording - Please speak your query")
        print("Audio recording - Please speak your query")
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
    

    def transcribe_audio(self, audio, recognizer):
        mono_audio = audio[::self.channels]
        max_sample_value = max(abs(min(mono_audio)), max(mono_audio))
        
        if max_sample_value > 0:
            norm_audio = np.array(mono_audio, dtype=np.float32) / max_sample_value
        else:
            print("⚠️ Warning: Audio normalization by zero. Using fallback normalization.")
            norm_audio = np.array(mono_audio, dtype=np.float32) / 1e-6
        
        audio_data = (norm_audio * 32767).astype(np.int16).tobytes()
        if not recognizer.AcceptWaveform(audio_data):
            print("Transcription error due to not accepted waveform.")
        
        result = json.loads(recognizer.Result()).get("text", "")
        print("Transcription complete. You said:\n", result)
        return result

    def _record_and_transcribe(self, recognizer):
        audio = self.record_audio()
        transcript = self.transcribe_audio(audio, recognizer)
        return transcript
    


class TerminateHandler(ModeHandler):
    def __init__(self, cleanup_funcs):
        print("Init TerminatedHandler!!")
        print("Init TerminatedHandler!!")
        """
        cleanup_funcs: list of 0-arg callables that each perform
                       one piece of teardown (e.g. stopping threads,
                       terminating processes, closing streams).
        """
        self._cleanup = cleanup_funcs

    def on_enter(self):
        print("[TerminateHandler] Cleaning up…")
        
        for fn in self._cleanup:
            try:
                fn()
            except Exception as e:
                print(f"⚠️  Error during cleanup: {e}")
                
        items_cleared = gc.collect()
        print("Cleanup complete!")
        print("Items_cleared:", items_cleared)

        

    def on_exit(self):
        print("Shutting down!")

    def on_frame(self, frame):
        pass

    def on_tts_done(self):
        pass






class Mode(Enum):
    WATCHING   = auto()
    GUIDING    = auto()
    ASSISTANT  = auto()
    CAPTIONING = auto()
    TERMINATE  = auto()

class EventType(Enum):
    KW_COMPUTER         = auto()
    KW_CAPTION          = auto()
    KW_GUIDANCE         = auto()
    WATCHING            = auto()
    ASSISTANT_DONE      = auto()
    LANGUAGE_SWITCH     = auto()
    FRAME_CAPTURED      = auto()
    QUIT                = auto()

@dataclass
class Event:
    type: EventType
    payload: Any = None








def load_vlm_model():
        model_name = "llava"
        print(f"Using model: {model_name}")
        return ollama_ifc.OllamaVLM(model_name)
        



class FSMEngine:
    def __init__(self,
                 initial: Mode,
                 handlers: Dict[Mode, ModeHandler], _event_queue: queue.Queue, global_running_flag):  #global_running_flag is of type "multiprocessing.Value"
        print("Init FSMEngine")
        print("Init FSMEngine")
        print("Init FSMEngine")
        self.current_mode = initial
        self.handlers = handlers
        self.transitions = self.define_transitions()
        self.event_queue = _event_queue
        #self.queue: "queue.Queue[Event]" = queue.Queue() 
        self._running = False
        self.global_running_flag = global_running_flag #idk if this does anything really
        

    def define_transitions(self) -> Dict[Mode, Dict[EventType, Mode]]:
        return {
        Mode.WATCHING: {
            EventType.KW_COMPUTER:       Mode.ASSISTANT,
            EventType.KW_CAPTION:        Mode.CAPTIONING,
            EventType.KW_GUIDANCE:       Mode.GUIDING,
            EventType.LANGUAGE_SWITCH: Mode.WATCHING,
        },
        Mode.CAPTIONING: {
            EventType.WATCHING:       Mode.WATCHING,
            EventType.KW_COMPUTER:       Mode.ASSISTANT,
            EventType.KW_GUIDANCE:       Mode.GUIDING,
        },
        Mode.GUIDING: {
            EventType.WATCHING:       Mode.WATCHING,
            EventType.KW_CAPTION:        Mode.CAPTIONING,
            EventType.KW_COMPUTER:       Mode.ASSISTANT,
        },
        Mode.ASSISTANT: {
            EventType.ASSISTANT_DONE: Mode.WATCHING,
        },
        Mode.TERMINATE: {}
        }


    def post(self, event: Event):
        self.event_queue.put(event)

    def run(self):
        self._running = True
        # fire initial on_enter
        self.handlers[self.current_mode].on_enter()

        while self._running and self.global_running_flag.value:
            event = self.event_queue.get()
            print(f"[FSM] ← Recieved event: {event.type}")
            # global quit
            if event.type is EventType.QUIT:
                #print("FSM ENGINE READS EVENTTYPE QUIT")
                #print("SWITCHING TO TERMINATE MODE")
                self._swap_to(Mode.TERMINATE)
                break
            
            if event.type is EventType.LANGUAGE_SWITCH:
                print(f"[FSM] handling LANGUAGE_SWITCH in {self.current_mode}")
                # call a new hook on the handler
                self.handlers[self.current_mode].on_language_switch()
                continue

            if event.type is EventType.FRAME_CAPTURED:
                self.handlers[self.current_mode].on_frame(event.payload)


            # mode-specific transitions
            nxt = self.transitions.get(self.current_mode, {}).get(event.type, self.current_mode)
            if nxt is not self.current_mode:
                self._swap_to(nxt)
            elif event.type is EventType.ASSISTANT_DONE:
                # let handlers see TTS‐done if they care
                self.handlers[self.current_mode].on_tts_done()

        # call final on_exit (e.g. TerminateHandler cleanup)
        self.handlers[self.current_mode].on_exit()
        self.global_running_flag.value = False

    def _swap_to(self, new_mode: Mode):
        print("swap to called with new mode:", new_mode)
        self.handlers[self.current_mode].on_exit()
        self.current_mode = new_mode
        self.handlers[new_mode].on_enter()
        if new_mode is Mode.TERMINATE:
            self._running = False

    def stop(self):
        self._running = False




import cv2
import textwrap
from collections import deque

class FrameOverlay:
    def __init__(self,
                 caption_queue: deque,
                 font=cv2.FONT_HERSHEY_SIMPLEX,
                 font_scale=0.7,
                 thickness=1,
                 padding=10):
        self.caption_queue = caption_queue
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness
        self.padding = padding

        self.show_help_flag = False
        self.help_text = [
            "'q': Quit",
            "'x': Camera only",
            "'c': Caption mode",
            "'v': Vision assistant",
            "'l': Toggle language",
            "'a': Toggle audio",
            "'1': Switch camera",
            "'h': Toggle help"
        ]

    def toggle_help(self):
        self.show_help_flag = not self.show_help_flag

    def replace_umlaute(self, text: str) -> str:
        return (text
            .replace("ä", "ae").replace("Ä", "Ae")
            .replace("ö", "oe").replace("Ö", "Oe")
            .replace("ü", "ue").replace("Ü", "Ue")
            .replace("ß", "ss").replace("\n", " ")
        )

    def wrap_text(self, text: str, img_width: int) -> list[str]:
        # rough split by pixel width using char count
        # fallback to Python's textwrap for simplicity
        max_chars = max(1, img_width // int(20 * self.font_scale))
        return textwrap.wrap(text, width=max_chars)

    def _draw_caption(self, frame):
        if not self.caption_queue:
            return frame
        try:
            caption = self.caption_queue[-1]
        except IndexError:
            return frame

        caption = self.replace_umlaute(caption)
        lines = self.wrap_text(caption, frame.shape[1])
        if not lines:
            return frame

        # Get the size of one line to compute heights
        (text_w, text_h), baseline = cv2.getTextSize(
        lines[0],
        self.font,
        self.font_scale,
        self.thickness
        )
        line_h = text_h + 5
        rect_h = line_h * len(lines) + 2 * self.padding
        h, w = frame.shape[:2]
        y0 = h - rect_h

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
        overlay,
        (0, y0),
        (w, h),
        (255, 255, 255),
        -1
        )
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Draw each line, centered
        y = y0 + self.padding + text_h
        for line in lines:
            (text_w, _), _ = cv2.getTextSize(
            line,
            self.font,
            self.font_scale,
            self.thickness
        )
            x = (w - text_w) // 2
            cv2.putText(
            frame,
            line,
            (x, y),
            self.font,
            self.font_scale,
            (0, 0, 0),
            self.thickness,
            cv2.LINE_AA
        )
            y += line_h

        return frame


    def _draw_help(self, frame):
        if not self.show_help_flag:
            return frame
        h, w = frame.shape[:2]
        # semi-transparent box in top-left
        overlay = frame.copy()
        box_w = w // 3
        box_h = len(self.help_text)*(20) + 2*self.padding
        cv2.rectangle(overlay, (0,0), (box_w, box_h),
                      (50,50,50), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        # draw text lines
        y = self.padding + 20
        for line in self.help_text:
            cv2.putText(frame, line, (self.padding, y),
                        self.font, 0.6, (255,255,255), 1, cv2.LINE_AA)
            y += 20
        return frame

    def render(self, frame):
        # draw caption at bottom
        frame = self._draw_caption(frame)
        # draw help overlay if toggled
        frame = self._draw_help(frame)
        return frame

def frame_producer(observer,
                   fsm_event_queue: queue.Queue,
                   display_event_queue: queue.Queue,
                   global_running_flag,
                   overlay=None,
                   pause_frame_flag=None,
                   fps: float = 20.0):
    """
    Continuously:
      • pull the latest RGB image from observer,
      • process it (rotate + BGR→RGB),
      • send it to the FSM as a FRAME_CAPTURED event (keeping only one in flight),
      • send the raw frame to the display queue,
      • sleep to maintain ~fps.
    """
    interval = 1.0 / fps
    last_time = time.time()

    while global_running_flag.value:
        if pause_frame_flag.value:
            time.sleep(0.05)
            continue

        # 1) throttle to target fps
        now = time.time()
        to_sleep = interval - (now - last_time)
        if to_sleep > 0:
            time.sleep(to_sleep)
        last_time = time.time()

        # 2) grab & preprocess the RGB image
        if aria.CameraId.Rgb not in observer.images:
            continue
        
        #print("len of observer.images before pop is", len(observer.images))
        img = observer.images.pop(aria.CameraId.Rgb)
        #print("len of observer.images after pop is", len(observer.images))
        frame = np.rot90(img, -1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3) enqueue to FSM queue, replacing only stale FRAME_CAPTURED
        evt = Event(EventType.FRAME_CAPTURED, frame)
        try:
            # fast‐path: if queue not full, just put
            fsm_event_queue.put_nowait(evt)
        except queue.Full:
            # queue is full—peek the old event
            old = fsm_event_queue.get_nowait()
            if old.type != EventType.FRAME_CAPTURED:
                # put back control events
                fsm_event_queue.put_nowait(old)
            # now enqueue the fresh frame
            fsm_event_queue.put_nowait(evt)

        # 4) publish raw frame for display (no cap)
        display_event_queue.put(frame)

        # 5) tiny yield to avoid spinning too tight
        time.sleep(0.001)


def display_loop(display_queue, fsm_queue, global_running_flag, overlay):
    window = "Aria View"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_count = 0
    t_start     = time.perf_counter()

    while global_running_flag.value:
        try:
            frame = display_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_count += 1
        now = time.perf_counter()
        if now - t_start >= 1.0:
            fps = frame_count / (now - t_start)
            print(f"[Perf] Display FPS: {fps:.1f}")
            frame_count = 0
            t_start     = now

        display_frame = overlay.render(frame)
        cv2.imshow(window, display_frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            fsm_queue.put(Event(EventType.QUIT))
            break

    cv2.destroyAllWindows()



class LanguageState:
    def __init__(self, initial: str = "en"):
        self.lang = initial

    def toggle(self):
        self.lang = "de" if self.lang == "en" else "en"
        print("Language switched to: ", self.lang)



def load_recognizers() -> dict[str, KaldiRecognizer]:
        model_paths = {
        "en": "./vosk-model-en-us-0.22",
        "de": "./vosk-model-de-0.21"
        }

        recognizers = {}
        for lang_code, path in model_paths.items():
            model = STTModel(path)
            recognizer = KaldiRecognizer(model, 48000) #48khZ
            recognizers[lang_code] = recognizer
        
       
        print("LOADING RECOGNIZERS FINISHED")
        return recognizers
    

#event_queue = queue.Queue()
fsm_queue = queue.Queue(maxsize=1)
display_queue = queue.Queue()

audio_queue = deque(maxlen=10)



def main():

    global_running_flag = multiprocessing.Value('b', True)
    tts_queue, tts_proc, kw_flag, tts_enabled_flag = start_tts_process(global_running_flag)
    tts = TTSService(tts_queue, tts_enabled_flag)

    recognizers = load_recognizers()

    pause_frame_flag   = multiprocessing.Value('b', False)
    pause_audio_flag    = multiprocessing.Value('b', False)


    


    annotation_queue = deque(maxlen=5)
    overlay = FrameOverlay(annotation_queue)

    

    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()
    
    streaming_client, device_client, observer, device, streaming_manager  = init_aria(args)

    def audio_collector(global_running_flag, pause_audio_flag):
        print("audio collector thread started!")
        channels = 7
        buffer = []
        MAX_CHUNKS_PER_CYCLE = 15000

        while global_running_flag.value:
            if pause_audio_flag.value:
                time.sleep(0.01)
                continue
            
            drained = 0
            buffer.clear()
            while drained < MAX_CHUNKS_PER_CYCLE and observer.audio:
                buffer.append(observer.audio.popleft())
                drained += 1

            if buffer:
                mono = buffer[::channels]
                audio_queue.append(mono)


            time.sleep(0.01) 
    
    collector_thread = threading.Thread(
    target=audio_collector,
    args=(global_running_flag,pause_audio_flag),
    daemon=True)

    collector_thread.start()
    




    model = load_vlm_model()
    vlm = VLMService(model)
    initial_language = LanguageState(initial="en")
    #initial_language = "en"

    cv2.namedWindow("Aria View", cv2.WINDOW_NORMAL)

    
    print("STARTING KWS THREAD")
    kws_thread = threading.Thread(
    target=keyword_listener,
    args=(fsm_queue, kw_flag, global_running_flag),
    daemon=True
    )
    frame_thread = threading.Thread(
        target=frame_producer,
        args=(observer, fsm_queue, display_queue, global_running_flag, overlay, pause_frame_flag),
        daemon=True
    )
    
    frame_thread.start()
    kws_thread.start()



    

   

    
    cleanup_funcs = [
        lambda: tts_queue.put(None),
        #lambda: cv2.destroyAllWindows(),
        lambda: streaming_client.unsubscribe(),
        lambda: streaming_manager.stop_streaming(),
        lambda: device_client.disconnect(device),
        lambda: tts_proc.terminate(),
        lambda: tts_proc.join(timeout=2),
        lambda: kws_thread.join(timeout=1),
        lambda: frame_thread.join(timeout=1)
        
    ]
    


    handlers = {
      Mode.WATCHING:   WatchingHandler(lang=initial_language),
      Mode.CAPTIONING: CaptioningHandler(vlm, tts, annotation_queue, lang=initial_language, fsm_queue=fsm_queue),
      Mode.GUIDING:    GuidingHandler(vlm, tts, annotation_queue, lang=initial_language),
      Mode.ASSISTANT:  AssistantHandler(vlm, tts, recognizers, annotation_queue,  observer, initial_language, kw_flag, event_queue=fsm_queue, pause_frame_flag=pause_frame_flag, pause_audio_flag=pause_audio_flag),
      Mode.TERMINATE:  TerminateHandler(cleanup_funcs),
    }

    # start event loop, kws listener, frame producer…
    fsm = FSMEngine(initial=Mode.WATCHING, handlers=handlers, _event_queue=fsm_queue, global_running_flag=global_running_flag)
    fsm_thread = threading.Thread(target=fsm.run)   # default daemon=False
    fsm_thread.start()

    display_loop(display_queue=display_queue, fsm_queue=fsm_queue, global_running_flag=global_running_flag, overlay=overlay)

    fsm_thread.join(timeout=1)
    print("fsm thread cleaned. final shutdown")





if __name__ == "__main__":
    #multiprocessing.set_start_method("spawn", force=True)
    multiprocessing.set_start_method("spawn", force=True)
    main()