# tts_process.py
import numpy as np
import pyaudio
from TTS.api import TTS
import sounddevice as sd
import time
from scipy.signal import resample

def speed_up_audio(input_audio, speed_factor):
    y = np.array(input_audio, dtype=np.float32)
    new_length = int(len(y) / speed_factor)
    y_resampled = resample(y, new_length)
    return y_resampled.astype(np.float32)

def tts_worker(queue, keyword_listening_flag, tts_enabled_flag):
    print("[TTS Worker] Process started")

    try:
        
        print("Loading English TTS model...")
        tts_engine_en = TTS(model_name="tts_models/en/ljspeech/fast_pitch", progress_bar=False, gpu=False)

        print("Loading German TTS model...")
        tts_engine_de = TTS(model_name="tts_models/de/css10/vits-neon", progress_bar=False, gpu=False)

        
        print("Running warm-up inference...")
        _ = tts_engine_en.tts("Warm-up on going. This speeds up the model", split_sentences=False)

    except Exception as e:
        print(f"❌ [TTS Worker] Failed to initialize TTS engines: {e}")
        import traceback
        traceback.print_exc()
        return

    p = pyaudio.PyAudio()

    while True:

        while not tts_enabled_flag.value:
            print("[TTS Worker] TTS is paused. Waiting...")
            time.sleep(0.1)
            
        try:
            item = queue.get()

            if item is None:
                print("[TTS Worker] Shutdown signal received.")
                break  # Shutdown signal

            text, lang = item
            print(f"🗣️ [TTS Worker] Speaking ({lang}): {text}")

            #urrent_session_id = tts_session_id.value  # Track session ID
            #print(f"🗣️ [TTS Worker] Speaking ({lang}): {text} | Session {current_session_id}")


            engine = tts_engine_en if lang == "en" else tts_engine_de

            audio = engine.tts(text, split_sentences=False)
            audio = np.array(audio, dtype=np.float32)

            #if current_session_id != tts_session_id.value:
            #    print(f"⚠️ [TTS Worker] Session mismatch detected (current {current_session_id} vs latest {tts_session_id.value}), skipping playback.")
            #    continue  # Skip playback and go to next item

            keyword_listening_flag.value = False
            print("Blocking keyword listening for TTS playback...")
            print("Blocking keyword listening for TTS playback...")
            print("Blocking keyword listening for TTS playback...")
            


            stream = p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=24000,
                            output=True)

            stream.write(audio.tobytes())
            stream.stop_stream()
            stream.close()

            time.sleep(0.8) 

            # Re-enable KWS
            keyword_listening_flag.value = True
            print("[TTS Worker] Keyword listening re-enabled after playback.")
            print("[TTS Worker] Keyword listening re-enabled after playback.")
            print("[TTS Worker] Keyword listening re-enabled after playback.")
            

        except Exception as e:
            print(f"❌ [TTS Worker] Error during synthesis/playback: {e}")
            import traceback
            traceback.print_exc()

    p.terminate()
    print("[TTS Worker] Terminated cleanly.")
