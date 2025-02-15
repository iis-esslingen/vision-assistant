import os
import zipfile
import requests
from TTS.api import TTS


def download_stt_packs():

    urls = [
        "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
        "https://alphacephei.com/vosk/models/vosk-model-de-0.21.zip",
    ]
    for url in urls:
        package_name = url.split("/")[-1].removesuffix(".zip")
        if os.path.isdir(package_name):
            print(f"Language pack {package_name} already downloaded.")
            continue

        # Download the zip file
        print(f"Downloading language pack: {package_name}")
        response = requests.get(url)
        zip_file_path = os.path.join(package_name + ".zip")

        # Save the zip file
        with open(zip_file_path, "wb") as file:
            file.write(response.content)

        # Unzip the file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall()

        # Delete the zip file
        os.remove(zip_file_path)

    print("Speech-to-text language packs downloaded successfully.\n")


def download_tts_packs():
    tts = TTS()
    tts.download_model_by_name(model_name="tts_models/en/ljspeech/glow-tts")
    tts.download_model_by_name(model_name="tts_models/de/thorsten/vits")


def main():
    download_stt_packs()
    download_tts_packs()


if __name__ == "__main__":
    main()
