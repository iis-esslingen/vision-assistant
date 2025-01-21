import os
import zipfile
import requests


def download_language_packs():

    urls = [
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
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

    print("Language packs downloaded and extracted successfully.")


def main():
    download_language_packs()


if __name__ == "__main__":
    main()
