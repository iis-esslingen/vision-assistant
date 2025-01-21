# vision-assistant
Get the world around you described while wearing the Aria glasses.

# Requirements
Python >= 3.11

# Get started
After downloading this project, run the setup.sh or setup.bat (depending on your system) script once.

# Start the tool
Make sure your venv is activated. Run `which python` on Linux/Mac or `where python` on Windows and check the received path. If it points to a file in the venv folder of this project, you should be fine.

Now run `python -m vision_assistant` to start the tool. Make sure your Aria glasses are connected via USB.

If you want to use a Wi-Fi connection, run `python -m vision_assistant --interface wifi --device-ip 192.168.0.1`. Replace the placeholder adress with the exact IP adress that you will find in your Aria App. [Read more](<https://facebookresearch.github.io/projectaria_tools/docs/ARK/mobile_companion_app>). Make sure that both your Aria glasses as well as the host computer are connected to the same Wi-Fi access point.

# Usage
While the camera window is in focus, press 'h' to print help.
There you will see the options of the tool.