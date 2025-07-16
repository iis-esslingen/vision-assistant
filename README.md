# vision-assistant

Get the world around you described while wearing the Aria glasses. <br>
Features the following modes: <br> - Assistant Mode - Ask a question while looking at a point of Interest to get contextual information <br> - Captioning Mode - Continuously receive descriptive captions of your environment <br> - Guiding Mode - Continuously receive navigation-relevant information based on your environement

  
  

# Requirements

Python == 3.11
Llava
  
  

# Get started

### 🔧 Setup Instructions

After downloading or cloning this project, you need to run the setup script  **once**  to install dependencies and configure the environment.

#### 🪟 For  **Windows**  users:
Open  **Command Prompt**  or  **PowerShell**, navigate to the project directory, and run:
`setup.bat` 
#### 🍎 For  **macOS/Linux**  users:
Open your  **terminal**, navigate to the project directory, and run:
`bash setup.sh`

  
  

# Start the tool

Make sure your virtual environment is activated before running the tool.

### Activate Virtual Environment

#### 🍎 On  **macOS/Linux**:

```bash
source bin/venv/activate
```

#### 🪟 On  **Windows**:

```bash
.\bin\venv\Scripts\activate
```

You can verify that you're using the virtual environment by running:

-   **Linux/macOS**:
    
    ```bash
    which python
    ```
    
-   **Windows**:
    
    ```cmd
    where python
    ```
    
If the path shown points to a file inside the  `venv`  folder of this project, you're good to go.

----------

##  Running the Application

Make sure your  **Aria glasses are connected via USB**.

> Only the  **USB Aria Interface**  is supported in this version.

Run the following command to start the tool:

```bash
python -m vision_assistant --interface usb
```

----------

## 🗣️ Usage

Once running, speak clearly to interact with the  **Keyword Spotting System (KWS)**.

### 📢 Available Keywords:

-   **Assistant**  – Ask a question regarding the current Aria frame.
    
-   **Caption**  – Receive a descriptive caption of the scene.
    
-   **Guiding**  – Get guidance-related information.
    
-   **Language**  – Toggle language from  **English**  to  **German** and vice-versa.
    
-   **Watching**  – Neutral state (no action).
    

----------

### 🧼 Exiting the Application

Press  **`Q`**  in the OpenCV window to  **exit the program cleanly**.

----------



  