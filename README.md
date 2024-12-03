# Eye Blink Detection

Welcome to **Eye Blink Detection** – a dynamic program that detects eye blinks and head orientation using **OpenCV**, **MediaPipe**, and custom threshold adjustments! Track your blinks, analyze head movements, and interactively tweak settings for a personalized experience.

---

## 🌟 Features

- **Blink Detection**: Accurately track blinks for each eye and detect whether one or both eyes are closed.
- **Head Pose Estimation**: Calculate pitch, yaw, and roll using facial landmarks for head orientation tracking.
- **Dynamic Sensitivity**: Adjust blink thresholds based on head pose to ensure reliable results.
- **Interactive Controls**: Modify the optimal pitch in real time with keyboard input.

---

## 🖥️ Requirements

- **Python 3.9**
- **Webcam** (for real-time video input)

---

## 🚀 Installation

To get started with the **Eye Blink Detection**, follow these steps:

### 1. Clone the Repository

Download the program from GitHub:

```bash
git clone https://github.com/Dorus-rutten/blinking-mediapipe
cd blinking-mediapipe
```
2. Set Up a Python Environment
We recommend using Conda or venv to create a dedicated Python environment.

Conda:
```bash
conda create -n blink-detection-env python=3.9
```
```bash
conda activate blink-detection-env
```
venv:
For macOS/Linux:

```bash
python3.9 -m venv blink-detection-env
source blink-detection-env/bin/activate
```
For Windows:

```bash
python3.9 -m venv blink-detection-env
blink-detection-env\Scripts\activate
```
3. Install Dependencies
Install the required libraries using the setup.py file:

```bash
python setup.py install
```
Alternatively, you can use pip:

```bash
pip install .
```
## 🎮 Running the Program
Once you have everything set up, launch the program with:

```bash
python blink_detection.py
```
## 🎛️ Controls
- Start/Stop Detection: The program starts detecting blinks as soon as the camera feed is active.
- Adjust Pitch Sensitivity: Press 'C' to set a new optimal pitch value during runtime.
- Quit Program: Press 'Q' to exit the program.
## 💡 How It Works
Landmark Detection: The program uses MediaPipe's Face Mesh to detect facial landmarks, including the eyes and key points for head pose estimation.
Eye Aspect Ratio (EAR): Blink detection is based on calculating the EAR for each eye.
Head Pose Analysis: Adjusts blink sensitivity dynamically based on the head's pitch angle to account for changes in posture.
## 💬 Contributions
Contributions are always welcome! If you have ideas for enhancements or spot any bugs, feel free to open an issue or submit a pull request.

Enjoy the game and happy blinking! ✨👀
