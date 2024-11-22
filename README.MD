# Hand HUD with Cool Effects

This project uses [MediaPipe](https://google.github.io/mediapipe/) to perform hand tracking and adds exciting Tony Stark-inspired HUD effects to your video feed. The system detects hand landmarks and overlays cool visual elements like rotating rings, pulsating circles, arcs, atoms, and a fire effect.

## Features

- **Real-Time Hand Tracking**: Utilizes MediaPipe's hand tracking for precise landmark detection.
- **HUD Effects**:
  - Rotating rings around fingertips.
  - Pulsating and rotating circles.
  - Floating numbers near fingertips.
  - Atom-like orbits with electrons at the palm center.
  - Fire effects at each fingertip.
- **Customizable Parameters**: Easily adjust detection/tracking confidence or the number of hands tracked.

## Demo

![HUD Demo](https://via.placeholder.com/800x400)  
_Sample visualization of the hand HUD effects._

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/hand-hud-cool-effects.git
    cd hand-hud-cool-effects
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have a webcam connected. Adjust the video capture index (`cv2.VideoCapture`) in the code if necessary.
2. Run the script:
    ```bash
    python main.py
    ```
3. Press `q` to exit the application.

## Dependencies

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

Dependencies can be installed via `requirements.txt`:
```bash
pip install -r requirements.txt
