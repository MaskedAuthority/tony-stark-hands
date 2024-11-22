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

2. Install the dependencies:
    ```bash
    pip install opencv-python mediapipe numpy
    ```

## Dependencies

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
