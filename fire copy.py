import cv2
import mediapipe as mp
import numpy as np
import math
import random

# Initialize MediaPipe for Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class handHUD:
    def __init__(self, maxhands=2, detection_confidence=0.7, tracking_confidence=0.7):
        # Initialize MediaPipe Hands
        self.hands = mp_hands.Hands(max_num_hands=maxhands,
                                    min_detection_confidence=detection_confidence,
                                    min_tracking_confidence=tracking_confidence)
        self.mpdraw = mp.solutions.drawing_utils
        self.top_idx = [4, 8, 12, 16, 20]  # Thumb tip, index tip, etc.
        self.palm_idx = [0, 1, 5, 9, 13, 17]  # Palm landmarks for atom center
        self.particles = []  # List to store fire particles

    def findhands(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                # Draw cool effects and fire effects on detected hands
                self.draw_hud_elements(frame, handlms)
        return frame

    def draw_hud_elements(self, frame, hand_landmarks):
        """ Draw HUD elements like rotating rings, arcs, numbers, and the atom. """
        # Compute the palm center for the atom
        palm_center = self.compute_palm_center(frame, hand_landmarks)

        # Loop over finger tips and apply cool effects
        for idx in self.top_idx:
            x = int(hand_landmarks.landmark[idx].x * frame.shape[1])
            y = int(hand_landmarks.landmark[idx].y * frame.shape[0])

            # Cool effects for each fingertip
            self.draw_rotating_rings(frame, (x, y))
            self.draw_pulsating_circle(frame, (x, y))
            self.draw_rotating_arc(frame, (x, y), 50)
            self.draw_floating_numbers(frame, (x, y))

            # Draw fire effect at each fingertip
            self.draw_fire(frame, (x, y))

        # Add the atom at the palm center
        self.draw_atom(frame, palm_center)

    def compute_palm_center(self, frame, hand_landmarks):
        """ Compute the average position of the palm landmarks to find the center of the palm. """
        palm_x = 0
        palm_y = 0
        for idx in self.palm_idx:
            palm_x += hand_landmarks.landmark[idx].x * frame.shape[1]
            palm_y += hand_landmarks.landmark[idx].y * frame.shape[0]

        palm_x = int(palm_x / len(self.palm_idx))
        palm_y = int(palm_y / len(self.palm_idx))

        return (palm_x, palm_y)

    def draw_atom(self, frame, center):
        """ Draw an atom-like structure at the palm center. """
        nucleus_radius = 40  # Larger nucleus
        cv2.circle(frame, center, nucleus_radius, (0, 0, 255), -1)  # Red nucleus

        # Electron orbit parameters
        orbit_radii = [80, 120, 160]  # More orbits, increasing radius
        num_electrons_per_orbit = [3, 6, 9]  # More electrons on outer orbits
        electron_radius = 15  # Larger electron size
        time = cv2.getTickCount() / cv2.getTickFrequency()  # For smooth movement

        # Loop through each orbit
        for orbit_index, orbit_radius in enumerate(orbit_radii):
            num_electrons = num_electrons_per_orbit[orbit_index]  # Electrons on this orbit

            for i in range(num_electrons):
                # Angle for each electron (spread evenly in orbit)
                angle = 2 * math.pi * i / num_electrons + time

                # Calculate electron's position in orbit
                electron_x = int(center[0] + orbit_radius * math.cos(angle))
                electron_y = int(center[1] + orbit_radius * math.sin(angle))

                # Draw the electron
                cv2.circle(frame, (electron_x, electron_y), electron_radius, (255, 255, 255), -1)  # White electrons

            # Draw the orbit (only once for each orbit)
            cv2.circle(frame, center, orbit_radius, (255, 255, 255), 2)  # White orbit line

    def draw_rotating_rings(self, frame, center):
        """ Draw concentric rings rotating around the fingertip. """
        time = cv2.getTickCount() / cv2.getTickFrequency()  # For smooth movement
        num_rings = 3
        for i in range(num_rings):
            radius = int(40 + 10 * i)  # Different radii for each ring
            angle_offset = int((time * 50) % 360)
            cv2.ellipse(frame, center, (radius, radius), angle_offset, 0, 360, (255, 255, 255), 2)

    def draw_pulsating_circle(self, frame, center):
        """ Draw a pulsating circle that expands and contracts. """
        time = cv2.getTickCount() / cv2.getTickFrequency()
        pulse_radius = int(40 + 10 * math.sin(time * 3))  # Change radius over time
        cv2.circle(frame, center, pulse_radius, (0, 255, 255), 2)

    def draw_rotating_arc(self, frame, center, radius):
        """ Draw an arc that rotates around the fingertip. """
        time = cv2.getTickCount() / cv2.getTickFrequency()
        start_angle = int(time * 100) % 360  # Angle rotates over time
        end_angle = (start_angle + 60) % 360  # 60-degree arc
        cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, (255, 0, 0), 3)

    def draw_floating_numbers(self, frame, center):
        """ Draw floating numbers near the fingertips. """
        time = cv2.getTickCount() / cv2.getTickFrequency()
        number = int(time * 10) % 100  # Changing number
        cv2.putText(frame, f'{number}', (center[0] - 20, center[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_fire(self, frame, center):
        """ Simulate a fire effect at the given center (finger tip). """
        # Add new fire particles
        if random.random() < 0.1:  # Control the frequency of fire particles
            self.particles.append({
                'pos': np.array([float(center[0]), float(center[1])], dtype='float64'),  # Ensure float64 type for smooth movement
                'size': random.randint(5, 15),
                'color': (random.randint(200, 255), random.randint(50, 150), 0),
                'vel': np.array([random.uniform(-2, 2), random.uniform(-2, -4)], dtype='float64')  # Float64 for velocity
            })

        # Update and draw particles if any exist
        if self.particles:  # Only proceed if the particle list is not empty
            for particle in self.particles:
                # Move particle upwards
                particle['pos'] += particle['vel']
                particle['size'] -= 0.1  # Shrink particle over time

                # Ensure the particle size remains valid (size > 0)
                if particle['size'] > 0:
                    # Draw particle as a circle
                    cv2.circle(frame, tuple(particle['pos'].astype(int)), int(particle['size']), particle['color'], -1)

            # Remove dead particles (those with size <= 0)
            self.particles = [p for p in self.particles if p['size'] > 0]

# Initialize Video Capture (adjust the index for the correct camera)
cap = cv2.VideoCapture(1)  # Use index 0 for the built-in camera

# Initialize Hand HUD with MediaPipe Hand Tracking
hand_hud = handHUD()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process hand tracking and HUD elements
    frame = hand_hud.findhands(frame)

    # Display the frame with hand effects
    cv2.imshow('Hand HUD with Cool Effects', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

       

