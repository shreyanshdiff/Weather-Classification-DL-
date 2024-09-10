import cv2
import mediapipe as mp
import pyautogui
import math
import time
import ctypes

class GestureControlApp:
    def __init__(self):
        self.video_source = cv2.VideoCapture(0)
        if not self.video_source.isOpened():
            print("Error: Could not open webcam.")
            return

        self.last_change_time = time.time()

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def set_volume(self, volume):
        # Set volume level using ctypes
        volume = max(0, min(volume, 100))  # Clamp volume between 0 and 100
        volume = int(volume * 0xFFFF / 100)  # Convert to range 0-0xFFFF
        ctypes.windll.winmm.waveOutSetVolume(0, volume)

    def adjust_brightness(self, increase=True):
        # Adjust brightness on Windows
        if increase:
            pyautogui.press("brightness up")
        else:
            pyautogui.press("brightness down")

    def run(self):
        while True:
            ret, frame = self.video_source.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            frame = cv2.flip(frame, 1)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = self.holistic.process(rgb_image)

            if output.right_hand_landmarks or output.left_hand_landmarks:
                if output.right_hand_landmarks:
                    hand_landmarks = output.right_hand_landmarks
                    control_type = "volume (Left Hand)"
                elif output.left_hand_landmarks:
                    hand_landmarks = output.left_hand_landmarks
                    control_type = "brightness (Right Hand)"

                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                h, w, _ = frame.shape
                thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_finger_tip_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

                cv2.line(frame, thumb_tip_coords, index_finger_tip_coords, (0, 255, 0), 2)

                distance = math.sqrt((index_finger_tip_coords[0] - thumb_tip_coords[0])**2 +
                                     (index_finger_tip_coords[1] - thumb_tip_coords[1])**2)

                distance_text = f"{control_type.capitalize()}: {distance:.2f} pixels"
                cv2.putText(frame, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                current_time = time.time()
                if distance > 100 and current_time - self.last_change_time > 1:
                    if control_type == "volume":
                        volume = min(100, int(distance / 2))  # Example volume control logic
                        print(f"Volume set to {volume} at distance: {distance}")
                        self.set_volume(volume)
                    elif control_type == "brightness":
                        print(f"Brightness up triggered at distance: {distance}")
                        self.adjust_brightness(increase=True)
                    self.last_change_time = current_time
                elif distance <= 100 and current_time - self.last_change_time > 1:
                    if control_type == "volume (Left Hand)":
                        print(f"Volume down triggered at distance: {distance}")
                        self.set_volume(0)
                    elif control_type == "brightness (Right Hand)":
                        print(f"Brightness down triggered at distance: {distance}")
                        self.adjust_brightness(increase=False)
                    self.last_change_time = current_time

            # Display the frame
            cv2.imshow('Gesture Control', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break

        self.video_source.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = GestureControlApp()
    app.run()
