# import cv2
# import mediapipe as mp
# import pyautogui
# import math
# import time
# import streamlit as st

# st.title("Volume and Brightness Control Using Hand Gestures")

# # Initialize the webcam
# webcam = cv2.VideoCapture(0)

# if not webcam.isOpened():
#     st.error("Error: Could not open webcam.")
#     st.stop()

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_holistic = mp.solutions.holistic

# # Last time the volume and brightness were changed
# last_change_time = time.time()

# FRAME_WINDOW = st.image([])

# with mp_holistic.Holistic(
#     static_image_mode=False,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as holistic:
    
#     while webcam.isOpened():
#         ret, image = webcam.read()
#         image = cv2.flip(image, 1)
#         if not ret:
#             st.error("Error: Failed to capture image")
#             break
        
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         output = holistic.process(rgb_image)
        
#         if output.right_hand_landmarks or output.left_hand_landmarks:
#             if output.right_hand_landmarks:
#                 hand_landmarks = output.right_hand_landmarks
#                 control_type = "volume"
#             elif output.left_hand_landmarks:
#                 hand_landmarks = output.left_hand_landmarks
#                 control_type = "brightness"
                
#             mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
#             # Get the coordinates of the thumb tip and index finger tip
#             thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#             index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

#             # Convert normalized coordinates to pixel coordinates
#             h, w, _ = image.shape
#             thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
#             index_finger_tip_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

#             # Draw a line between the thumb tip and index finger tip
#             cv2.line(image, thumb_tip_coords, index_finger_tip_coords, (0, 255, 0), 2)

#             # Calculate the Euclidean distance
#             distance = math.sqrt((index_finger_tip_coords[0] - thumb_tip_coords[0])**2 + 
#                                  (index_finger_tip_coords[1] - thumb_tip_coords[1])**2)

#             # Convert distance to string
#             distance_text = f"{control_type.capitalize()}: {distance:.2f} pixels"

#             # Display the distance on the image
#             cv2.putText(image, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
#             # Control the volume or brightness with a delay
#             current_time = time.time()
#             if distance > 100 and current_time - last_change_time > 1:  # Adjust delay as needed
#                 if control_type == "volume":
#                     st.write(f"Volume up triggered at distance: {distance}")
#                     pyautogui.press("volume up")
#                 elif control_type == "brightness":
#                     st.write(f"Brightness up triggered at distance: {distance}")
#                     pyautogui.press("brightness up")
#                 last_change_time = current_time
#             elif distance <= 100 and current_time - last_change_time > 1:
#                 if control_type == "volume":
#                     st.write(f"Volume down triggered at distance: {distance}")
#                     pyautogui.press("volume down")
#                 elif control_type == "brightness":
#                     st.write(f"Brightness down triggered at distance: {distance}")
#                     pyautogui.press("brightness down")
#                 last_change_time = current_time

#         # Show the frame with the distance overlayed
#         FRAME_WINDOW.image(image[:, :, ::-1])
       
#         if cv2.waitKey(10) & 0xFF == 27:  # ESC key to break
#             break

# # Release the webcam
# webcam.release()
# cv2.destroyAllWindows()

API = "cc472ad20172f2d6f386fd698b73534c"
