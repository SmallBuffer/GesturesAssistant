import pandas as pd
import mediapipe as mp
import cv2
import pyautogui
import numpy as np
import pickle


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)

las_gesture = "none"
las_pos=(-1,-1,-1)
screen_width = 192*25
screen_height = 108*25
det=4

with open('hand-pose.pkl', 'rb') as f:
    model = pickle.load(f)
cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        image.flags.writeable = False


        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)

        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # # 1. Draw face landmarks
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        #                           mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        #                           )

        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )

        # # 3. Left Hand
        # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        #                           mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        #                           )

        # # 4. Pose Detections
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        #                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        #                           )
        # Export coordinates
        try:
            # Extract Right——Hand landmarks
            right_hand = results.right_hand_landmarks.landmark
            right_hand_row = list(
                np.array(
                    [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())


            # Make Detections
            X = pd.DataFrame([right_hand_row])
            right_hand_class = model.predict(X)[0]
            right_hand_prob = model.predict_proba(X)[0]
            # print(right_hand_class, right_hand_prob)



            # Grab ear coords
            coords = tuple(np.multiply(
                np.array(
                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                     results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                , [640, 480]).astype(int))


            # Get status box
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, right_hand_class.split(' ')[0]
                        , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(right_hand_prob[np.argmax(right_hand_prob)], 2))
                        , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#Mouse Action
            wrist_pos=((1-right_hand[0].x)*screen_width-108*8,(right_hand[0].y)*screen_height*1.5-192*9,(right_hand[0].z)*100000)
            # print(wrist_pos)

            if right_hand_class=="point-down"  :#Done
                pyautogui.mouseUp(button='left')
            elif right_hand_class=="point-up":
                pyautogui.mouseUp(button='left')

            elif right_hand_class=="open":#Done
                if las_gesture=="open" :
                    if wrist_pos[1]<800:
                        pyautogui.scroll(-400)
                    elif wrist_pos[1]>1200:
                        pyautogui.scroll(400)

            elif right_hand_class == "fist":#Done
                pyautogui.moveTo(wrist_pos[0], wrist_pos[1])


            elif right_hand_class == "zoom-in":#Done
                pyautogui.keyDown('ctrl')
                pyautogui.keyDown('+')
                pyautogui.keyUp('+')
                pyautogui.keyUp('ctrl')
            elif right_hand_class == "zoom-out":#Done
                pyautogui.keyDown('ctrl')
                pyautogui.keyDown('-')
                pyautogui.keyUp('-')
                pyautogui.keyUp('ctrl')

            las_pos=wrist_pos
            las_gesture=right_hand_class
        except:
            pass

        cv2.imshow('Raw Webcam Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()