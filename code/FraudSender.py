import cv2
import requests
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Telegram bot token and chat ID
BOT_TOKEN = os.environ.get('past here ure bot token')
CHAT_ID = os.environ.get('past here ure bot id')

# Function to send a message and image to Telegram
def send_telegram_message(image):
    # Send a message
    message = f"🚨 Fraud detected in the exam! 🚨"
    send_text_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"
    requests.get(send_text_url)

    # Send the image
    img_path = 'detected_fraud.jpg'
    cv2.imwrite(img_path, image)

    # Prepare to send the photo
    send_photo_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(img_path, 'rb') as img_file:
        requests.post(send_photo_url, data={'chat_id': CHAT_ID}, files={'photo': img_file})

# ______________________________________make prediction on video frmaes_______________________________________________________
model = YOLO(r"C:\Users\user\Desktop\ci2\Projet\Online cheating in exam detection\code\results\best.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

skip_frames = 0  # Counter to skip frames after detecting fraud

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Only run detection if skip_frames is 0
    cheating = False
    if skip_frames == 0:
        results = model.predict(frame)
        labelsC = {}
        labelsBbx = {}
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                confidence = box.conf[0]
                label = int(box.cls[0])
                if confidence >= 0.6:
                    labelsC[label] = confidence
                    labelsBbx[label] = [x1, y1, x2, y2]

        # Check for cheating based on detected labels
        for lab in [0, 1, 2, 4, 5, 6, 7, 9, 10, 12]:
            if lab in labelsC and labelsC[lab] > 0.6:
                cheating = True
                skip_frames = 3  # Set to skip the next 3 frames after detecting fraud
                break

        # Draw bounding boxes and labels if cheating is detected
        if cheating:
            for label in labelsC:
                if label in [0, 1, 2, 4] and label in labelsBbx:
                    cv2.rectangle(frame, (labelsBbx[label][0], labelsBbx[label][1]), (labelsBbx[label][2], labelsBbx[label][3]), (0, 0, 255), 2)
                    cv2.putText(frame, 'head moved', (labelsBbx[label][0], labelsBbx[label][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                elif label in [5, 6, 7, 9, 10] and label in labelsBbx:
                    cv2.rectangle(frame, (labelsBbx[label][0], labelsBbx[label][1]), (labelsBbx[label][2], labelsBbx[label][3]), (0, 0, 255), 2)
                    cv2.putText(frame, 'eyes moved', (labelsBbx[label][0], labelsBbx[label][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                elif label == 12 and label in labelsBbx:
                    cv2.rectangle(frame, (labelsBbx[label][0], labelsBbx[label][1]), (labelsBbx[label][2], labelsBbx[label][3]), (0, 0, 255), 2)
                    cv2.putText(frame, 'talking', (labelsBbx[label][0], labelsBbx[label][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        elif 3 in labelsBbx:
            cv2.rectangle(frame, (labelsBbx[3][0], labelsBbx[3][1]), (labelsBbx[3][2], labelsBbx[3][3]), (0, 255, 0), 2)
            cv2.putText(frame, 'Not cheating', (labelsBbx[3][0], labelsBbx[3][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    else:
        # Skip detection for this frame
        skip_frames -= 1

    if cheating:
        send_telegram_message(frame)
                

    # Display the frame with bounding boxes
    cv2.imshow('Camera Feed', frame)

    # Press 'q' to quit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()