import cv2
import requests
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Telegram bot token and chat ID
BOT_TOKEN = '7826384882:AAECzwYmkvO-jnc0rSBvlGNHA_1u0of5iUs'
CHAT_ID = '6082963128'

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

# ______________________________________make prediction on video frmaes _______________________________________________________
model = YOLO(r"C:\Users\user\Desktop\ci2\Projet\Online cheating in exam detection\code\results\best.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()  
    if not ret:
        break  

    results = model.predict(frame)
    cheating = False

    for result in results:
        boxes = result.boxes

        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            confidence = box.conf[0]  # Confidence score
            label = int(box.cls[0])  # Class index

            if confidence >= 0.6:
                # Draw the bounding box on the frame 
                if label not in [0, 1, 2, 4, 5, 6, 7, 9, 10, 12]:
                    if label == 3:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, 'Not cheating', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                else:
                    cheating = True
                    if label in [0, 1, 2, 4]:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, 'head moved', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                    elif label in [5, 6, 7, 9, 10]:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, 'eyes moved', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                    elif label == 12:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, 'talking', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    if cheating:
        send_telegram_message(frame)
                

    # Display the frame with bounding boxes
    cv2.imshow('Camera Feed', frame)

    # Press 'q' to quit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()