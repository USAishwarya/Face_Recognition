import cv2
import os

name = input("Enter name: ")
os.makedirs(f'dataset/{name}', exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while count < 2000:
    ret, frame = cap.read()
    if not ret:
        break

    face = cv2.resize(frame, (224, 224))  # Resize to CNN input size
    cv2.imwrite(f'dataset/{name}/{count}.jpg', face)
    count += 1

    cv2.imshow("Collecting Images", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


