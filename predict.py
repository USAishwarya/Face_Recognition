import cv2
import numpy as np
from keras.models import load_model
import joblib

# Load trained model and label encoder
model = load_model('best_model.h5')
# le = joblib.load('label_encoder.pkl')



# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y + h, x:x + w]
        face = cv2.resize(roi_color, (224, 224))  # Resize to model input
        face = face / 255.0  # Normalize
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(face)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index]

        name = le.inverse_transform([class_index])[0]
        label = f"{name} ({confidence * 100:.1f}%)"

        # Display
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Recognition - Press Q to Exit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
