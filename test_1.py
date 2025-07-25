import cv2
from keras.models import load_model
import numpy as np

model = load_model('best_model.h5')

cap = cv2.VideoCapture(0)
while True:
    ret , frame = cap.read()
    if ret is False:
        break
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    dict = {0: "Aishwarya",
            1: "Anoop",
            2: "Amma"}

    x,y = 100,150
    pred = np.argmax(model.predict(img))


    pred = np.argmax(model.predict(img))
    name = dict[pred]

    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) == ord("q"):
        break




cap.release()
cv2.destroyAllWindows()