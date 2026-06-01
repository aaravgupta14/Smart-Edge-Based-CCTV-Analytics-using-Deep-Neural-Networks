import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(
    "VisionGuard_Embedding_final.keras"
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:

        face = frame[
            y:y+h,
            x:x+w
        ]

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Detected Face",
            face
        )

        face_input = cv2.resize(
            face,
            (128, 128)
        )

        face_input = face_input.astype(
            np.float32
        )

        face_input = face_input / 255.0

        face_input = np.expand_dims(
            face_input,
            axis=0
        )

        embedding = model.predict(
            face_input,
            verbose=0
        )

        embedding = embedding[0]

        norm = np.linalg.norm(
            embedding
        )

        if norm != 0:
            embedding = embedding / norm

        print(
            "Embedding Shape:",
            embedding.shape
        )

        print(
            "First 5 Values:",
            embedding[:5]
        )

    cv2.imshow(
        "VisionGuard Face Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()