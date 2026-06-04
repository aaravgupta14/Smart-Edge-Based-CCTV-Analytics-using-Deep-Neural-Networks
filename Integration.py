import cv2
import tensorflow as tf
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

model = tf.keras.models.load_model(
    "VisionGuard_Embedding_final.keras"
)

with open(
    "face_database.pkl",
    "rb"
) as f:
    database = pickle.load(f)

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

    if len(faces) > 0:
        largest_face = max(
        faces,
        key=lambda f: f[2] * f[3]
        )

        x, y, w, h = largest_face

        side_pad = 25
        top_pad = 20
        bottom_pad = 50

        x1 = max(0, x - side_pad)
        y1 = max(0, y - top_pad)

        x2 = min(frame.shape[1], x + w + side_pad)
        y2 = min(frame.shape[0], y + h + bottom_pad)

        face = frame[
            y1:y2,
            x1:x2
        ]

        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
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
        )[0]

        norm = np.linalg.norm(
            embedding
        )

        if norm != 0:

            embedding = (
                embedding /
                norm
            )

        best_person = "Unknown"
        best_score = -1

        all_scores = {}

        for person in database:

            scores = []

            for item in database[person]:

                stored_embedding = item[
                    "embedding"
                ]

                score = cosine_similarity(
                    embedding.reshape(
                        1,
                        -1
                    ),
                    stored_embedding.reshape(
                        1,
                        -1
                    )
                )[0][0]

                scores.append(
                    score
                )

            scores.sort(
                reverse=True
            )

            top_k = min(
                5,
                len(scores)
            )

            person_score = np.mean(
                scores[:top_k]
            )

            all_scores[person] = person_score

            if person_score > best_score:

                best_score = person_score
                best_person = person

        sorted_scores = sorted(
            all_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print(
            "\n========== TOP 4 =========="
        )

        for name, score in sorted_scores[:4]:

            print(
                name,
                round(
                    score,
                    4
                )
            )

        print(
            "Prediction:",
            best_person,
            round(
                best_score,
                4
            )
        )

        if best_score >= 0.95:

            label = (
                f"{best_person} "
                f"({best_score:.2f})"
            )

            color = (
                0,
                255,
                0
            )

        elif best_score >= 0.90:

            label = (
                f"Possible "
                f"{best_person} "
                f"({best_score:.2f})"
            )

            color = (
                0,
                255,
                255
            )

        else:

            label = (
                f"Unknown "
                f"({best_score:.2f})"
            )

            color = (
                0,
                0,
                255
            )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow(
    "VisionGuard",
    frame
    )
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
