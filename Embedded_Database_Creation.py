import tensorflow as tf
import numpy as np
import os
import pickle

model = tf.keras.models.load_model(
    "VisionGuard_Embedding_final.keras"
)

dataset_path = r"C:\Users\Aarav Gupta\OneDrive\Desktop\DATASET\train"

face_database = {}

for person in os.listdir(dataset_path):

    person_path = os.path.join(
        dataset_path,
        person
    )

    if not os.path.isdir(person_path):
        continue

    face_database[person] = []

    for image_name in os.listdir(person_path):

        if not image_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        img_path = os.path.join(
            person_path,
            image_name
        )

        try:

            img = tf.keras.preprocessing.image.load_img(
                img_path,
                target_size=(128, 128)
            )

            img = tf.keras.preprocessing.image.img_to_array(
                img
            )

            img = img / 255.0

            img = np.expand_dims(
                img,
                axis=0
            )

            embedding = model.predict(
                img,
                verbose=0
            )[0]

            embedding = (
                embedding /
                np.linalg.norm(embedding)
            )

            face_database[person].append({

                "image_path": img_path,

                "embedding": embedding

            })

        except Exception as e:

            print(f"Failed: {img_path}")

            print(e)

with open(
    "face_database.pkl",
    "wb"
) as f:

    pickle.dump(
        face_database,
        f
    )

adaptive_database = {}

with open(
    "adaptive_database.pkl",
    "wb"
) as f:

    pickle.dump(
        adaptive_database,
        f
    )

print("Database Created Successfully")

print(
    "Known Persons:",
    list(face_database.keys())
)