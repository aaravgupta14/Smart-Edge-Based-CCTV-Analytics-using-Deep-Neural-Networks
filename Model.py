import tensorflow as tf
import numpy as np
import os
import random

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Lambda
)

from sklearn.model_selection import train_test_split

dataset_path = r"C:\Users\Aarav Gupta\OneDrive\Desktop\DATASET\train"

IMG_SIZE = (128, 128)

class_images = {}

for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if os.path.isdir(person_path):

        images = []

        for img_name in os.listdir(person_path):

            img_path = os.path.join(person_path, img_name)

            try:
                img = tf.keras.preprocessing.image.load_img(
                    img_path,
                    target_size=IMG_SIZE
                )

                img = tf.keras.preprocessing.image.img_to_array(img)

                images.append(img)

            except:
                pass

        class_images[person] = images

pairs = []
labels = []

persons = list(class_images.keys())

for person in persons:

    imgs = class_images[person]

    for i in range(len(imgs)):

        for j in range(i + 1, len(imgs)):

            pairs.append([imgs[i], imgs[j]])
            labels.append(1)

for person1 in persons:

    for person2 in persons:

        if person1 == person2:
            continue

        imgs1 = class_images[person1]
        imgs2 = class_images[person2]

        num_pairs = len(imgs1) * 5

        for _ in range(num_pairs):

            img1 = random.choice(imgs1)
            img2 = random.choice(imgs2)

            pairs.append([img1, img2])
            labels.append(0)

pairs = np.array(pairs, dtype=np.float32) / 255.0
labels = np.array(labels)

X1 = pairs[:, 0]
X2 = pairs[:, 1]

X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
    X1,
    X2,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

embedding_network = Sequential([

    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(128,128,3)
    ),

    MaxPooling2D(),

    Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(),

    Conv2D(
        128,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(),

    Flatten(),

    Dense(
        256,
        activation="relu"
    ),

    Dense(
        128,
        activation="relu",
        name="embedding_layer"
    )
])

inputA = Input(shape=(128,128,3))
inputB = Input(shape=(128,128,3))

embeddingA = embedding_network(inputA)
embeddingB = embedding_network(inputB)

distance = Lambda(
    lambda x: tf.abs(x[0] - x[1])
)([embeddingA, embeddingB])

output = Dense(
    1,
    activation="sigmoid"
)(distance)

siamese_model = Model(
    inputs=[inputA, inputB],
    outputs=output
)

siamese_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = siamese_model.fit(
    [X1_train, X2_train],
    y_train,
    validation_data=(
        [X1_val, X2_val],
        y_val
    ),
    epochs=30,
    batch_size=16,
    callbacks=[early_stopping]
)

loss, accuracy = siamese_model.evaluate(
    [X1_val, X2_val],
    y_val
)

print(f"Validation Accuracy: {accuracy:.4f}")

siamese_model.save("VisionGuard_Siamese_final.keras")

embedding_network.save("VisionGuard_Embedding_final.keras")

print("Models Saved Successfully")