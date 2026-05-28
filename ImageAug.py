import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datset_path = r'C:\Users\Aarav Gupta\OneDrive\Desktop\DATASET\train'
datagen=ImageDataGenerator(rotation_range=20, 
                           width_shift_range=0.2, 
                           height_shift_range=0.2, 
                           shear_range=0.2, zoom_range=0.2, 
                           horizontal_flip=True, 
                           fill_mode='nearest')
for folder in os.listdir(datset_path):
    folder_path = os.path.join(datset_path, folder)
    if os.path.isdir(folder_path):
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path)
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=folder_path, save_prefix='aug', save_format='jpeg'):
                i += 1
                if i > 5:  
                    break