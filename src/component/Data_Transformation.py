import os,sys
import numpy as np
import pandas as pd
from src.Exception_Handler import Custom_Exception
from src.Logger import logging

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Data_Transformation:
    def __init__(self,train_df):
        self.X_train=train_df.drop("labels",axis=1)
        self.y_train=train_df["labels"]

    def get_data_augementation(self,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest"):
        data_aug_generate=ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            fill_mode=fill_mode
        )
        return data_aug_generate
    
    def intiate_transformation(self):
        try:
            #converting the dataframe to array
            # Convert X_train and y_train to numpy arrays if not already
            X_train = np.array(list(self.X_train))  # Assuming the images are in a column named "image_column"
        
            y_train = self.y_train.to_numpy()  # Ensure labels are also arrays
        
            # Check the shape to make sure it's correct
            print(f"X_train shape: {X_train.shape} \n y_train shape: {y_train.shape}")
            augmentation=self.get_data_augementation()
        
            train_generator=augmentation.flow(
                X_train,
                y_train,
                batch_size=32
            )


            return(
                train_generator
            )
        except Exception as e:
            raise Custom_Exception(e,sys)