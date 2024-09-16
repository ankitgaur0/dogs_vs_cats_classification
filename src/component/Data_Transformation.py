import os,sys
import pandas as pd
from src.Exception_Handler import Custom_Exception
from src.Logger import logging

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Data_Transformation:
    def __init__(self,train_df):
        self.X_train=train_df.drop("labels",axis=1)
        self.y_train=train_df["labels"]

    def get_data_augementation(self,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horiontal_flip=True,fill_mode="nearest"):
        data_aug_generate=ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horiontal_flip=horiontal_flip,
            fill_mode=fill_mode
        )
        return data_aug_generate
    
    def intiate_Transformation(self):
        try:
            augmentation=self.get_data_augementation()
            train_generator=augmentation.flow_from_dataframe(
                self.X_train,
                self.y_train,
                batch_size=32,
                target_size=(70,70),
                classes='binary'
            )


            return(
                train_generator
            )
        except Exception as e:
            raise Custom_Exception(e,sys)