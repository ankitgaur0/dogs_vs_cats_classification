import os,sys
from pathlib import Path
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.Logger import logging
from src.Exception_Handler import Custom_Exception
from src.Utils import resized_image
from src.Utils import save_obj
from dataclasses import dataclass
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

@dataclass
class Data_Ingestion_Config:
    images_data_path :str =os.path.join("artifacts","images_data.pickle")
    label_data_path :str =os.path.join("artifacts","labels_data.pickle")

class Data_Ingestion:
    def __init__(self,data_dir):
        self.Data_Ingestion_Config_obj=Data_Ingestion_Config()
        self.data_dir=data_dir



    def initiate_augmentation(self,rescale=1.0/255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest"):
        train_aug_data_gen=ImageDataGenerator(
            rescale=rescale,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            fill_mode=fill_mode
        )

        return train_aug_data_gen
    


    def initiate_data(self):
        try:
       
            images_path=self.data_dir

            aug_data=self.initiate_augmentation(rescale=1.0/255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")
            train_data=aug_data.flow_from_directory(
                images_path,
                target_size=(50,50),
                batch_size=32,
                class_mode='binary'
            )

            images,labels=next(train_data)
            logging.info(f"this is the shape of the input features {images.shape}")
            logging.info(f"the shape of the lables list is {labels.shape}")

            #now load the first images in the log
            logging.info(f"the first images is :\n label is {labels[0]} \n {plt.imshow(images[0]),plt.show()} ")

            
            save_obj(self.Data_Ingestion_Config_obj.images_data_path,obj=images)
            save_obj(self.Data_Ingestion_Config_obj.label_data_path,obj=labels)

            return train_data
        except Exception as e:
            raise Custom_Exception(e,sys)
       