import os,sys
from src.Logger import logging
from src.Exception_Handler import Custom_Exception

#for spliting the data use sklearn 
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
#from tensorflow.keras.utils import img_to_array
class Data_Ingestion:
    def __init__(self,data_path):
        self.data_path=data_path
        self.image_size=100

    def get_img_augmentation(self):
        augmentation_generator=ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        return augmentation_generator

    def initiate_data(self):
        try:
            images_path=self.data_path
            data_aug=self.get_img_augmentation()
            training_augdata =data_aug.flow_from_directory(
                images_path,
                target_size=(self.image_size,self.image_size),
                subset='training',
                batch_size=32,
                class_mode='binary'
            )
            testing_augdata=data_aug.flow_from_directory(
                images_path,
                target_size=(self.image_size,self.image_size),
                subset='validation',
                batch_size=32,
                class_mode='binary'

            )

            return(
                training_augdata,
                testing_augdata,
                self.image_size
            )


        except Exception as e:
            raise Custom_Exception(e,sys)