import os,sys
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from src.Logger import logging
from src.Exception_Handler import Custom_Exception
from src.Utils import resized_image
from src.Utils import save_obj
from dataclasses import dataclass

@dataclass
class Data_Ingestion_Config:
    features_data_path :str =os.path.join("artifacts","X.pickle")
    label_data_path :str =os.path.join("artifacts","y.pickle")

class Data_Ingestion:
    def __init__(self,data_dir,image_size=(50,50)):
        self.Data_Ingestion_Config_obj=Data_Ingestion_Config()
        self.data_dir=data_dir
        self.image_size=image_size
        self.data=[]
        self.label=[]


    def initiate_data(self):
        try:
            logging.info("initiating the data is starting")
            #data_dir=Path("/home/ankit/dogs_vs_cats_project/notebook/Data/animal")
            CATEGORIES=["cat","dog"]

            for category in CATEGORIES:
                dir_path=os.path.join(self.data_dir,category)
                class_label=CATEGORIES.index(category) # 0 for cat and 1 for cat

                for image_name in os.listdir(dir_path):
                    img_path=os.path.join(dir_path,image_name) #image_name provide the individuals names of the images
                    image=cv2.imread(img_path)
                    if image is None:
                        continue #if the image is currpted then skip and continue

                    resized_img=resized_image(image,self.image_size)
                    self.data.append(resized_img)
                    self.label.append(class_label)

            data_array=np.array([self.data])
            label_array=np.array([self.label])

            save_obj(
                self.Data_Ingestion_Config_obj.features_data_path,data_array
            )
            save_obj(self.Data_Ingestion_Config_obj.label_data_path,label_array)


            return(
                data_array,
                label_array
            )






        except Exception as e:
            raise Custom_Exception(e,sys)