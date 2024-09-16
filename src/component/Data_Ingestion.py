import os,sys
from pathlib import Path

import pandas as pd
import numpy as np
import cv2
from dataclasses import dataclass
from src.Logger import logging
from src.Exception_Handler import Custom_Exception

#for spliting the data use sklearn 
from sklearn.model_selection import train_test_split


@dataclass
class Data_Config:
    raw_data_path :str =os.path.join("artifacts","raw_data.csv")
    train_data_path :str =os.path.join("artifacts","train.csv")
    test_data_path :str =os.path.join("artifacts","test.csv")


class Data_Ingestion:
    def __init__(self,data_path) -> None:
        self.data_config_obj=Data_Config()
        self.data_path=data_path
        self.image_size=70

    def initiate_data(self):
        try:
            images_path=self.data_path

            categories=["cat","dog"]
            #storing the data and labels in the list
            data=[]
            labels=[]

            for category in categories:
                path=os.path.join(images_path,category)
                image_label=categories.index(category)

                for image_name in os.listdir(path):
                    full_image_path=os.path.join(path,image_name)
                    img=cv2.imread(full_image_path)

                    if img is not None:
                        new_img=cv2.resize(img,(self.image_size,self.image_size)) #resize the images of 70*70
                        #store the data into data list &store the label in the label list
                        data.append(new_img)
                        data.append(image_label)
            #converting the list to array
            data=np.array(data)
            labels=np.array(labels)

            #now concate the data and labels so that can do spliting
            full_data_array=np.c_[data,labels]

            #store the orinial datat to artifacts
            row_artifacts_path=os.path.join(os.path.dirname(self.data_config_obj.raw_data_path))
            #now make the artifacts folder
            os.makedirs(row_artifacts_path,exist_ok=True)
            
            full_data_array.to_csv(self.data_config_obj.raw_data_path,index=False)
            logging.info("store the raw data successfully")

            #now spliting the array into trian and test data
            train_data,test_data=train_test_split(full_data_array,test_data=0.20,random_state=42)

            # store the train and test data in artifacts
            train_data.to_csv(self.data_config_obj.train_data_path,index=False)
            logging.info("store the train_data in the artifacts completed")
            test_data.to_csv(self.data_config_obj.test_data_path,index=False)
            logging.info("storing the test_data to artifacts is compeleted")

            return(
                train_data,
                test_data
            )


        except Exception as e:
            raise Custom_Exception(e,sys)