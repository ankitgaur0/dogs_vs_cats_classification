import os,sys
from pathlib import Path
import pickle
import cv2

from src.Exception_Handler import Custom_Exception
from src.Logger import logging

#image_size should be a touple of two parameter
def resized_image(img,image_size):
    try:
        """
        this function is used for resized the image size
        becuase the image that is being ingest , may be in different size for individual image
        so resized_image function make the images in a same or a single size for all the images

        """
        resize_img=cv2.resize(img,image_size)
        return resize_img
    except Exception as e:
        raise Custom_Exception(e,sys)





def save_obj(obj_dir,obj):
    try:
        dir_path=os.path.dirname(obj_dir)
        os.makedirs(dir_path,exist_ok=True)

        with open(obj_dir,"wb") as f:
            pickle.dump(obj,f)
            logging.info("pickle file is succesfully save")

    except Exception as e:
        raise Custom_Exception(e,sys)


def load_obj(obj_dir):
    try:
        obj_dir=Path(obj_dir)
        with open(obj_dir,"rb") as f:
            pickle_file=pickle.load(f)
            return pickle_file

    except Exception as e:
        raise Custom_Exception(e,sys)