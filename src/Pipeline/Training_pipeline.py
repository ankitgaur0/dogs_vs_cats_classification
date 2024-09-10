import os,sys
from pathlib import Path
import pandas as pd
import numpy as np

#some local packages
from src.component.Data_Ingestion import Data_Ingestion
from src.Exception_Handler import Custom_Exception
from src.Logger import logging


try:
    #data dir where data is actually store
    data_dir=Path("/home/ankit/dogs_vs_cats_project/notebook/Data/animal")
    logging.info(f"the data is stored in the in this dir {data_dir}")
    #now make a obj of the Data_Ingestion class
    data_ingest=Data_Ingestion(data_dir)
    data_array,label_array=data_ingest.initiate_data()
    logging.info("the data in the form of array in store in data and label variable")

    print(data_array)
    print(label_array)






except Exception as e:
    raise Custom_Exception(e,sys)

