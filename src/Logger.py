import logging
import os,sys
from datetime import datetime
log_file_name=f"{datetime.now().strftime('%d-%m-%Y ,%H:%M:%S')}.log"

#now making the log_path
log_path=os.path.join(os.getcwd(),"Logs",log_file_name)
#now making the directory of log_path
os.makedirs(log_path,exist_ok=True)

#file name used in logging.basicConfig()
Log_file_path=os.path.join(log_path,log_file_name)

#define the custom log_format to show the stored log messages
log_format="[%(asctime)s]%(levelname)s -%(name)s -%(filename)s :%(lineno)d -%(message)s"


#now define the basicConfig
logging.basicConfig(
    filename=Log_file_path,
    level=logging.INFO,
    format=log_format,
    datefmt="%d-%m-%Y ,%H:%M:%S"
)