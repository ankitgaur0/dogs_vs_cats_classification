import os,sys
from pathlib import Path
import logging

log_format="[%(asctime)s] %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format, #appling the custom log format
    datefmt="%Y-%m-%d  %H :%M :%S" #appling the custom date format
)


list_of_files=[
    ".github/workflows/main.yaml",
    "notebook/__init__.py",
    "notebook/Data/__init__.py",
    "notebook/EDA.ipynb",
    "notebook/Research.ipynb",
    "src/__init__.py",
    "src/component/__init__.py",
    "src/component/Data_Ingestion.py",
    "src/component/Data_transformation.py",
    "src/component/Model_Trainer.py",
    "src/component/Model_Evaluation.py",
    "src/Pipeline/__init__.py",
    "src/Pipeline/Training_pipeline.py",
    "src/Pipeline/Prediction_pipeline.py",
    'src/Logger.py',
    "src/Exception_Handler.py",
    "src/Utils.py",
    "app.py",
    "template",
    "try.ipynb"
]


for file_path in list_of_files:
    file_path=Path(file_path)

    dir_path,file_name=os.path.split(file_path)

    if (dir_path != ""):
        logging.info(f"if {dir_path} is !="" then make the directory")
        os.makedirs(dir_path,exist_ok=True)

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path)==0):
        with open(file_path,"wb") as file_path_obj:
            pass

    else:
        print("file is already exists")
        logging.info("file is already exists")




