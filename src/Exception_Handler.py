import os,sys
from pathlib import Path


class Custom_Exception(Exception):
    def __init__(self,error_message,error_details:sys):
        self.error_message=error_message

        _,_,exc_tb=error_details.exc_info()

        self.file_name=exc_tb.tb_frame.f_code.co_filename
        self.line_number=exc_tb.tb_lineno







    def __str__(self) -> str:
        return f"the file name is: {self.file_name} \n the line number is :{self.line_number} \n the error is : {str(self.error_message)}"