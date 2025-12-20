import sys
from src.logger import logging

def error_message_details(error):
    _,_,exc_tb = sys.exc_info() ##exception traceback logging
    file_name = exc_tb.tb_frame.f_code.co_filename ##extract the file name from the traceback
    error_message = 'Error encountered in the script [{0}] at line number [{1}] error message [{2}]'.format(
        file_name,
        exc_tb.tb_lineno,
        str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self,error):
        super().__init__(str(error))
        self.error_message = error_message_details(error = error)
    
    def __str__(self):
        return self.error_message

if __name__ == '__main__':
    try:
        x = 1/0
    except Exception as e:
        logging.info('Divided by zero')
        raise CustomException(e)