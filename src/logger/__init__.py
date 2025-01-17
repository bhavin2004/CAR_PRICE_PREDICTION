import logging
from datetime import datetime
import os
import sys

LOG_FILE=f"{datetime.now().strftime('%d-%m-%Y-%H:%M:%S')}.log"

LOG_PATH = os.path.join(os.getcwd(),'logs',LOG_FILE)

LOG_FILE_PATH = os.path.join(LOG_PATH,LOG_FILE)
os.makedirs(LOG_PATH,exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(levelname)s: %(name)s %(module)s at line no %(lineno)d=> %(message)s",
    handlers=[
        logging.FileHandler(filename=LOG_FILE_PATH),
        # logging.StreamHandler(sys.stdout)
    ]
)
