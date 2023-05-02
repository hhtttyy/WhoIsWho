import logging

logger = logging.getLogger("whoiswho")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler = logging.StreamHandler()
c_handler.setFormatter(formatter)
c_handler.setLevel(logging.INFO)
logger.addHandler(c_handler)