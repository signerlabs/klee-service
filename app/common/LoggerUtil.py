import logging

# 配置日志记录
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler("logs/logs.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
