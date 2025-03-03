# config.py
import os
from dotenv import load_dotenv


class Config:
    _instance = None  # 单例实例

    def __new__(cls, env=None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.init_config(env)  # 初始化配置
        return cls._instance

    def init_config(self, env):
        if not env:
            raise ValueError("Environment must be specified (e.g., 'dev', 'prod')")

        # 加载对应的 .env 文件
        load_dotenv(f'.env.{env}')

        # 读取环境变量
        self.openai_key = os.getenv('OPENAI_KEY')
        self.llama_cloud_api_key = os.getenv('LLAMA_CLOUD_API_KEY')

# 全局配置实例
config = Config(env="local")  # 默认使用开发环境

