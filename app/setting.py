from fastapi import FastAPI
from pydantic_settings import BaseSettings
import subprocess
import multiprocessing

def get_mac_cpu_info():
    try:
        # 获取物理 CPU 核心数
        physical_cores = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).strip())
        
        # 获取逻辑 CPU 核心数（包括超线程）
        logical_cores = int(subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"]).strip())
        
        # 获取当前 CPU 使用率
        cpu_usage = float(subprocess.check_output(["top", "-l", "1", "-s", "0", "-n", "1"]).split(b'\n')[3].split()[2].decode().strip('%'))
        
        return {
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
            "cpu_usage": cpu_usage
        }
    except Exception as e:
        print(f"Error getting CPU info: {e}")
        # 如果出错，回退到使用 multiprocessing
        return {
            "physical_cores": multiprocessing.cpu_count(),
            "logical_cores": multiprocessing.cpu_count(),
            "cpu_usage": None
        }

def get_optimal_thread_count():
    cpu_info = get_mac_cpu_info()
    logical_cores = cpu_info["logical_cores"]
    cpu_usage = cpu_info["cpu_usage"]
    
    # 如果 CPU 使用率低于 50%，使用所有逻辑核心
    if cpu_usage is None or cpu_usage < 50:
        return logical_cores
    # 如果 CPU 使用率在 50% 到 75% 之间，使用 75% 的逻辑核心
    elif 50 <= cpu_usage < 75:
        return max(1, int(logical_cores * 0.75))
    # 如果 CPU 使用率高于 75%，使用 50% 的逻辑核心
    else:
        return max(1, int(logical_cores * 0.5))


class Settings(BaseSettings):
    port: int = 6190
    max_content_length: int = 6144


settings = Settings()


