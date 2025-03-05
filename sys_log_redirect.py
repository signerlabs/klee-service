import sys
import codecs
import chardet
import os

# 设置系统默认编码为UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

if not sys.stdout:
    class FakeStdOut:
        def __init__(self, filename="main-backend.log"):
            self.log = open(filename, "a", encoding="utf-8")
            self.encoding = 'utf-8'

        def write(self, message):
            try:
                if isinstance(message, bytes):
                    # 如果是bytes，先尝试解码
                    message = message.decode('utf-8', errors='ignore')
                elif isinstance(message, str):
                    # 如果是字符串，直接使用
                    pass
                else:
                    # 其他类型转换为字符串
                    message = str(message)
                
                # 确保message是UTF-8编码的字符串
                if not isinstance(message, str):
                    message = str(message)
                
                # 写入文件时强制使用UTF-8编码
                self.log.write(message)
                self.log.flush()
                
                # 同时尝试打印到控制台（用于调试）
                try:
                    # 使用replace策略处理无法编码的字符
                    sys.__stdout__.write(message.encode(sys.__stdout__.encoding or 'utf-8', errors='replace').decode(sys.__stdout__.encoding or 'utf-8'))
                    sys.__stdout__.flush()
                except:
                    pass  # 忽略控制台输出错误
                    
            except Exception as e:
                # 记录错误信息
                error_msg = f"Error writing to log: {str(e)}\n"
                try:
                    self.log.write(error_msg)
                    self.log.flush()
                except:
                    pass  # 如果连错误信息都无法写入，则忽略

        def flush(self):
            self.log.flush()
            try:
                sys.__stdout__.flush()
            except:
                pass

        def isatty(self):
            return False

    # 重定向标准输出
    sys.stdout = FakeStdOut()

