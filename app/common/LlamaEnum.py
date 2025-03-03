import os

from enum import Enum

user_home = os.path.expanduser("~")


class SystemTypeDiffVectorUrl(Enum):
    """
        The vector type enum for win and mac
    """
    MAC_OS = os.path.join(user_home, 'Library/Application Support/com.signerlabs.klee/vector/')
    WIN_OS = "C:/Users/Administrator/AppData/Local/com/signer_labs/klee/vector/"


class SystemTypeDiffConfigUrl(Enum):
    """
        Config url for win and mac
    """
    MAC_OS = os.path.join(user_home, 'Library/Application Support/com.signerlabs.klee/config/Config.yaml')
    MAC_OS_path = os.path.join(user_home, 'Library/Application Support/com.signerlabs.klee/config')
    WIN_OS = "C:/Users/Administrator/AppData/Local/com/signer_labs/klee/config/Config.yaml"
    WIN_OS_path = "C:/Users/Administrator/AppData/Local/com/signer_labs/klee/config"


class SystemTypeDiff(Enum):
    WIN = "Windows"
    MAC = "Linux/macOS"


class SystemTypeDiffTempFileUrl(Enum):
    WIN_PATH = "C:/Users/Administrator/AppData/Local/com/signer_labs/klee/temp/file/"
    MAC_PATH = os.path.join(user_home, 'Library/Application Support/com.signerlabs.klee/temp/file/')


class SystemTypeDiffLlmUrl(Enum):
    WIN_PATH = "C:/Users/Administrator/AppData/Local/com/signer_labs/klee/llm/"
    MAC_PATH = os.path.join(user_home, "Library/Application Support/com.signerlabs.klee/llm/")


class SystemTypeDiffModelType(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    CLAUDE = "anthropic"
    DEEPSEEK = "deepseek"
    LOCAL = "local"
    KLEE = "klee"


class SystemTiktokenUrl(Enum):
    WIN_PATH = "C:/Users/Administrator/AppData/Local/com/signer_labs/klee/tiktoken_encode/"
    MAC_PATH = os.path.join(user_home, "Library/Application Support/com.signerlabs.klee/tiktoken_encode/")

class SystemEmbedUrl(Enum):
    WIN_PATH = "C:/Users/Administrator/AppData/Local/com/signer_labs/klee/embed_model/"
    MAC_PATH = os.path.join(user_home, "Library/Application Support/com.signerlabs.klee/embed_model/")


class SystemKleePkgPath(Enum):
    WIN_PATH = "C:/Users/Administrator/AppData/Local/com/signer_labs/klee/"
    MAC_PATH = os.path.join(user_home, "Library/Application Support/com.signerlabs.klee/")

