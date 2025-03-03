import uuid
from dataclasses import dataclass

from sqlalchemy import Column, String, Float, Boolean

from app.model.base import Base


@dataclass
class GlobalSettings(Base):
    __tablename__ = 'global_setting_table'

    id: str = Column(
        String, primary_key=True, default=lambda: str(uuid.uuid4()),
        index=True
    )  # 小写的uuid

    local_mode: bool = Column(Boolean, default=False)  # 是否开启本地模式
    model_name: str = Column(String, default='')  # 模型名称
    model_path: str = Column(String, default='')  # 模型路径
    model_id: str = Column(String, default='')  # 模型ID
    provider_id: str = Column(String, default='')  # 服务商ID
    create_at = Column(Float)  # 创建时间
    update_at = Column(Float)  # 更新时间

    def __init__(self, **kwargs):
        try:
            super().__init__(**kwargs)
        except Exception as e:
            raise ValueError(f"初始化 GlobalSettings 失败: {e}")
