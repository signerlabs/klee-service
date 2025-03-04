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
    )  # UUID in lowercase

    local_mode: bool = Column(Boolean, default=False)  # Enable/disable local mode
    model_name: str = Column(String, default='')  # Model name
    model_path: str = Column(String, default='')  # Model file path
    model_id: str = Column(String, default='')  # Model ID
    provider_id: str = Column(String, default='')  # Service provider ID
    create_at = Column(Float)  # Creation timestamp
    update_at = Column(Float)  # Last update timestamp

    def __init__(self, **kwargs):
        try:
            super().__init__(**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to initialize GlobalSettings: {e}")
