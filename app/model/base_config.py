import uuid

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Double
from dataclasses import dataclass


Base = declarative_base()


@dataclass
class BaseConfig(Base):
    __tablename__ = 'base_config'
    id: str = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    apiKey: str = Column(String, default="", nullable=False)
    name: str = Column(String, default="", nullable=False)
    description: str = Column(String, default="", nullable=False)
    baseUrl: str = Column(String, default="", nullable=False)
    models: str = Column(String, default="", nullable=False)

    create_at = Column(Double)
    update_at = Column(Double)
    delete_at = Column(Double)
