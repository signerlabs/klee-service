from typing import Optional

from pydantic import BaseModel




class Model(BaseModel):
    brand: str
    download_url: str
    id: str
    model_size: str
    name: str
    require: str
    avatar_url: str
    store_size: str