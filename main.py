import argparse
import logging
import multiprocessing
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from llama_cloud.client import AsyncLlamaCloud

from app.config.env_config import Config, config
from app.setting import settings
from app.model.klee_settings import Settings as KleeSettings
from app.controllers.chat_controller import router as chat_router
from app.controllers.base_controller import router as base_router
from app.controllers.note_controller import router as note_router
from app.controllers.knowledge_controller import router as knowledge_router

from app.services.client_sqlite_service import init_db, engine, DATABASE_PATH
from app.model.db_schema import CREATE_TABLE_STATEMENTS
from sqlalchemy import text
from fastapi.middleware.cors import CORSMiddleware
from app.services.llama_index_service import LlamaIndexService
from app.controllers.llama_cloud_controller import router as llama_cloud_router

import os

os.environ["PYTHONIOENCODING"] = "utf-8"

llama_index_service = LlamaIndexService()

# 如何当前目录下面有.local文件优先加载.local文件
# 否则加载.env文件
app = FastAPI()
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: - %(asctime)s - %(message)s",
)

logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许任何域，或者指定特定域
    allow_credentials=True,
    allow_methods=["*"],  # 允许任何方法，或者指定特定方法
    allow_headers=["*"],  # 允许任何头，或者指定特定头
)


@app.exception_handler(Exception)
async def validation_exception_handler(request: Request, exc: Exception):
    # Change here to Logger
    # 记录日志，url。，method，参数，异常信息
    logger.error(
        f"Failed method {request.method} at URL {request.url}. Exception message is {exc!r}"
    )
    return JSONResponse(
        status_code=500,
        content={
            "message": (
                f"Failed method {request.method} at URL {request.url}."
                f" Exception message is {exc!r}."
            )
        },
    )

logging.getLogger("passlib").setLevel(logging.ERROR)
app.include_router(chat_router, prefix="/chat")
app.include_router(base_router, prefix="/base")
app.include_router(note_router, prefix="/note")
app.include_router(knowledge_router, prefix="/knowledge")

app.include_router(llama_cloud_router, prefix="/llama_cloud")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


async def startup_event():
    if not os.path.exists(DATABASE_PATH):
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        open(DATABASE_PATH, 'a').close()

    await init_db(engine)

    async with engine.begin() as conn:
        for statement in CREATE_TABLE_STATEMENTS:
            await conn.execute(text(statement))

    logger.info("Database initialized")


app.add_event_handler("startup", startup_event)
app.add_event_handler("startup", llama_index_service.init_config)

# app.add_event_handler("startup", start_tru_lens)
app.add_event_handler("startup", llama_index_service.init_global_model_settings)

# app.add_event_handler("startup", set_global_config)


def main():
    parser = argparse.ArgumentParser(description="FastAPI server")
    parser.add_argument('--port', type=int, default=settings.port, help='Port to run the server on')
    parser.add_argument("--env", type=str, default="local", help="Environment to run the server on")

    args = parser.parse_args()

    config = Config(env=args.env)

    if config.llama_cloud_api_key is not None:
        KleeSettings.async_llama_cloud = AsyncLlamaCloud(token=config.llama_cloud_api_key)

    KleeSettings.local_mode = True

    port = args.port

    multiprocessing.freeze_support()
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False, workers=1)


if __name__ == "__main__":
    main()
