import argparse
import logging
import multiprocessing
import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from llama_cloud.client import AsyncLlamaCloud
from sqlalchemy import text

from app.config.env_config import Config
from app.config.logging_config import configure_logging
from app.controllers import (
    base_controller,
    chat_controller,
    knowledge_controller,
    llama_cloud_controller,
    note_controller,
)
from app.model.db_schema import CREATE_TABLE_STATEMENTS
from app.model.klee_settings import Settings as KleeSettings
from app.services.client_sqlite_service import DATABASE_PATH, engine, init_db
from app.services.llama_index_service import LlamaIndexService
from app.setting import settings

def setup_environment():
    """
    Set up the environment variables for the server
    Returns:
        None
    """
    os.environ.update({
        "PYTHONIOENCODING": "utf-8",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1"
    })

def create_app() -> FastAPI:
    """
    Create the FastAPI application
    Returns:
        The FastAPI application
    """
    app = FastAPI(title="Klee Service", 
                 description="Klee Service API",
                 version="1.0.0")
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routers
    routers = [
        (chat_controller.router, "/chat"),
        (base_controller.router, "/base"),
        (note_controller.router, "/note"),
        (knowledge_controller.router, "/knowledge"),
        (llama_cloud_controller.router, "/llama_cloud"),
    ]
    
    for router, prefix in routers:
        app.include_router(router, prefix=prefix)
    
    return app

async def handle_exception(
        request: Request,
        exc: Exception
) -> JSONResponse:
    """
    Handle exceptions and log them
    Args:
        request: The request object
        exc: The exception raised
    Returns:
        A JSON response with the error message
    """
    logger = logging.getLogger(__name__)
    error_msg = f"Failed method {request.method} at URL {request.url}. Exception: {exc!r}"
    logger.error(error_msg)
    return JSONResponse(
        status_code=500,
        content={"message": error_msg}
    )

async def init_database():
    """
    Initialize the database and create the tables if they don't exist
    Returns:
        None
    """
    if not os.path.exists(DATABASE_PATH):
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        open(DATABASE_PATH, 'a').close()

    await init_db(engine)
    async with engine.begin() as conn:
        for statement in CREATE_TABLE_STATEMENTS:
            await conn.execute(text(statement))
    
    logging.getLogger(__name__).info("Database initialized successfully")

def configure_app(app: FastAPI, llama_index_service: LlamaIndexService):
    """
    Configure the FastAPI application
    Args:
        app: The FastAPI application
        llama_index_service: The Llama Index Service instance
    Returns:
        None
    """
    app.add_exception_handler(Exception, handle_exception)
    app.add_event_handler("startup", init_database)
    app.add_event_handler("startup", llama_index_service.init_config)
    app.add_event_handler("startup", llama_index_service.init_global_model_settings)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Klee Service FastAPI Server")
    parser.add_argument('--port', type=int, default=settings.port, 
                       help='Port to run the server on')
    parser.add_argument("--env", type=str, default="local", 
                       help="Environment to run the server on (local/dev/prod)")
    return parser.parse_args()

def setup_llama_cloud(api_key: Optional[str]):
    """
    Setup the Llama Cloud client.
    Args:
        api_key: The Llama Cloud API key
    Returns:
        None
    """
    if api_key:
        KleeSettings.async_llama_cloud = AsyncLlamaCloud(token=api_key)
    KleeSettings.local_mode = True

def main():
    configure_logging()
    logger = logging.getLogger(__name__)
    
    setup_environment()
    
    args = parse_arguments()
    
    try:
        config = Config(env=args.env)
        setup_llama_cloud(config.llama_cloud_api_key)
        
        llama_index_service = LlamaIndexService()
        
        app = create_app()
        configure_app(app, llama_index_service)
        
        logging.getLogger("passlib").setLevel(logging.ERROR)
        
        logger.info(f"Starting server on port {args.port}")
        multiprocessing.freeze_support()
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=args.port, 
            reload=False, 
            workers=1,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    main()
