import logging
import multiprocessing
import uvicorn
import sys_log_redirect
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.controllers.model_controller import router as model_router
from app.controllers.openai_controller import router as chat_router

# 如何当前目录下面有.local文件优先加载.local文件
# 否则加载.env文件
app = FastAPI()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# logging.getLogger(__name__)
logging.getLogger("uvicorn").setLevel(logging.DEBUG)


@app.exception_handler(Exception)
async def validation_exception_handler(request: Request, exc: Exception):
    # Change here to Logger
    # 记录日志，url。，method，参数，异常信息
    logging.error(
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


# logging.getLogger("passlib").setLevel(logging.DEBUG)
app.include_router(model_router, prefix="/models")
app.include_router(chat_router, prefix="/chat")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    uvicorn.run(app, host="0.0.0.0", port=6120, reload=False, workers=1)
