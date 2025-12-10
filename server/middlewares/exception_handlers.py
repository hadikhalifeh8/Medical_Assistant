from fastapi import Request
from fastapi.responses import JSONResponse
from logger import logger


async def catch_exception_middleware(requset:Request, call_next):
    try:
        return await call_next(requset)
    except Exception as e:
        logger.exception("Unhandled Exception")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )