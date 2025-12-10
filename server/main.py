from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middlewares.exception_handlers import catch_exception_middleware
from routes.upload_pdfs import router as upload_pdfs_router
from routes.ask_question import router as ask_question_router

app = FastAPI(title="Medical Assistant API", description="API for AI Medical Assistant Chatbot")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# middleware exception handlers
app.middleware("http")(catch_exception_middleware)

# routers

# 1- upload pdf's documents
app.include_router(upload_pdfs_router)
# 2- asking query
app.include_router(ask_question_router)