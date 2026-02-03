from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.api import endpoints, targets

app = FastAPI(title="WAF-LLM-Adversarial-Workbench API", version="0.1.0")

from fastapi.middleware.cors import CORSMiddleware

# Configure CORS
origins = [
    "http://localhost:5173",
    "http://localhost:3000", # Compatibility
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
# 注册路由
app.include_router(endpoints.router, prefix="/api")
app.include_router(targets.router, prefix="/target", tags=["Vulnerable Targets"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the WAF-LLM Adversarial Workbench Backend"}
