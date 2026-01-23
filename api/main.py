from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Loan Approval Prediction API"
)

app.include_router(router)
