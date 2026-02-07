from fastapi import APIRouter
from api.schema import LoanInput
from api.service import prediction,home

router=APIRouter()

router.get("/")(home)
router.post("/prediction")(prediction)
router.get("/welcome")(lambda: {
    "message": "Welcome from Shaiv 🚀",
    "version": "1.0",
    "author": "Shaiv"
})



