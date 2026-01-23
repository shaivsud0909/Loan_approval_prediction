from fastapi import APIRouter
from api.schema import LoanInput
from api.service import prediction

router=APIRouter()

# router.get("/")(home)
router.post("/prediction")(prediction)

