from fastapi import APIRouter, Query

router = APIRouter()

@router.get("/baseline")
async def target_baseline(id: str = Query(..., description="Payload to test")):
    """
    Unprotected Vulnerable Target.
    This endpoint serves as the backend for external WAFs (ModSecurity, SafeLine) to protect.
    It simulates a vulnerable application that reflects input or performs unsafe operations.
    """
    # A generic "vulnerability" check to simulate a successful attack if reached
    # In a real scenario, this would be a vulnerable SQL query or echoed input
    return {
        "status": "success", 
        "message": "Payload reached vulnerable backend (No Internal WAF)", 
        "payload": id,
        "reflected": id # Simulate XSS reflection
    }
