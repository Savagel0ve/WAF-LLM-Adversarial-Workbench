from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.core.generator import PayloadGenerator
from app.core.verifier import VulnerabilityVerifier
from app.core.planner import PentestPlanner

router = APIRouter()

class AttackRequest(BaseModel):
    target_url: str
    attack_type: str  # e.g., "sqli", "xss"
    llm_provider: Optional[str] = "openai"

class StructuredLog(BaseModel):
    round: int
    strategy: str
    payload: str
    status: str # "Bypass" | "Blocked"
    code: int # HTTP Status Code
    reasoning: str
    timestamp: str

class AttackResponse(BaseModel):
    task_id: str
    status: str
    structured_logs: List[StructuredLog]
    raw_logs: List[str]

# 实例化核心模块 (简单的单例模式)
planner = PentestPlanner()
generator = PayloadGenerator()
verifier = VulnerabilityVerifier()

@router.post("/attack/start", response_model=AttackResponse)
async def start_attack(request: AttackRequest):
    """
    Start the enhanced "Plan-Generate-Verify-Correct" Attack Loop.
    """
    # 1. Initialize Planner
    task_id = planner.create_task(request.target_url, request.attack_type)
    raw_logs = []
    structured_logs = []
    
    # Configuration for the Loop
    MAX_ROUNDS = 3 
    current_feedback = ""
    last_payload = ""

    from datetime import datetime

    for round_idx in range(MAX_ROUNDS):
        round_num = round_idx + 1
        raw_logs.append(f"--- Round {round_num} ---")
        
        # 1. Plan / Strategy
        strategy = planner.analyze_strategy(task_id, context=current_feedback)
        raw_logs.append(f"Planner Strategy: {strategy}")
        
        # 2. Generate (or Mutate if feedback exists)
        if round_idx == 0:
            # Initial generation using RAG + LLM
            # We take the first one for simplicity in the loop, or loop through all
            generated_list = await generator.generate_payloads(strategy, count=1)
            payloads = generated_list
        else:
            # Feedback-driven Mutation
            raw_logs.append("Applying Feedback-driven Mutation...")
            mutated_payload = await generator.mutate_payload(last_payload, current_feedback)
            payloads = [mutated_payload]
        
        # 3. Verify
        round_success = False
        for payload in payloads:
            raw_logs.append(f"Testing Payload: {payload}")
            is_success, summary, detailed_feedback = await verifier.verify(request.target_url, payload)
            
            result_log = f"Result: {'Bypass/Success' if is_success else 'Blocked/Fail'} - {summary}"
            raw_logs.append(result_log)
            planner.add_log(task_id, result_log)

            # Record Structured Log
            s_log = StructuredLog(
                round=round_num,
                strategy=strategy,
                payload=payload,
                status="BYPASSED" if is_success else "BLOCKED",
                code=detailed_feedback.get("status_code", 0),
                reasoning=summary,
                timestamp=datetime.now().strftime("%H:%M:%S")
            )
            structured_logs.append(s_log)
            
            if is_success:
                raw_logs.append(">>> Attack Successful! Stopping loop.")
                round_success = True
                break
            
            # Prepare feedback for next round if failed
            last_payload = payload
            if "status_code" in detailed_feedback:
                current_feedback = f"Status {detailed_feedback['status_code']}, Message: {summary}"
            else:
                current_feedback = summary

        if round_success:
            break
            
    return AttackResponse(
        task_id=task_id,
        status="completed",
        structured_logs=structured_logs,
        raw_logs=raw_logs
    )
