import httpx
import difflib
import os
from openai import AsyncOpenAI
from typing import Tuple, Dict, Any

# Environment variables for LLM Judge
API_KEY = os.getenv("OPENAI_API_KEY", "sk-xxxx")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://hiapi.online/v1") 

class LLMJudge:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    async def evaluate_response(self, response_text: str, payload: str) -> bool:
        """
        Uses a lightweight LLM call to judge if a response indicates a successful leak or bypass,
        especially when no obvious error strings are present.
        """
        prompt = f"""
        Analyze the following HTTP response body for signs of successful exploitation.
        
        Payload used: {payload}
        
        Response Body Snippet:
        {response_text[:1000]}
        
        Task: Determine if the payload successfully triggered a vulnerability (SQLi, XSS, etc.) or bypassed the WAF.
        Ignore standard 403 Forbidden pages. Look for database errors, reflected input, or implementation leaks.
        
        Return "TRUE" if successful/suspicious, "FALSE" otherwise.
        """
        try:
             response = await self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
             return "TRUE" in response.choices[0].message.content.upper()
        except:
            return False

class VulnerabilityVerifier:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.judge = LLMJudge()

    async def get_baseline(self, url: str) -> str:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=self.headers, timeout=5.0)
                return resp.text
        except:
            return ""

    async def verify(self, target_url: str, payload: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Multimodal Verification:
        1. Status Code Analysis
        2. Keyword Matching (Signatures)
        3. Response Differential Analysis
        4. LLM Judgement
        
        Returns:
            (is_success, summary, detailed_feedback_dict)
        """
        test_url = f"{target_url}?id={payload}" # Simplified injection point
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(test_url, headers=self.headers, timeout=10.0)
                
                feedback = {
                    "status_code": response.status_code,
                    "length": len(response.text),
                    "headers": dict(response.headers)
                }

                # 1. WAF Block Detection
                if response.status_code in [403, 406]:
                    return False, f"WAF Blocked ({response.status_code})", feedback
                
                # 2. Signature Matching
                error_signatures = [
                    "syntax error", "mysql_fetch", "ORA-", "PostgreSQL", "unterminated string", # SQLi
                    "unexpected end of file", "unserialize()", "ObjectInputStream", # Deserialization
                    "upload success", "move_uploaded_file" # File Upload (Positive logic usually)
                ]
                for sig in error_signatures:
                    if sig.lower() in response.text.lower():
                        return True, f"Vulnerability Indicator Detected: {sig}", feedback

                # 3. Differential Analysis (Simplified)
                # In a real scenario, we'd compare similarity ratios.
                # If the page is significantly different from a normal page but NOT a 403/500, it's interesting.
                
                # 4. LLM Judgement (The "Perceptor")
                if await self.judge.evaluate_response(response.text, payload):
                    return True, "LLM Judge detected successful exploitation indicators", feedback
                
                return False, "Request completed but no vulnerability verified", feedback

        except Exception as e:
            return False, f"Request Failed: {e}", {"error": str(e)}
