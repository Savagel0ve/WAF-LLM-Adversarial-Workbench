import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

try:
    from app.core.planner import PentestPlanner
    from app.core.generator import PayloadGenerator
    from app.core.verifier import VulnerabilityVerifier
    print("Core modules imported successfully.")
    
    planner = PentestPlanner()
    task_id = planner.create_task("http://example.com", "sqli")
    print(f"Planner Working: Task ID {task_id}")
    
except ImportError as e:
    print(f"Import Error: {e}")
    print("You may need to install dependencies: pip install -r backend/requirements.txt")
except Exception as e:
    print(f"Error: {e}")
