import uuid
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel

class TaskStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    GENERATING = "generating"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskNode(BaseModel):
    """
    Represents a node in the Task Tree (PTT).
    Each node works on a specific sub-goal or strategy.
    """
    node_id: str
    parent_id: Optional[str] = None
    strategy: str
    payloads: List[str] = []
    results: List[Dict[str, Any]] = []
    status: TaskStatus = TaskStatus.CREATED

class PentestPlanner:
    def __init__(self):
        # In-memory storage for the Task Tree (PTT)
        # tasks[task_id] -> List[TaskNode] (or a tree structure)
        # For simplicity, we store the root task info and a flat list of nodes for the PTT
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def create_task(self, target: str, attack_type: str) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "target": target,
            "type": attack_type,
            "status": TaskStatus.CREATED,
            "nodes": [], # The Task Tree nodes
            "logs": []
        }
        # Create the initial root node for the plan
        root_node = TaskNode(
            node_id=str(uuid.uuid4()),
            strategy=f"Initial Recon for {attack_type}",
            status=TaskStatus.PLANNING
        )
        self.tasks[task_id]["nodes"].append(root_node)
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.tasks.get(task_id)

    def add_log(self, task_id: str, message: str):
        if task_id in self.tasks:
            self.tasks[task_id]["logs"].append(message)

    def analyze_strategy(self, task_id: str, context: str = "") -> str:
        """
        Planner / Strategy Analyzer.
        Decides the next high-level strategy based on task type and previous feedback (context).
        """
        task = self.tasks.get(task_id)
        if not task:
            return "Unknown Task"
        
        attack_type = task.get("type")
        
        # In a real system, this would use an LLM or PDDL planner.
        # Here we simulate the logic.
        if "Blocked" in context:
            return f"Refine {attack_type} payload using obfuscation (User-Agent rotation + Encoding)"
        elif "Success" in context:
            return f"Verify {attack_type} depth and impact"
        
        # Initial strategies
        if attack_type == "sqli":
            return "Union-based SQL Injection combined with Error-based detection"
        elif attack_type == "xss":
            return "Polyglot XSS vectors with event handler injection"
        elif attack_type == "file_upload":
            return "File Upload bypass using Polyglot files (GIF/PHP) and double extensions"
        elif attack_type == "deserialization":
            return "Object Injection using common gadget chains (e.g., ysoserial payloads, PHP object injection)"
        else:
            return "General fuzzing and boundary testing"

    def update_node_result(self, task_id: str, node_id: str, result: Dict[str, Any]):
        """
        Updates the specific node in the task tree with verification results.
        """
        task = self.tasks.get(task_id)
        if task:
            for node in task["nodes"]:
                if node.node_id == node_id:
                    node.results.append(result)
                    # Logic to update status based on result could go here
                    break
