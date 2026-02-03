# WAF-LLM-Adversarial-Workbench

Project for LLM-assisted Breach and Attack Simulation (BAS) focusing on WAF evasion.

## Project Structure

This project follows a Client-Server architecture:

- **Frontend (`/frontend`)**: React + Vite application for the control dashboard.
- **Backend (`/backend`)**: Python FastAPI server for handling LLM interactions, payload generation, and verification.

## Getting Started

### 1. Backend Setup

Prerequisites: Python 3.8+

```bash
cd backend
pip install -r requirements.txt
```

Run the server:

```bash
cd backend
uvicorn app.main:app --reload
```
The API will be available at `http://localhost:8000`.
Docs: `http://localhost:8000/docs`

### 2. Frontend Setup

Prerequisites: Node.js 16+

```bash
cd frontend
npm install
npm run dev
```
The dashboard will be available at `http://localhost:5173`.

## Features (Implemented)

- **LLM Payload Generator**: Uses OpenAI API (configurable) to generate evasive SQLi/XSS payloads.
- **Vulnerability Verifier**: Automated verification of generated payloads against target URLs.
- **Pentest Planner**: Task management system for tracking attack sessions.

## Configuration

Set your OpenAI API key in `backend/app/core/generator.py` or export it as an environment variable:
`export OPENAI_API_KEY=your_key_here`
