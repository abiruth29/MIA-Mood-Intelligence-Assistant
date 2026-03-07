@echo off
REM Backend: --reload-dir app watches only app/*.py, NOT mia_data.db (avoids reload loop)
start cmd /k "cd backend && venv\Scripts\activate && uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app"
start cmd /k "set PATH=D:\nvm\v24.11.0;%PATH% && cd frontend && npm run dev"
