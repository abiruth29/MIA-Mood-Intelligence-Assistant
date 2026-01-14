@echo off
start cmd /k "cd backend && venv\Scripts\activate && uvicorn main:app --reload"
start cmd /k "set PATH=D:\nvm\v24.11.0;%PATH% && cd frontend && npm run dev"
