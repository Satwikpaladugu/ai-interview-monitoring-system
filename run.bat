@echo off
cd /d "%~dp0"
start "FastAPI Backend" cmd /k "cd /d ""%~dp0"" && python app.py"
timeout /t 3
start "Frontend Server" cmd /k "cd /d ""%~dp0"" && python -m http.server 3000"
timeout /t 2
start "" "http://localhost:3000"
start "" "http://localhost:3000/admin.html"