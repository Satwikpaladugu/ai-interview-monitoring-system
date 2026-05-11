@echo off
cd /d C:\Users\satwi\OneDrive\Desktop\detection
start "Flask Backend" cmd /k "cd /d C:\Users\satwi\OneDrive\Desktop\detection && py -3.10 app.py"
timeout /t 3
start "Frontend Server" cmd /k "cd /d C:\Users\satwi\OneDrive\Desktop\detection && py -3.10 -m http.server 3000"
timeout /t 2
start "" "http://localhost:3000"
