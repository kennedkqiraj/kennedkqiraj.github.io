# run.ps1 — start Streamlit minimized on port 8501
$project = "$PSScriptRoot"
$venvBin = Join-Path $project ".venv\Scripts"
$env:PATH = "$venvBin;$env:PATH"
Start-Process (Join-Path $venvBin "streamlit.exe") -ArgumentList @("run","$project\app_llama_rag.py","--server.port","8501") -WindowStyle Minimized
