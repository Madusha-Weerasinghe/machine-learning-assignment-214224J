# ---------- Base Image ----------
FROM python:3.10-slim

# ---------- Set Working Directory ----------
WORKDIR /app

# ---------- Copy Project Files ----------
COPY . /app

# ---------- Install Python Dependencies ----------
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

# ---------- Expose Streamlit Port ----------
EXPOSE 8501

# ---------- Run Streamlit App ----------
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]