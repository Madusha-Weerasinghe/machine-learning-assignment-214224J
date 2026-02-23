# ---------- Base Image ----------
FROM python:3.10-slim

# ---------- Set Working Directory ----------
WORKDIR /app

# ---------- Copy Project Files ----------
# Copies all files from your project into the container's /app directory
COPY . /app

# ---------- Install Python Dependencies ----------
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

# ---------- Expose Streamlit Port ----------
# Default port for Streamlit is 8501
EXPOSE 8501

# ---------- Run Streamlit App ----------
# Executes the main dashboard script
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]