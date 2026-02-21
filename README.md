# Wood Apple AI – Machine Learning Assignment

## 📌 Project Overview

This project is a machine learning-based application for analyzing and predicting wood apple-related data. It processes datasets, trains predictive models, and provides a Streamlit-based user interface for visualization and interaction.

## 🛠 Tech Stack

- Python 3.10
- Streamlit
- Pandas
- NumPy
- Scikit-Learn

## 🚀 Data Processing & Model Pipeline

Run the following steps in order:

### 1️⃣ Preprocess Data

```powershell
py data/raw/preprocessing.py
```

This cleans and prepares the dataset for training.

### 2️⃣ Train the Model

```powershell
py src/train.py
```

This trains the machine learning model.

### 3️⃣ Generate Explainability & XAI Outputs

```powershell
py src/explainability.py
```

This generates:

- Sensitivity analysis PNG
- Accuracy and XAI visualizations

## 🐳 Running with Docker (Recommended)

### Build Image

```powershell
docker build --platform linux/amd64 -t wood-apple-ai .
```

### Run Container

```powershell
docker run -p 8501:8501 wood-apple-ai
```

Then open:

```
http://localhost:8501
```

## 📁 Project Structure

```
.
├── app/
│   ├── main.py          # Streamlit app entry
│   ├── pages/           # UI pages
├── data/
│   ├── raw/              # Raw dataset
│   ├── processed/        # Cleaned dataset
├── models/              # Trained ML models
├── src/
│   ├── train.py          # Model training
│   ├── explainability.py # XAI generation
├── requirements.txt      # Dependencies
├── Dockerfile            # Docker configuration
└── README.md             # Documentation
```

## ⚙️ Troubleshooting

- If Docker build fails, restart Docker Desktop and WSL
- Use cache cleanup if needed:

  ```powershell
  docker system prune -a
  ```

- Build with platform flag:

  ```powershell
  docker build --platform linux/amd64 -t wood-apple-ai .
  ```

## ✨ Author

Developed for machine learning assignment and data-driven analysis.
