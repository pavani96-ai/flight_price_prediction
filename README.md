# Flight Price Prediction ✈️

## 📝 Project Overview
This project is an End-to-End Machine Learning solution designed to predict flight ticket prices based on various features like airline, date of journey, source, destination, and more. It leverages a modular architecture to ensure scalability, maintainability, and reproducibility.

## ⚙️ Workflow
The project follows a structured ML pipeline:

1.  **Data Ingestion**: 
    *   Extracts raw flight data from **MongoDB**.
    *   Stores data in a local **Feature Store** (CSV format).
    *   Performs **Train-Test Split** for model evaluation.
2.  **Data Validation**: 
    *   Ensures data integrity against a predefined schema (`schema.yaml`).
    *   Checks for missing values and data type consistency,data distribution.
3.  **Feature Engineering & Transformation**: 
    *   Handling categorical variables (OneHotEncoding).
    *   Extracting features from date and time components.
    *   Scaling numerical features.
4.  **Model Training**: 
    *   Training various regression models (Random Forest, XGBoost, etc.) with hyperparameter tuning.
5.  **Evaluation**:
    *   Calculating metrics like RMSE, MAE, R2-Score, adj R2 score acheived 85% R2-score.
6.  **Deployment**:
    *   Model wrapped in a FastAPI endpoint for high-throughput predictions.

    *   Containerized with Docker, pushed to AWS ECR.

    *   Automated deployment to AWS EC2 via GitHub Actions.

    *   Artifacts (model + preprocessor pickle files) stored in AWS S3.

    *   Secrets managed via .env and GitHub Secrets.
## 🚀 Accomplishments (What’s done so far)
- [x] **Modular Project Structure**: Established a clean package-based structure (`src/flight_price_prediction`).
- [x] **Data Integration**: Implemented scripts to push raw data to MongoDB (`push_mongodb.py`) and ingest it back into the pipeline.
- [x] **Logging & Exception Handling**: Integrated robust logging and custom exception handling for easier debugging.
- [x] **Configuration Management**: Centralized all paths and parameters using YAML configuration files.
- [x] **Data Ingestion Pipeline**: Fully functional component that pulls data from the cloud and prepares it for processing.
- [x] **DVC Integration**: Set up Data Version Control to track datasets and model versions efficiently.
- [x] **CI/CD Foundation**: Configured GitHub Actions for automated workflows.

## 🛠️ Tech Stack
*   **Language**: Python 3.10+
*   **Database**: MongoDB Atlas
*   **ML Libraries**: Pandas, Scikit-learn, NumPy
*   **DevOps/MLOps**: DVC,mlflow,Dagshub,FastAPI, GitHub Actions, Docker
*   **Configuration**: YAML

## 📁 Project Structure
```text
.
├── artifacts/          # Generated data and model files
├── config/             # Configuration files (YAML)
├── research/           # Jupyter notebooks for experimentation
├── schema/             # Data validation schemas
├── src/                # Source code
│   └── flight_price_prediction/
│       ├── components/ # Pipeline components (Ingestion, Validation, etc.)
│       ├── entity/     # Data classes for config and artifacts
│       ├── pipeline/   # Orchestration of components
│       └── utils/      # Utility functions
├── app.py              # Web application entry point
├── main.py             # Pipeline execution script
└── push_mongodb.py     # Script to upload data to MongoDB
```

## 🛠️ Getting Started
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/pavani96-ai/flight_price_prediction.git](https://github.com/pavani96-ai/flight_price_prediction.git)
   cd flight_price_prediction
   ```
2. **Install dependencies**:
   ```bash
   conda create -p venv python=3.10 -y
   conda cativate ./venv
   pip install -r requirements.txt
   ```
3. **Environment Setup**:
   Create a `.env` file and add your MongoDB URL:
   ```env
   MONGO_DB_URL="your_mongodb_atlas_connection_string"
   DAGSHUB_TOKEN="your_dagshub_secret_app_token"
   MLFLOW_TRACKING_URI="your_dagshub_mlflow_url"
   MLFLOW_TRACKING_USERNAME="pavani96-ai"
   MLFLOW_TRACKING_PASSWORD="your_dagshub_secret_app_token"
   ```
4. **Fetch Versioned Data
   dvc pull

5. **Run the Pipeline**:
   ```bash
   python main.py
   ```

6. **Spin up the FastAPI Web Server Locally**
  ```bash
  uvicorn app:app --host 0.0.0.0 --port 8000
  ```
  Visit http://localhost:8000/docs in your browser to interactively test the prediction payload endpoints via the Swagger UI.

7. **Containerization & Deployment**
  ```bash
  # Build the Docker Image
  docker build -t flight-price-pipeline:latest .

  # Run the Containerized API Instance
   docker run -p 8080:8080 --env-file .env flight-price-pipeline:latest
   
8. **Continuous Delivery**
   Merging updates directly into your production main trunk branch immediately invokes an automated GitHub Actions runner flow. The automated pipeline will run code sanity checks, construct a clean production Docker footprint, push it live into your AWS ECR instance registry, and seamlessly issue image refresh commands over to your target cloud AWS EC2 instance host.
