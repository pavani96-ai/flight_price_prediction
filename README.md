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
    *   Checks for missing values and data type consistency.
3.  **Feature Engineering & Transformation**: 
    *   Handling categorical variables (OneHotEncoding/TargetEncoding).
    *   Extracting features from date and time components.
    *   Scaling numerical features.
4.  **Model Training**: 
    *   Training various regression models (Random Forest, XGBoost, etc.).
    *   Hyperparameter tuning using configurations from `params.yaml`.
5.  **Evaluation**:
    *   Calculating metrics like RMSE, MAE, and R2-Score.
6.  **Deployment**:
    *   Ready for deployment as a web application using cleaFastAPI.

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
*   **DevOps/MLOps**: DVC, GitHub Actions, Docker
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
   git clone https://github.com/your-username/flight_price_prediction.git
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Setup**:
   Create a `.env` file and add your MongoDB URL:
   ```env
   MONGO_DB_URL=your_mongodb_connection_string
   ```
4. **Run the Pipeline**:
   ```bash
   python main.py
   ```
