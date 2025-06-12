# Traffic Prediction Project

This project predicts traffic patterns using two machine learning models: XGBoost and a scikit-learn-based Random Forest. It leverages modern Python libraries for data processing, model training, and deployment via web interfaces.

## Features

- Machine learning model training and prediction using XGBoost and scikit-learn Random Forest.
- FastAPI for building high-performance APIs.
- Streamlit for interactive web-based dashboards.
- Real-time updates with streamlit-autorefresh.
- Efficient computation with NumPy and joblib.
- ASGI server support with Uvicorn.

## Models

### 1. XGBoost Model

- **Algorithm:** Gradient Boosted Decision Trees (XGBoost)
- **Description:** XGBoost is a powerful ensemble learning method that builds trees sequentially, optimizing for accuracy and speed.
- **Accuracy:** Achieved an accuracy of **92%** on the test dataset.

### 2. Random Forest Model

- **Algorithm:** Random Forest (scikit-learn)
- **Description:** Random Forest is an ensemble of decision trees, combining their outputs for robust predictions and reduced overfitting.
- **Accuracy:** Achieved an accuracy of **89%** on the test dataset.

## Dependencies

The project requires the following Python packages (see `requirements.txt`):

- **fastapi** (>=0.110.0): For building RESTful APIs.
- **uvicorn[standard]** (>=0.29.0): ASGI server for running FastAPI applications.
- **xgboost** (>=2.0.0): Gradient boosting library for machine learning.
- **scikit-learn** (>=1.4.0): Essential machine learning tools and utilities.
- **numpy** (>=1.26.0): Fundamental package for numerical computations.
- **joblib** (>=1.3.0): For efficient model serialization and parallel computation.
- **streamlit** (>=1.33.0): Framework for building interactive web apps.
- **streamlit-autorefresh** (>=0.0.4): Enables automatic refreshing of Streamlit apps.

## Installation

1. **Clone the repository:**

   ```bash
   git clone [<repository-url>](https://github.com/bava-kurian/traffic_predict_2)
   cd traffic_predict_2
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Train the Models

1. Prepare your dataset in the required format (see `data/` folder or documentation).
2. Run the training script:
   ```bash
   python train.py
   ```
   This will train both the XGBoost and Random Forest models and save them as serialized files (e.g., `xgboost_model.pkl`, `rf_model.pkl`).

### Run the API Server

Start the FastAPI server to serve predictions:

```bash
uvicorn main:app --reload
```

- The API will expose endpoints for making predictions using both models.

### Launch the Streamlit Dashboard

Start the Streamlit dashboard for interactive visualization:

```bash
streamlit run dashboard.py
```

- The dashboard allows users to input data and view predictions in real time.

## Project Structure

```
traffic_predict_2/
│
├── data/                  # Datasets and sample data
├── models/                # Saved model files
├── main.py                # FastAPI app
├── dashboard.py           # Streamlit dashboard
├── train.py               # Model training script
├── requirements.txt
└── README.md
```

## License

This project is licensed under the MIT License.
