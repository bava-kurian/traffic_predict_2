🚦 Traffic Prediction System

This project predicts traffic patterns using two powerful machine learning models: XGBoost and Random Forest. It combines modern Python libraries with interactive web interfaces for easy visualization and real-time interaction.


---

🔍 Overview

The system offers:

Accurate traffic prediction using ML models.

FastAPI backend for API-based interaction.

Streamlit dashboard for user-friendly data input and visualization.

Real-time updates with streamlit-autorefresh.

Efficient computation with NumPy and joblib.

High-performance serving via Uvicorn (ASGI server).



---

🤖 Machine Learning Models

1. XGBoost Model

Algorithm: Gradient Boosted Decision Trees

Description: Sequential tree boosting for optimal speed and accuracy.

Accuracy: ✅ 92% on the test dataset


2. Random Forest Model

Algorithm: Random Forest (via scikit-learn)

Description: Ensemble learning with multiple decision trees to reduce overfitting.

Accuracy: ✅ 89% on the test dataset



---

📦 Dependencies

Ensure the following packages are installed (see requirements.txt):

fastapi>=0.110.0

uvicorn[standard]>=0.29.0

xgboost>=2.0.0

scikit-learn>=1.4.0

numpy>=1.26.0

joblib>=1.3.0

streamlit>=1.33.0

streamlit-autorefresh>=0.0.4



---

🚀 Installation

1. Clone the Repository

git clone https://github.com/bava-kurian/traffic_predict_2
cd traffic_predict_2


2. Install Required Packages

pip install -r requirements.txt




---

🧠 Model Training

1. Prepare your dataset (refer to the data/ folder for examples).


2. Run the training script:

python train.py



This will train both models and save them as .pkl files in the models/ directory:

xgboost_model.pkl

rf_model.pkl



---

⚙️ Run the API

Start the FastAPI server to serve predictions:

uvicorn main:app --reload

Access API docs at: http://127.0.0.1:8000/docs


---

📊 Launch the Dashboard

To launch the real-time interactive dashboard:

streamlit run dashboard.py

Features:

Live predictions

Auto-refresh support

User-friendly interface



---

📁 Project Structure

traffic_predict_2/
│
├── data/             # Datasets and sample inputs
├── models/           # Trained model files (.pkl)
├── main.py           # FastAPI backend
├── dashboard.py      # Streamlit UI
├── train.py          # ML training script
├── requirements.txt
└── README.md


---

📌 Future Enhancements

Real-time data ingestion from sensors or IoT

Model selection toggle on dashboard

Advanced traffic visualization (e.g., heatmaps, maps integration)

Scheduling model retraining for continuous learning



---

🧑‍💻 Author

Bava Kurian
GitHub • LinkedIn
