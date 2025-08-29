# Auction Price Prediction System

This project predicts auction prices for LPG cylinders and valves using machine learning models trained on historical auction and market data. It includes:

- Data preprocessing and feature engineering  
- Model training and quantile-based inference  
- FastAPI endpoints for real-time prediction  
- Reporting utilities and simulation notebooks  

## Project Structure

### Project Structure
```
├── app/                   # FastAPI app: config, routers, services, utils
│   ├── main.py            # FastAPI entrypoint
│   ├── config.py          # Settings and environment variables
│   ├── routers/
│   │   └── predict.py     # API endpoints for prediction
│   ├── services/
│   │   ├── data_loader.py # Data loading utilities
│   │   ├── model_inference.py # Model loading and inference logic
│   │   └── preprocessor.py    # Feature engineering and preprocessing
│   └── utils/
│       └── logger.py      # Logging setup
├── models/                # Trained model files (.joblib)
├── data/                  # Raw and processed data files (.xlsx, .csv)
├── notebook/              # Jupyter notebooks for EDA, training, simulation
├── results/               # Output plots, reports, and evaluation results
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation (this file)
```

### Setup:
Clone Repository:
``` git clone https://github.com/AnoushkaBhadra/ml-auction-prediction-model.git```

### Create a virtual environment

```
python -m venv .venv
.venv\Scripts\activate
```

###
Install Dependecies:
``` pip install -r requirements.txt```


### Environment Variables

```copy .env.example .env``` 


### Prepare data
Place auction and market data in the data directory.
Run EDA and feature engineering notebooks as needed.


### Train models
Use Training_Model.ipynb to preprocess data and train models.
Trained models are saved in the models directory
### Run API:
``` uvicorn app.main:app --reload```

### Usage
1. Model Training
See Training_Model.ipynb for full training pipeline.
2. Prediction API
  Main entrypoint: main.py
  Prediction logic: model_inference.py


### Key Files & Notebooks
  API: main.py, predict.py
  Model Inference: model_inference.py
  Preprocessing: preprocessor.py
  Training: Training_Model.ipynb
  EDA: EDA_analysis_Auction_Data.ipynb, EDA_market_data.ipynb
  Simulation: Model_Prediction_Sim.ipynb

**Model Details**
  Features: Temporal features, EWM, price/quantity trends, market indices (brass index for valves)
  Models: Quantile regression with ensemble HistGradientBoostingRegressor
  Outputs: Median, lower/upper quantiles, confidence intervals



### Contributing
Fork the repo and create your branch.
Make changes and add tests.
Submit a pull request.
