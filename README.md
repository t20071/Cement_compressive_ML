# Cement Strength Prediction using Machine Learning

This project aims to build a robust regression pipeline for predicting **concrete compressive strength** based on multiple features related to concrete composition. The solution involves end-to-end automation of data validation, preprocessing, model training, prediction, and deployment.

---

## 📊 Data Description

Each record includes input variables (quantitative measures in kg/m³ or days) and one target variable—**Concrete Compressive Strength (MPa)**.

| Name | Type | Unit | Description |
|------|------|------|-------------|
| Cement (component 1) | Quantitative | kg/m³ | Input Variable |
| Blast Furnace Slag (component 2) | Quantitative | kg/m³ | Input Variable |
| Fly Ash (component 3) | Quantitative | kg/m³ | Input Variable |
| Water (component 4) | Quantitative | kg/m³ | Input Variable |
| Superplasticizer (component 5) | Quantitative | kg/m³ | Input Variable |
| Coarse Aggregate (component 6) | Quantitative | kg/m³ | Input Variable |
| Fine Aggregate (component 7) | Quantitative | kg/m³ | Input Variable |
| Age | Quantitative | Days (1–365) | Input Variable |
| Concrete Compressive Strength | Quantitative | MPa | **Target Variable** |

---

## 📁 Schema File

The schema file provided by the client includes:
- Expected filenames
- Length of date/time components in filenames
- Number and names of columns
- Data types for each column

---

## ✅ Data Validation

Before training or prediction, files go through the following validations:
1. **File Name Validation** using regex and schema constraints
2. **Column Count Check** against schema
3. **Column Names Check**
4. **Data Type Validation** during DB insertion
5. **Null Column Check** — files with entire columns of missing values are discarded

Files passing all validations are moved to `Good_Data_Folder`; others to `Bad_Data_Folder`.

---

## 🗃️ Data Insertion in Database

- **Database Creation/Connection** is handled programmatically.
- **Table Creation:** A `Good_Data` table is created based on schema definitions.
- **Insertion:** Validated files are inserted; invalid files are flagged and discarded.

---

## 🤖 Model Training Pipeline

1. **Data Export** from the database into a CSV format.
2. **Preprocessing:**
   - KNN imputation for missing values
   - Log transformation
   - Feature scaling
3. **Clustering:**
   - KMeans clustering with optimal cluster count via `KneeLocator`
   - Separate models trained per cluster
4. **Model Selection:**
   - `RandomForestRegressor` and `LinearRegression`
   - Models evaluated using R² score
   - Best model per cluster saved for prediction

---

## 🔮 Prediction Pipeline

1. **Data Validation & Insertion** (as above)
2. **Preprocessing:** Same as training
3. **Clustering:** Assign cluster using pre-trained KMeans model
4. **Prediction:**
   - Predict with best model for assigned cluster
   - Save predictions in CSV and return path to client

---
## File structure 
cement_strength_reg/
├── main.py                          # Main Flask application file
├── requirements.txt                 # Python dependencies
├── app.yaml                         # Google App Engine configuration file
├── templates/                       # Folder for HTML templates
│   └── index.html                   # Home page template
├── static/                          # Folder for static files (CSS, JS, images, etc.)
├── prediction_Validation_Insertion.py  # Module for prediction validation and insertion
├── trainingModel.py                 # Module for training the machine learning model
├── training_Validation_Insertion.py # Module for training data validation and insertion
├── predictFromModel.py              # Module for generating predictions from the model
├── README.md                        # Project documentation
└── LICENSE                          # License file (optional)

