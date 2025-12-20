import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # backend/
ROOT_DIR = os.path.dirname(BASE_DIR) # root/
DATA_PATH = os.path.join(ROOT_DIR, "combined_dataset_with_soil_moisture.csv")
APP_DIR = os.path.join(BASE_DIR, "app")
MODEL_PATH = os.path.join(APP_DIR, "final_model.keras")
PREPROCESSOR_PATH = os.path.join(APP_DIR, "dl_preprocessor.joblib")

print(f"Loading data from: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print("❌ Data file not found!")
    exit(1)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# ... (Data Engineering from Notebook) ...
np.random.seed(42)
# Nitrogen
df['fert_N_kg_ha'] = (df['yield_t_ha'] * 50) - df['soil_N_status_kg_ha'] + np.random.normal(0, 2, len(df))
df['fert_N_kg_ha'] = df['fert_N_kg_ha'].apply(lambda x: max(5.0, x) if x > -20 else 0)
# Phosphorus
df['fert_P_kg_ha'] = (df['yield_t_ha'] * 25) - df['soil_P_status_kg_ha'] + np.random.normal(0, 2, len(df))
df['fert_P_kg_ha'] = df['fert_P_kg_ha'].apply(lambda x: max(2.0, x) if x > -10 else 0)
# Potassium
df['fert_K_kg_ha'] = (df['yield_t_ha'] * 35) - df['soil_K_status_kg_ha'] + np.random.normal(0, 2, len(df))
df['fert_K_kg_ha'] = df['fert_K_kg_ha'].apply(lambda x: max(3.0, x) if x > -15 else 0)

# Prepare Data
X = df[['crop', 'yield_t_ha', 'soil_N_status_kg_ha', 'soil_P_status_kg_ha', 'soil_K_status_kg_ha', 'mean_temp_gs_C', 'soil_pH', 'soil_moisture_pct']]
y = df[['fert_N_kg_ha', 'fert_P_kg_ha', 'fert_K_kg_ha']]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['yield_t_ha', 'soil_N_status_kg_ha', 'soil_P_status_kg_ha', 'soil_K_status_kg_ha', 'mean_temp_gs_C', 'soil_pH', 'soil_moisture_pct']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['crop'])
    ])

X_processed = preprocessor.fit_transform(X)
y_processed = y.values

X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

# Save Preprocessor
joblib.dump(preprocessor, PREPROCESSOR_PATH)
print(f"✅ Preprocessor saved to {PREPROCESSOR_PATH}")

# Build Model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Training model...")
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Save Model
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
