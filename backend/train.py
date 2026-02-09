import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import pickle
import os

# =========================
# 1. LOAD DATA
# =========================
print("Loading data...")
train_df = pd.read_csv("data.csv")
test_df = pd.read_csv("test.csv")

# Ensure we don't accidentally train on IDs or indices
train_df = train_df.drop(columns=['index'], errors='ignore')
test_ids = test_df['index']
test_df = test_df.drop(columns=['index'], errors='ignore')

# =========================
# 2. TARGET TRANSFORMATION
# =========================
print("Transforming target variable...")
y = np.log1p(train_df['price'])
train_df = train_df.drop(columns=['price'])

# =========================
# 3. COMBINE FOR FEATURE ENGINEERING
# =========================
print("Combining datasets for feature engineering...")
train_df['is_train'] = 1
test_df['is_train'] = 0
full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# =========================
# 4. FEATURE ENGINEERING
# =========================
print("Engineering features...")
full_df['building_age'] = 2026 - full_df['year']
full_df['is_modern'] = (full_df['year'] > 2015).astype(int)
full_df['is_historic'] = (full_df['year'] < 1960).astype(int)
full_df['room_size'] = full_df['total_area'] / (full_df['rooms_count'] + 1)
full_df['area_per_bath'] = full_df['total_area'] / (full_df['bath_count'] + 1)
full_df['kitchen_ratio'] = full_df['kitchen_area'] / (full_df['total_area'] + 1)
full_df['service_area_total'] = (
    full_df['kitchen_area'] + full_df['bath_area'] + full_df['other_area']
)
full_df['living_to_service_ratio'] = (
    full_df['total_area'] / (full_df['service_area_total'] + 1)
)
full_df['floor_ratio'] = full_df['floor'] / (full_df['floor_max'] + 1)
full_df['is_top_floor'] = (full_df['floor'] == full_df['floor_max']).astype(int)
full_df['is_first_floor'] = (full_df['floor'] == 1).astype(int)
full_df['is_high_rise'] = (full_df['floor_max'] > 12).astype(int)

# --- Categorical Conversion ---
cat_cols = ['gas', 'hot_water', 'central_heating', 'extra_area_type_name', 'district_name']
for col in cat_cols:
    full_df[col] = full_df[col].astype('category')

# =========================
# 5. SPLIT BACK
# =========================
print("Splitting datasets...")
X = full_df[full_df['is_train'] == 1].drop(columns=['is_train'])
X_test = full_df[full_df['is_train'] == 0].drop(columns=['is_train'])

# =========================
# 6. TRAIN / VALIDATION SPLIT
# =========================
print("Creating train/validation split...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 7. MODEL TRAINING
# =========================
print("\n" + "="*50)
print("Training Model")
print("="*50 + "\n")

# ----------------------------------------------------
# HYPERPARAMETERS
# ----------------------------------------------------
cb_model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    early_stopping_rounds=100,
    verbose=200
)

# ----------------------------------------------------
# TRAIN WITH VALIDATION
# ----------------------------------------------------
cb_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_cols
)

# ----------------------------------------------------
# VALIDATION METRICS
# ----------------------------------------------------
print("\nCalculating validation metrics...")
val_preds_log = cb_model.predict(X_val)

y_val_actual = np.expm1(y_val)
y_val_pred = np.expm1(val_preds_log)

cb_rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
cb_mae = mean_absolute_error(y_val_actual, y_val_pred)
cb_r2 = r2_score(y_val_actual, y_val_pred)
cb_mape = np.mean(np.abs((y_val_actual - y_val_pred) / y_val_actual)) * 100

print(f"\n{'='*50}")
print(f"Validation Results for CatBoost")
print(f"{'='*50}")
print(f"RMSE: {cb_rmse:,.2f}")
print(f"MAE:  {cb_mae:,.2f}")
print(f"R²:   {cb_r2:.4f}")
print(f"MAPE: {cb_mape:.2f}%")
print(f"{'='*50}\n")

# =========================
# 8. FINAL TRAINING ON FULL DATASET
# =========================
print("Training final model on full dataset...")
cb_model_final = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=200
)

cb_model_final.fit(
    X, y,
    cat_features=cat_cols
)

# =========================
# 9. TEST PREDICTION
# =========================
print("\nGenerating test predictions...")
test_preds_log = cb_model_final.predict(X_test)
final_prices = np.expm1(test_preds_log)

# =========================
# 10. SAVE MODEL
# =========================
print("\nSaving model and metadata...")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model
cb_model_final.save_model("models/flat_price_prediction_model.cbm")
print("✅ Model saved to models/flat_price_prediction_model.cbm")

# Save categorical columns list
with open("models/cat_cols.pkl", "wb") as f:
    pickle.dump(cat_cols, f)
print("✅ Categorical columns saved to models/cat_cols.pkl")

# Save feature names
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)
print("✅ Feature names saved to models/feature_names.pkl")

print("\n" + "="*50)
print("Training Complete!")
print("="*50)
