
#
# 1. Verwende die vier Features als Eingabedaten (`X`) und `CO2EMISSIONS` als Target (`y`).
# 2. Teile die Daten in Trainings- und Testdaten.
# 3. Trainiere ein lineares Regressionsmodell.
# 4. Werte das Modell mit folgenden Metriken aus:
#    * Mean Absolute Error (MAE)
#    * Mean Squared Error (MSE)
#    * R²-Score
# **Ziel:**
# * Grundidee der linearen Regression verstehen
# * Modellgüte korrekt interpretieren
# ### 2) Polynomielle Regression
# 1. Erweitere die Eingabedaten mit polynomialen Features (z. B. Grad 2). Probiere verschiedene Grade aus.
# 2. Trainiere erneut ein Regressionsmodell.
# 3. Vergleiche die Ergebnisse mit der linearen Regression.

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)





df = pd.read_csv(
        "https://s3-api.us-geo.objectstorage.softlayer.net/"
        "cf-courses-data/CognitiveClass/ML0101ENv3/labs/"
        "FuelConsumptionCo2.csv"
    )
# Erstes Mal die Daten anschaun
print(df.head())
print(df.describe())
print(df.columns)
print(df.info())
#exit()
# 8   FUELCONSUMPTION_CITY      1067 non-null   float64
#  9   FUELCONSUMPTION_HWY       1067 non-null   float64
#  10  FUELCONSUMPTION_COMB      1067 non-null   float64
#  11  FUELCONSUMPTION_COMB_MPG  1067 non-null   int64
#  12  CO2EMISSIONS              1067 non-null   int64
#



# Features als Eingabedaten
X = df[["FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY","FUELCONSUMPTION_COMB","FUELCONSUMPTION_COMB_MPG"]].to_numpy()
print(f"Features  :",X.shape, type(X))

# Target
y = df["CO2EMISSIONS"].to_numpy()
print("Target Vektor :",y.shape, type(y))

X_train, X_test, y_train, y_test = train_test_split(
    X,                    # *arrays
    y,                    # *arrays
    test_size=0.3,        # 30 % Testdaten
    random_state=42,      # reproduzierbarer Zufall
    #shuffle=True,         # Daten mischen
)
print("X_train:", X_train)
print("X_test :", X_test)
print("y_train:", y_train)
print("y_test :", y_test)

model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metriken
mean_absolute_error = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error:", mean_absolute_error)
print("Mean Squared Error:", mse)
print("R2:", r2)

# * Modellgüte korrekt interpretieren
# ### 2) Polynomielle Regression
# 1. Erweitere die Eingabedaten mit polynomialen Features (z. B. Grad 2). Probiere verschiedene Grade aus.
# 2. Trainiere erneut ein Regressionsmodell.
# 3. Vergleiche die Ergebnisse mit der linearen Regression.



