# Advertising : Lineare Regression
# 1.Aufgabe:df aus advertising.csv erstellen

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


BASE_DIR = Path(__file__).parent

df=pd.read_csv(BASE_DIR / "advertising.csv", index_col=0)
#print(df)
#print(df.sales.shape)
#print(df["sales"].sales.shape)
# sales ist Target Vektor
# X ist der Feature Matrix und radio,tv,new pepers
# 2.Aufgabe : Feature-Matrix X und Target Vektor y bestimmen
# jede spalte der print(df) ist ein sries
# hier macht man den Target Vektor
#y = df["sales"]
y = df["sales"].to_numpy()  # alternativ df.sales=>pandas.series macht numpy array
print("Target Vektor :",y.shape, type(y)) #Numpy Array
# print("Feature Matrix:",X.shape(200,3)) #Numpy Array
#X = df[["TV", "radio", "newspaper"]].to_numpy() # pandas.DataFrame
X = df[["TV"]].to_numpy()
print("Target Vektor :",X.shape, type(y))

# 3.Aufgabe: Train-Test-Split erzeugen (kein stratify)
# Importieren der entsprechenden Funktion ,und erstellen eines Train-Test-Splits
X_train, X_test, y_train, y_test = train_test_split(
    X,                    # *arrays
    y,                    # *arrays
    test_size=0.3,        # 30 % Testdaten
    random_state=42,      # reproduzierbarer Zufall
    shuffle=True,         # Daten mischen
)
print("X_train:", X_train)
print("X_test :", X_test)
print("y_train:", y_train)
print("y_test :", y_test)

# 4.Aufgabe: Was ist der nächste Schritt?
# Importieren des LinearRegression_Models und Training(model.fit)
model = LinearRegression()
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
print(y_pred)

# 5.Aufgabe,Metriken: Welche Metriken ? Bitte mal 2 Metriken ausgeben
#print(df.describe())
mean_absolute_error=mean_absolute_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
print("mean_absolute_error:",r2_score(y_test, y_pred))
print("r2:",r2_score(y_test, y_pred))
print(df.describe())

# Es soll eine lineare Regression auf dem Advertising Datensatz gemacht werden.
# 1 Feature (TV)
# Ein Feature können wir in einem 2D-Plot darstellen. Eine Gerade kann gezeichnet
# werden, wenn man den y-Achsenabschnitt (intercept) und die Steigung (m) kennt.
# Beide Werte erhält man nach dem Training via:
# model.intercept_ (Intercept, Bias)
# model.coef_[0] (Steigung)
print("Intercept (Bias):", model.intercept_)  # -2938
#print(df.columns[:3])
print(df.columns[:1])
print(model.coef_)
# Koeffizienten der Features
coef_df = pd.DataFrame(
    {
        "Feature": df.columns[:1],
         #"Feature": df.columns[:3],
        "Koeffizient": model.coef_,
    }
)
print(coef_df)

m = model.coef_[0]
b = model.intercept_
x_line = [X.min(),X.max()]  # [3, 10]
y_line = [m * x + b for x in x_line]  # [1, 5] m * x + b
plt.scatter(X, y)  # Punkte
plt.plot(x_line, y_line) # Linie
plt.show()