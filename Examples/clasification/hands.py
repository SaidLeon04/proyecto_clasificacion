import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Cargar datos desde CSV
df = pd.read_csv("hands.csv")

# Separar características (todas las columnas menos 'clase')
X = df.drop(columns=["clase"])

# Etiqueta (0: cerrada, 1: abierta)
y = df["clase"]

# Dividir en entrenamiento y prueba (75% - 25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# Crear y entrenar el modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predecir sobre el set de prueba
y_pred = model.predict(X_test)

# Resultados
print("Precisión:", accuracy_score(y_test, y_pred))

# Opcional: mostrar algunas predicciones y sus etiquetas reales
for i in range(len(y_pred)):
    print(f"Predicho: {y_pred[i]} - Real: {y_test.iloc[i]}")
