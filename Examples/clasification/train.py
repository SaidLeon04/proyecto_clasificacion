import numpy as nu
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load


data = {
    "tiempo_minutos" : [5,8,12,2,20,15,3,12,45,10],
    "paginas_visitadas": [3,5,7,1,10,8,2,6,8,3],
    "compro" : [0,0,1,0,1,1,0,0,1,0] 
}

df = pd.DataFrame(data)

X = df[["paginas_visitadas"]]
y = df["compro"]

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print(X_test)
print(y_pred)
print(f"Precision {accuracy_score(y_test, y_pred):0%}")
