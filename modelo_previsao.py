# Importar bibliotecas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib  # Para salvar o modelo

# Criar um dataset fictício
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100 amostras, 1 feature
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + ruído

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Avaliar o modelo
score = modelo.score(X_test, y_test)
print(f"Coeficiente de determinação R²: {score:.2f}")

# Salvar o modelo treinado
joblib.dump(modelo, 'modelo.pkl')
print("Modelo salvo como 'modelo.pkl'")