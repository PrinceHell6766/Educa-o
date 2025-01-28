from flask import Flask, request, jsonify
import joblib
import numpy as np

# Carregar o modelo salvo
modelo = joblib.load('modelo.pkl')

# Inicializar o Flask
app = Flask(__name__)

# Definir o endpoint de previsão
@app.route('/predict', methods=['POST'])
def predict():
    # Receber os dados da requisição
    dados = request.json['dados']
    dados = np.array(dados).reshape(-1, 1)  # Transformar em formato adequado

    # Fazer a previsão
    previsao = modelo.predict(dados)

    # Retornar a previsão como JSON
    return jsonify({'previsao': previsao.tolist()})

# Rodar a aplicação
if __name__ == '__main__':
    app.run(debug=True)