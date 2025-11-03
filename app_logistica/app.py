from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Cargar el modelo
model = joblib.load('modelo_regresion_logistica.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener valores del formulario
    edad = float(request.form['edad'])
    salario = float(request.form['salario'])

    # Crear array con las características
    features = np.array([[edad, salario]])

    # Predicción
    prediction = model.predict(features)[0]
    resultado = "Pertenece a la clase 1" if prediction == 1 else "Pertenece a la clase 0"

    # --------- Gráfica ---------
    fig, ax = plt.subplots()
    
    # Simulación: podrías graficar la frontera de decisión o solo el punto ingresado
    ax.scatter(edad, salario, color='red', label='Dato ingresado')
    ax.set_xlabel('Edad')
    ax.set_ylabel('Salario')
    ax.set_title('Visualización de entrada')
    ax.legend()

    # Convertir gráfico a imagen base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)

    # Renderizar con la imagen incluida
    return render_template('index.html', prediction=resultado, plot_url=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
