from flask import Flask, render_template, request
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # necesario para usar matplotlib sin interfaz gráfica
import matplotlib.pyplot as plt
import io, base64

# Cargar el modelo entrenado
model = joblib.load('modelo_salario.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener el valor ingresado
    experiencia = float(request.form['experiencia'])
    features = np.array([[experiencia]])
    prediccion = model.predict(features)[0]

    # Generar datos para la gráfica (por ejemplo, 0 a 10 años)
    x = np.linspace(0, 10, 50).reshape(-1, 1)
    y = model.predict(x)

    # Crear la gráfica
    plt.figure(figsize=(6,4))
    plt.plot(x, y, color='blue', label='Modelo de regresión')
    plt.scatter(experiencia, prediccion, color='red', s=80, label='Tu predicción')
    plt.title('Predicción de Salario vs Experiencia')
    plt.xlabel('Años de experiencia')
    plt.ylabel('Salario estimado')
    plt.legend()
    plt.grid(True)

    # Guardar la gráfica en memoria
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template(
        'index.html',
        prediction_text=f"Salario estimado: ${prediccion:,.2f}",
        plot_url=plot_url
    )

if __name__ == "__main__":
    app.run(debug=True)
