from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar modelo y columnas
modelo = joblib.load("modelo_knn.pkl")
columnas = joblib.load("columnas_modelo.pkl")

# Opciones con valor numérico y descripción
opciones_categoricas = {
    'Gender': {'1 (Male)': 1, '0 (Female)': 0},
    'Ever_married': {'1 (Yes)': 1, '0 (No)': 0},
    'Work_type': {
        '1 (children)': 1,
        '2 (Govt_job)': 2,
        '3 (Never_worked)': 3,
        '4 (Private)': 4,
        '5 (Self-employed)': 5
    },
    'Residence_type': {'1 (Urban)': 1, '0 (Rural)': 0},
    'Smoking_status': {
        '0 (never smoked)': 0,
        '1 (formerly smoked)': 1,
        '2 (smokes)': 2
    },
    'Hypertension': {'1 (Yes)': 1, '0 (No)': 0},
    'Heart_disease': {'1 (Yes)': 1, '0 (No)': 0}
}

# Formulario HTML
formulario_html = """
<!doctype html>
<html>
<head>
    <title>Predicción de Accidente Cerebrovascular</title>
</head>
<body>
    <h2>Formulario para predecir riesgo de accidente cerebrovascular</h2>
    <form method="post" action="/predecir_formulario">
        {% for col in columnas %}
            <label>{{ col.replace('_', ' ').capitalize() }}:</label><br>
            {% if col in opciones %}
                <select name="{{ col }}">
                    {% for texto, valor in opciones[col].items() %}
                        <option value="{{ valor }}">{{ texto }}</option>
                    {% endfor %}
                </select><br><br>
            {% else %}
                <input type="number" name="{{ col }}" step="any" required><br><br>
            {% endif %}
        {% endfor %}
        <input type="submit" value="Predecir">
    </form>
</body>
</html>
"""

@app.route('/')
def formulario():
    return render_template_string(formulario_html, columnas=columnas, opciones=opciones_categoricas)

@app.route('/predecir_formulario', methods=['POST'])
def predecir_formulario():
    try:
        datos = {}
        for col in columnas:
            valor = request.form[col]
            datos[col] = float(valor)

        df = pd.DataFrame([datos])
        pred = modelo.predict(df)[0]
        prob = modelo.predict_proba(df)[0].tolist()

        return f"""
        <h2>Resultado:</h2>
        <p><strong>Predicción:</strong> {"Sí" if int(pred) == 1 else "No"}</p>
        <p><strong>Probabilidades:</strong> No: {round(prob[0]*100, 2)}%, Sí: {round(prob[1]*100, 2)}%</p>
        <a href="/">Volver al formulario</a>
        """

    except Exception as e:
        return f"<h2>Error:</h2><p>{str(e)}</p><a href='/'>Volver</a>"

if __name__ == '__main__':
    app.run(debug=True)