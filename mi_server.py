from flask import Flask, request, render_template, send_file
import joblib
import pandas as pd
import io
from xhtml2pdf import pisa

app = Flask(__name__)

modelo = joblib.load("modelo_knn.pkl")
columnas = joblib.load("columnas_modelo.pkl")

opciones_categoricas = {
    'Gender': {'Masculino': 1, 'Femenino': 0},
    'Ever_married': {'Sí': 1, 'No': 0},
    'Work_type': {
        'Niño/a': 1,
        'Trabajo gubernamental': 2,
        'Nunca ha trabajado': 3,
        'Privado': 4,
        'Independiente': 5
    },
    'Residence_type': {'Urbano': 1, 'Rural': 0},
    'Smoking_status': {
        'Nunca fumó': 0,
        'Fumó anteriormente': 1,
        'Fuma': 2
    },
    'Hypertension': {'Sí': 1, 'No': 0},
    'Heart_disease': {'Sí': 1, 'No': 0}
}

traducciones = {
    'Gender': 'Género',
    'Age': 'Edad',
    'Hypertension': 'Hipertensión',
    'Heart_disease': 'Enfermedad cardíaca',
    'Ever_married': '¿Alguna vez casado/a?',
    'Work_type': 'Tipo de trabajo',
    'Residence_type': 'Tipo de residencia',
    'Avg_glucose_level': 'Nivel promedio de glucosa',
    'Bmi': 'Índice de masa corporal (IMC)',
    'Smoking_status': 'Estado de fumador'
}

@app.route('/')
def formulario():
    return render_template('formulario.html', columnas=columnas, opciones=opciones_categoricas, traducciones=traducciones)

@app.route('/predecir_formulario', methods=['POST'])
def predecir_formulario():
    datos = {col: float(request.form[col]) for col in columnas}
    df = pd.DataFrame([datos])
    pred = modelo.predict(df)[0]
    prob = modelo.predict_proba(df)[0].tolist()

    resultado = {
        "prediccion": "Sí" if int(pred) == 1 else "No",
        "prob_no": round(prob[0] * 100, 2),
        "prob_si": round(prob[1] * 100, 2),
        "datos": datos,
        "pred": int(pred)
    }

    return render_template('formulario.html', columnas=columnas, opciones=opciones_categoricas, traducciones=traducciones, resultado=resultado)

@app.route('/descargar_pdf', methods=['POST'])
def descargar_pdf():
    datos = request.form.to_dict()
    pred = datos.pop('pred')
    prob_no = datos.pop('prob_no')
    prob_si = datos.pop('prob_si')

    # Invertimos los diccionarios para decodificar valores numéricos a etiquetas
    opciones_invertidas = {
        clave: {str(v): k for k, v in opciones.items()}
        for clave, opciones in opciones_categoricas.items()
    }

    datos_formateados = []
    for clave, valor in datos.items():
        nombre_mostrado = traducciones.get(clave, clave.replace('_', ' ').capitalize())

        # Verificamos si es una variable categórica
        if clave in opciones_invertidas:
            valor_mostrado = opciones_invertidas[clave].get(str(int(float(valor))), valor)
        else:
            # Convertimos a entero si no tiene parte decimal significativa
            valor_float = float(valor)
            valor_mostrado = int(valor_float) if valor_float.is_integer() else round(valor_float, 3)

        datos_formateados.append(f"<li><strong>{nombre_mostrado}:</strong> {valor_mostrado}</li>")

    html = f"""
    <h1>Resultado de Predicción</h1>
    <p><strong>Predicción:</strong> {'Sí' if int(pred) == 1 else 'No'}</p>
    <p><strong>Probabilidades:</strong> No: {prob_no}%, Sí: {prob_si}%</p>
    <h2>Datos ingresados:</h2>
    <ul>
        {''.join(datos_formateados)}
    </ul>
    """

    pdf_buffer = io.BytesIO()
    pisa.CreatePDF(html, dest=pdf_buffer)
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, download_name="resultado_prediccion.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
