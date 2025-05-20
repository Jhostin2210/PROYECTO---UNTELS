from flask import Flask, request, render_template, send_file
import joblib
import pandas as pd
import io
from xhtml2pdf import pisa
import requests

app = Flask(__name__)

modelo = joblib.load("modelo_randomforest.pkl")
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

    email_usuario = request.form.get("email_usuario")

    # Construir string con datos ingresados, en formato texto plano con saltos de línea
    datos_str = ""
    for clave, valor in datos.items():
        nombre_mostrado = traducciones.get(
            clave, clave.replace('_', ' ').capitalize())
        if clave in opciones_categoricas:
            inverso = {v: k for k, v in opciones_categoricas[clave].items()}
            valor_mostrado = inverso.get(valor, valor)
        else:
            valor_mostrado = valor
        datos_str += f"{nombre_mostrado}: {valor_mostrado}<br>"

    # Armar contenido HTML con datos primero y luego resultados
    contenido_html = f"""
    <h3>Datos ingresados:</h3>
    {datos_str}
    <br>
    <h2>Resultado de Predicción</h2>
    <p><strong>Predicción:</strong> {resultado['prediccion']}</p>
    <p><strong>Probabilidad NO:</strong> {resultado['prob_no']}%</p>
    <p><strong>Probabilidad SÍ:</strong> {resultado['prob_si']}%</p>
    """

    if email_usuario:
        codigo, respuesta = enviar_correo(
            destinatario=email_usuario,
            asunto="Resultado de predicción de ACV",
            contenido_html=contenido_html
        )
        print(f"Correo enviado: {codigo}, Respuesta: {respuesta}")

    return render_template('formulario.html',
                           columnas=columnas,
                           opciones=opciones_categoricas,
                           traducciones=traducciones,
                           resultado=resultado)

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

def enviar_correo(destinatario, asunto, contenido_html):
    # reemplázala con tu API Key real
    api_key = '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'

    url = "https://api.brevo.com/v3/smtp/email"
    headers = {
        "accept": "application/json",
        "api-key": api_key,
        "content-type": "application/json"
    }
    data = {
        "sender": {"name": "Predicción ACV", "email": "2113110269@untels.edu.pe"},
        "to": [{"email": destinatario}],
        "subject": asunto,
        "htmlContent": contenido_html
    }

    response = requests.post(url, json=data, headers=headers)
    return response.status_code, response.json()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
