from flask import Flask, request, render_template, send_file
import joblib
import pandas as pd
import io
from xhtml2pdf import pisa

app = Flask(__name__)

modelo = joblib.load("modelo_knn.pkl")
columnas = joblib.load("columnas_modelo.pkl")

opciones_categoricas = {
    'Gender': {'1 (Masculino)': 1, '0 (Femenino)': 0},
    'Ever_married': {'1 (Sí)': 1, '0 (No)': 0},
    'Work_type': {
        '1 (Niño/a)': 1,
        '2 (Trabajo gubernamental)': 2,
        '3 (Nunca ha trabajado)': 3,
        '4 (Privado)': 4,
        '5 (Independiente)': 5
    },
    'Residence_type': {'1 (Urbano)': 1, '0 (Rural)': 0},
    'Smoking_status': {
        '0 (Nunca fumó)': 0,
        '1 (Fumó anteriormente)': 1,
        '2 (Fuma)': 2
    },
    'Hypertension': {'1 (Sí)': 1, '0 (No)': 0},
    'Heart_disease': {'1 (Sí)': 1, '0 (No)': 0}
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
        "datos": datos
    }

    return render_template('resultado.html', resultado=resultado)

@app.route('/descargar_pdf', methods=['POST'])
def descargar_pdf():
    datos = request.form.to_dict()
    pred = datos.pop('pred')
    prob_no = datos.pop('prob_no')
    prob_si = datos.pop('prob_si')

    html = f"""
    <h1>Resultado de Predicción</h1>
    <p><strong>Predicción:</strong> {'Riesgo Alto' if int(pred) == 1 else 'Riesgo Bajo'}</p>
    <p><strong>Probabilidades:</strong> No: {prob_no}%, Sí: {prob_si}%</p>
    <h2>Datos ingresados:</h2>
    <ul>
        {''.join(f"<li>{k}: {v}</li>" for k, v in datos.items())}
    </ul>
    """

    pdf_buffer = io.BytesIO()
    pisa.CreatePDF(html, dest=pdf_buffer)
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, download_name="resultado_prediccion.pdf", as_attachment=True)

@app.route('/descargar_excel', methods=['POST'])
def descargar_excel():
    datos = request.form.to_dict()
    pred = datos.pop('pred')
    prob_no = datos.pop('prob_no')
    prob_si = datos.pop('prob_si')

    datos['Prediccion'] = 'Riesgo Alto' if int(pred) == 1 else 'Riesgo Bajo'
    datos['Probabilidad_No'] = f"{prob_no}%"
    datos['Probabilidad_Si'] = f"{prob_si}%"

    df = pd.DataFrame([datos])
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)
    return send_file(excel_buffer, download_name="resultado_prediccion.xlsx", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
