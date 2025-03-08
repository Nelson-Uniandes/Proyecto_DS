from flask import Flask, render_template, request, jsonify
import random
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO)

def load_data():
    """Carga los productos y meses desde un archivo CSV"""
    df = pd.read_csv('data_form/data.csv')
    products = df['Product'].dropna().unique().tolist()
    months = df['Month'].dropna().unique().tolist()
    return products, months

def load_data_excel():
    """Carga el archivo Excel en partes para evitar alto consumo de memoria."""
    try:
        chunks = pd.read_excel('data/Online Retail.xlsx', chunksize=10000)  # Cargar en bloques de 10,000 filas
        df = pd.concat(chunks, ignore_index=True)  # Combinar los chunks en un solo DataFrame
        return df
    except Exception as e:
        logging.error(f"Error al cargar el archivo Excel: {e}")
        return pd.DataFrame()

def generate_historical_sales_chart(df, stock_code):
    """Genera la gráfica de ventas históricas basado en StockCode"""
    
    # Filtrar datos por StockCode
    product_data = df[df['StockCode'] == stock_code].copy()
    
    if product_data.empty:
        return None

    # Calcular las ventas totales (Cantidad * Precio Unitario)
    product_data["Sales"] = product_data["Quantity"] * product_data["UnitPrice"]

    # Agrupar ventas por Mes y Año
    product_data = product_data.groupby(["Year", "Month"])["Sales"].sum().reset_index()

    # Crear el eje X con "Month-Year"
    product_data["Month-Year"] = product_data["Month"].astype(str) + "-" + product_data["Year"].astype(str)

    # Variables para graficar
    x = product_data["Month-Year"]
    y = product_data["Sales"]

    # Crear la gráfica
    plt.figure(figsize=(6, 3))
    plt.plot(x, y, marker='o', linestyle='-', color='b', label="Actual Sales")
    plt.xlabel('Month-Year')
    plt.ylabel('Total Sales ($)')
    plt.title(f'Historical Sales Trend - {stock_code}')
    plt.xticks(rotation=45)
    plt.legend()

    # Guardar imagen en formato base64
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return encoded_img

def generate_demand_prediction_chart(month, year, predictions):
    """Genera la gráfica de predicciones de demanda"""
    x = [f"{year}-{m}" for m in range(month, month + len(predictions))]
    y = predictions

    plt.figure(figsize=(6, 3))
    plt.plot(x, y, marker='o', linestyle='--', color='r', label="Predicted Demand")
    plt.xlabel('Month-Year')
    plt.ylabel('Predicted Units Sold')
    plt.title('Demand Prediction Trend')
    plt.xticks(rotation=45)
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return encoded_img

app = Flask(__name__)

# Cargar el modelo guardado
modelo = joblib.load("ml_models/demand_model.pkl")
le = joblib.load("ml_models/label_encoder.pkl")

@app.route('/')
def home():
    products, months = load_data()
    return render_template('index.html', months=months, products=products)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        df_excel = load_data_excel()
        datos = request.json  # Recibir datos en formato JSON

        # Extraer datos de la solicitud
        product = datos.get("Product")  # Usar .get() en lugar de request.form[]
        month = datos.get("Month")
        year = datos.get("Year")

        if not product:
            return jsonify({"error": "Product is required"}), 400

        # Convertir Product a número usando LabelEncoder
        #product_num = le.transform([product])[0]

        # Crear DataFrame con los datos
        #X_nuevo = pd.DataFrame({"Month": [month], "Year": [year], "StockCode": [product]})

        # Hacer la predicción
        #predict = float(modelo.predict(X_nuevo)[0])

         # Generar predicciones para los próximos 6 meses
        predictions = []
        for i in range(6):
            X_nuevo = pd.DataFrame({"Month": [month], "Year": [year], "StockCode": [product]})
            predict = float(modelo.predict(X_nuevo)[0])
            predictions.append(predict)

        # Generar gráficas
        #historical_sales_chart = generate_historical_sales_chart(df_excel, product)
        #demand_prediction_chart = generate_demand_prediction_chart(month, year, predictions)  

        #return jsonify({"Month": month, "Year": year, "Predict": round(predict, 2)})
        return jsonify({
                "Month": month,
                "Year": year,
                "Predict": round(predictions[0], 2),
                #"HistoricalSalesChart": historical_sales_chart,
                #"DemandPredictionChart": demand_prediction_chart
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)