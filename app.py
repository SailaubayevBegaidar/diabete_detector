from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Загрузка модели и scaler
model = joblib.load("diabetes_model.joblib")
scaler = joblib.load("scaler.joblib")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Получение значений из формы
        features = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["blood_pressure"]),
            float(request.form["skin_thickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["diabetes_pedigree_function"]),
            float(request.form["age"])
        ]

        # Преобразуем в массив и масштабируем
        data = np.array([features])
        data_scaled = scaler.transform(data)

        # Предсказание
        prediction = model.predict(data_scaled)[0]

        result = "✅ Диабет обнаружен" if prediction == 1 else "✅ Диабет не обнаружен"
        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"⚠️ Ошибка: {e}")

if __name__ == "__main__":
    app.run(debug=True)
