from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# تحميل النموذج والمعالج
model = joblib.load("model.pkl")
encoder = joblib.load("preprocessor.pkl")  # هذا فقط OneHotEncoder

# الأعمدة
categorical_cols = ['Food Product', 'Main Ingredient', 'Sweetener', 'Fat/Oil', 'Seasoning', 'Allergens']
numerical_cols = ['Price ($)', 'Customer rating']

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # استلام بيانات المستخدم
        data = request.get_json()
        df = pd.DataFrame([data])

        # معالجة الأعمدة الفئوية
        cat_data = df[categorical_cols].astype(str)
        cat_encoded = encoder.transform(cat_data).toarray()

        print(cat_data.head())

        # استخراج الأعمدة الرقمية كما هي
        num_data = df[numerical_cols].values

        # دمج البيانات
        final_input = np.hstack([cat_encoded, num_data])

        # التنبؤ
        prediction = model.predict(final_input)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
