from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import flask_cors
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from PIL import Image


# Định nghĩa lại hàm focal_loss và tải model
@tf.keras.utils.register_keras_serializable()
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)

        alpha_t = y_true * alpha + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = -alpha_t * tf.keras.backend.pow((tf.keras.backend.ones_like(y_true) - p_t), gamma) * tf.keras.backend.log(
            p_t)

        return tf.keras.backend.mean(tf.keras.backend.sum(fl, axis=-1))

    return focal_loss_fixed


model_path = 'best_model_lan2.keras'  # Đường dẫn tới model của bạn
model = tf.keras.models.load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.25)})

app = Flask(__name__)
CORS(app)


# Hàm chuẩn bị ảnh đầu vào
def prepare_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        image = Image.open(BytesIO(file.read()))
        processed_image = prepare_image(image)

        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=-1)[0]
        confidence = np.max(prediction)

        if predicted_class == 0:
            result = "Không có dấu hiệu bệnh lý."
        elif predicted_class == 1:
            result = "Dấu hiệu bệnh lý nhẹ."
        elif predicted_class == 2:
            result = "Dấu hiệu bệnh lý trung bình."
        elif predicted_class == 3:
            result = "Bệnh lý nghiêm trọng."
        elif predicted_class == 4:
            result = "Bệnh lý rất nghiêm trọng."
        else:
            result = "Không xác định được tình trạng bệnh."

        return jsonify({"result": result, "confidence": f"{confidence:.2f}"})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == '__main__':
    app.run(debug=True)
