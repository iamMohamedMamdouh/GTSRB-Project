import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

model = tf.keras.models.load_model("traffic_model.h5")

class_indices = {
    '0': 0,
    '1': 1,
    '3': 8,
    '4': 9,
    '12': 2,
    '15': 3,
    '16': 4,
    '21': 5,
    '22': 6,
    '25': 7
}
idx_to_class = {v: k for k, v in class_indices.items()}

class_labels = {
    "0": "Speed Limit (20 km/h)",
    "1": "Speed Limit (30 km/h)",
    "3": "Speed Limit (60 km/h)",
    "4": "Speed Limit (70 km/h)",
    "12": "Priority Road",
    "15": "No Entry",
    "16": "Yield",
    "21": "Double Curve",
    "22": "Slippery Road",
    "25": "Road Work"
}
# ========== 3. دالة لتحويل الصورة Base64 ==========
def get_base64_of_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = "111.jpg"
bg_base64 = get_base64_of_image(bg_image)

# ========== 4. CSS لإضافة الخلفية ==========
page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
}}
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-color: rgba(255,255,255,0.1);
    z-index: -1;
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("🚦 Traffic Sign Classifier")
st.write("Upload a photo of a traffic light and the model will classify it ✅")

uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Photo Uploaded", use_container_width=True)

    # تجهيز الصورة للموديل
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # توقع
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)

    # اسم الكلاس
    class_code = idx_to_class[class_idx]
    label = class_labels.get(class_code, class_code)

    # النتيجة
    st.markdown(f"### ✅ Prediction: **{label}**")
    st.markdown(f"### 🔥 Confidence: **{confidence*100:.2f}%**")
