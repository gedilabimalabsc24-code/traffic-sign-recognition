
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦")

st.title("🚦 Traffic Sign Recognition")
st.write("Upload a traffic sign image to predict its class using a CNN model.")

class_names = [
    "Speed limit (20 km/h)",
    "Speed limit (30 km/h)",
    "Speed limit (50 km/h)",
    "Speed limit (60 km/h)",
    "Speed limit (70 km/h)",
    "Speed limit (80 km/h)",
    "End of speed limit (80 km/h)",
    "Speed limit (100 km/h)",
    "Speed limit (120 km/h)",
    "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing for vehicles over 3.5 tons"
]

@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__), "models", "traffic_sign_model.keras"
    )
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

uploaded_file = st.file_uploader(
    "Choose a traffic sign image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Traffic Sign", width=300)

    resized_image = image.resize((32, 32))
    img_array = np.array(resized_image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    predicted_class = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]) * 100)

    st.success("Predicted Traffic Sign: " + class_names[predicted_class])
    st.write("Class Number:", predicted_class)
    st.write("Model Confidence:", round(confidence, 2), "%")
    st.caption("Confidence is not a guarantee that the prediction is correct.")


    # Generate Grad-CAM visualization
    import cv2

    # Find the last convolutional layer
    conv_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    last_conv_layer = conv_layers[-1]

    # Create Grad-CAM model
    # Rebuild the forward path using the saved model layers
    grad_input = tf.keras.Input(shape=(32, 32, 3))
    x = grad_input
    feature_maps = None

    for layer in model.layers:
        x = layer(x)
        if layer is last_conv_layer:
            feature_maps = x

    grad_model = tf.keras.Model(
        inputs=grad_input,
        outputs=[feature_maps, x]
    )

    # Calculate gradients
    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(
            img_array, training=False
        )
        predicted_index = tf.argmax(predictions[0])
        loss = predictions[:, predicted_index]

    grads = tape.gradient(loss, feature_maps)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(
        feature_maps[0] * pooled_grads, axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    # Resize and color the heatmap
    heatmap_resized = cv2.resize(heatmap.numpy(), (32, 32))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(
        heatmap_colored, cv2.COLOR_BGR2RGB
    )

    # Overlay heatmap on original image
    original_img = np.uint8(img_array[0] * 255)

    overlay = cv2.addWeighted(
        original_img, 0.6,
        heatmap_colored, 0.4, 0
    )

    st.subheader("Grad-CAM Visualization")
    st.image(overlay, caption="CNN Attention Heatmap", width=300)
