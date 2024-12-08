import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
from vision_transformer import VisionTransformer
import tensorflow as tf
import numpy as np

# Load models based on selection
@st.cache_resource
def load_model(model_name):
    if model_name == "Vision Transformer (ViT)":
        model_path = "logo_classifier.pth"
        embed_dim = 256
        hidden_dim = 512
        num_channels = 3
        num_heads = 8
        num_layers = 6
        num_classes = 2
        patch_size = 16
        image_size = 128

        model = VisionTransformer(embed_dim, hidden_dim, num_channels, num_heads,
                                  num_layers, num_classes, patch_size, image_size)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model, "pytorch"

    elif model_name == "ResNet50":
        model_path = "resnet-50.h5"
        model = tf.keras.models.load_model(model_path, compile=False)

        # Use appropriate loss function based on the model's purpose
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        return model, "tensorflow"

    elif model_name == "VGG16":
        model_path = "vgg16.h5"
        model = tf.keras.models.load_model(model_path, compile=False)

        # Use appropriate loss function based on the model's purpose
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        return model, "tensorflow"

    elif model_name == "Custom CNN":
        model_path = "custom-cnn.h5"
        model = tf.keras.models.load_model(model_path, compile=False)

        # Use appropriate loss function based on the model's purpose
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        return model, "tensorflow"

# Preprocessing function for PyTorch
def preprocess_image_pytorch(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Preprocessing function for TensorFlow
def preprocess_image_tensorflow(image, target_size=(256, 256)):
    image = image.resize(target_size)  # Resize image
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    if len(image_array.shape) == 2:  # If grayscale, convert to RGB
        image_array = np.stack([image_array] * 3, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit app interface
st.title("Real vs Fake Image Classifier")
st.write("Upload an image to classify it as Real or Fake.")

# Add model selection dropdown
model_name = st.selectbox(
    "Choose a Model",
    ["Vision Transformer (ViT)", "ResNet50", "VGG16", "Custom CNN"]
)

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image:
    # Display uploaded image
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Load selected model
    model, framework = load_model(model_name)

    # Preprocess image based on framework
    if framework == "pytorch":
        input_tensor = preprocess_image_pytorch(image)
    else:  # TensorFlow
        input_tensor = preprocess_image_tensorflow(image, target_size=(256, 256))

    # Predict and display results
    if framework == "pytorch":
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs[0], dim=0)
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities[prediction].item() * 100


    elif framework == "tensorflow":

        predictions = model.predict(input_tensor)
        yhat = predictions[0][0]  # Assuming binary output (single value)
        confidence = yhat * 100 if yhat > 0.5 else (1 - yhat) * 100
        prediction = 1 if yhat > 0.5 else 0  # 1 for Real, 0 for Fake

    # Display prediction
    labels = ["Fake", "Real"]
    st.write(f"**Model Used**: {model_name}")
    st.write(f"Prediction: **{labels[prediction]}**")
    st.write(f"Confidence: {confidence:.2f}%")
