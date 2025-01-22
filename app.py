import streamlit as st 
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
import torch.nn.functional as F
import cv2
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Set the page configuration
st.set_page_config(
    page_title="Cancer Detection App",
    page_icon="🩺",
)

CLASS_NAMES = ['glioma', 'meningioma', 'no tumor', 'pituitary']


# Load the ResNet model
model = models.resnet18(pretrained=False)
num_classes = len(CLASS_NAMES)

model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, num_classes)
)
state_dict = torch.load("/home/skhotijah/project/microsoft_cup/model_resnet18_2.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# Preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Grad-CAM generation
def generate_grad_cam(model, image_tensor, target_class):
    gradients = []
    activations = []

    def save_gradient(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def save_activation(module, input, output):
        activations.append(output)

    final_conv_layer = model.layer4[-1].conv2
    final_conv_layer.register_forward_hook(save_activation)
    final_conv_layer.register_backward_hook(save_gradient)

    output = model(image_tensor)
    class_score = output[0, target_class]
    model.zero_grad()
    class_score.backward(retain_graph=True)

    grad = gradients[0].cpu().numpy()
    activation = activations[0].cpu().detach().numpy()
    weights = np.mean(grad, axis=(2, 3), keepdims=True)
    cam = np.sum(weights * activation, axis=1)[0]
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    return cam

# Heatmap overlay
def overlay_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    image_array = np.array(image)
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(image_array, 0.6, heatmap, 0.4, 0)
    return overlay

# Extract ground truth from filename
def extract_ground_truth(file_name): 
    file_name = file_name.replace('.jpg', '')
    if "/" in file_name:
        folder_name = file_name.split("/")[-3]
    else:
        folder_name = file_name.split("_")[0]
    
    label = folder_name.lower()
    return label if label in CLASS_NAMES else None

# Calculate metrics for each class
def calculate_metrics_for_all_classes(ground_truths, predictions, class_names):
    metrics = {}

    for class_name in class_names:
        # Calculate class-specific metrics
        acc = accuracy_score(ground_truths, predictions)
        precision = precision_score(ground_truths, predictions, average='weighted', zero_division=1)
        recall = recall_score(ground_truths, predictions, average='weighted', zero_division=1)
        f1 = f1_score(ground_truths, predictions, average='weighted', zero_division=1)

        metrics[class_name] = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }

    return metrics


# Streamlit App
def main():
    #st.title("Computed Tomography Scan Image Analysis for Cancer and Tumor Detection")
    st.markdown(
    "<h1 style='text-align: center;'>Computed Tomography Scan Image Analysis for Cancer and Tumor Detection</h1>", 
    unsafe_allow_html=True
)


    st.write("""
Upload the CT Scan Image you desire to analyze, our Neural Network Model will automatically analyze and display the results:
    """)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    all_ground_truths, all_predictions = [], []

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Extract ground truth
        ground_truth = extract_ground_truth(uploaded_file.name)
        all_ground_truths.append(ground_truth)

        # Preprocess the image
        processed_image = preprocess_image(image)

        if processed_image is not None:
            with st.spinner("Predicting..."):
                with torch.no_grad():
                    output = model(processed_image)
                    probabilities = F.softmax(output, dim=1).squeeze().numpy() 
                    prediction = np.argmax(probabilities)
                    confidence = probabilities[prediction]
            
                # Convert probabilities to percentages (multiply by 100) and round to 2 decimal places
                probabilities_percent = [f"{prob * 100:.2f}" for prob in probabilities]


            all_predictions.append(CLASS_NAMES[prediction])

            with st.spinner("Generating Grad-CAM..."):
                cam = generate_grad_cam(model, processed_image, prediction)
                overlay = overlay_heatmap(image, cam)

            # Display results
            st.markdown("<h2 style='font-size: 30px;'>This are the result for the provided images:</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 25px;'>Predicted: <b>{CLASS_NAMES[prediction]}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 25px;'>Confidence: <b>{confidence * 100:.2f}%</b></p>", unsafe_allow_html=True)


            # Ground truth
            # st.subheader("Ground Truth")
            # st.write(f"Ground Truth: **{ground_truth}**")
            
            # Image
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            with col2:
                st.image(overlay, caption="Grad-CAM Visualization", use_container_width=True)

            # Prediction probabilities
            # st.subheader("Prediction Probabilities")

            # Convert probabilities to percentages and format them
            probabilities_percent = [f"{prob * 100:.2f}" for prob in probabilities]

            # Create DataFrame for plotting
            probabilities_df = pd.DataFrame({
                'Class': CLASS_NAMES,
                'Probability': probabilities
            })



            # Plotting with matplotlib
            fig, ax = plt.subplots(figsize=(8, 6))  # Width = 8, Height = 6
            ax.bar(probabilities_df['Class'], probabilities_df['Probability'] * 100, color='skyblue')

            # Add text labels on top of the bars
            for i, v in enumerate(probabilities_df['Probability'] * 100):
                ax.text(i, v + 1, f"{v:.2f}", ha='center', va='bottom', fontweight='bold')

            # Customize the plot
            ax.set_ylabel('Probability (%)')
            ax.set_xlabel('Class')
            ax.set_ylim(0, 120)
            ax.set_title("Class Prediction Probabilities")

            # Display the plot in Streamlit
            st.pyplot(fig)
            

        st.write("""
Our model provides fast and accurate results, helping healthcare professionals identify potential issues for timely diagnosis and treatment.
    """)

if __name__ == "__main__":
    main()
