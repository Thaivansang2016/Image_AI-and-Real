import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models

# 1️⃣ Khởi tạo model giống lúc training:
model = models.convnext_tiny(pretrained=False)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)  # Giả sử bạn phân 2 lớp

# 2️⃣ Load weights:
checkpoint = torch.load(r'D:\ĐỒ ÁN TTNT\demo\best_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

# 3️⃣ Set model sang eval mode:
model.eval()
# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit app
st.set_page_config(page_title="AI vs. Real Image Classifier", page_icon="🤖")

st.title("🖼️ AI vs. Real Image Classifier")
st.write("Upload an image to check if it's AI-generated or real!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        conf, pred = torch.max(prob, 1)
        label = 'AI-generated' if pred.item() == 1 else 'Real'

    st.markdown(
        f"""<p style="font-size:60px;"><b>Prediction:</b> {label}</p>""",unsafe_allow_html=True
    )

    st.markdown(
        f"""<p style="font-size:60px;"><b>Confidence:</b> {conf.item() * 100:.2f}%</p>""",unsafe_allow_html=True
    )