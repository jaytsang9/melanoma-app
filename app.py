import streamlit as st
from PIL import Image
from torchvision import transforms
import torch
import os
from neural_net import AlexNet


PATH = "melanoma_CNN.pt"

model = torch.load(PATH, map_location=torch.device('cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

classes = ['Benign', 'Malignant']



upload = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
if upload is not None:
    # To read file as bytes:
    for u in upload:
        image = Image.open(u)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        batch_t = torch.unsqueeze(transform(image), 0)
        output = model(batch_t)

        ps = torch.exp(output)
        _, pred = torch.max(ps,1)
        prob = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.max(prob, 1)
        st.subheader((classes[pred[0]], f'{round(prediction[0].item()*100, 3)}%'))
