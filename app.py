import streamlit as st
from PIL import Image
from torchvision import transforms
import torch
import os
from neural_net import AlexNet, prediction


st.title('Welcome to Melanoma Image Recognition!')

file_types = ['jpg', 'jpeg', 'png']

upload = st.file_uploader("Choose a file", type=file_types, accept_multiple_files=True)
col1, col2 = st.columns(2)

if upload is not None:
    # To read file as bytes:
    for u in upload:
        image = Image.open(u)
        label, percentage = prediction(image)
        st.image(image, caption=f'{u.name}', width=300)
