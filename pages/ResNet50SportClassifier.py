import streamlit as st
from models import sasha_model
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision import io
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import time
import requests
from io import BytesIO

device='cpu'
CLASS_NAMES = sorted(['bobsled',
                        'hurdles',
                        'snow boarding',
                        'fly fishing',
                        'luge',
                        'sidecar racing',
                        'ampute football',
                        'volleyball',
                        'billiards',
                        'giant slalom',
                        'tennis',
                        'horse racing',
                        'pole dancing',
                        'figure skating women',
                        'rollerblade racing',
                        'fencing',
                        'olympic wrestling',
                        'golf',
                        'ultimate',
                        'harness racing',
                        'football',
                        'frisbee',
                        'bungee jumping',
                        'shot put',
                        'ice climbing',
                        'figure skating men',
                        'rowing',
                        'bike polo',
                        'swimming',
                        'judo',
                        'axe throwing',
                        'archery',
                        'baseball',
                        'wheelchair basketball',
                        'log rolling',
                        'lacrosse',
                        'rock climbing',
                        'horse jumping',
                        'hydroplane racing',
                        'formula 1 racing',
                        'curling',
                        'jousting',
                        'javelin',
                        'water cycling',
                        'speed skating',
                        'barell racing',
                        'bull riding',
                        'horseshoe pitching',
                        'croquet',
                        'chuckwagon racing',
                        'hammer throw',
                        'rugby',
                        'pole climbing',
                        'nascar racing',
                        'snowmobile racing',
                        'boxing',
                        'mushing',
                        'track bicycle',
                        'canoe slamon',
                        'disc golf',
                        'bmx',
                        'air hockey',
                        'rings',
                        'trapeze',
                        'motorcycle racing',
                        'tug of war',
                        'ski jumping',
                        'field hockey',
                        'gaga',
                        'polo',
                        'ice yachting',
                        'jai alai',
                        'wheelchair racing',
                        'bowling',
                        'weightlifting',
                        'wingsuit flying',
                        'surfing',
                        'pole vault',
                        'water polo',
                        'basketball',
                        'skydiving',
                        'uneven bars',
                        'pommel horse',
                        'cricket',
                        'sky surfing',
                        'steer wrestling',
                        'hang gliding',
                        'parallel bar',
                        'shuffleboard',
                        'cheerleading',
                        'baton twirling',
                        'balance beam',
                        'hockey',
                        'high jump',
                        'table tennis',
                        'figure skating pairs',
                        'sailboat racing',
                        'sumo wrestling',
                        'arm wrestling',
                        'roller derby'])
model = sasha_model.MyResNet50()
model.to(device)
model.load_state_dict(torch.load('models/weights_sasha.pt', map_location=torch.device('cpu')))

# Transformations for input images
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

def load_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        st.error("Failed to load image from URL")
        return None


# Prediction function
def predict(model, image_path):
    model.eval()
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    start_time = time.time()
    with torch.no_grad():
        pred = model(image)
        pred_class = pred.argmax(dim=1).item()
        outputs = model(image)
        end_time = time.time()
        prediction_time = end_time - start_time
        confidence = torch.nn.functional.softmax(pred, dim=1)[0][pred_class].item()
        st.write(f"Predicted class: {CLASS_NAMES[pred_class]}")
        st.write(f"Confidence in predicted class: {confidence:.2f}")
        st.write(f"Prediction time: {prediction_time:.4f} seconds")
    _, predicted = torch.max(outputs, 1)
    return predicted.item()


def predict_url(model, image_url):
    model.eval()
    image = load_image_from_url(image_url)
    image = transform(image).unsqueeze(0)
    start_time = time.time()
    with torch.no_grad():
        pred = model(image)
        pred_class = pred.argmax(dim=1).item()
        outputs = model(image)
        end_time = time.time()
        prediction_time = end_time - start_time
        confidence = torch.nn.functional.softmax(pred, dim=1)[0][pred_class].item()
        st.write(f"Predicted class: {CLASS_NAMES[pred_class]}")
        st.write(f"Confidence in predicted class: {confidence:.2f}")
        st.write(f"Prediction time: {prediction_time:.4f} seconds")
    _, predicted = torch.max(outputs, 1)
    return predicted.item()




# Blood cell classification interface
def sport_classification():
    st.title("Classification of sports using ResNet50 and kaggle dataset of sports")
    st.subheader('Example of prediction:')
    st.image('images/exaple_of_prediction.jpg')


    st.subheader('You can try upload your own image of sport or put the URL!')

    uploaded_file = st.file_uploader("Upload image file", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("Enter image URL:")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded image', use_column_width=True)
        st.write("")
        prediction = predict(model, uploaded_file)

    if image_url:
        image = load_image_from_url(image_url)

        if image:
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            prediction = predict_url(model, image_url)

if __name__ == '__main__':
    sport_classification()