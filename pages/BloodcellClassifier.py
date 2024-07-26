import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Define path to the saved model
BLOOD_CELL_MODEL_PATH = 'model.pth'

# Custom ResNet model definition
class CustomResNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load the custom model
def load_custom_model(model_path, num_classes):
    model = CustomResNet(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()
    return model

# Transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prediction function
def predict(model, image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Blood cell classification interface
def blood_cell_classification():
    st.title("Классификация изображений клеток крови")
    st.text("Загрузите изображение клетки крови для классификации.")

    uploaded_file = st.file_uploader("Выберите файл изображения", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        st.write("")
        st.write("Классификация...")

        class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
        
        # Load the blood cell model
        model = load_custom_model(BLOOD_CELL_MODEL_PATH, num_classes=4)
        
        # Predict the class
        prediction = predict(model, uploaded_file)

        st.write(f"Предсказанный класс: {class_names[prediction]}")

if __name__ == '__main__':
    blood_cell_classification()