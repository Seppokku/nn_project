import streamlit as st

st.title('Model: ResNet18')
st.title('Traing history of acc and loss during 10 epochs')
st.image('images/bludcellshistory.png')
st.subheader('Train loss: 0.4381, valid loss = 0.4695 Train acc 0.8794 Valid acc 0.8496 on last epoch')
st.title('Time of training: 100 minutes')
st.title('Training dataset consist of 4 classes with 650 images in each and valid dataset with only 5 images in each')