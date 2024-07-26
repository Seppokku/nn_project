import streamlit as st
st.title('Model: ResNet50')
st.title('Traing history of acc and loss during 15 epochs')
st.image('images/loss_accuracy_history_resnet50.png')
st.subheader('Train loss: 0.2381, valid loss = 0.1695 Train acc 0.9394 Valid acc 0.9396 on last epoch')
st.title('Time of training: 18 minutes')
st.title('Training dataset consist of 100 classes with 100 images in each and valid dataset with only 5 images in each')
