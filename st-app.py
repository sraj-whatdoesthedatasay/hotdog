import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

st.title('Hotdog or Not?')

page = st.sidebar.selectbox(
'Select a page:',
('Hotdog or not','Contact Info')
)

if page == 'Contact Info':
    st.write('Name:  S Raj')
    st.write('Thanks for visiting')
    st.write('Contact info: 804 123-4567')

if page == 'Hotdog or not':
    st.title("Upload + HotDog Classification Example")
    new_model = tf.keras.models.load_model('saved_model/my_model')

    file = st.file_uploader("Upload your image here...", type=["png","jpg","jpeg"])

    if file is not None:
        hotdog = Image.open(file)
        st.image(hotdog)
        rgb_im = hotdog.convert('RGB')
        hotdog_arr = tf.keras.preprocessing.image.img_to_array(rgb_im) / 255
        resized_hotdog = tf.image.resize(hotdog_arr, (256, 256))
        hotdog_array = np.array(resized_hotdog).reshape(1,256,256,3)

        label = new_model.predict(hotdog_array)

        #st.write('The probability of recognizing as hotdog')
        #st.write(label[0][0])
        if label[0][0] > 0.8:
            st.write('It is definitely a hotdog! Enjoy!')
        elif label[0][0] > 0.5:
            st.write('It is probably a hot dog!')
        elif label[0][0] > 0.3:
            st.write('It is probably not a hot dog!')
        else:
            st.write('Its deinitely not a hotdog')


        #uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    # if uploaded_file is not None:
    #     image = Image.open(uploaded_file)
    #     st.write('p1')
    #     image = image.convert('RGB')
    #     #hd = load_img(image, target_size=(256, 256))
    #     st.write('p3')
    #     hd_arr = img_to_array(image.resize((256,256))) / 255
    #     hd_arr2 = np.reshape(hd_arr, (1,256,256,3))
    #     #tf.expand_dims(hd_arr, axis=0)
    #     st.write(hd_arr2.shape)
    #     #hd_arr2= np.array([1, hd_arr])
    #     #st.image(image, caption='Uploaded Image.', use_column_width=True)
    #     #st.write("")
    #     #st.write("Classifying...")
    #     st.write('p4')
    #     label = new_model.predict(hd_arr2)
    #     st.write('%s (%.2f%%)' % (label[1], label[2]*100))
