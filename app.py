import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fun With Filters", page_icon="📸")
st.title("Fun with Filters")

st.sidebar.title("Filters")

upl_img = st.file_uploader("Upload Image to Edit: ", type=['jpg', 'png', 'jpeg'])

if upl_img is not None:
    image = Image.open(upl_img)
    wpercent = 300/float(image.size[0])
    hsize = int((float(image.size[1])*float(wpercent)))
    image = image.resize((300, hsize))

    col1, col2 = st.columns([0.5,0.5])
    with col1:
        st.write("Original")
        st.image(image, width=300)
    
    with col2:
        st.write("Edited")
        filter = st.sidebar.radio('What filter to use?: ', ['Original','Greyscale', 'Blur', 'Motion Blur','Vignette','Pencil Sketch','Emboss'])
        
        if filter == 'Original':
            st.image(image, width=300)
        
        elif filter=='Greyscale':
            conv_img = np.array(image.convert('RGB'))
            r,g,b = conv_img[:,:,0], conv_img[:,:,1], conv_img[:,:,2]
            gamma = st.sidebar.slider("Gamma",0.0, 3.0, 1.04)
            rc, gc, bc = 0.2989, 0.5870, 0.1140
            gray_img = rc*r**gamma + gc*g**gamma + bc*b**gamma
            fig = plt.figure(1)
            img1 = fig.add_subplot()
            img1.imshow(gray_img, cmap="gray")
            plt.axis('off')
            plt.savefig("greyimg.png",bbox_inches='tight',pad_inches=0)
            show_img = Image.open("greyimg.png")
            st.image(show_img, clamp=True, width=300)

        elif filter=='Blur':
            conv_img = np.array(image.convert('RGB'))
            def gkern(l=5, sig=1.):
                """
                creates gaussian kernel with side length `l` and a sigma of `sig`
                """
                ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
                gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
                kernel = np.outer(gauss, gauss)
                return kernel / np.sum(kernel)
            sig = st.sidebar.slider("Intensity: ",0.1,2.0,1.)
            kernel = gkern(10, sig)
            blur_image = cv2.filter2D(conv_img, -1, kernel)
            st.image(blur_image, channels='RGB', width=300)
        
        elif filter=='Motion Blur':
            conv_img = np.array(image.convert('RGB'))
            i = st.sidebar.slider("Intensity: ",1,20,5)
            kernel = np.zeros((i,i))
            np.fill_diagonal(kernel, 1)
            kernel /= kernel.sum()
            mblur_image = cv2.filter2D(conv_img, -1, kernel)
            st.image(mblur_image, channels='RGB', width=300)
        
        elif filter=='Vignette':
            conv_img = np.array(image.convert('RGB'))
    
st.markdown("<p style='text-align:center'>You can right-click and save the image if you want to!</p>",unsafe_allow_html=True)


