# from black import out
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

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
        # filter = st.sidebar.radio('What filter to use?: ', ['Original','Greyscale', 'Blur', 'Motion Blur','Vignette','Pencil Sketch','Emboss'])
        filter = st.sidebar.radio('What filter to use?: ', ['Original','Greyscale', 'Blur', 'Motion Blur','Pencil Sketch','Sharpen'])
        conv_img = np.array(image.convert('RGB'))

        if filter == 'Original':
            st.image(image, width=300)
        
        elif filter=='Greyscale':
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
            sig = st.sidebar.slider("Intensity: ",0.1,2.0,1.)
            kernel = gkern(10, sig)
            blur_image = cv2.filter2D(conv_img, -1, kernel)
            st.image(blur_image, channels='RGB', width=300)
        
        elif filter=='Motion Blur':
            i = st.sidebar.slider("Intensity: ",1,20,5)
            kernel = np.zeros((i,i))
            np.fill_diagonal(kernel, 1)
            kernel /= kernel.sum()
            mblur_image = cv2.filter2D(conv_img, -1, kernel)
            st.image(mblur_image, channels='RGB', width=300)
        
        # elif filter=='Vignette':
        #     image.save('vig_img.jpg')
        #     conv_img = cv2.imread('vig_img.jpg')
        #     rows, cols = conv_img.shape[:2]

        #     # generating vignette mask using Gaussian kernels
        #     kernel_x = cv2.getGaussianKernel(cols,200)
        #     kernel_y = cv2.getGaussianKernel(rows,200)
        #     #rowsXcols
        #     kernel = kernel_y * kernel_x.T

        #     #Normalizing the kernel
        #     kernel = kernel/np.linalg.norm(kernel)

        #     #Genrating a mask to image
        #     mask = 255 * kernel
        #     output = np.copy(conv_img)

        #     # applying the mask to each channel in the input image
        #     for i in range(3):
        #         output[:,:,i] = output[:,:,i] * mask
        #     st.image(output)

        elif filter=='Pencil Sketch':
            gray_scale = cv2.cvtColor(conv_img, cv2.COLOR_RGB2GRAY)
            inv_gray = 255 - gray_scale
            slider = st.sidebar.slider('Intensity: ', 25, 255, 125, step=2)
            blur_image = cv2.GaussianBlur(inv_gray, (slider,slider), 0, 0)
            sketch = cv2.divide(gray_scale, 255 - blur_image, scale=256)
            st.image(sketch, width=300) 
        
        elif filter=='Sharpen':
            i = st.sidebar.selectbox(label='Type: ', options=['Normal','Subtle','Excessive'])
            if i == 'Normal':
                kernel = np.array([[-1.,-1.,-1.],[-1.,9.,-1.],[-1.,-1.,-1.]])
            elif i == 'Subtle':
                kernel = np.array([[-1.,-1.,-1.,-1.,-1.],[-1., 2.,2.,2.,-1.],[-1.,2.,8.,2.,-1.],[-1.,2.,2.,2.,-1.],[-1.,-1.,-1.,-1.,-1.]])
            else:
                kernel = np.array([[1.,1.,1.],[1.,-7.,1.],[1.,1.,1.]])
            
            kernel /= kernel.sum()
            sharp_img = cv2.filter2D(conv_img, -1, kernel)
            st.image(sharp_img, channels='RGB', width=300)






st.markdown("<p style='text-align:center'>You can right-click and save the image if you want to!</p>",unsafe_allow_html=True)


footer="""
<style>
a:link , a:visited{
color: white;
background-color: transparent;
text-decoration: underline;
}
.footer {
display: flex;
justify-content: left;
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: #0E1117;
color: white;
text-align: left;
}
</style>
<div class="footer">
<p>Developed by <a style='text-align: left;' href="https://github.com/horizon3902/" target="_blank">Kshitij Agarkar</a></p>
<a style="margin-left: 73%" href="https://github.com/horizon3902/fun-with-filters" data-color-scheme="no-preference: dark; light: light; dark: dark;" aria-label="Watch horizon3902/movie-recommender-salsa on GitHub">View on Github</a>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)


