# Import stuff
import cv2
# from matplotlib import pyplot as plt
# %matplotlib inline
# plt.rcParams['image.cmap']='gray'
import streamlit as st
import numpy as np

st.title("Awesome Edge Detector")
st_file = st.file_uploader("Upload image for edge detection", type=['jpg', 'jpeg', 'png'])



# Function to pre-process image input
def pre_process(image):
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_gaussian5 = cv2.GaussianBlur(img_gray,(5,5),0,0)
    return img_gray, img_gaussian5

# Function to perform canny edge detection
def detect_edge(input_img,lthresh):
    org_edges = cv2.Canny(img_gray,threshold1=lthresh, threshold2=200)
    blur_edges = cv2.Canny(img_gaussian5,threshold1=lthresh, threshold2=200)
    return org_edges, blur_edges

if st_file is not None:
    # convert uploaded image to a numpy array
    raw_bytes = np.asarray(bytearray(st_file.read()), dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    # run function created earlier
    img_gray, img_gaussian5 = pre_process(image)
    
    # Create placeholders to display input and output images.
    placeholders = st.columns(3)
    # Display Input image in the first placeholder.
    placeholders[0].image(image, channels='BGR')
    placeholders[0].markdown("Input Image")

    # Display Grayscale image in the second placeholder
    placeholders[1].image(img_gray, channels='GRAY')
    placeholders[1].markdown("Grayscale Image")

    # Display Gaussian Blurred image in the third placeholder
    placeholders[2].image(img_gaussian5, channels='GRAY')
    placeholders[2].markdown("Gaussian Blurred Image, 5x5 Kernal")

    lthresh = st.slider("Set Lower Minimum Threshold for Hysteresis",min_value=0,max_value=200,value=50)

    st.title("Edge Detection Results:")

    # Run detect_edge function
    org_edges, blur_edges = detect_edge(img_gray,lthresh=lthresh)

    # show edge detection results
    results1 = st.columns(2)

    results1[0].markdown(f"Original, {lthresh} Lower Thresh")
    results1[0].image(org_edges, channels='GRAY')

    results1[1].markdown(f"Gaussian Pre-Processing, {lthresh} Lower Thresh")
    results1[1].image(blur_edges, channels='GRAY')

