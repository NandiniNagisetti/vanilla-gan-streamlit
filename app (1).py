import streamlit as st
import numpy as np
from inference import load_generator_model, generate_images

st.title("Vanilla GAN Image Generator")

st.write("Generate synthetic handwritten digits using GAN")

num_images = st.slider("Number of images", 1, 10, 5)

if st.button("Generate"):
    generator = load_generator_model()
    images = generate_images(generator, num_images)

    for img in images:
        st.image(img.squeeze(), width=150)
