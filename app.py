import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Title of the Streamlit app
st.title("Image to Text with Stable Diffusion")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)

    # Display the image in the Streamlit app
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Perform the image to text conversion
    st.write("Generating caption...")
    inputs = processor(image, return_tensors="pt").to("cpu")

    with torch.no_grad():
        generated_ids = model.generate(pixel_values=inputs.pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Display the generated text
    st.write("Generated Caption:")
    st.write(generated_text)

