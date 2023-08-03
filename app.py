import streamlit as st
import pickle
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import time
import cv2

st.title("Image Caption Genrator")

uploaded_file = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])

if uploaded_file is not None:
    st.image(uploaded_file)
    st.write("Image Uploaded Successfully")

time.sleep(25)

model = pickle.load(open("model.pkl", "rb"))
feature_extractor = pickle.load(open("feature.pkl", "rb"))
tokenizer = pickle.load(open("token.pkl", "rb"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

caption=predict_step([uploaded_file])
st.subheader("Caption of the image :")
st.write(caption)
