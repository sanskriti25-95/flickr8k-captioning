"""Streamlit demo: upload an image, get a caption.   streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import streamlit as st
import keras
import cap

st.title("Image captioning (CNN + LSTM, Flickr8k)")
enc = st.selectbox("Encoder", list(cap.ENCODERS))
k = st.slider("Beam width (1 = greedy)", 1, 7, 3)


@st.cache_resource(max_entries=1)  # keep only one encoder in memory (free hosting has little RAM)
def load(enc):
    ctor, size, pp = cap.ENCODERS[enc]
    base = getattr(keras.applications, ctor)(include_top=False, weights="imagenet", pooling="avg")
    return base, cap.Captioner(enc)


up = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
if up:
    from PIL import Image
    img = Image.open(up).convert("RGB")
    st.image(img)
    base, cp = load(enc)
    _, size, pp = cap.ENCODERS[enc]
    x = np.asarray(img.resize((size, size)), np.float32)[None]
    f = base.predict(getattr(keras.applications, pp).preprocess_input(x), verbose=0)[0]
    st.subheader(cp.greedy_batch(f[None])[0] if k == 1 else cp.beam(f, k))
