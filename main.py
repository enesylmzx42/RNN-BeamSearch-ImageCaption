import os
import streamlit as st
import tensorflow as tf
import yaml
import gdown

from data_utils.preprocessor import Preprocessor
from model_utils.feature_extractor import FeatureExtractor
from model_utils.tokenizer import Tokenizer
from RNN_model.rnn_model import RNNImageCaptioner
from keras.applications import EfficientNetV2B3
from model_utils.utils import load_image

WEIGHTS_PATH = "./weights/model4-ep.28-loss.2.34.weights.h5"


def app_render():
    st.title("Image Captioner")
    uploaded_image = st.file_uploader("upload a image", type=["jpg", "jpeg", "png"])
    st.markdown("<h4 style='font-family: Arial;'>Generate Options</h4>", unsafe_allow_html=True)
    method = st.radio(
        "Select Generation Algorithm",
        horizontal=True,
        options=["Greedy", "Beam Search"],
    )
    if method == "Greedy":
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0)
    elif method == "Beam Search":
        kbeams = st.number_input(
            "Number of Beams (1 to 10) ", min_value=1, max_value=10, value=5
        )

    if uploaded_image is not None:
        _, center, _ = st.columns(3)
        with center:
            st.image(uploaded_image, width=300)
        image = load_image(uploaded_image.read())
        if st.button("Generate Captions", use_container_width=True):
            info = st.info("generating...")
            if method == "Greedy":
                caption = model.greedy_gen(image, temperature=temperature)
            elif method == "Beam Search":
                caption = model.beam_search_gen(image, Kbeams=kbeams)

            if caption:
                info.empty()
                st.write("Generated Captions")
                st.success(caption[0])
            else:
                info.error("Error")




def load_model(weights_path):
    if config["model"]["type"] == "rnn":
        model = RNNImageCaptioner(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            **config["model"]["params"],
        )
        model.build(
            input_shape=((None, *config["model"]["img_features_shape"]), (None, None))
        )
        model.load_weights(weights_path)

    elif config["model"]["type"] == "transformer":
        raise NotImplementedError("model can not implemented")
    return model


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    config = load_config("./LSTM_GRU/lstm_config.yaml")

    pooling = None
    if config["model"]["params"]["pooling"] == False:
        pooling = "avg"
    effnet = EfficientNetV2B3(include_top=False, pooling=pooling)
    effnet.trainable = False
    feature_extractor = FeatureExtractor(
        feature_extractor=effnet, features_shape=config["model"]["img_features_shape"]
    )

    tokenizer = Tokenizer.from_vocabulary(
        path="./vocab/tokenizer_vocab.pkl",
        standardize=Preprocessor(),
        ragged=True,
    )
    model = load_model(WEIGHTS_PATH)
    app_render()