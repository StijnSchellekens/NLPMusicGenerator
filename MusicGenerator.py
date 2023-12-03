import streamlit as st
from src.bert import get_bert_embedding, get_closest_feature
from src.music_vae import getSample

feature_embeddings = {
    "calm_soothing_1": get_bert_embedding("a calm and soothing melody"),
    "calm_soothing_2": get_bert_embedding("gentle relaxing music for a quiet evening"),
    "upbeat_energetic_1": get_bert_embedding("a fast-paced, energetic beat"),
    "upbeat_energetic_2": get_bert_embedding("uplifting and lively music for working out"),
    "ambient_relaxing": get_bert_embedding("ambient soundscape for stress relief"),
}


st.title("Music Generator")

st.error("This is a demo of how NLP could be used to influence music generation. Generated music is of low quality and might not sound all that different with differing descriptions due to how limited the influence on the model is for now.")
st.error("Currently the only factors that are influenced by input data are the temperature used during generation and the waveform type, but can be expanded on in the future.")

user_input = st.text_input("Describe the general mood of the melody you want to create", "create an ambient melody suitable for in the background.")



generate = st.button("Generate")

if generate:
    user_embedding = get_bert_embedding(user_input)

    closest_feature = get_closest_feature(feature_embeddings, user_embedding)

    st.info(f"Generating a(n) {closest_feature} song...")
    st.audio(getSample(closest_feature), format='audio/wav')
