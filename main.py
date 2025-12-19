from tensorflow.keras.utils import pad_sequences
from tensorflow import keras
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import streamlit as st
import json
import numpy as np

@st.cache_resource
def load_model():
    from tensorflow import keras
    model = keras.models.load_model("model2.keras")
    return model

model = load_model()

st.title("English to Hindi Translator 🌍➡️🇮🇳")
@st.cache_resource
def load_tokenizer(path):
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    with open(path, 'r', encoding='utf-8') as f:
        return tokenizer_from_json(f.read().strip())
tokenizer_in = load_tokenizer("tokenizer_in2.json")
tokenizer_out = load_tokenizer("tokenizer_out2.json")


text = st.text_input("ENTER HERE")


def translate_simple(sentence, model, tokenizer_in, tokenizer_out, max_len=20):
    sentence = "<start> " + sentence + " <end>"
    enc_input = pad_sequences(
        tokenizer_in.texts_to_sequences([sentence]), maxlen=max_len, padding="post"
    )

    # Start decoding
    start_id = tokenizer_out.word_index["start"]
    dec_input = [start_id]
    result = []

    for _ in range(max_len):
        # Pad decoder input
        dec_input_padded = pad_sequences([dec_input], maxlen=max_len-1, padding="post")
        preds = model.predict([enc_input, dec_input_padded], verbose=0)
        next_id = np.argmax(preds[0, len(dec_input)-1, :])
    
        if next_id == tokenizer_out.word_index["end"]:
            break
    
        word = tokenizer_out.index_word.get(next_id, "")
        if word != "":
            result.append(word)
    
        dec_input.append(next_id)
    
    return " ".join(result)
if st.button("predict"):
    st.success(translate_simple(text, model, tokenizer_in, tokenizer_out))
