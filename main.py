from tensorflow.keras.utils import pad_sequences
from tensorflow import keras
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import streamlit as st
import json
model = keras.models.load_model("model2.keras")
with open("tokenizer_in2.json" , 'r') as f:
    tokenizer_in = tokenizer_from_json(json.load(f))
with open("tokenizer_out2.json" , 'r') as f1:
    tokenizer_out = tokenizer_from_json(json.load(f1))
st.title("English to Hindi Translator 🌍➡️🇮🇳")

text = st.text_input("ENTER HERE")

if st.button("predict"):
    def prepare_input(sentence, tokenizer, max_len):
        seq = tokenizer.texts_to_sequences([sentence])
        seq = pad_sequences(seq, maxlen=max_len, padding="post")
        return seq
    import numpy as np
    def translate_simple(sentence, model, tokenizer_in, tokenizer_out):

        sentence = "<start> " + sentence + " <end>"

        enc_input = prepare_input(sentence, tokenizer_in, 20)

        start_id = tokenizer_out.word_index["start"]
        dec_input = np.zeros((1, 19))
        dec_input[0, 0] = start_id

        preds = model.predict([enc_input, dec_input])

        word_ids = np.argmax(preds[0], axis=-1)

        result = []
        for wid in word_ids:
            word = tokenizer_out.index_word.get(wid, "")
            if word == "<end>":
                break
            if word not in ["<start>", ""]:
                result.append(word)

        return " ".join(result)

    st.success(translate_simple(text,model,tokenizer_in,tokenizer_out))
