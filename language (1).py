import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect

st.title("🌐 AI Language Translator")

text = st.text_area("Enter text to translate:")
target_lang = st.selectbox("Select target language", ["fr", "de", "hi", "es"])

# Map langdetect codes to Helsinki codes
LANG_MAP = {
    "en": "en",
    "fr": "fr",
    "de": "de",
    "hi": "hi",
    "es": "es",
    "it": "it",
    "ja": "ja",
    "ko": "ko",
    "ru": "ru"
}

if st.button("Translate"):
    if not text.strip():
        st.warning("Please enter some text to translate.")
    else:
        src_lang = detect(text)
        src_lang = LANG_MAP.get(src_lang, "en")  # fallback to English if unsupported
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{target_lang}"

        st.write(f"🔍 Detected source language: `{src_lang}`")
        st.write(f"📦 Using model: `{model_name}`")

        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)

            inputs = tokenizer(text, return_tensors="pt", padding=True)
            translated_tokens = model.generate(**inputs)
            translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            st.success(translation)

        except OSError:
            st.error(f"⚠️ Sorry, translation from `{src_lang}` to `{target_lang}` is not supported.")
