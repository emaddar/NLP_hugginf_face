from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import streamlit as st

st.header("test")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = st.text_area("you text")

ner_results = nlp(example)

import pandas as pd

# assuming your data is stored in a variable called 'data'
df = pd.DataFrame(ner_results)

st.write(df)
