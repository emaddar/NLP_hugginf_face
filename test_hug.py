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


st.write(" ")
st.write(" ")
st.write(" ")

st.write("flair/ner-french")

from flair.data import Sentence
from flair.models import SequenceTagger

# load tagger
tagger = SequenceTagger.load("flair/ner-french")

sentence = Sentence(example)

# predict NER tags
tagger.predict(sentence)

# print sentence
st.write(sentence)

# print predicted NER spans
st.write('The following NER tags are found:')
# iterate over entities and print
for entity in sentence.get_spans('ner'):
    st.write(entity)







