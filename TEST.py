# install the necessary libraries using pip
!pip install transformers
!pip install sentencepiece
!pip install spacy
!python -m spacy download es_core_news_sm
!python -m spacy download en_core_web_sm

# import the necessary libraries
import torch
from transformers import pipeline

# get user input for the Spanish text to summarize and translate
input_text = input("Enter Spanish text to summarize and translate: ")

# set up the summarization pipeline
summarization_pipeline = pipeline(
    "summarization",
    model="t5-base",
    tokenizer="t5-base",
    device=0 if torch.cuda.is_available() else -1
)

# generate the summary using the pipeline
max_chunk_size = 500
input_chunks = [input_text[i:i+max_chunk_size] for i in range(0, len(input_text), max_chunk_size)]
summary_text = ""
for chunk in input_chunks:
    summary = summarization_pipeline(chunk, max_length=200, min_length=30, do_sample=False)
    summary_text += summary[0]['summary_text']

# print the Spanish summary in lines of 20 words
print("Spanish summary:")
words = summary_text.split()
for i in range(0, len(words), 20):
    print(' '.join(words[i:i+20]))

# set up the translation pipeline
translation_pipeline = pipeline(
    "translation_es_to_en",
    model="Helsinki-NLP/opus-mt-es-en",
    tokenizer="Helsinki-NLP/opus-mt-es-en",
    device=0 if torch.cuda.is_available() else -1
)

# generate the English translation using the pipeline
translation = translation_pipeline(summary_text)

# print the English translation in lines of 20 words
print("\nEnglish translation:")
words = translation[0]['translation_text'].split()
for i in range(0, len(words), 20):
    print(' '.join(words[i:i+20]))
