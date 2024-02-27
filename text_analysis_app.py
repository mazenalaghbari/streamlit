import streamlit as st
from docx import Document
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from heapq import nlargest
from PyPDF2 import PdfReader
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from googletrans import Translator
from transformers import pipeline
import spacy

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Function to extract keywords from text
def extract_keywords(text, num_keywords=5):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    word_freq = Counter(filtered_words)
    return dict(nlargest(num_keywords, word_freq.items(), key=lambda item: item[1]))

# Function to extract main sentences from text
def extract_main_sentences(text, num_sentences=3):
    sentences = sent_tokenize(text)
    return nlargest(num_sentences, sentences, key=len)

def process_docx(uploaded_file):
    document = Document(uploaded_file)
    full_text = ""
    for para in document.paragraphs:
        full_text += para.text + "\n"
    return full_text

def process_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text()
    return full_text

def generate_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Change 3 to the number of sentences you want in the summary
    return ' '.join([str(sentence) for sentence in summary])

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return polarity

def plot_keyword_frequency(keywords):
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(keywords)), [freq for _, freq in keywords.items()], align='center')
    plt.yticks(range(len(keywords)), list(keywords.keys()))
    plt.xlabel('Frequency')
    plt.ylabel('Keyword')
    plt.title('Keyword Frequency')
    st.pyplot(plt)

def plot_topic_distribution(document_topics):
    plt.figure(figsize=(8, 6))
    topic_weights = np.sum(document_topics, axis=0)
    plt.bar(range(len(topic_weights)), topic_weights, align='center')
    plt.xlabel('Topic')
    plt.ylabel('Weight')
    plt.title('Topic Distribution')
    st.pyplot(plt)

def perform_topic_modeling(text):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    document_topics = lda.fit_transform(X)
    return document_topics

def translate_text(text, target_language='en'):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text

def perform_translation(text):
    translated_text = translate_text(text)
    return translated_text

def analyze_entities(text):
    entities = [(ent.text, ent.label_) for ent in nlp(text).ents]
    return entities

def explore_page():
    st.header("Text Analysis App")
    # Add Google logo
    st.image("https://logowik.com/content/uploads/images/abc-australian-broadcasting-corporation2950.jpg", caption="", use_column_width=True)

    # Dummy content
    st.write("Here is some dummy content for the index page:")

    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx"])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            full_text = process_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            full_text = process_docx(uploaded_file)

        keywords = extract_keywords(full_text)
        main_sentences = extract_main_sentences(full_text)
        summary = generate_summary(full_text)
        sentiment_score = analyze_sentiment(full_text)
        document_topics = perform_topic_modeling(full_text)
        translated_text = perform_translation(full_text)
        entities = analyze_entities(full_text)

        st.subheader("Keywords:")
        st.write(keywords)
        plot_keyword_frequency(keywords)

        st.subheader("Main Sentences:")
        for sentence in main_sentences:
            st.write(sentence)

        st.subheader("Summary:")
        st.write(summary)

        st.subheader("Sentiment Analysis:")
        st.write(f"Sentiment Score: {sentiment_score:.2f}")

        st.subheader("Topic Modeling:")
        plot_topic_distribution(document_topics)

        st.subheader("Translation:")
        st.write(translated_text)

        st.subheader("Named Entities:")
        st.write(entities)

        st.subheader("Explore More with Keywords:")
        for keyword in keywords:
            if st.button(f"Explore '{keyword}'"):
                st.subheader(f"Sentences containing '{keyword}' in Original Text:")
                sentences_containing_keyword = [sentence for sentence in sent_tokenize(full_text) if keyword in word_tokenize(sentence)]
                for sentence in sentences_containing_keyword:
                    st.write(sentence)

if __name__ == "__main__":
    explore_page()
