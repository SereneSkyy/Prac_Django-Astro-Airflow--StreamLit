FROM apache/airflow:2.9.3

# Copy requirements file
COPY requirements.txt /requirements.txt

# Install the packages
RUN pip install --no-cache-dir -r /requirements.txt

# ---- Bake BERT and NLTK into the image ----
USER root
RUN mkdir -p /bert_model && chown -R airflow: /bert_model

USER airflow

# 1. Pre-download BERT into a static folder (Fast & Offline)
RUN python -c "from transformers import BertTokenizer, BertModel; \
t = BertTokenizer.from_pretrained('bert-base-uncased'); \
m = BertModel.from_pretrained('bert-base-uncased'); \
t.save_pretrained('/bert_model'); \
m.save_pretrained('/bert_model')"

# 2. Pre-download NLTK data (Prevents LookupError/Network errors)
RUN python -m nltk.downloader stopwords wordnet omw-1.4

RUN python -m spacy download en_core_web_sm

# Force Transformers to run in offline mode
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1