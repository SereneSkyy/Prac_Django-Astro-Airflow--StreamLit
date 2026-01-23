FROM apache/airflow:2.9.3

# Copy requirements file
COPY requirements.txt /requirements.txt

# Install the packages
RUN pip install --no-cache-dir -r /requirements.txt

# ---- Bake Hugging Face model into the image ----
USER root

# Make a cache dir that airflow user can write/read
RUN mkdir -p /opt/airflow/.hf && chown -R airflow: /opt/airflow/.hf

USER airflow

# Point HF/Transformers cache to that dir
ENV HF_HOME=/opt/airflow/.hf \
    HUGGINGFACE_HUB_CACHE=/opt/airflow/.hf/hub \
    TRANSFORMERS_CACHE=/opt/airflow/.hf/transformers

# Download model + tokenizer during build (cached into /opt/airflow/.hf)
RUN python -c "from transformers import BertTokenizer, BertModel; \
BertTokenizer.from_pretrained('bert-base-uncased'); \
BertModel.from_pretrained('bert-base-uncased')"

# Optional: make runtime fully offline (prevents any download attempts)
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1
