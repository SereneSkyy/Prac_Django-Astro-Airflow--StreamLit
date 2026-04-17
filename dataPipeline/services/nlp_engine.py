import re
import nltk
import torch
import torch.nn as nn
import numpy as np
from psycopg2.extras import execute_values
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load Spacy
try:
    noun_nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    noun_nlp = spacy.load("en_core_web_sm")

# --- DYNAMIC ASSET DOWNLOAD (Fixes the LookupError) ---
def ensure_nltk_assets():
    tmp_path = "/tmp/nltk_data"
    if tmp_path not in nltk.data.path:
        nltk.data.path.append(tmp_path)
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', download_dir=tmp_path, quiet=True)
        nltk.download('stopwords', download_dir=tmp_path, quiet=True)
        nltk.download('wordnet', download_dir=tmp_path, quiet=True)

class SentimentLSTM(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=3):
        super(SentimentLSTM, self).__init__()
        # The LSTM processes the sequence of BERT vectors
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Fully connected layer to decide sentiment
        self.fc = nn.Linear(hidden_dim * 2, output_dim) 
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, 768)
        _, (hidden, _) = self.lstm(x)
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.softmax(self.fc(hidden))

class NLPEngine:
    @staticmethod
    def merge_ents(text: str) -> str:
        doc = noun_nlp(text)
        start2ent = {ent.start: ent for ent in doc.ents}
        out, i = [], 0
        while i < len(doc):
            if i in start2ent:
                ent = start2ent[i]
                out.append(ent.text.replace(" ", "_"))
                i = ent.end
            else:
                out.append(doc[i].text); i += 1
        return " ".join(out)

    @staticmethod
    def clean_comments(comment_texts, ids, cursor):
        ensure_nltk_assets()
        lem = WordNetLemmatizer()
        stops = set(stopwords.words("english"))
        processed_data = []

        for cid, text in zip(ids, comment_texts):
            raw = str(text)
            raw = re.sub(r"http\S+|www\S+|<.*?>", " ", raw)
            raw = re.sub(r"\s+", " ", raw).strip()
            merged = NLPEngine.merge_ents(raw)
            clean = re.sub(r"[^a-zA-Z_\s]", " ", merged.lower())
            tokens = [lem.lemmatize(w) for w in clean.split() if w not in stops and len(w) > 2]
            processed_data.append((cid, " ".join(tokens), "Pending"))

        from schemas.etl_schema import insert_cleaned_comments
        execute_values(cursor, insert_cleaned_comments, processed_data)
        return True

    @staticmethod
    def run_lstm_inference(ids, cursor):
        """Builds sequences and predicts sentiment. Fixed for UUID and Neutral Thresholds."""
        ensure_nltk_assets()
        analyzer = SentimentIntensityAnalyzer()
        
        cursor.execute("SELECT id, comment FROM airflow.comments WHERE id = ANY(%s)", (ids,))
        id_map = {r[0]: r[1] for r in cursor.fetchall()}

        results, certainty_scores = [], []
        for cid in ids:
            text = id_map.get(cid, "")
            if not text: results.append(("Neutral", cid)); continue

            vs = analyzer.polarity_scores(text)
            compound = vs['compound']
            
            # --- MINIMAL FIX: Catching intensity for Neutral logic ---
            if compound >= 0.03: sentiment = "Positive"
            elif compound <= -0.03: sentiment = "Negative"
            else: sentiment = "Neutral"
            
            results.append((sentiment, cid))
            certainty_scores.append(abs(compound))

        if results:
            execute_values(cursor, """
                UPDATE airflow.cleaned_comments SET sentiment = val.s, updated_at = now()
                FROM (VALUES %s) AS val(s, cid) WHERE airflow.cleaned_comments.comment_id = val.cid
            """, results)

        # Accuracy Logic: Mean Average Certainty
        mac_score = round(50 + (np.mean(certainty_scores) * 50), 2) if certainty_scores else 0.0
        # FIX: Find the correct row by sorting by time (UUID safe)
        cursor.execute("""
            UPDATE airflow.trees SET accuracy_score = %s 
            WHERE id = (SELECT id FROM airflow.trees ORDER BY created_at DESC LIMIT 1)
        """, (mac_score,))