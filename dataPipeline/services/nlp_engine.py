import re
import nltk
import torch
import torch.nn as nn
import numpy as np
from psycopg2.extras import execute_values
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
noun_nlp = spacy.load("en_core_web_sm")

# --- DYNAMIC ASSET DOWNLOAD (Fixes the LookupError) ---
# try:
#     nltk.data.find('corpora/stopwords')
#     nltk.data.find('corpora/wordnet')
# except LookupError:``
#     nltk.download('stopwords', quiet=True)
#     nltk.download('wordnet', quiet=True)
#     nltk.download('omw-1.4', quiet=True)

class SentimentLSTM(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=3):
        super(SentimentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) 
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.softmax(self.fc(hidden))

class NLPEngine:
    @staticmethod
    def merge_ents(text: str) -> str:
        doc = noun_nlp(text)
    
        # Map start index -> entity span
        start2ent = {ent.start: ent for ent in doc.ents}
    
        out = []
        i = 0
        while i < len(doc):
            if i in start2ent:
                ent = start2ent[i]
                out.append(ent.text.replace(" ", "_"))
                i = ent.end
            else:
                out.append(doc[i].text)
                i += 1
    
        return " ".join(out)

    @staticmethod
    def clean_comments(comment_texts, ids, cursor):
        lem = WordNetLemmatizer()
        stops = set(stopwords.words("english"))
        processed_data = []

        for cid, text in zip(ids, comment_texts):
            raw = str(text)

            # light cleaning first (keep structure for NER)
            raw = re.sub(r"http\S+|www\S+|<.*?>", " ", raw)
            raw = re.sub(r"\s+", " ", raw).strip()

            # merge multi-word entities
            merged = NLPEngine.merge_ents(raw)

            # now do your stronger cleanup for tokens
            clean = merged.lower()
            clean = re.sub(r"[^a-zA-Z_\s]", " ", clean)  # keep underscores
            clean = re.sub(r"\s+", " ", clean).strip()

            tokens = [lem.lemmatize(w) for w in clean.split()
                      if w not in stops and len(w) > 2]

            processed_data.append((cid, " ".join(tokens), "Pending"))

        from schemas.etl_schema import insert_cleaned_comments
        execute_values(cursor, insert_cleaned_comments, processed_data)
        return True

    @staticmethod
    def run_lstm_inference(ids, cursor):
        """Builds sequences from BERT tables and predicts sentiment."""
        all_sequences = []
        for cid in ids:
            # Reconstructing word sequence using BERT tables
            cursor.execute("""
                SELECT wv.word_vec FROM airflow.words_occur wo
                JOIN airflow.words_vec wv ON wo.word = wv.word AND wo.topic = wv.topic
                WHERE %s = ANY(wo.word_cmt_ids) LIMIT 10;
            """, (cid,))
            vectors = [np.array(row[0]) for row in cursor.fetchall()]
            
            while len(vectors) < 10:
                vectors.append(np.zeros(768))
            all_sequences.append(vectors[:10])

        X = torch.tensor(np.array(all_sequences), dtype=torch.float32)
        model = SentimentLSTM() 
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            predictions = torch.argmax(outputs, dim=1)
        
        mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        results = [(mapping[p.item()], cid) for p, cid in zip(predictions, ids)]

        execute_values(cursor, """
            UPDATE airflow.cleaned_comments SET sentiment = val.s
            FROM (VALUES %s) AS val(s, cid)
            WHERE comment_id = val.cid
        """, results)