import re
import nltk
import psycopg2
from psycopg2.extras import execute_values
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import TfidfModel, LdaModel, LsiModel, Nmf
from gensim.models.coherencemodel import CoherenceModel

# Initialization
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class NLPEngine:
    @staticmethod
    def clean_comments(text_list):
        lem = WordNetLemmatizer()
        stops = set(stopwords.words('english')).union({'phone', 'video', 'get', 'like', 'would'})
        processed = []
        for text in text_list:
            clean = re.sub(r'http\S+|www\S+|<.*?>|[^a-zA-Z\s]', '', str(text).lower())
            tokens = [lem.lemmatize(w) for w in clean.split() if w not in stops and len(w) > 2]
            processed.append(tokens)
        return processed

    @staticmethod
    def compare_models(tfidf_corpus, dictionary, tokens, n_topics=3):
        """Runs competition between LDA, LSA, and NMF based on U_Mass Coherence."""
        models = {
            'LDA': LdaModel(tfidf_corpus, id2word=dictionary, num_topics=n_topics, random_state=42),
            'LSA': LsiModel(tfidf_corpus, id2word=dictionary, num_topics=n_topics),
            'NMF': Nmf(tfidf_corpus, id2word=dictionary, num_topics=n_topics, random_state=42)
        }
        
        scores = {}
        for name, model in models.items():
            try:
                # U_Mass is best for small-to-medium datasets
                cm = CoherenceModel(model=model, corpus=tfidf_corpus, dictionary=dictionary, coherence='u_mass')
                scores[name] = float(cm.get_coherence())
            except:
                scores[name] = -20.0 # Fallback
        
        # Determine winner (closest to 0 is best)
        winner_name = max(scores, key=scores.get)
        return winner_name, scores

    @staticmethod
    def run_full_pipeline(english_data):
        """Processes English comments and saves only the preprocessed text."""
        if not english_data:
            return "NO_EN", {}

        ids = [item['id'] for item in english_data]
        texts = [item['comment'] for item in english_data]
        
        # cleaning and tokenization
        tokens = NLPEngine.clean_comments(texts)
        cleaned_strings = [" ".join(t) if t else "[no meaningful content]" for t in tokens]

        # Topic Modeling 
        # We use TF-IDF internally to improve accuracy, but we don't save it.
        dictionary = corpora.Dictionary(tokens)
        winner = "None"
        scores = {}
        
        if len(dictionary) > 0:
            bow = [dictionary.doc2bow(t) for t in tokens]
            tfidf_model = TfidfModel(bow)
            tfidf_corpus = tfidf_model[bow]
            winner, scores = NLPEngine.compare_models(tfidf_corpus, dictionary, tokens, n_topics=3)

        # save only cleaned text to DB
        try:
            conn = psycopg2.connect(host="127.0.0.1", database="data_pipeline", user="admin", password="admin")
            cur = conn.cursor()
            
            data_to_update = list(zip(cleaned_strings, ids))
            
            # Optimized SQL No vector_tfidf, just the cleaned text
            query = """
                UPDATE airflow.processed_comments 
                SET cleaned_text = val.txt, updated_at = CURRENT_TIMESTAMP
                FROM (VALUES %s) AS val (txt, cid) 
                WHERE comment_id = val.cid;
            """
            execute_values(cur, query, data_to_update)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB Error] {e}")
            
        return winner, scores