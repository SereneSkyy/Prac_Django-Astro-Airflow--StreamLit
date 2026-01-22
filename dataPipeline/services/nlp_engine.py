import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import TfidfModel, LdaModel, LsiModel, Nmf
from gensim.models.coherencemodel import CoherenceModel
from schemas.etl_schema import execute_processed_comments_sql, insert_cleaned_comments

# Initialization
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class NLPEngine:
    @staticmethod
    def save_processed_comments(proc_texts, ids, cursor):
        try:
            cursor.execute(execute_processed_comments_sql)

            # turn token lists into a single string per comment (easy to store/search)
            processed_strings = [" ".join(tokens) for tokens in proc_texts]

            values = [(cid, ptxt) for cid, ptxt in zip(ids, processed_strings)]

            cursor.executemany(insert_cleaned_comments, values)

        except Exception as e:
            print(f"[DB Error] {e}")

    @staticmethod
    def clean_comments(comment_texts, ids, cursor):
        lem = WordNetLemmatizer()
        stops = set(stopwords.words('english'))
        processed = []
        for text in comment_texts:
            clean = re.sub(r'http\S+|www\S+|<.*?>|[^a-zA-Z\s]', '', str(text).lower())
            tokens = [lem.lemmatize(w) for w in clean.split() if w not in stops and len(w) > 2]
            processed.append(tokens)
        return NLPEngine.save_processed_comments(processed, ids, cursor)

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
            
        return winner, scores