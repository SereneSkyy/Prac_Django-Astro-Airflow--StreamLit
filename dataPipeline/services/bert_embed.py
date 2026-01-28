import json
import torch
import numpy as np
import os  
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

class TaxonomyAndTreeBuilder:
    def __init__(self, threshold, pro_cmts, target_words):
        # --- OFFLINE FIX START ---
        # Look for the baked-in folder from the Dockerfile
        model_path = "/bert_model" if os.path.exists("/bert_model") else 'bert-base-uncased'
        is_offline = os.path.exists("/bert_model")

        self.tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=is_offline)
        self.model = BertModel.from_pretrained(model_path, local_files_only=is_offline)
        # --- OFFLINE FIX END ---
        
        self.pro_cmts = pro_cmts
        self.target_words = target_words
        self.threshold = threshold
    
    def calculate_dynamic_abstractness(self, word_occur, cmt_vecs):
        """
            word_occurrences: list of vectors for a specific word from different cmts
            cmt_vecs: the summarized vector for each comment
        """
        if not word_occur:
            return 0.5, np.zeros(768)
        
        # average the vectors to get one 'global' word embedding for the tree
        mean_word_vec = np.mean(word_occur, axis=0)
        
        # calculate similarities to ALL comments to see 'distributional breadth'
        sims = cosine_similarity([mean_word_vec], cmt_vecs)[0]
        
        # abstractness = 1 - Variance (Lower variance = more abstract/consistent)
        abs_score = 1 - np.var(sims)
        return abs_score, mean_word_vec

    def _tokenizer(self):
        # Joined as values if pro_cmts is a dict
        cmts = [" ".join(words) for words in self.pro_cmts.values()]
        inputs = self.tokenizer(cmts, return_tensors="pt", padding=True, truncation=True) # reutrn pytorch tensors
        return inputs

    def run_bert(self):
        inputs = self._tokenizer()
        outputs = self.model(**inputs)

        # get last hidden state
        # vector for each word (cmts separate)
        last_hidden = outputs.last_hidden_state      # (batch, seq_len, 768)
        mask = inputs["attention_mask"].unsqueeze(-1)    # (batch, seq_len, 1)

        # mean pooling (ignore PAD tokens)
        summed = (last_hidden * mask).sum(dim=1)   # (batch, 768)
        
        # mask.sum(dim=1) is for knowing how many real words are present as 1's for real word and 0's for padding
        # .clamp is there to avoid division by zero. It ensures the divisor is at least a tiny value but not zero
        counts = mask.sum(dim=1).clamp(min=1e-9)         # (batch, 1)
        comment_vecs = summed / counts                   # (batch, 768)
        return comment_vecs, last_hidden

    def create_tree(self, word_metadata, word_vectors, imp_score, max_nodes=25):
        """
        Adaptive Pruning: Keeps only Top-K significant words.
        Automatically removes sub-branches if parent is missing.
        """
        # 1. Adapt pruning floor based on sample size (0.01 floor if many comments, 0 if few)
        floor = 0.01 if len(self.pro_cmts) > 10 else 0.0
        
        # Sort by importance and take top N
        sorted_active = sorted(imp_score.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        active_words = [w[0] for w in sorted_active if w[1] >= floor]
        
        # sort words by abstractness (Descending: Highest score first)
        active_sorted = sorted(active_words, key=lambda w: word_metadata[w]["abs_score"], reverse=True)
        
        tree = {word: [] for word in active_sorted}
        roots = []

        # 2. Branching Sensitivity: Use a slightly lower threshold internaly if passed threshold is too high
        # This prevents the "Line-shaped tree" by allowing multiple branches per node
        effective_threshold = min(self.threshold, 0.28)

        for i, word in enumerate(active_sorted):
            # the most abstract word is automatically a root
            if i == 0:
                roots.append(word); continue

            best_match = None
            max_sim = -1
            child_vec = word_vectors[word].reshape(1, -1)

            # compare current word against all words more abstract than it (potential parents)
            for j in range(i):
                potential_parent = active_sorted[j]
                parent_vec = word_vectors[potential_parent].reshape(1, -1)
                sim = cosine_similarity(parent_vec, child_vec)[0][0]
                if sim > max_sim:
                    max_sim = sim
                    best_match = potential_parent

            # assign to parent if it passes threshold, otherwise it's a new root
            if best_match and max_sim >= effective_threshold:
                tree[best_match].append(word)
            else:
                roots.append(word)

        return tree, roots

    def save_tree(self, tree, roots, cursor, topic, imp_score, occur):
        """
        Saves the tree using UUID hierarchy and inputs LSTM values.
        """
        # 1. Map LSTM Predicted Sentiments from DB to individual words
        # Professional Mapping: Negative (0), Neutral (0.5), Positive (1.0)
        mapping = {"Positive": 1.0, "Neutral": 0.5, "Negative": 0.0}
        word_sentiment_vals = {}
        
        for word, cids in occur.items():
            # Convert cids to a list for the ANY(%s) SQL syntax
            cursor.execute("SELECT sentiment FROM airflow.cleaned_comments WHERE comment_id = ANY(%s)", (list(cids),))
            sents = [r[0] for r in cursor.fetchall() if r[0] is not None and r[0] != 'Pending']
            
            if sents:
                # Find the most frequent sentiment for this concept
                mode_sent = max(set(sents), key=sents.count)
                word_sentiment_vals[word] = mapping.get(mode_sent, 0.5)
            else:
                # Default to 0.5 (Neutral) if no valid classifications found
                word_sentiment_vals[word] = 0.5

        # 2. Insert Tree Root Record
        cursor.execute("INSERT INTO airflow.trees (name) VALUES (%s) RETURNING id;", (topic,))
        tree_uuid = cursor.fetchone()[0]

        # 3. Recursive function to maintain Parent ID chain (UUID Handshake)
        def insert_node_recursive(word, parent_uuid=None):
            cursor.execute(
                """
                INSERT INTO airflow.tree_nodes (tree_id, parent_id, text, imp_val, lstm_val)
                VALUES (%s, %s, %s, %s, %s) RETURNING id;
                """,
                (tree_uuid, parent_uuid, word, imp_score.get(word, 0), word_sentiment_vals.get(word, 0.5))
            )
            node_id = cursor.fetchone()[0]
            # Recursion: process children linked to this node
            for child in tree.get(word, []):
                insert_node_recursive(child, node_id)

        for root in roots:
            insert_node_recursive(root)

    def build_tree(self):
        # tokenize and get comments embeddings
        cmts_vec_t, words_vec_t = self.run_bert()
        cmts_vec = cmts_vec_t.detach().numpy()
        words_vec = words_vec_t.detach().numpy()

        # map the words to its abs score
        word_metadata, word_vectors, occur, imp_score = {}, {}, {}, {}
        n_cmts = len(self.pro_cmts)

        # for every unq word, find its context and score
        for word in self.target_words:
            temp_occur = []
            for i, (ids, cmt) in enumerate(self.pro_cmts.items()):
                if word in cmt:
                    idx = cmt.index(word)
                    if word not in occur: occur[word] = []
                    occur[word].append(ids)
                    # BERT adds a [CLS] token at index 0, so words at idx + 1
                    temp_occur.append(words_vec[i, idx+1, :])
            if not temp_occur:
                # get a standalone vector if word not in comments
                inputs = self.tokenizer(word, return_tensors="pt")
                output = self.model(**inputs)
                mean_vec = output.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
                score = 0.5 # neutral abs for unseen words
            else:
                score, mean_vec = self.calculate_dynamic_abstractness(temp_occur, cmts_vec)
            
            word_metadata[word] = {"abs_score": score}
            word_vectors[word] = mean_vec
            imp_score[word] = len(occur.get(word, [])) / n_cmts if n_cmts > 0 else 0
        
        return cmts_vec, occur, word_vectors, word_metadata, imp_score