import spacy
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import Counter, defaultdict

class TaxonomyAndTreeBuilder:
    def __init__(self, threshold, pro_cmts, topic, model_name="all-MiniLM-L6-v2"):
        # --- RESTORED OFFLINE FIX ---
        # If the baked-in folder exists, use it; otherwise, download from HF
        model_path = "/bert_model" if os.path.exists("/bert_model") else 'all-MiniLM-L6-v2'
        
        # Load the embedder using the local path if available
        self.embedder = SentenceTransformer(model_path)
        
        # Load Spacy (Note: this still requires 'en_core_web_sm' to be installed via pip/download)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Original logic continues...
        self.pro_cmts = pro_cmts
        self.threshold = threshold
        self.topic = topic
        self.title_vec = self.embedder.encode(topic, convert_to_tensor=True)
        self.title_words = {t.text.lower() for t in self.nlp(topic) if not t.is_stop}

    def _extract_candidates(self, text):
        doc = self.nlp(text)
        candidates = []
        for chunk in doc.noun_chunks:
            tokens = [t.text for t in chunk if not t.is_stop and t.pos_ in ["NOUN", "PROPN", "ADJ"]]
            phrase = " ".join(tokens).replace(" - ", "-").strip().lower()
            if phrase and len(phrase) > 2: candidates.append(phrase)
        for t in doc:
            if t.pos_ in ["NOUN", "VERB"] and not t.is_stop:
                lemma = t.lemma_.lower()
                if len(lemma) > 2: candidates.append(lemma)
        return list(set(candidates))

    def build_tree(self):
        # 1. Prep Comments
        cid_list = list(self.pro_cmts.keys())
        raw_comments = [" ".join(words) for words in self.pro_cmts.values()]
        
        # 2. Extract & Embed
        comment_candidate_map = [self._extract_candidates(c) for c in raw_comments]
        unique_candidates = list(set([c for sub in comment_candidate_map for c in sub]))
        
        if not unique_candidates: return None # Handle empty case
        
        cand_vecs = self.embedder.encode(unique_candidates, convert_to_tensor=True)
        cmts_vec_t = self.embedder.encode(raw_comments, convert_to_tensor=True)

        # 3. Scoring
        l_sims = util.cos_sim(cand_vecs, cmts_vec_t).mean(dim=1).tolist()
        g_sims = util.cos_sim(cand_vecs, self.title_vec).flatten().tolist()
        
        occurrence_counter = Counter()
        focus_counter = Counter()
        occur = defaultdict(list)

        for i, (comment, extracted) in enumerate(zip(raw_comments, comment_candidate_map)):
            doc = self.nlp(comment)
            focus_terms = {t.lemma_.lower() for t in doc if t.dep_ in ("nsubj", "nsubjpass", "ROOT")}
            for cand in extracted:
                occurrence_counter[cand] += 1
                occur[cand].append(cid_list[i])
                if any(word in focus_terms for word in cand.split()):
                    focus_counter[cand] += 1

        # 4. Final Scores & Domains (Hierarchy Logic)
        final_scores = {}
        max_occ = max(occurrence_counter.values()) or 1
        for idx, cand in enumerate(unique_candidates):
            overlap = 1.0 if any(w in cand for w in self.title_words) else 0.0
            base = (l_sims[idx] * 0.8) + (g_sims[idx] * 0.19) + (overlap * 0.01)
            occ_s = occurrence_counter[cand] / max_occ
            foc_s = focus_counter[cand] / (max(focus_counter.values()) or 1)
            final_scores[cand] = (0.5 * base) + (0.2 * occ_s) + (0.3 * foc_s)

        # Domain/Subdomain Assignment
        max_f = max(final_scores.values()) or 1
        domains = [c for c, s in final_scores.items() if s >= 0.75 * max_f]
        subdomains = defaultdict(list)
        
        dom_list = list(domains)
        if dom_list:
            dom_vecs = self.embedder.encode(dom_list, convert_to_tensor=True)
            for cand in unique_candidates:
                if cand in domains: continue
                c_vec = cand_vecs[unique_candidates.index(cand)]
                sims = util.cos_sim(c_vec, dom_vecs).flatten()
                best_idx = sims.argmax().item()
                if sims[best_idx] >= 0.30:
                    subdomains[dom_list[best_idx]].append(cand)

        # Return same structure + new hierarchy
        word_vectors = {c: vec.cpu().numpy() for c, vec in zip(unique_candidates, cand_vecs)}
        word_metadata = {c: {"abs_score": final_scores[c]} for c in unique_candidates}
        imp_score = {c: occurrence_counter[c]/max_occ for c in unique_candidates}
        
        return cmts_vec_t.cpu().numpy(), dict(occur), word_vectors, word_metadata, imp_score, domains, dict(subdomains)

    def save_hierarchy(self, cursor, topic, domains, subdomains, imp_score, occur):
        """Utilizes the domain/subdomain logic for DB insertion."""
        # Get Sentiment Mapping (from your existing logic)
        mapping = {"Positive": 1.0, "Neutral": 0.5, "Negative": 0.0}
        word_sent_vals = {}
        for word, cids in occur.items():
            cursor.execute("SELECT sentiment FROM airflow.cleaned_comments WHERE comment_id = ANY(%s)", (list(cids),))
            sents = [r[0] for r in cursor.fetchall() if r[0] not in [None, 'Pending']]
            word_sent_vals[word] = mapping.get(max(set(sents), key=sents.count), 0.5) if sents else 0.5

        # Create Tree Entry
        cursor.execute("INSERT INTO airflow.trees (name) VALUES (%s) RETURNING id;", (topic,))
        tree_id = cursor.fetchone()[0]

        word_to_id = {}
        # 1. Insert Domains (Roots)
        for dom in domains:
            cursor.execute(
                "INSERT INTO airflow.tree_nodes (tree_id, text, imp_val, lstm_val, parent_id) VALUES (%s, %s, %s, %s, NULL) RETURNING id;",
                (tree_id, dom, imp_score.get(dom, 0), word_sent_vals.get(dom, 0.5))
            )
            word_to_id[dom] = cursor.fetchone()[0]

        # 2. Insert Subdomains (Children)
        for dom, subs in subdomains.items():
            parent_id = word_to_id.get(dom)
            for sub in subs:
                cursor.execute(
                    "INSERT INTO airflow.tree_nodes (tree_id, text, imp_val, lstm_val, parent_id) VALUES (%s, %s, %s, %s, %s);",
                    (tree_id, sub, imp_score.get(sub, 0), word_sent_vals.get(sub, 0.5), parent_id)
                )