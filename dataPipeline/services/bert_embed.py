import json
import torch
import numpy as np
import os
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

class TaxonomyAndTreeBuilder:
    def __init__(self, threshold, pro_cmts, target_words):
        model_path = "/bert_model" if os.path.exists("/bert_model") else 'bert-base-uncased'
        is_offline = os.path.exists("/bert_model")

        # Load from the static folder if it exists
        self.tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=is_offline)
        self.model = BertModel.from_pretrained(model_path, local_files_only=is_offline)
        
        self.pro_cmts = pro_cmts
        self.target_words = target_words
        self.threshold = threshold
    
    def calculate_dynamic_abstractness(self, word_occur, cmt_vecs):
        if not word_occur:
            return 0.5, np.zeros(768)
        mean_word_vec = np.mean(word_occur, axis=0)
        sims = cosine_similarity([mean_word_vec], cmt_vecs)[0]
        abs_score = 1 - np.var(sims)
        return abs_score, mean_word_vec

    def _tokenizer(self):
        cmts = [" ".join(words) for words in self.pro_cmts.values()]
        inputs = self.tokenizer(cmts, return_tensors="pt", padding=True, truncation=True)
        return inputs

    def run_bert(self):
        inputs = self._tokenizer()
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        comment_vecs = summed / counts
        return comment_vecs, last_hidden

    def _create_tree(self, word_metadata, word_vectors):
        sorted_words = sorted(word_metadata.keys(), key=lambda w: word_metadata[w]["abs_score"], reverse=True)
        tree = {word: [] for word in sorted_words}
        roots = []
        for i, word in enumerate(sorted_words):
            if i == 0:
                roots.append(word)
                continue
            best_match, max_sim = None, -1
            child_vec = word_vectors[word].reshape(1, -1)
            for j in range(i):
                potential_parent = sorted_words[j]
                parent_vec = word_vectors[potential_parent].reshape(1, -1)
                sim = cosine_similarity(parent_vec, child_vec)[0][0]
                if sim > max_sim:
                    max_sim, best_match = sim, potential_parent
            if best_match and max_sim >= self.threshold:
                tree[best_match].append(word)
            else:
                roots.append(word)
        return tree, roots

    def _save_to_json(self, tree, roots, filename="hierarchy_tree.json"):
        export_data = {"tree": tree, "roots": roots}
        def default_serializer(obj):
            if isinstance(obj, (np.ndarray, np.float32)): return obj.tolist()
            raise TypeError(f"Type {type(obj)} not serializable")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4, default=default_serializer)

    def build_tree(self):
        cmts_vec_tensor, words_vec_tensor = self.run_bert()
        cmts_vec = cmts_vec_tensor.detach().numpy()
        words_vec = words_vec_tensor.detach().numpy()

        word_metadata, word_vectors, occur = {}, {}, {}
        
        for word in self.target_words:
            temp_occur = []
            for i, (ids, cmt) in enumerate(self.pro_cmts.items()):
                if word in cmt:
                    idx = cmt.index(word)
                    if word not in occur: occur[word] = []
                    occur[word].append(ids)
                    if idx + 1 < words_vec.shape[1]:
                        temp_occur.append(words_vec[i, idx+1, :])
            
            if not temp_occur:
                inputs = self.tokenizer(word, return_tensors="pt")
                with torch.no_grad():
                    output = self.model(**inputs)
                    mean_vec = output.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
                score = 0.5
            else:
                score, mean_vec = self.calculate_dynamic_abstractness(temp_occur, cmts_vec)
            word_metadata[word] = {"abs_score": score}
            word_vectors[word] = mean_vec
        
        tree, roots = self._create_tree(word_metadata, word_vectors)
        self._save_to_json(tree, roots) 
        return cmts_vec, occur, word_vectors