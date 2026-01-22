import json
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

class TaxonomyAndTreeBuilder:
    def __init__(self, threshold, pro_cmts, target_words):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
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
        cmts = [" ".join(words) for words in self.pro_cmts]
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

    def _create_tree(self, word_metadata, word_vectors):
        """
        word_metadata: dict of {word: {"abs_score": float}}
        word_vectors: dict of {word: np.array}
        """
        # sort words by abstractness (Descending: Highest score first)
        sorted_words = sorted(word_metadata.keys(), 
                              key=lambda w: word_metadata[w]["abs_score"], 
                              reverse=True)
        
        tree = {word: [] for word in sorted_words}
        roots = []

        for i, word in enumerate(sorted_words):
            # the most abstract word is automatically a root
            if i == 0:
                roots.append(word)
                continue

            best_match = None
            max_sim = -1
            child_vec = word_vectors[word].reshape(1, -1)

            # compare current word against all words more abstract than it (potential parents)
            for j in range(i):
                potential_parent = sorted_words[j]
                parent_vec = word_vectors[potential_parent].reshape(1, -1)
                
                sim = cosine_similarity(parent_vec, child_vec)[0][0]
                
                if sim > max_sim:
                    max_sim = sim
                    best_match = potential_parent

            # assign to parent if it passes threshold, otherwise it's a new root
            if best_match and max_sim >= self.threshold:
                tree[best_match].append(word)
            else:
                roots.append(word)

        return tree, roots

    def _draw_tree(self, tree, nodes, indent=0):
        for node in nodes:
            # create the visual prefix (the "branch" look)
            # we use four spaces per level of depth
            prefix = "    " * indent + "└── "
            
            print(f"{prefix}{node}")

            # check if this word has children in our dictionary
            if node in tree and tree[node]:
                # Recursion: draw the children, but increase the indent
                self._draw_tree(tree, tree[node], indent + 1)

    def _save_to_json(self, tree, roots, filename="hierarchy_tree.json"):
        """
        Saves the generated tree and root nodes to a JSON file.
        """
        #cCreate the data structure for export
        export_data = {
            "tree": tree,
            "roots": roots
        }

        # helper function to handle non-serializable objects (like NumPy arrays)
        def default_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.float32):
                return float(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        try:
            with open(filename, "w", encoding="utf-8") as f:
                # indent=4 makes the file human-readable
                json.dump(export_data, f, indent=4, default=default_serializer)
            print(f"Successfully saved taxonomy to {filename}")
        except Exception as e:
            print(f"Error saving to JSON: {e}")

    def build_tree(self):
        # tokenize and get comments embeddings
        # summarized vector embed for each comment i.e., cmts_vec[0], cmts_vec[1], ...
        cmts_vec, words_vec = self.run_bert()
        cmts_vec = cmts_vec.detach().numpy()

        # map the words to its abs score
        word_metadata = {}
        # separate map for 1D vecs later used in draw tree
        word_vectors = {}

        # for every unq word, find its context and score
        occur = {}
        for word in self.target_words:
            temp_occur = []
            for i, cmt in enumerate(self.pro_cmts):
                if word in cmt:
                    # find pos of word to get the specific BERT vector
                    idx = cmt.index(word)
                    # store word and its associated comment idx
                    if word in occur.keys():
                        occur[word].append(i)
                    else: occur[word] = [i]
                    # BERT adds a [CLS] token at index 0, so words at idx + 1
                    temp_occur.append(words_vec[i, idx+1, :].detach().numpy())
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
        
        tree, roots = self._create_tree(word_metadata, word_vectors)
        self._draw_tree(tree, roots)
        self._save_to_json(tree, roots) 