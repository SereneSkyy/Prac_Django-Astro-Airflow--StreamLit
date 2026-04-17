from services.nlp_engine import NLPEngine
from services.bert_embed import TaxonomyAndTreeBuilder
from schemas.etl_schema import *

def create_embeddings(vid_ids, cursor, topic):
    # 1. Fetch raw comments for the processed videos (English only)
    cursor.execute(
        """
        SELECT DISTINCT c.id, c.comment
        FROM airflow.processed_vidIds p
        JOIN airflow.comment_lang cl ON cl.comment_id = p.cmt_id
        JOIN airflow.comments c ON c.id = p.cmt_id
        WHERE p.vid_id = ANY(%s)
          AND cl.language = 'en';
        """,
        (vid_ids,)               # -> have to be tuple so comma at the end
    )
    rows = cursor.fetchall()
    
    if not rows:
        return print("create_embeddings: No English Rows Were Fetched!!!")
    
    comment_texts = [comment for (_id, comment) in rows]
    ids = [_id for(_id, _comment) in rows]

    # 2. Clean (preprocess) the comments and store them (Initial state: 'Pending')
    if NLPEngine.clean_comments(comment_texts, ids, cursor):
        
        # --- REQUIREMENT: RUN LSTM FIRST ---
        # We calculate sentiment and update the DB immediately. 
        # This ensures the Taxonomy Tree can read real sentiment labels instead of 'Pending'.
        print("[*] Phase 1: Running LSTM Sentiment Inference...")
        try:
            NLPEngine.run_lstm_inference(ids, cursor)
        except Exception as e:
            print(f"[!] LSTM Error: {e}")

        # 3. Fetch the cleaned data to prepare for the BERT Taxonomy Builder
        cursor.execute(
            """
            SELECT comment_id, cleaned_text from airflow.cleaned_comments
            WHERE comment_id = ANY(%s)
            """, (ids,)
        )
        proc_rows = cursor.fetchall()
        # Fix: Ensure inputs are strings
        proc_cmts = {cid: comment for(cid, comment) in proc_rows}

        # 4. Build the BERT Taxonomy Tree (Partner's Logic)
        taxTree = TaxonomyAndTreeBuilder(threshold=0.30, pro_cmts=proc_cmts, topic=topic) 
        
        # Unpack the 7 variables returned by your updated bert_embed.py
        results = taxTree.build_tree()
        if not results:
            return print("[!] Tree Builder returned no results.")
            
        comments_vec, words_occur, word_vectors, word_metadata, imp_score, domains, subdomains = results

        # --- REQUIREMENT: SAVE HIERARCHY SECOND ---
        # Now that the DB has real sentiments, we create and save the tree
        print("[*] Phase 2: Saving Taxonomy Tree...")
        cursor.execute(execute_trees_sql)
        cursor.execute(execute_tree_nodes_sql)
        taxTree.save_hierarchy(cursor, topic, domains, subdomains, imp_score, words_occur)

        # 5. Save BERT Vectors and Features
        # ---------- embed_comments ----------
        comment_embeddings = comments_vec.tolist()
        cmts_vec_rows = [(cid, emb) for cid, emb in zip(ids, comment_embeddings)]

        # ---------- words_vec (topic, word, word_vec) ----------
        words_vec_rows = []
        for word, vec in word_vectors.items():
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            words_vec_rows.append((topic, word, vec))

        # ---------- words_occur normalized ----------
        # Fixed: Corrected variable 'w' to 'word' to fix NameError
        words_occur_rows = [(topic, word, cids) for word, cids in words_occur.items()]

        # Final Batch Inserts
        cursor.execute(execute_embed_comments_sql)
        cursor.execute(execute_words_vec_sql)
        cursor.execute(execute_words_occur_sql)

        cursor.executemany(insert_embed_comments, cmts_vec_rows)
        cursor.executemany(insert_words_vec, words_vec_rows)
        cursor.executemany(insert_words_occur, words_occur_rows)
        
        print("[SUCCESS] BERT Taxonomy, LSTM Sentiment, and MAC Score saved successfully.")

    else:
        print("[X] NLP Cleaning phase failed. Pipeline aborted.")