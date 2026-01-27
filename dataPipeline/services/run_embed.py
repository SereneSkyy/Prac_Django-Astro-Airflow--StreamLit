from services.nlp_engine import NLPEngine
from services.bert_embed import TaxonomyAndTreeBuilder
from schemas.etl_schema import (execute_embed_comments_sql, execute_words_vec_sql, execute_words_occur_sql, 
                                insert_embed_comments, insert_words_vec, insert_words_occur)

def create_embeddings(vid_ids, cursor, topic):
    cursor.execute(
    """
        SELECT DISTINCT c.id, c.comment
        FROM airflow.processed_vidIds p
        JOIN airflow.comment_lang cl
          ON cl.comment_id = p.cmt_id
        JOIN airflow.comments c
          ON c.id = p.cmt_id
        WHERE p.vid_id = ANY(%s)
          AND cl.language = 'en';
    """,
        (vid_ids,)              # -> has to be tuple so comma at the end
    )
    rows = cursor.fetchall()
    if not rows:
        return print("create_embeddings: No Rows Were Fetched!!!")
    
    comment_texts = [comment for (_id, comment) in rows]
    ids = [_id for(_id, _comment) in rows]

    # cleans (preprocesses) the comments and stores them
    if NLPEngine.clean_comments(comment_texts, ids, cursor):
        cursor.execute(
            """
            SELECT comment_id, cleaned_text from airflow.cleaned_comments
            """
        )
        rows = cursor.fetchall()

        proc_cmts = {cid: comment.split() for(cid, comment) in rows}

        target_words = ['protest', 'genz', 'kpoli', 'balenshah', 'corruption', 'singhadurbar', 'gaganthapa', 'youth', 'frustration', 'anti-corruption', 'government', 'nepal',
                    'accountability', 'baneshwor', 'political', 'curfew', 'clash', 'nepobaby']

        taxTree = TaxonomyAndTreeBuilder(threshold=0.45, pro_cmts=proc_cmts, target_words=target_words) 

        comments_vec, words_occur, word_vectors, word_metadata, imp_score = taxTree.build_tree()

        # ---------- create tree ----------
        tree, roots = taxTree.create_tree(word_metadata, word_vectors)
        taxTree.save_tree(tree, roots, cursor, topic, imp_score)

        # ---------- embed_comments ----------
        comment_embeddings = comments_vec.tolist()
        cmts_vec_rows = [(cid, emb) for cid, emb in zip(ids, comment_embeddings)]

        # ---------- words_vec (topic, word, word_vec) ----------
        words_vec_rows = []
        for word, vec in word_vectors.items():
            # vec might be numpy or torch
            if hasattr(vec, "detach"):
                vec = vec.detach().cpu().tolist()
            elif hasattr(vec, "tolist"):
                vec = vec.tolist()
            words_vec_rows.append((topic, word, vec))

        # ---------- words_occur normalized (topic, word, word_cmt_id) ----------
        words_occur_rows = [
            (topic, word, cmt_ids)
            for word, cmt_ids in words_occur.items()
        ]

        # create tables + insert
        cursor.execute(execute_embed_comments_sql)
        cursor.execute(execute_words_vec_sql)
        cursor.execute(execute_words_occur_sql)

        cursor.executemany(insert_embed_comments, cmts_vec_rows)
        cursor.executemany(insert_words_vec, words_vec_rows)
        cursor.executemany(insert_words_occur, words_occur_rows)
        print("[+] BERT features saved successfully.")

        # TRIGGER LSTM SENTIMENT ANALYSIS
        # This function fetches the vectors we just saved and runs the Bidirectional LSTM
        print("[*] Triggering LSTM Sentiment Inference using BERT features...")
        try:
            NLPEngine.run_lstm_inference(ids, cursor)
            print("[+] Sentiment analysis complete.")
        except Exception as e:
            print(f"[X] LSTM Error: {e}")

    else:
        print("[X] NLP Cleaning phase failed. Pipeline aborted.")

