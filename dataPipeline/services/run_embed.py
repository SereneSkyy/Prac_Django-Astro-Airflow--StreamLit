from services.nlp_engine import NLPEngine
from services.bert_embed import TaxonomyAndTreeBuilder
from schemas.etl_schema import *

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
        (vid_ids,)               # -> have to be tuple so comma at the end
    )
    rows = cursor.fetchall()
    if not rows:
        return print("create_embeddings: No English Rows Were Fetched!!!")
    
    comment_texts = [comment for (_id, comment) in rows]
    ids = [_id for(_id, _comment) in rows]

    # cleans (preprocesses) the comments and stores them
    if NLPEngine.clean_comments(comment_texts, ids, cursor):
        cursor.execute("SELECT comment_id, cleaned_text from airflow.cleaned_comments")
        rows = cursor.fetchall()
        proc_cmts = {cid: comment.split() for(cid, comment) in rows}

        target_words = ['protest', 'genz', 'kpoli', 'balenshah', 'corruption', 'singhadurbar', 'gaganthapa', 'youth', 'frustration', 'government', 'nepal', 'political', 'curfew', 'clash', 'rights', 'nepobaby']

        # setting threshold to 0.30 
        taxTree = TaxonomyAndTreeBuilder(threshold=0.30, pro_cmts=proc_cmts, target_words=target_words) 

        comments_vec, words_occur, word_vectors, word_metadata, imp_score = taxTree.build_tree()

        # ---------- Requirement: Run LSTM FIRST ----------
        print("[*] Triggering LSTM Sentiment Inference...")
        try:
            NLPEngine.run_lstm_inference(ids, cursor)
        except Exception as e:
            print(f"[X] LSTM Error: {e}")

        # ---------- Requirement: Create and Save Tree ----------
        cursor.execute(execute_trees_sql)
        cursor.execute(execute_tree_nodes_sql)
        tree, roots = taxTree.create_tree(word_metadata, word_vectors, imp_score, max_nodes=20)
        taxTree.save_tree(tree, roots, cursor, topic, imp_score, words_occur)

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
        words_occur_rows = [(topic, word, cids) for word, cids in words_occur.items()]

        # insert other features
        cursor.execute(execute_embed_comments_sql)
        cursor.execute(execute_words_vec_sql)
        cursor.execute(execute_words_occur_sql)

        cursor.executemany(insert_embed_comments, cmts_vec_rows)
        cursor.executemany(insert_words_vec, words_vec_rows)
        cursor.executemany(insert_words_occur, words_occur_rows)
        
        print("[+] BERT Taxonomy and features saved successfully.")

    else:
        print("[X] NLP Cleaning phase failed. Pipeline aborted.")