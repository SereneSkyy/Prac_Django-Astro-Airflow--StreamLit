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
        return print("create_embeddings: No English Rows Found")
    
    comment_texts = [comment for (_id, comment) in rows]
    ids = [_id for(_id, _comment) in rows]

    # cleans (preprocesses) the comments and stores them
    if NLPEngine.clean_comments(comment_texts, ids, cursor):
        cursor.execute("SELECT comment_id, cleaned_text from airflow.cleaned_comments")
        rows = cursor.fetchall()
        proc_cmts = {cid: comment.split() for(cid, comment) in rows}

        target_words = ['protest', 'genz', 'kpoli', 'balenshah', 'corruption', 'singhadurbar', 'gaganthapa', 'youth', 'frustration', 'government', 'nepal', 'political', 'curfew', 'clash']

        # 1. Initialize BERT and build taxonomy structure
        taxTree = TaxonomyAndTreeBuilder(threshold=0.30, pro_cmts=proc_cmts, target_words=target_words) 
        comments_vec, words_occur, word_vectors, word_metadata, imp_score = taxTree.build_tree()

        # 2. RUN LSTM INFERENCE FIRST (To ensure sentiments exist for the tree)
        print("[*] Triggering LSTM Sentiment Inference...")
        try:
            NLPEngine.run_lstm_inference(ids, cursor)
        except Exception as e:
            print(f"[X] LSTM Error: {e}")

        # 3. BUILD AND SAVE PRUNED TREE
        cursor.execute(execute_trees_sql)
        cursor.execute(execute_tree_nodes_sql)
        tree, roots = taxTree.create_tree(word_metadata, word_vectors, imp_score, max_nodes=20)
        taxTree.save_tree(tree, roots, cursor, topic, imp_score, words_occur)
        
        print("[+] BERT Taxonomy and features saved successfully.")
    else:
        print("[X] NLP Cleaning phase failed.")