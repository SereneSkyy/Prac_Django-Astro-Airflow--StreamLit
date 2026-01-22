from services.nlp_engine import NLPEngine
from services.bert_embed import TaxonomyAndTreeBuilder
from schemas.etl_schema import execute_embed_comments_sql, insert_embed_comments

nlpEngine = NLPEngine()

def create_embeddings(vid_ids, cursor):
    cursor.execute(
        """
        SELECT id, comment
        FROM airflow.comments
        WHERE id = ANY(%s)
        """,
        (vid_ids,)  # note the trailing comma -> tuple
    )
    rows = cursor.fetchall()

    comment_texts = [comment for (_id, comment) in rows]
    ids = [_id for(_id, _comment) in rows]

    # cleans (preprocesses) the comments and stores them
    nlpEngine.clean_comments(comment_texts, ids, cursor)

    cursor.execute(
        """
        SELECT comment_id, cleaned_text from airflow.processed_comments
        """
    )
    rows = cursor.fetchall()

    proc_cmts = [comments.split() for(_id, comments) in rows]

    target_words = ['protest', 'genz', 'kpoli', 'balenshah', 'corruption', 'singhadurbar', 'gaganthapa', 'youth', 'frustration', 'anti-corruption', 'government', 'nepal',
                'accountability', 'baneshwor', 'political', 'curfew', 'clash', 'nepobaby']
    
    taxTree = TaxonomyAndTreeBuilder(threshold=0.45, pro_cmts=proc_cmts, target_words=target_words)  
    # feed comments into BERT -> return comment and word vectors
    comment_vecs, _ = taxTree.run_bert()
    # convert to list[list[float]]
    embeddings = comment_vecs.detach().cpu().tolist()

    values = [(cid, emb) for cid, emb in zip(ids, embeddings)]

    cursor.executemany(insert_embed_comments, values)

