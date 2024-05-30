import pandas as pd
import numpy
import sys

from arxiv_topics.config import Config
from arxiv_topics.db import DB
from arxiv_topics.transformer import Transformer


if __name__ == "__main__":
    """ convert text abstracts to vectors using SentenceTransformer """


    config = Config()

    db = DB(config.database)

    # load SentenceTransformer
    embedding_model = Transformer.EmbeddingModel(config.embeddings_model)


    print('Calculating embeddings', end='')
    sys.stdout.flush()
    
    while True:
        
        papers = db.get_papers_for_embedding(config.embeddings_chunk_size)

        if papers.empty:
            break
       
        # Encode abstracts
        embeddings = embedding_model.encode(papers['abstract'], precision=config.embeddings_precision, normalize_embeddings=True, show_progress_bar=True)

        embeddings = [numpy.array(emb) for emb in embeddings]
            
        # Save to database
        db.import_embeddings(papers['id'], embeddings)

        print('.', end='')
        sys.stdout.flush()


    print()
    print('Done.')

