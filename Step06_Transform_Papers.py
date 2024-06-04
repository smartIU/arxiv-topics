import sys
import gc

from arxiv_topics.config import Config
from arxiv_topics.db import DB
from arxiv_topics.pipeline import Pipeline


if __name__ == "__main__":
    """ assign papers to existing topics with the original UMAP & HDBScan models
       (better accuracy than the BERTopic approximate_distribution method, but can only assign a single topic and leads to more outliers)
    """


    config = Config()

    db = DB(config.database)


    models = db.get_models_for_transform()

    if models.empty:
        print('no papers to transform')
        
    else:
        # Iterate models to transform
        for m in models.itertuples():        

            hierarchy = m.hierarchy
            model = m.model

            print(f'Loading model {model}')
            topic_model = Pipeline.load_model(model, drop_llm=True)


            print('Predicting topics', end='')
            sys.stdout.flush()

            while True:
                
                papers = db.get_papers_for_transform(model, config.transform_chunk_size)        

                if papers.empty:
                    break
             
                embeddings = db.convert_embeddings(papers['embedding'])                

                # predict topics for new papers
                topics, probs = topic_model.transform(papers['abstract'], embeddings)


                # Save topics and probabilities to db        
                db.import_paper_topics(model, papers['id'], topics, probs)

                print('.', end='')
                sys.stdout.flush()

            print()
            
            # free memory
            del topic_model
            gc.collect()


    print('Done.')
