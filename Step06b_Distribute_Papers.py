import numpy as np
import pandas as pd
import gc

from arxiv_topics.config import Config
from arxiv_topics.db import DB
from arxiv_topics.pipeline import Pipeline


if __name__ == "__main__":
    """ use BERTopic approximate_distribution method to assign outliers to one or more existing topics 
       (worse accuracy than using the UMAP & HDBScan models, but faster and can assign multiple topics to each paper)
    """

    #TODO: Make arg
    model = 'arxiv_0'
    hierarchy = 0 
    min_similarity = 0.15


    config = Config()

    db = DB(config.database)

    print(f'collecting papers')    
    papers = db.get_papers_for_training('outliers', hierarchy, 100)
    papers.drop(columns=['embedding'])

    print('loading model')    
    topic_model = Pipeline.load_model(model, drop_llm=True)

    topic_model.umap_model = None
    topic_model.hdbscan_model = None
    gc.collect()
    
    paper_count = len(papers.index)
        
    for chunk in np.array_split(papers, paper_count // 10000):

        print(f'creating approximate distributions for {len(chunk.index)} papers')
        distributions, _ = topic_model.approximate_distribution(chunk['abstract'])

        df = pd.DataFrame(distributions.tolist())

        df.index = chunk['id'].tolist()             
        
        df = df.stack(future_stack=True).reset_index().rename(columns={'level_0':'paper_id','level_1':'topic_id',0:'prob'})

        df = df[df['prob'] > min_similarity]
        
        
        print('importing to database')    
        db.import_paper_topics(model, df['paper_id'], df['topic_id'], df['prob'], distributed=True)


    print('Done.')