import pandas as pd
import numpy
import gc

from arxiv_topics.config import Config
from arxiv_topics.db import DB
from arxiv_topics.pipeline import Pipeline


if __name__ == "__main__":
    """ create models for subsets of the arXiv dataset, based on arXiv categories """


    config = Config()    

    # arxiv taxonomy
    taxonomy = pd.concat([pd.read_json(tax).loc['name'] for tax in config.input_taxonomy])
    taxonomy = taxonomy[~taxonomy.index.duplicated()] 
    

    # BERTopic model
    bert_params = config.bert_params[config.subset_bert_params]
    bertopic = Pipeline(config.embeddings_precision, bert_params
                      , config.llm_model, config.llm_prompt, config.llm_diversity, config.llm_stopwords)
    
    # Get subsets
    db = DB(config.database)
    
    subsets = db.get_subsets(config.subset_archive_max_count, config.subset_category_min_count)
    for subset in subsets:

        model = f'subset_{subset[0]}'
        
        model_exists = db.check_model_exists(model)

        if model_exists:            
             print(f'model {model} already created')
             continue
                      

        # Get abstracts and embeddings for subset
        print(f'collecting papers')
        categories = subset[1]['category_id'].tolist()
        papers = db.get_papers_by_categories(categories)

        embeddings = db.convert_embeddings(papers['embedding'])
        papers.drop(columns=['embedding'])


        # Train model        
        print(f'creating model {model} for {len(papers.index)} papers')

        topic_model = bertopic.fit(papers['abstract'], embeddings, outlier_label=taxonomy[subset[0]])


        # Save model to disk
        print('saving model')        
        Pipeline.save_model(topic_model, model)

        # Save topics and probabilities to db        
        db.import_topics(-1, model, topic_model, papers['id'])
        
        # free memory
        del topic_model
        gc.collect()


    print('Done.')
