import pandas as pd
import numpy as np
import gc

from arxiv_topics.config import Config
from arxiv_topics.db import DB
from arxiv_topics.pipeline import Pipeline
from arxiv_topics.transformer import Transformer


if __name__ == "__main__":
    """ train main model(s) """


    #TODO: make arg
    create_heatmap = True
    create_barchart = False
    create_hierarchy = True


    config = Config()

    db = DB(config.database)        

    # Iterate models to train
    for model in config.train_models:        

        model_params = config.train_models[model]

        model_exists = db.check_model_exists(model)

        delete_model = False
        if model_exists:
            if model_params.overwrite:
                delete_model = True
            else:
                print(f'model {model} already created')
                continue
        
 
        # Get abstracts and embeddings for model
        print(f'collecting papers for {model}')
        papers = db.get_papers_for_training(model_params.model_filter, model_params.filter_value
                                          , model_params.percent_of_papers)
                
        embeddings = db.convert_embeddings(papers['embedding'])
        papers.drop(columns=['embedding'])

        if model_params.model_filter == 'hierarchy':
            labels = papers['topic_id']
            embedding_model = Transformer.EmbeddingModel(config.embeddings_model)
        else:
            labels = None
            embedding_model = None

        # Train model        
        print(f'creating model {model} for {len(papers.index)} papers')

        bert_params = config.bert_params[model_params.bert_params]  
        bertopic = Pipeline(config.embeddings_precision, bert_params
                           ,config.llm_model, config.llm_prompt, config.llm_diversity, config.llm_stopwords)

        topic_model = bertopic.fit(papers['abstract'], embeddings, embedding_model=embedding_model, labels=labels)
  
  
        # Save model to disk
        print('saving model')        
        Pipeline.save_model(topic_model, model)

        if delete_model:
            # cleaning database
            db.delete_model(model)
     
        # Save topics and probabilities to db        
        db.import_topics(model_params.hierarchy, model, topic_model, papers['id'])


        if create_heatmap or create_barchart or create_hierarchy:
            Pipeline.evaluate_model(topic_model, model, papers['abstract'], create_heatmap,create_barchart,create_hierarchy)
        
        # free memory
        del topic_model
        gc.collect()
        
        
    print('Done.')    