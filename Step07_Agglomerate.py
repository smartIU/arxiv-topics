import pandas as pd
import numpy
import gc
from ast import literal_eval

from arxiv_topics.config import Config
from arxiv_topics.db import DB
from arxiv_topics.pipeline import Pipeline


if __name__ == "__main__":
    """ create a new model by merging the topics of an existing model, based on cluster similarity
       (only used to assign the same color to similar topics in the dash app for now
       ,but could be used to create an interactive deep dive with multiple hierarchies)
    """


    config = Config()

    db = DB(config.database)

    # Iterate models to agglomerate
    for new_model in config.agg_models:

        agg_params = config.agg_models[new_model]
        
        model = agg_params.parent_model

        model_exists = db.check_model_exists(new_model)

        delete_model = False
        if model_exists:
            if agg_params.overwrite:
                delete_model = True
            else:
                print(f'model {new_model} already created')
                continue        


        print('collecting papers')
         
        papers = db.get_papers_by_model(model)

   
        print('loading model')
        
        topic_model = Pipeline.load_model(model)    


        print('assessing topics to merge')

        # read in output from Pipeline.evaluate_model(hierarchy=True)
        df = pd.read_csv(f'./output/{model}_hierarchy.csv', sep='\t')

        # filter topics by distance 
        df = df[df['Distance'].astype(float) < agg_params.max_cluster_distance]

        # extract last levels of hierarchy
        df = df[['Topics']]
        df['Topics'] = df['Topics'].transform(literal_eval)

        df['TopicCount'] = df['Topics'].str.len()
        df.sort_values(by='TopicCount', ascending=False)
        
        topics = df['Topics'].explode()
        topics = topics[topics.duplicated()]
            
        df = df[~df.index.isin(topics.index.unique())]

        topics_to_merge = df['Topics'].tolist()


        # merge topics
        print('merging...')    
        Pipeline.merge_topics(topic_model, papers['abstract'], topics_to_merge)


        # save new model
        print('saving model')        
        Pipeline.save_model(topic_model, new_model)


        if delete_model:
            # cleaning database
            db.delete_model(new_model)

        # save topics and probabilities
        db.import_topics(agg_params.hierarchy, new_model, topic_model
                       , papers['id'], excl_outliers=agg_params.excl_outliers)


        # update hierarchy
        db.update_hierarchy(model, new_model)

        # free memory
        del topic_model
        gc.collect()
        
                             
    print('Done.')
