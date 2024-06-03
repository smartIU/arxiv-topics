import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from arxiv_topics.config import Config
from arxiv_topics.db import DB
from arxiv_topics.pipeline import Pipeline


if __name__ == "__main__":
    """ creates evaluations for a specific model again, if not activated in Step04_Train_Models """


    #TODO: Make arg
    model = 'arxiv_0'
    
    create_heatmap = True
    create_barchart = False
    create_hierarchy = True


    config = Config()

    db = DB(config.database)    

    if create_hierarchy:
        print(f'collecting papers')    
        abstracts = db.get_papers_by_model(model)['abstract']
    else:
        abstracts = None


    print('loading model')    
    topic_model = Pipeline.load_model(model)

    # Evaluation
    Pipeline.evaluate_model(topic_model, model, abstracts, create_heatmap, create_barchart, create_hierarchy)  

    print('Done.')
