import gc

from arxiv_topics.config import Config
from arxiv_topics.db import DB
from arxiv_topics.pipeline import Pipeline


if __name__ == "__main__":
    """ use LLM model to regenerate labels
        only affects topics without a label, so for unappealing results first manually set the llm_label in the sqlite database to an empty string
    """

    config = Config()

    db = DB(config.database)


    # Get topics    
    models = db.get_topics_without_label()
        
    for model in models:

        model_name = model[0]

        print(f'Loading model {model_name}')
        topic_model = Pipeline.load_model(model_name)


        print('Generating labels')
        abstracts = model[1]['abstract']
        topic_ids = model[1]['topic_id']

        ids, labels = Pipeline.regenerate_labels(topic_model, abstracts, topic_ids
                                               , config.llm_prompt, config.llm_diversity, config.llm_stopwords)

        print('Updating database')
        db.import_topic_labels(model_name, ids, labels)

        # free memory
        del topic_model
        gc.collect()

    print('Done.')
