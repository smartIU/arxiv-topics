from arxiv_topics.config import Config
from arxiv_topics.db import DB


if __name__ == "__main__":
    """ computes monthly counts and percentages for all categories and topics """
    

    config = Config()

    db = DB(config.database)

    print('updating category stats...')

    db.update_category_stats()


    print('updating topic stats...')

    db.update_topic_stats()

                             
    print('Done.')
