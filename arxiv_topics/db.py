import io
import os
import sqlite3
import time
import numpy as np
import pandas as pd

from contextlib import closing

class DB:
    """ encapsulates all database requests """

    def __init__(self, connection : str):
        """ instanciate database encapsulation
        
          Arguments:
            connection: path to sqlite3 database file
        """

        self._connection = connection
        self._journal = f'{connection}-journal'
        
        # automatically convert np.array to BLOB when inserting
        sqlite3.register_adapter(np.ndarray, DB._adapt_array)

    
    def connection(self):
        """ returns a database connection """       
        return sqlite3.connect(self._connection, detect_types=sqlite3.PARSE_DECLTYPES)        


    @staticmethod
    def _adapt_array(arr):
        """ convert np.array to BLOB """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())
    
    @staticmethod
    def _convert_array(text):
        """ convert BLOB to np.array """        
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)
    
    
    def convert_embeddings(self, embeddings):
        """ convert series of BLOBs to ndarray (not done directly during sqlite read to improve performance)
         
          Arguments:
            embeddings: binary embeddings as read from database 
        """
             
        return np.stack(embeddings.apply(DB._convert_array), axis=0)


    def assure_database(self):
        """ creates database file and schema objects if neccessary
        
        Tables:
            paper(id,arxiv_id,creation_year,creation_month,title,abstract,abstract_length,embedding,update_date)
            category(id,archive,name,paper_count)
            topic(id,parent_id,hierarchy,model,orig_id,name,llm_label,paper_count)
            
            paper_categories(paper_id,category_id,is_primary)            
            paper_topics(paper_id,topic_id,probability,transformed)
            
            category_stats(category_id,year,month,paper_count,total_percent,parent_percent) #NOTE: is only updated via update_category_stats()
            topic_stats(topic_id,year,month,transformed,paper_count,total_percent,parent_percent) #NOTE: is only updated via update_topic_stats()            
            
        View:            
            archives(archive,paper_count) #NOTE: archive paper_count is inflated, because one paper can be in multiple categories
        """

        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:

                # Main tables
                cmd.execute('''CREATE TABLE IF NOT EXISTS paper(
                               id INTEGER PRIMARY KEY                              
                              ,arxiv_id TEXT NOT NULL       
                              ,creation_year INTEGER NOT NULL                              
                              ,creation_month INTEGER NOT NULL
                              ,title TEXT NOT NULL
                              ,abstract TEXT NOT NULL
                              ,abstract_length INTEGER NOT NULL
                              ,embedding ARRAY
                              ,update_date TEXT NOT NULL DEFAULT(date('today', 'localtime'))                              
                              )''')
                              
                cmd.execute('''CREATE UNIQUE INDEX IF NOT EXISTS index_paper_arxiv_id ON paper(arxiv_id)''')
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_paper_year ON paper(creation_year)''')
                
                cmd.execute('''CREATE TABLE IF NOT EXISTS category(
                               id INTEGER PRIMARY KEY
                              ,archive TEXT NOT NULL
                              ,name TEXT NOT NULL
                              ,paper_count INTEGER NOT NULL DEFAULT(0)
                              )''')
                
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_category_archive ON category(archive)''')
                cmd.execute('''CREATE UNIQUE INDEX IF NOT EXISTS index_category_name ON category(name)''')                
                             
                cmd.execute('''CREATE TABLE IF NOT EXISTS paper_categories(
                               paper_id INTEGER NOT NULL
                              ,category_id INTEGER NOT NULL    
                              ,is_primary INTEGER NOT NULL DEFAULT(0)
                              ,FOREIGN KEY(paper_id) REFERENCES paper(id) ON DELETE CASCADE
                              ,FOREIGN KEY(category_id) REFERENCES category(id) ON DELETE CASCADE
                              )''')

                cmd.execute('''CREATE INDEX IF NOT EXISTS index_paper_categories_1 ON paper_categories(paper_id)''')
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_paper_categories_2 ON paper_categories(category_id)''')
                
                cmd.execute('''CREATE TABLE IF NOT EXISTS topic (
                               id INTEGER PRIMARY KEY
                              ,parent_id INTEGER
                              ,hierarchy INTEGER NOT NULL DEFAULT(-1)
                              ,model TEXT NOT NULL
                              ,orig_id INTEGER NOT NULL
                              ,name TEXT NOT NULL
                              ,llm_label TEXT NOT NULL
                              ,paper_count INTEGER NOT NULL DEFAULT(0)
                              ,FOREIGN KEY(parent_id) REFERENCES topic(id) ON DELETE SET NULL)''')

                cmd.execute('''CREATE INDEX IF NOT EXISTS index_topic_model ON topic(model)''')
                cmd.execute('''CREATE UNIQUE INDEX IF NOT EXISTS index_topic_orig_id ON topic(model, orig_id)''')
                
                cmd.execute('''CREATE TABLE IF NOT EXISTS paper_topics(
                               paper_id INTEGER NOT NULL
                              ,topic_id INTEGER NOT NULL                              
                              ,probability REAL
                              ,transformed INTEGER NOT NULL DEFAULT(0)
                              ,FOREIGN KEY(paper_id) REFERENCES paper(id) ON DELETE CASCADE
                              ,FOREIGN KEY(topic_id) REFERENCES topic(id) ON DELETE CASCADE
                              )''')

                cmd.execute('''CREATE UNIQUE INDEX IF NOT EXISTS index_paper_topics ON paper_topics(paper_id,topic_id)''')
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_paper_topics_1 ON paper_topics(paper_id)''')
                cmd.execute('''CREATE INDEX IF NOT EXISTS index_paper_topics_2 ON paper_topics(topic_id)''')

                cmd.execute('''CREATE TABLE IF NOT EXISTS category_stats(
                               category_id INTEGER NOT NULL
                              ,year INTEGER NULL
                              ,month INTEGER NULL
                              ,paper_count INTEGER
                              ,total_percent REAL
                              ,parent_percent REAL
                              ,FOREIGN KEY(category_id) REFERENCES category(id) ON DELETE CASCADE
                              )''')

                cmd.execute('''CREATE UNIQUE INDEX IF NOT EXISTS index_category_stats ON category_stats(category_id,year,month)''')

                cmd.execute('''CREATE TABLE IF NOT EXISTS topic_stats(
                               topic_id INTEGER NOT NULL
                              ,year INTEGER NULL
                              ,month INTEGER NULL
                              ,transformed INT NOT NULL DEFAULT(1)
                              ,paper_count INTEGER
                              ,total_percent REAL
                              ,parent_percent REAL
                              ,FOREIGN KEY(topic_id) REFERENCES topic(id) ON DELETE CASCADE
                              )''')

                cmd.execute('''CREATE UNIQUE INDEX IF NOT EXISTS index_topic_stats ON topic_stats(topic_id,year,month,transformed)''')


                # Import table (other import tables will be created (and dropped) dynamically by pandas
                #               but this one has to be created by hand to define 'ARRAY' type)
                cmd.execute('''CREATE TABLE IF NOT EXISTS import_embeddings(
                               paper_id INTEGER                              
                              ,embedding ARRAY                              
                              )''')

                # Views
                cmd.execute('''CREATE VIEW IF NOT EXISTS archives AS
                               SELECT [archive], SUM(paper_count) AS paper_count 
                               FROM category                                
                               GROUP BY [archive]''')
                
       
    def import_snapshot(self):
        """ import new papers and possibly categories """
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:            

                cmd.execute('''PRAGMA foreign_keys = ON;''')
                
                cmd.execute('''INSERT OR IGNORE INTO paper(arxiv_id, title, abstract, abstract_length, creation_year, creation_month, update_date)
                            SELECT id, title, abstract, abstract_length, creation_year, creation_month, update_date
                            FROM import_abstracts
                            ''')
                
                cmd.execute('''WITH categories AS
                            (
                             SELECT cat_0 AS cat FROM import_abstracts
                             UNION
                             SELECT cat_1 FROM import_abstracts
                             UNION
                             SELECT cat_2 FROM import_abstracts
                            )
                            INSERT OR IGNORE INTO category (archive, name)
                            SELECT CASE WHEN INSTR(cat, '.') = 0 THEN cat ELSE SUBSTR(cat, 1, INSTR(cat, '.') - 1) END AS 'archive', cat
                            FROM categories
                            WHERE cat IS NOT NULL''')

                cmd.execute('''INSERT OR IGNORE INTO paper_categories(paper_id, category_id, is_primary)
                            SELECT paper.id, category.id, CASE WHEN category.name = import_abstracts.cat_0 THEN 1 ELSE 0 END
                            FROM import_abstracts
                            INNER JOIN paper ON import_abstracts.id = paper.arxiv_id
                            INNER JOIN category ON category.name IN (import_abstracts.cat_0, import_abstracts.cat_1, import_abstracts.cat_2)''')
               
                cmd.execute('''UPDATE category SET paper_count =
                              (SELECT COUNT(*) FROM paper_categories WHERE category.id = paper_categories.category_id)''')

                cmd.execute('''DROP TABLE import_abstracts''')
                
                conn.commit()

                #necessary after json import, since sqlite3 does not clean up automatically
                cmd.execute('''vacuum;''')


    def await_access(self):
        """ check sqlite3 journal file to handle concurrent database requests """
        counter = 0
        while os.path.isfile(self.journal) and counter < 10:
            time.sleep(3)
            counter += 1
    

    def get_papers_for_embedding(self, chunksize):
        """ get all papers without an embedding
        
          Arguments:
            chunksize: number of papers to return
        """

        return pd.read_sql_query(f'''SELECT p.id, p.abstract
                                     FROM paper p
                                     WHERE p.embedding IS NULL
                                     LIMIT {chunksize}''', self.connection())                          


    def import_embeddings(self, paper_ids, embeddings):
        """ import embeddings
          
          Arguments:
            paper_ids: paper ids
            embeddings: embeddings from SentenceTransformer
        """

        df = pd.DataFrame({'paper_id': paper_ids, 'embedding': embeddings})
                
        df.to_sql(con=self.connection(), name='import_embeddings', index=False, if_exists='append')


        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:

                cmd.execute('''UPDATE paper                    
                               SET embedding = I.embedding
                               FROM (SELECT paper_id, embedding FROM import_embeddings) AS I
                               WHERE paper.id = I.paper_id''')

                cmd.execute('''DELETE FROM import_embeddings''')

                conn.commit()


    def get_subsets(self, archive_max, category_min):
        """ get a mapping from arxiv category to either itself or the parent archive
        
          Arguments:
            archive_max: maximum number of papers in an archive, before creating subsets for its categories
            category_min: minimum number of papers in a category to form its own subset
        """
        
        return pd.read_sql_query(f'''SELECT c.id AS category_id
                                           ,CASE WHEN c.name != c.archive AND a.paper_count > {archive_max} AND c.paper_count > {category_min} THEN c.name
                                                 ELSE c.archive END AS subset
                                     FROM category c
                                     INNER JOIN archives a
                                      ON a.archive = c.archive''', self.connection()).groupby('subset')


    def get_papers_by_categories(self, categories):
        """ get all papers belonging to a list of categories
        
          Arguments:
            categories: list of category ids
        """

        categories = str(categories).strip('[]')

        papers = pd.read_sql_query(f'''SELECT p.id, p.abstract, p.embedding
                                       FROM paper AS p                                   
                                       INNER JOIN paper_categories AS pc
                                        ON pc.paper_id = p.id
                                       WHERE pc.category_id IN ({categories})
                                       AND p.embedding IS NOT NULL
                                       ORDER BY p.id''', self.connection())
        
        return papers[~papers['id'].duplicated()]
    

    def get_papers_for_training(self, model_filter, filter_value, percent_of_papers):
        """ get papers for training a model
        
          Arguments:
            model_filter: way of selection [None, 'archives', 'outliers', 'hierarchy']
            filter_value: hierarchy level to filter 'outliers' or 'hierarchy'
            percent_of_papers: percent of papers to return for 'archives', 'outliers' or 'hierarchy'
            
          Returns:
            DataFrame with paper ids, abstracts and embeddings
            NOTE: ordered by id - important to be able to select the same papers in order in get_papers_by_model()
        """

        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:

                if model_filter is None or model_filter == 'None':
                # return all papers

                    query = f'''SELECT p.id, p.abstract, p.embedding
                                FROM paper AS p
                                WHERE p.embedding IS NOT NULL
                                ORDER BY p.id'''

                    return pd.read_sql_query(query, self.connection())  


                # to improve performance use temp tables for selection involving percent_of_papers 
                # TODO: handle percent_of_papers = 100 seperately
                cmd.execute('''DROP TABLE IF EXISTS temp_var_papers''')
                cmd.execute('''DROP TABLE IF EXISTS temp_selection''')
                cmd.execute('''CREATE TABLE temp_var_papers(paper_id INTEGER)''')                    
                cmd.execute('''CREATE TABLE temp_selection(paper_id INTEGER, topic_id INTEGER)''')                    
                cmd.execute('''CREATE UNIQUE INDEX temp_index ON temp_var_papers(paper_id)''')
                 
                 
                if model_filter == 'archives':
                # return selection proportional to archives
                
                    archives = cmd.execute('''SELECT a.archive, a.paper_count
                                              FROM archives a''').fetchall()
   
                    for archive in archives:
                        
                        query = f'''INSERT INTO temp_selection(paper_id)
                                    SELECT p.id
                                    FROM paper p
                                    INNER JOIN paper_categories pc
                                     ON p.id = pc.paper_id
                                    INNER JOIN category c
                                     ON pc.category_id = c.id
                                    WHERE c.[archive] = '{archive[0]}' '''
                        
                        if percent_of_papers > 0 and percent_of_papers < 100:
                            query += f'''ORDER BY p.abstract_length DESC
                                         LIMIT {archive[1] * percent_of_papers // 100}'''
                        
                        cmd.execute(query)
                    
                    conn.commit()
                    
                    papers = pd.read_sql_query('''SELECT p.id, p.abstract, p.embedding 
                                                  FROM temp_selection s
                                                  INNER JOIN paper p
                                                   ON s.paper_id = p.id
                                                  ORDER BY p.id''', self.connection())
                                                
                elif model_filter == 'outliers':
                # return selection of outliers proportional to years
                
                    years = cmd.execute('''SELECT p.creation_year, COUNT(*) as paper_count
                                           FROM paper p
                                           GROUP BY p.creation_year''').fetchall()

                    for year in years:
                        
                        query = f'''INSERT INTO temp_selection(paper_id)
                                    SELECT p.id
                                    FROM paper p
                                    LEFT OUTER JOIN (paper_topics pt	 
                                    INNER JOIN topic t
                                     ON pt.topic_id = t.id
                                     AND t.hierarchy = {filter_value}
                                     AND t.orig_id != -1)
                                     ON p.id = pt.paper_id
                                    WHERE pt.paper_id IS NULL
                                      AND p.creation_year = {year[0]} '''
                        
                        if percent_of_papers > 0 and percent_of_papers < 100:
                            query += f'''ORDER BY p.abstract_length DESC
                                         LIMIT {year[1] * percent_of_papers // 100}'''
                        
                        cmd.execute(query)
                    
                    conn.commit()
                    
                    papers = pd.read_sql_query('''SELECT p.id, p.abstract, p.embedding 
                                                  FROM temp_selection s
                                                  INNER JOIN paper p
                                                   ON s.paper_id = p.id
                                                  ORDER BY p.id''', self.connection())
                                                
                elif model_filter == 'hierarchy':
                # return selection from hierarchy level proportional to topics
                    
                    topic_ids = cmd.execute(f'''SELECT t.id, t.paper_count
                                                FROM topic t
                                                WHERE t.hierarchy = {filter_value}
                                                 AND t.orig_id != -1 ''').fetchall()

                    cmd.execute(f'''INSERT INTO temp_var_papers(paper_id)
                                    SELECT p.id
                                    FROM paper p
                                    INNER JOIN paper_topics pt
                                     ON p.id = pt.paper_id
                                    INNER JOIN topic t
                                     ON t.id = pt.topic_id
                                    WHERE t.hierarchy = {filter_value}
                                      AND t.orig_id != -1
                                    GROUP BY p.id
                                    HAVING COUNT(*) > 1''')
                    
                    conn.commit()
                 
                    for topic_id in topic_ids:
                        
                        query = f'''INSERT INTO temp_selection(paper_id, topic_id)
                                    SELECT p.id
                                         , CASE WHEN EXISTS (SELECT paper_id FROM temp_var_papers WHERE paper_id = p.id) THEN -1
                                                ELSE {topic_id[0]} END
                                    FROM paper AS p
                                    INNER JOIN paper_topics AS pt
                                     ON pt.paper_id = p.id                                    
                                    WHERE pt.topic_id = {topic_id[0]} '''
                                                   
                        if percent_of_papers > 0 and percent_of_papers < 100:
                            query += f'''ORDER BY pt.probability DESC
                                         LIMIT {topic_id[1] * percent_of_papers // 100}'''
                        
                        cmd.execute(query)
                    
                    conn.commit()
                    
                    papers = pd.read_sql_query('''SELECT p.id, p.abstract, p.embedding, s.topic_id 
                                                  FROM temp_selection s
                                                  INNER JOIN paper p
                                                   ON s.paper_id = p.id
                                                  ORDER BY p.id''', self.connection())
                    
                # drop temp tables
                cmd.execute('''DROP INDEX temp_index''')
                cmd.execute('''DROP TABLE temp_var_papers''')
                cmd.execute('''DROP TABLE temp_selection''')
                
                conn.commit()
                    
                return papers[~papers['id'].duplicated()]


    def get_papers_by_model(self, model_name):
        """ get all papers that were used originally to train a model """        
        return pd.read_sql_query(f'''SELECT p.id, p.abstract
                                     FROM paper_topics pt
                                     INNER JOIN topic t 
                                      ON t.id = pt.topic_id
                                     INNER JOIN paper p
                                      ON p.id = pt.paper_id
                                     WHERE pt.transformed = 0
                                      AND t.model = '{model_name}'
                                     ORDER BY p.id''', self.connection())


    def get_models_for_transform(self):
        """ get models (except subsets) that are missing entries in paper_topics """
        return pd.read_sql_query('''SELECT t.hierarchy, t.model
                                    FROM topic t
                                    WHERE t.hierarchy > -1
                                    GROUP BY t.hierarchy, t.model
                                    HAVING SUM(paper_count) < (SELECT COUNT(*) FROM paper)''', self.connection())


    def get_papers_for_transform(self, model_name, chunksize):
        """ get all papers that have no entry in paper_topics for a given model """
        query = f'''SELECT p.id, p.abstract, p.embedding
                    FROM paper AS p
                    LEFT OUTER JOIN (paper_topics pt
                    INNER JOIN topic t
                     ON pt.topic_id = t.id
                    AND t.model = '{model_name}')
                     ON p.id = pt.paper_id                                     
                    WHERE t.id IS NULL'''

        if chunksize > 0:
            query += f' LIMIT {chunksize}'

        return pd.read_sql_query(query, self.connection())        


    def get_topics_without_label(self):
        """ get all topics without an llm_label """
        return pd.read_sql_query(f'''SELECT t.model, t.orig_id AS topic_id, p.abstract
                                     FROM topic t
                                     INNER JOIN paper_topics pt
                                      ON pt.topic_id = t.id
                                     INNER JOIN paper p
                                      ON pt.paper_id = p.id
                                     WHERE t.llm_label = '' ''', self.connection()).groupby('model')        


    def check_model_exists(self, model_name):
        """ check if any topics exists for a given model """
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:

                res = cmd.execute(f'''SELECT t.id
                                      FROM topic t
                                      WHERE t.model = '{model_name}'
                                      LIMIT 1''').fetchone()

                if res is None:
                    return False

                return True
                

    def import_topics(self, hierarchy, model_name, topic_model, paper_ids, excl_outliers=False):
        """ imports new model incl. topics and paper assignments
        
          Arguments:
            hierarchy: hierarchy level for the new model (should be -1 for subsets and 0+ for others)
            model_name: name of the trained model
            topic_model: BERTopic topic_model to extract topic names, labels and assignments
            paper_ids: paper ids in the same order as for training
            excl_outliers: set to true to exclude topic with orig_id -1 (for agglomerated models)
        """

        topics = topic_model.get_topic_info()

        labels = topic_model.custom_labels_ if isinstance(topic_model.representation_model, dict) else topics['Name']
        
        df = pd.DataFrame({'id':topics['Topic'], 'name':topics['Name'], 'llm_label':labels})
                
        df.to_sql(con=self.connection(), name='import_topics', index=False, if_exists='replace')        


        df = pd.DataFrame({'paper_id': paper_ids, 'topic_id': topic_model.topics_, 'probability': topic_model.probabilities_})

        if excl_outliers:
            df = df[df['topic_id'] != -1]
                
        df.to_sql(con=self.connection(), name='import_paper_topics', index=False, if_exists='replace')


        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:

                cmd.execute('''PRAGMA foreign_keys = ON;''')
    
                cmd.execute(f'''INSERT OR IGNORE INTO topic (hierarchy, model, orig_id, name, llm_label)
                                SELECT {hierarchy}, '{model_name}', id, name, llm_label
                                FROM import_topics''')

                cmd.execute(f'''INSERT OR IGNORE INTO paper_topics (paper_id, topic_id, probability)
                                SELECT pt.paper_id, t.id, probability
                                FROM import_paper_topics AS pt
                                INNER JOIN topic AS t
                                 ON t.model = '{model_name}' AND pt.topic_id = t.orig_id''')
                
                cmd.execute('''UPDATE topic SET paper_count =
                              (SELECT COUNT(*) FROM paper_topics
                               WHERE topic.id = paper_topics.topic_id)''')
                conn.commit()


                cmd.execute('''DROP TABLE import_topics''')
                cmd.execute('''DROP TABLE import_paper_topics''')
                 
                conn.commit()


    def import_paper_topics(self, model_name, paper_ids, topic_ids, probs, distributed=False):
        """ import paper assignments for exisiting topic
        
          Arguments:
            model_name: name of the trained model
            paper_ids: paper ids
            topic_ids: (original model) topic id assignment for each paper
            probs: probability for each topic assignment
            distributed: probabilities were calculated by ctfidf distribution instead of hdbscan transformation
        """        

        df = pd.DataFrame({'paper_id': paper_ids, 'topic_id': topic_ids, 'probability': probs})
                
        df.to_sql(con=self.connection(), name='import_paper_topics', index=False, if_exists='replace')


        transformed = 2 if distributed else 1

        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:

                cmd.execute('''PRAGMA foreign_keys = ON;''')
    
                cmd.execute(f'''INSERT OR IGNORE INTO paper_topics (paper_id, topic_id, probability, transformed)
                                SELECT pt.paper_id, t.id, probability, {transformed}
                                FROM import_paper_topics AS pt
                                INNER JOIN topic AS t
                                 ON t.model = '{model_name}' AND pt.topic_id = t.orig_id''')

                cmd.execute('''UPDATE topic SET paper_count =
                              (SELECT COUNT(*) FROM paper_topics
                               WHERE topic.id = paper_topics.topic_id)''')
                
                conn.commit()

              
                cmd.execute('''DROP TABLE import_paper_topics''')
                 
                conn.commit()


    def import_topic_labels(self, model_name, topic_ids, labels):
        """ import new llm_labels for existing topics
        
          Arguments:
            model_name: name of trained model
            topic_ids: (original model) topic ids
            labels: new labels
        """
        
        df = pd.DataFrame({'topic_id':topic_ids, 'llm_label':labels})
                
        df.to_sql(con=self.connection(), name='import_labels', index=False, if_exists='replace')        

        
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:


                cmd.execute(f'''UPDATE topic AS t
                                SET llm_label = i.llm_label
                                FROM import_labels AS i
                                WHERE t.model = '{model_name}'
                                  AND i.topic_id = t.orig_id''')
                
                conn.commit()

              
                cmd.execute('''DROP TABLE import_labels''')
                 
                conn.commit()
         

    def update_hierarchy(self, model_name, parent_model_name):
        """ set new agglomerated model as parent for exisiting model and transfer paper assignments
        
          Arguments:
            model_name: name of prior model
            parent_model_name: name of new agglomerated model
        """

        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:

                cmd.execute(f'''UPDATE topic
                                SET parent_id = (SELECT t_p.id
                                                 FROM topic t_p
                                                 INNER JOIN paper_topics pt_p
                                                  ON t_p.id = pt_p.topic_id
                                                 INNER JOIN paper_topics pt_c
                                                  ON pt_c.paper_id = pt_p.paper_id
                                                 WHERE pt_c.topic_id = topic.id
                                                  AND t_p.model = '{parent_model_name}'
                                                 LIMIT 1)
                                WHERE model = '{model_name}'
                                  AND orig_id != -1''')
                
                conn.commit()

                cmd.execute('''PRAGMA foreign_keys = ON;''')
    
                cmd.execute(f'''INSERT OR IGNORE INTO paper_topics (paper_id, topic_id, probability, transformed)
                               SELECT pt.paper_id, t.parent_id, pt.probability, pt.transformed
                               FROM paper_topics pt
                               INNER JOIN topic t
                                ON t.id = pt.topic_id
                               WHERE t.model = '{model_name}'
                                 AND t.parent_id IS NOT NULL''')

                cmd.execute('''UPDATE topic SET paper_count =
                              (SELECT COUNT(*) FROM paper_topics
                               WHERE topic.id = paper_topics.topic_id)''')

                conn.commit()


    def update_category_stats(self):
        """ update the category_stats table to be used for visualization """
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:

                cmd.execute('''DELETE FROM category_stats''')

#                cmd.execute('''INSERT INTO category_stats(category_id, year, paper_count)
#                               SELECT pt.category_id, p.creation_year, COUNT(p.id)
#                               FROM paper p
#                               INNER JOIN paper_categories pt
#                                ON p.id = pt.paper_id
#                               GROUP BY pt.category_id, p.creation_year''')

                cmd.execute('''INSERT INTO category_stats(category_id, year, month, paper_count)
                               SELECT pt.category_id, p.creation_year, p.creation_month, COUNT(p.id)
                               FROM paper p
                               INNER JOIN paper_categories pt
                                ON p.id = pt.paper_id
                               GROUP BY pt.category_id, p.creation_year, p.creation_month''')

#                cmd.execute('''WITH Years AS
#                               (
#                                SELECT p.creation_year AS total_year, COUNT(*) AS total_count
#                                FROM paper p
#                                GROUP BY p.creation_year
#                               ) 
#                               UPDATE category_stats
#                               SET total_percent = CAST(paper_count * 100 AS float) / (SELECT total_count
#                                                                                       FROM Years
#                                                                                       WHERE total_year = year)
#                               WHERE month IS NULL''')

                cmd.execute('''WITH Months AS
                               (
                                SELECT p.creation_year AS total_year, p.creation_month AS total_month, COUNT(*) AS total_count
                                FROM paper p
                                GROUP BY p.creation_year, p.creation_month
                               ) 
                               UPDATE category_stats
                               SET total_percent = CAST(paper_count * 100 AS float) / (SELECT total_count
                                                                                       FROM Months
                                                                                       WHERE total_year = year AND total_month = month)
                               WHERE month IS NOT NULL''')

                conn.commit()
     

    def update_topic_stats(self):
        """ update the topic_stats table to be used for visualization """
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:

                cmd.execute('''UPDATE topic SET paper_count =
                              (SELECT COUNT(*) FROM paper_topics
                               WHERE topic.id = paper_topics.topic_id)''')

                cmd.execute('''DELETE FROM topic_stats''')

#                cmd.execute('''INSERT INTO topic_stats(topic_id, year, paper_count)
#                               SELECT pt.topic_id, p.creation_year, COUNT(p.id)
#                               FROM paper p
#                               INNER JOIN paper_topics pt
#                                ON p.id = pt.paper_id
#                               GROUP BY pt.topic_id, p.creation_year''')

                cmd.execute('''INSERT INTO topic_stats(topic_id, year, month, transformed, paper_count)
                               SELECT pt.topic_id, p.creation_year, p.creation_month, 1, COUNT(p.id)
                               FROM paper p
                               INNER JOIN paper_topics pt
                                ON p.id = pt.paper_id
                               WHERE pt.transformed < 2
                               GROUP BY pt.topic_id, p.creation_year, p.creation_month''')

                cmd.execute('''INSERT INTO topic_stats(topic_id, year, month, transformed, paper_count)
                               SELECT pt.topic_id, p.creation_year, p.creation_month, 2, COUNT(p.id)
                               FROM paper p
                               INNER JOIN paper_topics pt
                                ON p.id = pt.paper_id
                               GROUP BY pt.topic_id, p.creation_year, p.creation_month''')

#                cmd.execute('''WITH Years AS
#                               (
#                                SELECT p.creation_year AS total_year, COUNT(*) AS total_count
#                                FROM paper p
#                                GROUP BY p.creation_year
#                               ) 
#                               UPDATE topic_stats
#                               SET total_percent = CAST(paper_count * 100 AS float) / (SELECT total_count
#                                                                                       FROM Years
#                                                                                       WHERE total_year = year)
#                               WHERE month IS NULL''')

                cmd.execute('''WITH Months AS
                               (
                                SELECT p.creation_year AS total_year, p.creation_month AS total_month, COUNT(*) AS total_count
                                FROM paper p
                                GROUP BY p.creation_year, p.creation_month
                               ) 
                               UPDATE topic_stats
                               SET total_percent = CAST(paper_count * 100 AS float) / (SELECT total_count
                                                                                       FROM Months
                                                                                       WHERE total_year = year AND total_month = month)
                               WHERE month IS NOT NULL''')

#                cmd.execute('''UPDATE topic_stats
#                               SET parent_percent = CAST(paper_count * 100 AS float) / (SELECT ts_p.paper_count
#                                                                                        FROM topic t_c
#                                                                                        INNER JOIN topic_stats ts_p
#                                                                                         ON ts_p.topic_id = t_c.parent_id
#                                                                                        AND ts_p.year = topic_stats.year
#                                                                                        AND IFNULL(ts_p.month,0) = IFNULL(topic_stats.month,0)
#                                                                                        WHERE t_c.id = topic_stats.topic_id)''')
                conn.commit()
        
        
    def get_category_stats_by_month(self):
        """ get monthly category stats, ordered by date """
        
        return pd.read_sql_query(f'''SELECT c.id, c.archive as parent
                                          , c.name AS label, cs.year || '-' || printf('%02d', cs.month) || '-01' AS publication
                                          , cs.paper_count, cs.total_percent / 100 AS total_percent, 0 AS parent_percent
                                     FROM category_stats cs
                                     INNER JOIN category c
                                      ON c.id = cs.category_id
                                     WHERE cs.month IS NOT NULL
                                     ORDER BY c.name, cs.year || '-' || printf('%02d', cs.month) || '-01' ''', self.connection()) 

                                
    def get_topic_stats_by_month(self, hierarchy = 0, top_n_topics = 100, incl_approximation = False):
        """ get monthly topic stats, ordered by date
        
          Arguments:
            hierarchy: hierarchy level            
            incl_approximation: incl. topic assignments from BERTopic approximation
        """
        
        transformed = 2 if incl_approximation else 1
        
        return pd.read_sql_query(f'''WITH topics AS
                                     (
                                        SELECT ts.topic_id
                                        FROM topic_stats ts
                                        INNER JOIN topic t
                                         ON ts.topic_id = t.id
                                        WHERE t.hierarchy = {hierarchy}
                                        GROUP BY topic_id
                                        ORDER BY MAX(total_percent) DESC
                                        LIMIT {top_n_topics}
                                     )
                                     SELECT t.id, t.parent_id as parent
                                          , t.llm_label AS label, ts.year || '-' || printf('%02d', ts.month) || '-01' AS publication
                                          , ts.paper_count, ts.total_percent / 100 AS total_percent, ts.parent_percent / 100 AS parent_percent
                                     FROM topic_stats ts
                                     INNER JOIN topic t
                                      ON t.id = ts.topic_id
                                     WHERE t.orig_id != -1
                                      AND t.id IN (SELECT topic_id FROM topics)
                                      AND ts.month IS NOT NULL
                                      AND ts.transformed = {transformed}
                                     ORDER BY t.llm_label, ts.year || '-' || printf('%02d', ts.month) || '-01' ''', self.connection()) 


    def get_representative_papers_for_category(self, category_id, year, month, top_n_papers):
        """ get papers representative for a given category
        
          Arguments:
            category_id: database category id
            year: creation_year of papers
            month: creation_month of papers
            top_n_papers: how many papers to return
        """
        
        return pd.read_sql_query(f'''SELECT p.arxiv_id, p.title
                                     FROM paper p
                                     INNER JOIN paper_categories pt
                                      ON p.id = pt.paper_id
                                     WHERE pt.category_id = {category_id}
                                       AND p.creation_year = {year}
                                       AND p.creation_month = {month}
                                       AND p.title NOT LIKE '%(formula)%'                                     
                                     LIMIT {top_n_papers} ''', self.connection())     
    
    
    def get_representative_papers_for_topic(self, topic_id, year, month, top_n_papers, approximation=False):
        """ get papers representative for a given topic
        
          Arguments:
            topic_id: database topic id
            year: creation_year of papers
            month: creation_month of papers
            top_n_papers: how many papers to return
            approximation: get papers assigned by approximation first
        """
        
        transformed = 2 if approximation else 0
        
        return pd.read_sql_query(f'''SELECT p.arxiv_id, p.title
                                     FROM paper p
                                     INNER JOIN paper_topics pt
                                      ON p.id = pt.paper_id
                                     WHERE pt.topic_id = {topic_id}
                                       AND p.creation_year = {year}
                                       AND p.creation_month = {month}
                                       AND pt.transformed = {transformed}
                                       AND p.title NOT LIKE '%(formula)%'
                                     ORDER BY pt.probability DESC
                                     LIMIT {top_n_papers} ''', self.connection())
    

    def delete_model(self, model):
        """ delete all topics belonging to a given model """
        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:

                cmd.execute('''PRAGMA foreign_keys = ON;''')
            
                cmd.execute(f'''DELETE FROM topic
                               WHERE model = '{model}' ''')

                conn.commit()

                
    def vacuum(self):
        """ helper function to clean and compress the database """

        with closing(self.connection()) as conn:
            with closing(conn.cursor()) as cmd:

                cmd.execute('''VACUUM''')               