import json

class Config:
     """ encapsulates .json config file """

     def __init__(self):

          with open('config.json') as file:
               conf = json.load(file)

               self.database = conf['database']

               self.input_snapshot = conf['input_snapshot']
               if isinstance(conf['input_taxonomy'], list):
                    self.input_taxonomy = conf['input_taxonomy']
               else:
                    self.input_taxonomy = [conf['input_taxonomy']]
               self.input_update_date = conf['input_update_date']

               self.pre_year_min = conf['pre_year_min']

               self.subset_archive_max_count = conf['subset_archive_max_count']
               self.subset_category_min_count = conf['subset_category_min_count']
               self.subset_bert_params = conf['subset_bert_params']            

               self.embeddings_model = conf['embeddings_model']
               self.embeddings_precision = conf['embeddings_precision']
               self.embeddings_chunk_size = conf['embeddings_chunk_size']
               
               self.llm_model = conf['llm_model']
               self.llm_prompt = conf['llm_prompt']
               self.llm_stopwords = conf['llm_stopwords']
               self.llm_diversity = conf['llm_diversity'] 
                 
               self.transform_chunk_size = conf['transform_chunk_size']               
               
               self.train_models = dict()
               for model in conf['train_models']:
                    self.train_models[model] = TrainModel(conf['train_models'][model])

               self.agg_models = dict()
               for model in conf['agg_models']:
                    self.agg_models[model] = AggModel(conf['agg_models'][model])

               self.bert_params = dict()
               for bert in conf['bert_params']:
                    self.bert_params[bert] = BertParams(conf['bert_params'][bert])
          

class TrainModel:

     def __init__(self, params):

          self.hierarchy = params['hierarchy']
          self.model_filter = params['model_filter']
          self.filter_value = params['filter_value']          
          self.percent_of_papers = params['percent_of_papers']          
          self.bert_params = params['bert_params']
          self.overwrite = params['overwrite']

class AggModel:

     def __init__(self, params):

          self.hierarchy = params['hierarchy']
          self.parent_model = params['parent_model']
          self.max_cluster_distance = params['max_cluster_distance']
          self.excl_outliers = params['excl_outliers']         
          self.overwrite = params['overwrite']

class BertParams:

     def __init__(self, params):

          self.keyword_min_freq = params['keyword_min_freq']
          self.keyword_diversity = params['keyword_diversity']

          self.ngram_start = params['ngram_start']
          self.ngram_end = params['ngram_end']          
          
          self.umap_parametric = params['umap_parametric']
          self.umap_neighbors = params['umap_neighbors']
          self.umap_components = params['umap_components']
          self.umap_min_dist = params['umap_min_dist']
          self.umap_random_state = params['umap_random_state']

          cluster_sizes = params['hdbscan_cluster_size']

          if isinstance(cluster_sizes, list):
               self.hdbscan_cluster_sizes = cluster_sizes
          else:
               self.hdbscan_cluster_sizes = [cluster_sizes]
           
          sample_sizes = params['hdbscan_sample_size']

          if isinstance(sample_sizes, list):
               self.hdbscan_sample_sizes = sample_sizes
          else:
               self.hdbscan_sample_sizes = [sample_sizes]
               
          self.hdbscan_min_clusters = params['hdbscan_min_clusters']
          self.hdbscan_predict = params['hdbscan_predict']

          self.generate_labels = params['generate_labels']
          
          self.top_n_words = params['top_n_words']
          self.reduce_topics = params['reduce_topics']
          self.outlier_threshold = params['outlier_threshold']

          self.verbose = params['verbose']
