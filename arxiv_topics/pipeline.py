import pandas as pd
import numpy as np
import types
import gc

from bertopic import BERTopic
from bertopic._bertopic import TopicMapper
from bertopic.cluster._utils import hdbscan_delegator, is_supported_hdbscan
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, LlamaCPP
from collections import Counter
from hdbscan import HDBSCAN
from joblib import Memory
from llama_cpp import Llama
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from typing import List, Tuple
from umap import UMAP


class Pipeline:
     """ wrapper around BERTopic as clustering pipeline """

     LLM_ASPECT = 'Label'
     STOP_WORDS = list(ENGLISH_STOP_WORDS.union(['(formula)', 'formula']))


     def __init__(self, embeddings_precision, bert_params
                 ,llm_model = None, llm_prompt = None, llm_diversity = None, llm_stopwords = None):

         ## BERTopic
         self.n_gram_range = (bert_params.ngram_start, bert_params.ngram_end)
         self.top_n_words = bert_params.top_n_words
         self.reduce_topics = bert_params.reduce_topics
         self.outlier_threshold = bert_params.outlier_threshold
         self.verbose = bert_params.verbose
         
         
         ## Vectorizer         
         self.vectorizer_model = CountVectorizer(ngram_range=self.n_gram_range, min_df=bert_params.keyword_min_freq, stop_words=self.STOP_WORDS)
         
         
         ## UMAP
         umap_metric = 'bit_hamming' if embeddings_precision in ['binary', 'ubinary'] else 'cosine'
         umap_random_state = None if bert_params.umap_random_state else 42
         
         if bert_params.umap_parametric:
              #use tensorflow+keras network to fit UMAP
              #NOTE: not recommended - will result in worse results unless implementing your own network
              from umap.parametric_umap import ParametricUMAP
              
              self.umap_model = ParametricUMAP(n_neighbors=bert_params.umap_neighbors, n_components=bert_params.umap_components, min_dist=bert_params.umap_min_dist
                              , metric=umap_metric, random_state=umap_random_state, verbose=bert_params.verbose)                 
         else:
              self.umap_model = UMAP(n_neighbors=bert_params.umap_neighbors, n_components=bert_params.umap_components, min_dist=bert_params.umap_min_dist
                              , metric=umap_metric, random_state=umap_random_state, verbose=bert_params.verbose, n_epochs=300)
          
          
         ## HDBScan
         gen_span_tree = True if len(bert_params.hdbscan_cluster_sizes) > 1 or len(bert_params.hdbscan_sample_sizes) > 1 or bert_params.hdbscan_predict else False

         self.hdbscan_model = HDBSCAN(min_cluster_size=bert_params.hdbscan_cluster_sizes[0]
                                    , min_samples=bert_params.hdbscan_sample_sizes[0], metric='minkowski', p=2 #bert_params.umap_components
                                    , prediction_data=bert_params.hdbscan_predict, gen_min_span_tree=gen_span_tree)
         
         if bert_params.hdbscan_min_clusters == 1:
              self.hdbscan_model.allow_single_cluster = True
         
         #workaround for p values different from 2
         #NOTE: needs an additional fix in hdbscan module - see https://github.com/scikit-learn-contrib/hdbscan/pull/637
         self.hdbscan_model._metric_kwargs['p'] = self.hdbscan_model.p
         
         #custom parameters
         self.hdbscan_model.cluster_sizes = bert_params.hdbscan_cluster_sizes
         self.hdbscan_model.sample_sizes = bert_params.hdbscan_sample_sizes
         self.hdbscan_model.min_clusters = bert_params.hdbscan_min_clusters
         
         
         ## LLM
         
         #chained "Main" representation model plus aspect is not supported
         #self.representation_model = [KeyBERTInspired(top_n_words=bert_params.top_n_words), MaximalMarginalRelevance(diversity=bert_params.keyword_diversity)]
         self.representation_model = MaximalMarginalRelevance(diversity=bert_params.keyword_diversity)                     

         #only set LLM params, init model later
         self.generate_labels = bert_params.generate_labels
         self.llm_model = llm_model
         self.llm_prompt = llm_prompt
         self.llm_diversity = llm_diversity
         self.llm_stopwords = llm_stopwords  
         

     def fit(self, abstracts, embeddings, labels = None, embedding_model = None, outlier_label = 'Outliers'):
          """ Fit the models on a collection of papers, generate topics, and return the probabilities and topic per paper.

            Arguments:
                abstracts: list of paper abstracts to fit on
                embeddings: pre-trained abstract embeddings
                labels: pre-determined labels
                embedding_model: embedding_model to use, when not providing pre-trained embeddings or providing labels
                outlier_label: label for abstracts that cannot be fitted
                
            Returns:
                fitted BERTopic topic_model
          """
          
          topic_model = BERTopic(
               
               embedding_model=embedding_model,
               
               umap_model=self.umap_model,
               hdbscan_model=self.hdbscan_model,
               vectorizer_model=self.vectorizer_model,
               representation_model = self.representation_model,
               top_n_words=self.top_n_words,
               nr_topics = 'auto' if self.reduce_topics else None,

               verbose = self.verbose
             )

                  
          if self.umap_model.metric == 'bit_hamming' and len(abstracts.index) < 4096:
               #UMAP has a more precise calculation for "small data", but it fails for bit embeddings
               self.umap_model.force_approximation_algorithm = True               
               
          if len(self.hdbscan_model.cluster_sizes) > 1 or len(self.hdbscan_model.sample_sizes) > 1:          
               #override hdbscan method to try multiple parameters and choose max relative DBCV score
               topic_model._cluster_embeddings = types.MethodType(_cluster_embeddings_best, topic_model)

          if (self.generate_labels == True) and (self.llm_model is not None):
               #override bertopic method to lazy load LLM model (thereby leaving more memory to UMAP & HDBScan)
               topic_model.llm_model = self.llm_model
               topic_model.llm_prompt = self.llm_prompt
               topic_model.llm_diversity = self.llm_diversity
               topic_model.llm_stopwords = self.llm_stopwords                 
               topic_model._extract_topics = types.MethodType(_lazy_extract_topics, topic_model)
               
          
          # Train
          topics = topic_model.fit_transform(abstracts, embeddings, y=labels)

          if self.verbose:
               print(topic_model.get_topic_info())
          
          if topic_model._outliers != 1:
               outlier_label = None
          elif self.outlier_threshold < 1:
               print('reducing outliers')
               topic_model.reduce_outliers(abstracts, topics, strategy="distributions", threshold=self.outlier_threshold)

          # Set custom labels
          self.set_custom_labels(topic_model, outlier_label)

          return topic_model


     @classmethod
     def save_model(cls, topic_model, model):
          """ save BERTopic topic_model to file
          
            Arguments:
                topic_model: BERTopic topic_model
                model: model name
          """
          
          if model.startswith('subset'):
               model_path = f'./output/subset/{model.replace("subset_","")}.pkl'
          else:
               model_path = f'./output/{model}.pkl'
        
          Path(model_path).parents[0].mkdir(parents=True, exist_ok=True)

          topic_model.save(model_path)
          
          # save ParametricUMAP network
          if 'parametric' in str(type(topic_model.umap_model)).lower():               
               topic_model.umap_model.save(model_path.replace('.pkl', '_pUMAP'))
               
               
     @classmethod
     def load_model(cls, model, drop_llm = False):
          """ load BERTopic topic_model from file
          
            Arguments:
                model: name of the model
                drop_llm: release llm model to free memory e.g. for transform (make sure not to save the topic_model afterwards)
          """

          if model.startswith('subset'):
               model_path = f'./output/subset/{model.replace("subset_","")}.pkl'
          else:
               model_path = f'./output/{model}.pkl'


          # making sure pickle can find the override methods
          BERTopic._cluster_embeddings_best = _cluster_embeddings_best
          BERTopic._lazy_extract_topics = _lazy_extract_topics          
          
          # load model
          # NOTE: if using bit_hamming metric and the model fails after loading, update pynndescent
          # https://github.com/lmcinnes/pynndescent/commit/1c0d2e93a2b064e581ada09c9d8b980e87b22c02
          topic_model = BERTopic.load(model_path)

          # reset stop words (they get lost when pickling)
          topic_model.vectorizer_model.stop_words = cls.STOP_WORDS

          # load ParametricUMAP network
          if 'parametric' in str(type(topic_model.umap_model)).lower():
               from umap.parametric_umap import load_ParametricUMAP
               topic_model.umap_model = load_ParametricUMAP(model_path.replace('.pkl', '_pUMAP'))

          # release llm model to free memory 
          if drop_llm and (isinstance(topic_model.representation_model, dict)):
               topic_model.generate_labels = False
               topic_model.representation_model.pop(cls.LLM_ASPECT)
               gc.collect()
                    

          return topic_model


     @classmethod
     def merge_topics(cls, topic_model, abstracts, topics_to_merge):
          """ merge given topics and generate new representations
          
            Arguments:
              topic_model: BERTopic topic_model
              abstracts: all abstracts used to train the model (in the same order)
              topics_to_merge: list of lists of (original model) topic ids
          """
          
          topic_model.merge_topics(abstracts, topics_to_merge)

          cls.set_custom_labels(topic_model)


     @classmethod
     def regenerate_labels(cls, topic_model, abstracts, topic_ids
                         , llm_prompt = None, llm_diversity = None, llm_stopwords = None):
          """ regenerate LLM labels for selected topics
             
            Arguments:
                topic_model: BERTopic topic_model
                abstracts: all abstracts from the topics to be merged
                topic_ids: topic_ids of the provided abstracts
                llm_prompt: new prompt for llm (optional)
                llm_diversity: new diversity value for llm (optional)
                llm_stopwords: new stopwords for llm (optional)
            
            Returns:
                topic_ids, cleaned llm labels
          """

          documents = pd.DataFrame({"Document": abstracts,
                                    "ID": range(len(abstracts)),
                                    "Topic": topic_ids,
                                    "Image": None})      

          words = topic_model.vectorizer_model.get_feature_names_out()
          
          labels = sorted(list(documents.Topic.unique()))
          labels = [int(label) for label in labels]


          # Get top indices and values per row in a sparse c-TF-IDF matrix
          top_n_words = max(topic_model.top_n_words, 30)
          indices = topic_model._top_n_idx_sparse(topic_model.c_tf_idf_, top_n_words)
          scores = topic_model._top_n_values_sparse(topic_model.c_tf_idf_, indices)
          sorted_indices = np.argsort(scores, 1)
          indices = np.take_along_axis(indices, sorted_indices, axis=1)
          scores = np.take_along_axis(scores, sorted_indices, axis=1)


          # Get top words per topic based on c-TF-IDF score
          topics = {label: [(words[word_index], score)
                          if word_index is not None and score > 0
                          else ("", 0.00001)
                          for word_index, score in zip(indices[label + topic_model._outliers][::-1]
                                                      , scores[label + topic_model._outliers][::-1])
                          ]
                     for label in labels}


          # Generate new labels          
          if isinstance(topic_model.representation_model, dict) and topic_model.representation_model.get(cls.LLM_ASPECT):
               if isinstance(topic_model.representation_model[cls.LLM_ASPECT], list):
                    for tuner in topic_model.representation_model[cls.LLM_ASPECT]:
                         if isinstance(tuner, LlamaCPP) and llm_prompt is not None:
                            tuner.prompt=llm_prompt
                            tuner.diversity=llm_diversity
                            tuner.pipeline_kwargs={'stop':llm_stopwords}
                         topics = tuner.extract_topics(topic_model, documents, topic_model.c_tf_idf_, topics)
               else:
                    tuner = topic_model.representation_model[cls.LLM_ASPECT]
                    if isinstance(tuner, LlamaCPP) and llm_prompt is not None:
                            tuner.prompt=llm_prompt
                            tuner.diversity=llm_diversity
                            tuner.pipeline_kwargs={'stop':llm_stopwords}
                    topics = tuner.extract_topics(topic_model, documents, topic_model.c_tf_idf_, topics)
          else:
               print('no LLM representation defined')
               return


          #TODO: update topic_model and set custom labels

          return topics.keys(), [label[0][0].replace('&quot;','').strip(' \'`"*.?') for label in topics.values()]

               
     @staticmethod
     def set_custom_labels(topic_model, outlier_label = None):
          ''' Set cleaned LLM topic names and predefined outlier label as custom labels
              (custom labels are used for creating heatmap, barchart and hierarchical topics)
            
            Arguments:
                topic_model: BERTopic topic_model
                outlier_label: label for abstracts that cannot be fitted
          '''

          if not isinstance(topic_model.representation_model, dict):
               return
          
          all_topics = topic_model.get_topics(full=True)
          
          topic_labels = pd.DataFrame.from_dict(all_topics[Pipeline.LLM_ASPECT])
          topic_labels = topic_labels.iloc[0].apply(lambda x: x[0].replace('&quot;','').strip(' \'`"*.?')).tolist()

          if outlier_label is not None:
               topic_labels[0] = outlier_label

          topic_model.set_topic_labels(topic_labels)
          

     @classmethod
     def evaluate_model(cls, topic_model, model, abstracts=None, heatmap=True, barchart=False, hierarchy=False):
          """ create various visualizations from the BERTopic model to subjectively evaluate performance """
          
          Path('./output').mkdir(exist_ok=True)
          
          if heatmap:
               print('creating heatmap')
    
               # force using c-TF-IDFs
               topic_model.topic_embeddings_ = None
        
               fig = topic_model.visualize_heatmap(custom_labels=True)
               
               fig.write_html(f'./output/{model}_heatmap.html')

               fig.show()

          if barchart:
               print('creating barchart')
        
               fig = topic_model.visualize_barchart(custom_labels=True)
               
               fig.write_html(f'./output/{model}_barchart.html')

               fig.show()

          if hierarchy:

               if abstracts is None:
                    print('abstracts used when generating the model (in the same order) needed to create hierarchy')
                    
               else:
                    print('creating hierarchy')

                    hierarchy = topic_model.hierarchical_topics(abstracts)

                    hierarchy.to_csv(f'./output/{model}_hierarchy.csv', sep='\t') 

                    fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchy, custom_labels=True)

                    fig.write_html(f'./output/{model}_hierarchy.html')

                    fig.show()
         

### BERTopic overrides

def _lazy_extract_topics(self, documents: pd.DataFrame, embeddings: np.ndarray = None, mappings=None, verbose: bool = False):
     """ Extract topics from the hdbscan clusters using a class-based TF-IDF
        ,load aspect model beforehand if not loaded yet

     Arguments:
        documents: Dataframe with documents and their corresponding IDs
        embeddings: The document embeddings
        mappings: The mappings from topic to word        
     """

     if (not isinstance(self.representation_model, dict)) and (self.llm_model is not None):
          #lazy load LLM model
          llm = Llama(model_path=self.llm_model, n_gpu_layers=-1, n_ctx=2048)
                    
          self.representation_model = {
             'Main': self.representation_model
            ,Pipeline.LLM_ASPECT: [self.representation_model
                             , LlamaCPP(llm, prompt=self.llm_prompt, diversity=self.llm_diversity, nr_docs=5, pipeline_kwargs={'stop':self.llm_stopwords})]
          }  
    
     if verbose:
          print("Representation - Extracting topics from clusters using representation models.")
        
     documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
     self.c_tf_idf_, words = self._c_tf_idf(documents_per_topic)
     self.topic_representations_ = self._extract_words_per_topic(words, documents)
     self._create_topic_vectors(documents=documents, embeddings=embeddings, mappings=mappings)
     self.topic_labels_ = {key: f"{key}_" + "_".join([word[0] for word in values[:4]])
                          for key, values in
                          self.topic_representations_.items()}
                          
     if verbose:
          print("Representation - Completed.")


def _cluster_embeddings_best(self,
                            umap_embeddings: np.ndarray,
                            documents: pd.DataFrame,
                            partial_fit: bool = False,
                            y: np.ndarray = None) -> Tuple[pd.DataFrame,
                                                      np.ndarray]:
     """ Cluster UMAP embeddings with variable HDBSCAN parameters

     Arguments:
       umap_embeddings: The reduced sentence embeddings from UMAP
       documents: Dataframe with documents and their corresponding IDs
       partial_fit: for online learning only (not supported)

     Returns:
       documents: Updated dataframe with documents, their corresponding IDs and newly added Topics
       probabilities: The distribution of probabilities
     """

     if partial_fit:
          raise ValueError('partial_fit cannot be used with variable hdbscan parameters')

     if self.verbose:
          print('finding optimal hdbscan parameters')
     
     best_dbcv = 0
     best_cluster_size = 0
     best_sample_size = 0
     best_p = 0     
     
     recompute = False
     
     # use memory to change 'min_cluster_size' without recomputing
     self.hdbscan_model.memory = Memory(location='/tmp/joblib', verbose=0)
     
     for min_samples in self.hdbscan_model.sample_sizes:
          for p_value in [2]: # [1,1.6,2,self.umap_model.n_components]: #needs fix in hdbscan module - see https://github.com/scikit-learn-contrib/hdbscan/pull/637
               for min_cluster_size in self.hdbscan_model.cluster_sizes:
               
                    self.hdbscan_model._relative_validity = None
                     
                    self.hdbscan_model.min_cluster_size = min_cluster_size
                    self.hdbscan_model.min_samples = min_samples
                    self.hdbscan_model.p = p_value
                    self.hdbscan_model._metric_kwargs['p'] = p_value
                    self.hdbscan_model.fit(umap_embeddings, y=y)
                     
                    # relative DBCV score
                    dbcv = self.hdbscan_model.relative_validity_                    

                    cluster_count = len(self.hdbscan_model.cluster_persistence_)

                    if self.verbose:                         
                         labels_ = Counter(self.hdbscan_model.labels_.tolist())
                         if labels_.get(-1) is not None:
                              outliers = labels_[-1]
                         else:
                              outliers = 0
                         print(f'cluster_size:{min_cluster_size}, samples:{min_samples}, p:{p_value} - topics:{cluster_count}, outliers:{outliers}, score:{dbcv}')
                   
                    if (dbcv > best_dbcv) and (cluster_count >= self.hdbscan_model.min_clusters):
                         best_dbcv = dbcv
                         best_cluster_size = min_cluster_size
                         best_sample_size = min_samples
                         best_p = p_value
                         recompute = False
                    else:
                         recompute = True

     if recompute: #TODO: save best model instead(?)
          self.hdbscan_model._relative_validity = None
                
          self.hdbscan_model.min_cluster_size = best_cluster_size
          self.hdbscan_model.min_samples = best_sample_size
          self.hdbscan_model.p = best_p
          self.hdbscan_model._metric_kwargs['p'] = best_p
          self.hdbscan_model.fit(umap_embeddings, y=y)

     
     labels = self.hdbscan_model.labels_    
     documents['Topic'] = labels
     self._update_topic_size(documents)

     self._outliers = 1 if -1 in set(labels) else 0

     # Extract probabilities
     probabilities = None
     if hasattr(self.hdbscan_model, "probabilities_"):
          probabilities = self.hdbscan_model.probabilities_

          if self.calculate_probabilities and is_supported_hdbscan(self.hdbscan_model):
               probabilities = hdbscan_delegator(self.hdbscan_model, "all_points_membership_vectors")

     self.topic_mapper_ = TopicMapper(self.topics_)
     
     return documents, probabilities
