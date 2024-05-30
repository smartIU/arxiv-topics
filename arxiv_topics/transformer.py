from sentence_transformers import SentenceTransformer

class Transformer:     

     def EmbeddingModel(model):
          
          transformer = SentenceTransformer(model, device='cuda')
          transformer.max_seq_length = 256

          return transformer
