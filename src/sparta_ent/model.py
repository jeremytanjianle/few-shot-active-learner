import spacy 
from torch import nn
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModel
# from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
from ..data import Doc_Tokens, Doc_Span

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class Sparta_Ent(nn.Module):
    
    def __init__(self, bert_name = 'albert-base-v2', use_proj = True):
        super().__init__()
        
        # encoder
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name, use_fast=False)
        self.encoder = AutoModel.from_pretrained(bert_name)

        # scorer parameters
        self.threshold_bias = nn.Parameter(torch.zeros(1))
        self.act = nn.Sigmoid() # nn.ReLU()
        self.proj = nn.Sequential(nn.Linear(768,768), nn.Linear(768,768))
        self.use_proj = use_proj

        # supports
        self.references = {}
    
    def encode_span(self, doc_tokens: Doc_Tokens, span_width_tuple: tuple, verbose=False):
        """
        :param doc_tokens: Doc_Tokens object that contains the spans
        :param span_width_tuple: tuple indicating start and end of spacy token spans
        :return span_encoding: (span length x embed dimensions) torch array
        """
        # get span 
        span_start, span_end = span_width_tuple
        span_start, span_end = doc_tokens.subword_idx[span_start], doc_tokens.subword_idx[span_end+1]

        # encode the pieces and slice according to span
        # span starts and end +1 to account for inserted CLS token
        piece_id = self.tokenizer.encode(doc_tokens.subword_tokens, return_tensors='pt')
        encodings = self.encoder(piece_id)['last_hidden_state']
        span_encoding = encodings[0, span_start+1: span_end+1, :]

        if verbose:
            print(doc_tokens.subword_tokens[span_start: span_end])
        
        return span_encoding
    
    
    def encode_spans(self, doc_tokens: Doc_Tokens, span_width_tuples: list, verbose=False):
        """
        :param doc_tokens: Doc_Tokens object that contains the spans
        :param span_width_tuples: list of tuple indicating start and end of spacy token spans
        :return span_encoding: (span length x embed dimensions) torch array
        """
        # encode the pieces and slice according to span
        piece_id = self.tokenizer.encode(doc_tokens.subword_tokens, return_tensors='pt')
        encodings = self.encoder(piece_id)['last_hidden_state']
        if self.use_proj: encodings = self.proj(encodings)
        
        # get ecodings of spans
        span_encodings = []
        for span_width_tuple in span_width_tuples:
            span_start, span_end = span_width_tuple
            # end +1 to get the index of end subword token
            span_start, span_end = doc_tokens.subword_idx[span_start], doc_tokens.subword_idx[span_end+1]
            # span starts and end +1 to account for inserted CLS token
            span_encoding = encodings[0, span_start+1: span_end+1, :]
            if verbose:
                print(doc_tokens.subword_tokens[span_start: span_end])
            span_encodings.append(span_encoding)
        
        return span_encodings
    
    def encoding_sim_score(self, query_encoding: torch.Tensor, reference_encoding: torch.Tensor):
        sim = sim_matrix(query_encoding, reference_encoding)
        max_vals = torch.mean( sim, axis=1 ) 
        max_vals += self.threshold_bias    
        score = torch.mean( self.act(max_vals) )
        # score = torch.sum( self.act(max_vals) )
        return score