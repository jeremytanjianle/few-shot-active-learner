import spacy 
import torch
import json
from transformers import AutoTokenizer
# from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans

unsqueeze = lambda x: torch.unsqueeze(x, 0)
tensorify_tuple = lambda span_tuple: unsqueeze( unsqueeze( torch.tensor(span_tuple) ) )

class Doc_Tokens:
    def __init__(self, doc, fullword_tokens, subword_tokens, subword_idx):
        self.doc = doc
        self.fullword_tokens = fullword_tokens
        self.subword_tokens = subword_tokens
        self.subword_idx = subword_idx

    def __getitem__(self, val):
        """
        Slice the doc. This is wrt to AllenNLP's inclusive spans.
        Meaning that it is not compatible with Spacy's exclusive spans
        
        For example, the input doc[0:4] in allennlp's inclusive span 
        will return the equivalent of self.doc[0:5] in spacy's terms

        reference for __getitem__:
        https://stackoverflow.com/questions/2936863/implementing-slicing-in-getitem
        """
        if isinstance(val, slice):
            # +1 because allennlp spans are inclusive
            return self.doc[val.start: val.stop+1]
            if val.stop == val.start:
                return self.doc[val.start: val.stop+1]
            else:
                return self.doc[val.start: val.stop]
        else:
            return self.doc[val]

# class Doc_Span(Doc_Tokens):
#     def __init__(self, doc_tokens:Doc_Tokens, span_tuple:tuple):
#         """
#         Span tuple follows AllenNLP's inclusive text, ie input (1,2) means that 2nd and 3rd tokens will both be included
#         """
#         self.doc = doc_tokens.doc
#         self.fullword_tokens = doc_tokens.fullword_tokens
#         self.subword_tokens = doc_tokens.subword_tokens
#         self.subword_idx = doc_tokens.subword_idx
#         self.span_tuple = span_tuple
#         self.span_text = self.doc[self.span_tuple[0]: self.span_tuple[1]+1].text
#         self.torch_span = tensorify_tuple(span_tuple)
#         self.encoding = 0


class Data_Handler:
    def __init__(self, bert_name = 'albert-base-v2', spacy_name = 'en_core_web_sm'):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name, use_fast=False)
        self.nlp = spacy.load(spacy_name)
        
    def process_sentence(self, text):
        """
        :param text: text sentence
        :return Doc_Tokens: Doc_Tokens objects for encoding
        """
        # spacy tokenize
        doc = self.nlp(text)
        doc_tokens = [spacy_word_token.text for spacy_word_token in doc]
        
        # transformers tokenize
        fullword_tokens = [self.tokenizer.tokenize(text) for text in doc_tokens]
        subword_tokens = [subword for subword_list in fullword_tokens for subword in subword_list]
        subword_idx, k = [], 0
        for subword in fullword_tokens:
            k+=len(subword)
            subword_idx.append(k)
        subword_idx = ([0] + subword_idx)
        
        return Doc_Tokens(doc, fullword_tokens, subword_tokens, subword_idx)
    
    def process_prodigy_annot(self, spacy_annot):
        """
        prodigy is inclusive but spacy is not
        :param text: text sentence
        :return Doc_Tokens: Doc_Tokens objects for encoding
        """
        doc_ = self.process_sentence(spacy_annot['text'])
        
        # span = spacy_annot['spans'][0] # assume only 1 span
        # return Doc_Span(doc_, span['token_start'], span['token_end']-1)
        # return [Doc_Span(doc_, ( span['token_start'], span['token_end']-1) ) for span in spacy_annot['spans'] ]
        return [ doc_, [(span['token_start'], span['token_end'])  for span in spacy_annot['spans']] ]

    def load_seeds(self, path = 'data/seeds/adversary-org.jsonl'):
        # Reading from jsonlines file
        with open(path, 'rb') as f:
            lines = f.readlines()
        lines = [json.loads(line.decode('utf-8')) for line in lines]

        # process token and spans
        references = []
        for line in lines:
            references.append( self.process_spacy_annot(line) )
        return references


