import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AdamW
from allennlp.modules.span_extractors.self_attentive_span_extractor import SelfAttentiveSpanExtractor
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
from sklearn.metrics import precision_recall_fscore_support

from ..data import Doc_Tokens, tensorify_tuple
from ..util import get_unique_spans

class Dygie_Ent(nn.Module):
    
    def __init__(self, bert_name = 'albert-base-v2', use_proj = True):
        super().__init__()
        
        # encoder
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name, use_fast=False)
        self.encoder = AutoModel.from_pretrained(bert_name)
        self.hidden_size = self.encoder.config.hidden_size

        # scorer parameters
        self.span_extractor = SelfAttentiveSpanExtractor(self.hidden_size)
        self.batch_norm = nn.BatchNorm1d(self.hidden_size) # , affine = False)
        self.proj = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size), nn.Linear(self.hidden_size,self.hidden_size))
        self.use_proj = use_proj

        # supports
        self.references = {}

        # optimizer & scheduler
        self.optimizer = AdamW(params=self.parameters(), lr=0.00001)
        # self.schedule = get_linear_schedule_with_warmup(self.optimizer,
        #                                         num_warmup_steps=batch_num * config.warmup_epoch,
        #                                         num_training_steps=batch_num * config.max_epoch)
        self.eval()
    
    def accept_reference(self, doc, label, ent_spans):
        # check if label is inside references
        if label not in self.references.keys():
            self.reference[label] = [[doc, ent_spans]]

        # check if doc is inside label 
        stored_texts = self.references[label]
        stored_texts = [stored_text[0].doc.text for stored_text in stored_texts]
        if doc.doc.text not in stored_texts:
            self.references[label].append([doc, ent_spans])
        else:
            stored_doc_idx = stored_texts.index(doc.doc.text)
            self.references[label][stored_doc_idx][1].append(ent_spans) 

    def encode_span(self, doc_tokens: Doc_Tokens, span_width_tuple: tuple, verbose=False):
        """
        :param doc_tokens: Doc_Tokens object that contains the spans
        :param span_width_tuple: tuple indicating start and end of spacy token spans
        :return span_encoding: (span length x embed dimensions) torch array
        """
        # get span 
        span_start, span_end = span_width_tuple 
        span_start, span_end = doc_tokens.subword_idx[span_start], doc_tokens.subword_idx[span_end+1]-1
        torch_span = tensorify_tuple( (span_start +1, span_end +1 ) ) # add to account for CLS token
        
        # encode the pieces and slice according to span
        piece_id = self.tokenizer.encode(doc_tokens.subword_tokens, return_tensors='pt')
        encodings = self.encoder(piece_id)['last_hidden_state']
        span_representation = self.span_extractor(encodings, torch_span)

        if verbose:
            print(span_start)
            print(span_end)
            print(doc_tokens.subword_tokens[span_start: span_end+1]) # slice is exclusive, so +1
            print(f"start: {self.tokenizer.decode(piece_id[0,span_start+1])}")
            print(f"end: {self.tokenizer.decode(piece_id[0,span_end+1 ])}")
        
        if self.use_proj: span_representation = self.proj(span_representation)
            
        return span_representation
    
    
    def encode_spans(self, doc_tokens: Doc_Tokens, span_width_tuples: list, verbose=False):
        """
        :param doc_tokens: Doc_Tokens object that contains the spans
        :param span_width_tuples: list of tuple indicating start and end of spacy token spans
        :return span_encoding: (span length x embed dimensions) torch array
        """
        # encode the pieces and slice according to span
        piece_id = self.tokenizer.encode(doc_tokens.subword_tokens, return_tensors='pt')
        encodings = self.encoder(piece_id)['last_hidden_state']
        
        # get ecodings of spans
        torch_spans = []
        for span_width_tuple in span_width_tuples:
            span_start, span_end = span_width_tuple
            # end +1 to get the index of end subword token
            span_start, span_end = doc_tokens.subword_idx[span_start], doc_tokens.subword_idx[span_end+1]-1
            # span starts and end +1 to account for inserted CLS token
            torch_span = tensorify_tuple( (span_start +1, span_end +1 ) ) # add to account for CLS token
            torch_spans.append(torch_span)
            
        torch_spans = torch.cat(torch_spans, axis=1)
        span_representations = self.span_extractor(encodings, torch_spans)
        
        if self.use_proj: span_representations = self.proj(span_representations)
        span_representations = span_representations.squeeze(0)
        span_representations = self.batch_norm(span_representations)
        return span_representations
    
    def encoding_sim_score(self, query_encoding: torch.Tensor, reference_encoding: torch.Tensor):
        return torch.cdist(query_encoding, reference_encoding)

    def calc_prototype(self, doc_span_pairs):
        """
        Sample input:
        ------------
        [[<src.data.Doc_Tokens at 0x2793506e4c8>,
         [(12, 12), (45, 45), (89, 89), (72, 73)]],
        [<src.data.Doc_Tokens at 0x27935047e48>, [(0, 1), (76, 77)]],
        [<src.data.Doc_Tokens at 0x27938cf2388>, [(2, 2), (5, 8), (31, 31)]]]
        
        :param doc_span_pairs: list of lists which contains a pair of docs and list of spans
        :return prototype_encoding: Torch tensor of shape (1 x embedding)
        """

        support_encodings = []
        for doc, span in doc_span_pairs:
            encodings = self.encode_spans(doc, span)
            support_encodings.append(encodings) # (num spans, hidden_size)
        prototype_encoding = torch.cat(support_encodings, axis=0)
        prototype_encoding = torch.mean(prototype_encoding, axis=0, keepdim=True)

        return prototype_encoding

    def get_prototypes(self):
        """
        calculate new set of prototypes
        """
        # calculate the prototypes
        labels, prototype_encodings = [], []
        for label, references in self.references.items():
            # doc_span_pairs = get_unique_spans(references)
            prototype_encoding = self.calc_prototype(references)
            prototype_encodings.append(prototype_encoding)
            labels.append(label)
        prototype_encodings = torch.cat(prototype_encodings, axis=1)
        
        # save the labels and prototype encodings
        self.labels = labels
        self.prototype_encodings = prototype_encodings

        return labels, prototype_encodings

    def forward(self, doc: Doc_Tokens, max_span_width=4):
        """
        :return prob_class: torch tensor of shape (1, no. of spans, no. of prototype classes + 1)
        """
        # get span encodings
        all_spans = enumerate_spans(doc.doc, max_span_width=max_span_width)
        encodings = self.encode_spans(doc, all_spans)
        # encodings = self.batch_norm(encodings)
        
        # get one-class support encodings
        support_encodings = torch.cat( [ torch.zeros(self.prototype_encodings.shape) , self.prototype_encodings ], axis=0 )
        # support_encodings = torch.cat( [ self.batch_norm(self.prototype_encodings), torch.zeros(self.prototype_encodings.shape)], axis=1 )
        distances = torch.cdist(encodings, support_encodings)

        # return prob
        prob_class = torch.nn.LogSoftmax(dim=1)(distances)

        # return selected spans
        selected_spans_mask = prob_class.argmax(dim=1).detach().numpy()
        selected_spans = np.array(all_spans)[selected_spans_mask==1].tolist()

        return prob_class , selected_spans
    
    def update(self, doc, ent_spans, max_span_width=4):
        self.train()

        # get labels
        all_spans = enumerate_spans(doc.doc, max_span_width=max_span_width)
        labels = [ int(span in ent_spans) for span in all_spans]
        
        # get prob
        prob_class, sel_spans = self.forward(doc, max_span_width=max_span_width)

        # get loss
        loss = nn.CrossEntropyLoss()
        loss_val = loss(prob_class.squeeze(0), torch.LongTensor(labels))

        # backward
        loss_val.backward()
        self.optimizer.step()
        # self.schedule.step()
        self.optimizer.zero_grad()

        # set back to eval and reset prototypes
        self.eval()
        self.get_prototypes()

    def evaluate(self, list_of_doc_span_pairs, max_span_width=4):

        y_trues = []
        y_preds = []
        for  doc, ent_spans in list_of_doc_span_pairs:
            # get labels
            all_spans = enumerate_spans(doc.doc, max_span_width=max_span_width)
            labels = [ int(span in ent_spans) for span in all_spans]
            y_true = np.array(labels)
            y_trues.append(y_true)
            
            # get prob
            probs, sel_spans = self.forward(doc, max_span_width=max_span_width)
            y_pred = probs.argmax(dim=1).detach().numpy()
            y_preds.append(y_pred)
        y_trues = np.hstack(y_trues)
        y_preds = np.hstack(y_preds)
        # return precision_recall_fscore_support(y_trues, y_preds)
        prec, recall, f1, _ = precision_recall_fscore_support(y_trues, y_preds)

        return prec[1:].tolist(), recall[1:].tolist(), f1[1:].tolist()

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
