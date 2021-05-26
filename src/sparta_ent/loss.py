import numpy as np
import torch
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
from ..data import Doc_Span


def brute_sum_tensors(list_of_tensors):
    return torch.sum( torch.stack( list_of_tensors ) )

def xentropy_l2r(model, query_span: Doc_Span, references: list, negatives: list):
    """
    Cross-entropy learning to rank loss as defined in:
    https://arxiv.org/pdf/2009.13013.pdf
    
    A stronger reference for this loss is eq (3) in:
    https://papers.nips.cc/paper/2009/file/2f55707d4193dc27118a0f19a1985716-Paper.pdf
    
    # TODO: this should get batched
    """
    # perform encoding
    query_encoding = model.encode_span(query_span, query_span.span_tuple)
    reference_encodings = [model.encode_span(reference_span, reference_span.span_tuple) 
                           for reference_span in references]  if type(references[0]) != torch.Tensor else references
    negative_encodings = [model.encode_span(neg_span, neg_span.span_tuple) 
                         for neg_span in negatives] if type(negatives[0]) != torch.Tensor else negatives

    # calc loss
    pos_sim = [model.encoding_sim_score(query_encoding, reference_encoding) 
               for reference_encoding in reference_encodings]
    neg_sim = [torch.exp( model.encoding_sim_score(query_encoding, negative_encoding) ) 
               for negative_encoding in negative_encodings]
    loss = torch.log( brute_sum_tensors(neg_sim) ) - brute_sum_tensors(pos_sim)
    return loss


def intra_doc_loss(model, answer_spans: list, references_spans: list, max_span_width=7):
    
    # genereate correct encodings
    answer_spans_encodings = model.encode_spans( answer_spans[0], [answer_span.span_tuple for answer_span in answer_spans] )
    
    # generate wrong spans
    all_possible_spans = enumerate_spans( answer_spans[0].doc, max_span_width = max_span_width)
    all_wrong_spans = [span for span in all_possible_spans 
                       if span not in [answer_span.span_tuple for answer_span in answer_spans]]
    all_wrong_span_encodings = model.encode_spans( answer_spans[0], all_wrong_spans )
    
    # iterate through references_spans & sum loss
    losses = []
    for reference in references_spans:
        loss_ = xentropy_l2r(model, reference, answer_spans_encodings, all_wrong_span_encodings)
        losses.append(loss_)
        
    return brute_sum_tensors(losses)

def reference_separator(model):
    raise NotImplementedError