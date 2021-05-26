from .data import Doc_Tokens

def get_unique_spans(references):
    """
    :param references: list of Doc_Span objects
    """
    unique_texts = list(set([reference.doc.text for reference in references]))
    
    text_2_spans = []
    for unique_text in unique_texts:
        spans = [ref.span_tuple for ref in references if ref.doc.text == unique_text]
        doc = [ref for ref in references if ref.doc.text == unique_text][0]
        doc = Doc_Tokens( doc.doc, doc.fullword_tokens, doc.subword_tokens, doc.subword_idx)
        text_2_spans.append([doc, spans])
    
    return text_2_spans