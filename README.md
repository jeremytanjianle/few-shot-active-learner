# Few-Shot entity extraction with active learning
Ever labelled a few entities, only to realize you need to change the labels and re-annotate all over again?  
Here's a few-shot entity extraction that will annotate with you, and adapt to changing labels with you. 

## Usage
### 1. Get Data
```
# Reading from jsonlines file
with open('data/seeds/adversary-org.jsonl', 'rb') as f:
    lines = f.readlines()
lines = [json.loads(line.decode('utf-8')) for line in lines]

from src.data import Data_Handler

handler = Data_Handler()
references = []
for line in lines:
    references.append( handler.process_prodigy_annot(line) )
```

The references will look like this:  
```
[[<src.data.Doc_Tokens at 0x1b5aa8b04c8>, [(0, 2), (76, 78)]],
 [<src.data.Doc_Tokens at 0x1b5aa8b0bc8>, [(2, 3), (5, 9), (31, 32)]],
 [<src.data.Doc_Tokens at 0x1b5abbbfa88>,
  [(12, 13), (45, 46), (89, 90), (72, 74)]]]
```

### 2. Init Model
Start a model from scratch
```
from src.dygie_ent import Dygie_Ent
model = Dygie_Ent()
model.references['adversary'] = references
model.get_prototypes();
```

### 3. Run model on raw string
```
text = "The activity of the advanced hacker group the researchers call Silence has increased significantly over the past year. Victims in the financial sector are scattered across more than 30 countries and financial losses have quintupled.\n The group started timidly in 2016, learning the ropes by following the path beaten by other hackers. Since then, it managed to steal at least $4.2 million, initially from banks in the former Soviet Union, then from victims in Europe, Latin America, Africa, and Asia.\n Researchers at Group-IB, Singapore-based cybersecurity company specializing in attack prevention, tracked Silence early on and judged its members to be familiar with white-hat security activity.\n A report last year\xa0details the roles of Silence hackers, their skills, failures, and successful bank heists"
doc = handler.process_sentence(text)
probs, spans = model(doc)
```
`probs` is a torch tensor of size (number of spans x 2) indicating the probability that this span corresponds to the provided prototypes  
`spans` is a list of span tuples indicating the entities per prototype class.
### 4. Train model
```
ent_spans = [(10,10), (38,39), (111, 111), (138,139)]
model.evaluate([[doc, ent_spans]])
```
### In development
1. Prodigy recipe
2. Pretrained model
3. Further multi-class support