from transformers import pipeline
from transformers import BertForMaskedLM, BertTokenizerFast, BertModel, BertConfig, BertTokenizer, BertForPreTraining

import torch
import json

config = BertConfig.from_json_file('output/config.json')
model = BertForMaskedLM.from_pretrained("output/pytorch_model.bin", config = config)

nlp = pipeline("fill-mask")

tokenizer = BertTokenizer(vocab_file = "output/vocab.txt")
print(tokenizer.mask_token)
sequence = f"Espero que a {tokenizer.mask_token} de cabeça passe depressa."
input = tokenizer.encode(sequence, return_tensors="pt")

mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

token_logits = model(input, return_dict=True).logits
mask_token_logits = token_logits[0, mask_token_index, :]
top_n_tokens = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist()

out = []
for token in top_n_tokens:
    #print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
    token_rep = sequence.replace(tokenizer.mask_token, tokenizer.decode([token]))
    out.append(token_rep)
with open('out.json', 'w') as f:
    json.dump(out, f)
