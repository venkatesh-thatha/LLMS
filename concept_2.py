import torch
import torch.nn as nn
from tokenizers import Tokenizer


tokenizer=Tokenizer.from_pretrained("gpt2")

text="world Hello"
output=tokenizer.encode(text)

input_ids=torch.tensor([output.ids])

vocab_size=tokenizer.get_vocab_size()

small_embed_dim=16
small_embedding_layer=nn.Embedding(vocab_size, small_embed_dim)

embedded_output=small_embedding_layer(input_ids)
print(f"Input IDs: {input_ids}")
print(f"Embedded output shape: {embedded_output.shape}")
print(f"Embedded output: {embedded_output}")
print(f"Original text: {text}")

