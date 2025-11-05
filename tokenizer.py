from tokenizers import Tokenizer

tokenizer=Tokenizer.from_pretrained("gpt2")

text=input("Enter text to tokenize: ")
encoded=tokenizer.encode(text)

print(f"original text: {text}")
print(f'tokenized text: {encoded.tokens}')
print(f"token ids: {encoded.ids}")

decoded=tokenizer.decode(encoded.ids)
print(f"decoded text: {decoded}")