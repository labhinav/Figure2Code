from transformers import BertTokenizerFast

# Tokenizer without clean_text option
tokenizer_no_clean = BertTokenizerFast.from_pretrained('bert-base-uncased', clean_text=False)
tokens_no_clean = tokenizer_no_clean.tokenize("Hello world\n\nThis is a test")
token_ids_no_clean = tokenizer_no_clean.convert_tokens_to_ids(tokens_no_clean)

print("Without clean_text:")
for token, token_id in zip(tokens_no_clean, token_ids_no_clean):
    print(f"Token: {token}, Token ID: {token_id}")

# Tokenizer with clean_text option
tokenizer_clean = BertTokenizerFast.from_pretrained('bert-base-uncased', clean_text=True)
tokens_clean = tokenizer_clean.tokenize("Hello world\nThis is a test")
token_ids_clean = tokenizer_clean.convert_tokens_to_ids(tokens_clean)

print("\nWith clean_text:")
for token, token_id in zip(tokens_clean, token_ids_clean):
    print(f"Token: {token}, Token ID: {token_id}")