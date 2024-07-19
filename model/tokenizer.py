import tiktoken
from model.config import *

encoding = tiktoken.get_encoding("cl100k_base")

def tokenize(text):
    tokenized_text = encoding.encode(text)
    max_token_value = max(tokenized_text) + 1
    tokenized_text = torch.tensor(tokenized_text, dtype=torch.long,device=device)
    
    return tokenized_text,max_token_value