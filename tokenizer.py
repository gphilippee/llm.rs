import tiktoken
enc = tiktoken.get_encoding("gpt2")
print(enc.encode_ordinary("What is the capital of France ?"))