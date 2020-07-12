from transformers import T5Tokenizer, T5Model

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5Model.from_pretrained('t5-small')
input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)
last_hidden_states = outputs[0]
