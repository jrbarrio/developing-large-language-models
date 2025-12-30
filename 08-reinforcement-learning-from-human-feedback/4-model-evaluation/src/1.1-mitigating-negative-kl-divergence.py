generation_kwargs = {
    # Set min length and top k parameters
    "min_length": -1,
	"top_k": 0.0, 
  	"top_p": 1.0,
  	"do_sample": True,  
  	"pad_token_id": tokenizer.eos_token_id, 
  	"max_new_tokens": 32}