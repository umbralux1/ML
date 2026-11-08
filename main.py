
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", attn_implementation='eager')

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
inputs = tok('hello world', return_tensors='pt').to(model.device)

out = model(**inputs, output_attentions=True)
print(len(out.attentions), out.attentions[0].shape)
