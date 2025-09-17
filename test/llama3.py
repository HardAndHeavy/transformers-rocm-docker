# Johannisbaer / LLama3_AMD.md
# https://gist.github.com/Johannisbaer/1a1435dd0cbc01b4a477dad3017e960a
import transformers
import torch
import gc

model_id = "IlyaGusev/saiga_llama3_8b"
pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"dtype": torch.bfloat16}, device_map="auto")

messages = [
    {"role": "system", "content": "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."},
    {"role": "user", "content": "Сочини длинный рассказ, обязательно упоминая следующие объекты. Дано: Таня, мяч"}
]

prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

outputs = pipeline(prompt, max_new_tokens=256, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
print(outputs[0]["generated_text"][len(prompt):])

del pipeline
torch.cuda.empty_cache()
gc.collect()
