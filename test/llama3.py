# Johannisbaer / LLama3_AMD.md
# https://gist.github.com/Johannisbaer/1a1435dd0cbc01b4a477dad3017e960a
import transformers
import torch
import gc

model_id = "IlyaGusev/saiga_llama3_8b"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {"role": "system", "content": "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."},
    {"role": "user", "content": "Сочини длинный рассказ, обязательно упоминая следующие объекты. Дано: Таня, мяч"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text[len(prompt):])

del model, tokenizer, inputs, outputs
torch.cuda.empty_cache()
gc.collect()
