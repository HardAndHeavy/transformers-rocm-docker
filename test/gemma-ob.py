from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("OpenBuddy/openbuddy-gemma-7b-v19.1-4k")
model = AutoModelForCausalLM.from_pretrained("OpenBuddy/openbuddy-gemma-7b-v19.1-4k", device_map="auto")

input_text = "Почему трава зеленая?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config,
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()

generation_config = GenerationConfig(
    temperature=0.2,
    do_sample=True,
    top_p=0.95,
    top_k=30,
    max_length=512,
    eos_token_id=tokenizer.eos_token_id
)

system_prompt = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
inputs = ["Почему трава зеленая?", "Что такое тупой угол?", "Сочини длинный рассказ, обязательно упоминая следующие объекты. Дано: Таня, мяч", "Что такое глазунья?"]
for inp in inputs:
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n"
    output = generate(model, tokenizer, prompt, generation_config)
    print(inp)
    print(output)
    print()
    print("==============================")
    print()
