"""Smoke-test Transformers text generation on a ROCm GPU with LFM3."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.getenv("MODEL_ID", "LiquidAI/LFM3-1.7B")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm GPU is not visible to PyTorch. Run `make test` on a host "
            "with /dev/kfd and /dev/dri and a compatible AMDGPU driver."
        )

    device = torch.device("cuda:0")
    print(f"PyTorch: {torch.__version__}; HIP: {torch.version.hip}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("The model was not loaded onto the ROCm GPU")

    messages = [
        {
            "role": "system",
            "content": "Ты — полезный русскоязычный ассистент. Отвечай кратко.",
        },
        {
            "role": "user",
            "content": "Назови столицу Франции. Ответь одним словом.",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    # Sampling values follow the model card. Generation is not checked against
    # fixed text: this test validates ROCm execution, not model knowledge.
    torch.manual_seed(0)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.1,
            top_k=50,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if generated_ids.numel() == 0 or not response:
        raise RuntimeError("The model returned an empty response")

    print("\nGenerated response:")
    print(response)
    print(f"\nGenerated tokens: {generated_ids.numel()}")
    print("ROCm Transformers smoke test passed.")


if __name__ == "__main__":
    main()
