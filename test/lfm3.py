"""Chat with LFM3 or run an automatic Transformers smoke test on ROCm."""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "LiquidAI/LFM3-1.7B")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
SYSTEM_PROMPT = "Ты — полезный русскоязычный ассистент. Отвечай кратко."
AUTO_TEST_PROMPT = "Назови столицу Франции. Ответь одним словом."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open an interactive LFM3 chat by default, or run a non-interactive "
            "ROCm smoke test with --auto-test."
        )
    )
    parser.add_argument(
        "--auto-test",
        action="store_true",
        help="run one fixed prompt, validate the response, and exit",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID (default: MODEL_ID or %(default)s)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="generation limit per response (default: MAX_NEW_TOKENS or %(default)s)",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be greater than zero")
    return args


def initial_messages() -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def check_rocm() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm GPU is not visible to PyTorch. Run `make auto-test` on a host "
            "with /dev/kfd and /dev/dri and a compatible AMDGPU driver."
        )

    device = torch.device("cuda:0")
    print(f"PyTorch: {torch.__version__}; HIP: {torch.version.hip}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    return device


def load_model(model_id: str, device: torch.device):
    print(f"Model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("The model was not loaded onto the ROCm GPU")

    return tokenizer, model


def generate_response(
    messages: list[dict[str, str]],
    tokenizer,
    model,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str, int]:
    """Generate one assistant response without changing the chat history."""
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    # Sampling values follow the model card. Do not compare generated text with
    # a fixed answer: this test validates ROCm execution, not model knowledge.
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
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
    # LFM3 does not register its thinking delimiters as special tokens. With
    # thinking disabled, expose only the final answer if it still emits one.
    response = response.rsplit("</think>", 1)[-1].strip()

    if generated_ids.numel() == 0 or not response:
        raise RuntimeError("The model returned an empty response")

    return response, generated_ids.numel()


def run_auto_test(tokenizer, model, device: torch.device, max_new_tokens: int) -> None:
    messages = initial_messages()
    messages.append({"role": "user", "content": AUTO_TEST_PROMPT})
    response, generated_tokens = generate_response(
        messages, tokenizer, model, device, max_new_tokens
    )

    print(f"\nPrompt: {AUTO_TEST_PROMPT}")
    print(f"Generated response: {response}")
    print(f"Generated tokens: {generated_tokens}")
    print("ROCm Transformers auto-test passed.")


def print_chat_help() -> None:
    print("Команды: /clear — очистить историю; /exit или /quit — выйти.")


def run_chat(
    model_id: str, tokenizer, model, device: torch.device, max_new_tokens: int
) -> None:
    messages = initial_messages()
    print(f"\nДиалог с {model_id}. Модель загружена, история сохраняется между репликами.")
    print_chat_help()

    while True:
        try:
            prompt = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДиалог завершён.")
            return

        if not prompt:
            continue

        command = prompt.lower()
        if command in {"/exit", "/quit", "/q"}:
            print("Диалог завершён.")
            return
        if command == "/clear":
            messages = initial_messages()
            print("История очищена.")
            continue
        if command == "/help":
            print_chat_help()
            continue

        messages.append({"role": "user", "content": prompt})
        try:
            response, _ = generate_response(
                messages, tokenizer, model, device, max_new_tokens
            )
        except KeyboardInterrupt:
            # Do not leave an unanswered user message in the next prompt.
            messages.pop()
            print("\nГенерация прервана. Можно продолжить диалог или ввести /exit.")
            continue

        messages.append({"role": "assistant", "content": response})
        print(f"\nLFM3: {response}")


def main() -> None:
    args = parse_args()
    device = check_rocm()
    tokenizer, model = load_model(args.model_id, device)

    # Reproducible sampling for the auto-test and consistent chat sessions.
    torch.manual_seed(0)
    if args.auto_test:
        run_auto_test(tokenizer, model, device, args.max_new_tokens)
    else:
        run_chat(args.model_id, tokenizer, model, device, args.max_new_tokens)


if __name__ == "__main__":
    main()
