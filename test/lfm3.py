"""Chat with LFM3 or run an automatic Transformers smoke test on ROCm."""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "LiquidAI/LFM3-1.7B")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
SYSTEM_PROMPT = "Ты — полезный русскоязычный ассистент. Отвечай кратко."
TEST_PROMPT = "Назови столицу Франции. Ответь одним словом."

# The model accepts 262144 positions, but the chat keeps a far smaller window so
# a long session degrades by dropping old turns instead of failing outright.
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8192"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open an interactive LFM3 chat by default, or run a non-interactive "
            "ROCm smoke test with --test."
        )
    )
    parser.add_argument(
        "--test",
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
            "ROCm GPU is not visible to PyTorch. Run `make test` on a host "
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


def trim_history(
    messages: list[dict[str, str]],
    tokenizer,
    max_new_tokens: int,
    context_limit: int = MAX_CONTEXT_TOKENS,
) -> list[dict[str, str]]:
    """Drop the oldest turns so the next generation fits in the window.

    The system prompt is always kept and the last user message is never dropped,
    so a single oversized prompt fails loudly instead of being silently cut.
    """
    budget = context_limit - max_new_tokens

    def length(candidate: list[dict[str, str]]) -> int:
        # Render first, then tokenize: apply_chat_template(tokenize=True) returns
        # a mapping in recent Transformers versions, and len() of it is not a
        # token count. Rendering the same way generation does keeps the estimate
        # honest (no thinking block, generation prompt appended).
        text = tokenizer.apply_chat_template(
            candidate,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(tokenizer(text)["input_ids"])

    trimmed = list(messages)
    while len(trimmed) > 2 and length(trimmed) > budget:
        # Drop a whole exchange (user + assistant) so the history always starts
        # with a user turn; a leading assistant turn confuses the template.
        del trimmed[1]
        if len(trimmed) > 2 and trimmed[1]["role"] == "assistant":
            del trimmed[1]
    if length(trimmed) > budget:
        raise RuntimeError(
            f"The prompt needs more than {budget} tokens even after trimming the "
            "history. Shorten the message or raise MAX_CONTEXT_TOKENS."
        )
    return trimmed


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


def run_test(tokenizer, model, device: torch.device, max_new_tokens: int) -> None:
    messages = initial_messages()
    messages.append({"role": "user", "content": TEST_PROMPT})
    response, generated_tokens = generate_response(
        messages, tokenizer, model, device, max_new_tokens
    )

    print(f"\nPrompt: {TEST_PROMPT}")
    print(f"Generated response: {response}")
    print(f"Generated tokens: {generated_tokens}")
    print("ROCm Transformers test passed.")


def print_chat_help() -> None:
    print(
        "Команды: /clear — очистить историю; /help — эта справка; "
        "/exit, /quit — выйти."
    )


def run_chat(
    model_id: str, tokenizer, model, device: torch.device, max_new_tokens: int
) -> None:
    messages = initial_messages()
    print(f"\nДиалог с {model_id}. Модель загружена, история сохраняется между репликами.")
    print(f"Окно контекста: {MAX_CONTEXT_TOKENS} токенов; старые реплики отбрасываются.")
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
        # Trimming is applied to a copy so the stored history keeps its turns;
        # only the request that goes to the model is shortened.
        try:
            response, _ = generate_response(
                trim_history(messages, tokenizer, max_new_tokens),
                tokenizer,
                model,
                device,
                max_new_tokens,
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

    # Reproducible sampling for the test and for consistent chat sessions.
    torch.manual_seed(0)
    if args.test:
        run_test(tokenizer, model, device, args.max_new_tokens)
    else:
        run_chat(args.model_id, tokenizer, model, device, args.max_new_tokens)


if __name__ == "__main__":
    main()
