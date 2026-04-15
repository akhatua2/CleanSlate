"""Chat formatting utilities."""


def format_chat(
    tokenizer,
    messages: list[dict],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            "Tokenizer does not support apply_chat_template. "
            "Use a chat model/tokenizer or provide a preformatted prompt."
        )

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError as e:
        raise RuntimeError(
            "Tokenizer apply_chat_template does not support enable_thinking."
        ) from e
