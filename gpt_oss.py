"""GPT-OSS prompt rendering and output splitting without loading model weights."""
from datetime import date

THINK_START = "<|channel|>analysis<|message|>"
THINK_END = "<|end|><|start|>assistant<|channel|>final<|message|>"
FINAL_START = "<|channel|>final<|message|>"


def render_prompt(tokenizer, messages, reasoning_effort, current_date):
    date.fromisoformat(current_date)
    # The HF template uses strftime_now, ignoring an ordinary date kwarg.
    template = tokenizer.get_chat_template()
    template = template.replace('strftime_now("%Y-%m-%d")', 'current_date')
    template = template.replace("strftime_now('%Y-%m-%d')", 'current_date')
    prompt = tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
        reasoning_effort=reasoning_effort, current_date=current_date,
    )
    if f"Current date: {current_date}" not in prompt:
        raise ValueError("GPT-OSS chat template did not render the requested date")
    return prompt


def split_response(output, prompt=""):
    """Return analysis and final, including completion of a prefilled header."""
    # Some templates prefill assistant analysis; recover only that header.
    header = prompt.rsplit("<|start|>assistant", 1)[-1] if "<|start|>assistant" in prompt else ""
    text = header + output
    if FINAL_START not in text:
        # Truncated analysis is not a final answer.
        return text.split(THINK_START, 1)[-1].split("<|end|>", 1)[0].strip(), ""
    before, final = text.split(FINAL_START, 1)
    think = before.split(THINK_START, 1)[-1] if THINK_START in before else ""
    think = think.split("<|end|>", 1)[0].strip()
    for token in ("<|fim_suffix|>", "<|im_end|>", "<|end|>", "<|ghissue|>"):
        final = final.split(token, 1)[0]
    return think, final.strip()
