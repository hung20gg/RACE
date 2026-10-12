import os
import sys
import argparse
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
from gpt_oss import render_prompt, split_response

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., qwen3-14b). Please name the thinking model with a name that ends with the letter b")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--output_dir", type=str, default="./modeloutput", help="Directory to save model outputs")
    parser.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--current_date", default="2026-10-12", help="Fixed date in the GPT-OSS system prompt (YYYY-MM-DD)")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Validate config and render a prompt; do not load weights")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.num_samples < 1 or (args.max_examples is not None and args.max_examples < 1):
        parser.error("Token, sample and example limits must be positive")
    return args

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.dry_run else args.gpu_id

model_name = args.model_name
dataset_name = args.dataset_name
model_path = args.model_path

output_path = os.path.join(args.output_dir, dataset_name, model_name)

config = AutoConfig.from_pretrained(model_path, local_files_only=args.local_files_only)
is_gpt_oss = config.model_type == "gpt_oss"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.local_files_only)
if args.dry_run:
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    prompt = render_prompt(tokenizer, messages, args.reasoning_effort, args.current_date) if is_gpt_oss else tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(json.dumps({"model_type": config.model_type, "dtype": "auto" if is_gpt_oss else "float16",
                      "attention": "sdpa" if is_gpt_oss else "flash_attention_2",
                      "completion_kwargs": {"extra_body": {"skip_special_tokens": False}} if is_gpt_oss else {},
                      "max_new_tokens": args.max_new_tokens, "num_samples": args.num_samples,
                      "prompt": prompt}, indent=2))
    sys.exit(0)

os.makedirs(output_path, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", local_files_only=args.local_files_only,
    torch_dtype="auto" if is_gpt_oss else torch.float16,
    attn_implementation="sdpa" if is_gpt_oss else "flash_attention_2",
)

# Load dataset
if dataset_name == "HotpotQA":
    dataset = json.load(open("./dataset/HotpotQA/hotpot_dev_fullwiki_v1.json"))
elif dataset_name == "NQ":
    dataset = []
    with open("./dataset/NQ-open.efficientqa.dev.1.1.jsonl") as f:
        for line in f.readlines():
            dataset.append(json.loads(line))
elif dataset_name == "squad":
    dataset = json.load(open("./dataset/squad.json"))
elif dataset_name == "triviaqa":
    dataset = json.load(open("./dataset/triviaqa.json"))

with open("./cot_prompt.txt") as f:
    cot_prompt = f.read()

def generate_response(text, greedy=True):
    # Prepare the messages with think tokens
    if "cot" in model_name:
        messages = [
            {"role": "user", "content": cot_prompt.format(text)}
        ]
    else:
        messages = [
            {"role": "user", "content": text}
        ]
    
    if greedy:
        generation_config = dict(
                            top_k=0,
                            top_p=1.0,
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=args.max_new_tokens)
    else:
        generation_config = dict(
                            top_p=0.95,
                            temperature=1,
                            do_sample=True,
                            num_beams=1,
                            num_return_sequences=args.num_samples,
                            max_new_tokens=args.max_new_tokens)
    
    if is_gpt_oss:
        # <|end|> closes analysis, not the assistant turn; do not stop there.
        generation_config["eos_token_id"] = tokenizer.convert_tokens_to_ids("<|fim_suffix|>")

    # Apply chat template
    if is_gpt_oss:
        # Render once; both generation and channel parsing use this exact prompt.
        prompt = render_prompt(tokenizer, messages, args.reasoning_effort, args.current_date)
        input_text = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        prompt = ""
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    attention_mask = torch.ones(len(input_text)).to(model.device)
    
    # Generate
    outputs = model.generate(torch.tensor([input_text]).to(model.device), attention_mask=attention_mask.unsqueeze(0), pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, **generation_config)
    
    # Decode
    if greedy:
        response = tokenizer.decode(outputs[0][len(input_text):], skip_special_tokens=not is_gpt_oss)
    else:
        response = [tokenizer.decode(output[len(input_text):], skip_special_tokens=not is_gpt_oss) for output in outputs]

    return (response, prompt)

result = []
sample_result = []

for i in tqdm(range(min(len(dataset), args.max_examples or len(dataset)))):
    question = dataset[i]['question']
    answer = dataset[i]['answer']
    output, prompt = generate_response(question)
    
    if is_gpt_oss:
        think, final_answer = split_response(output, prompt)
    elif model_name.endswith("b") or "thinking" in model_name:
        think = output.split('</think>')[0].split("<think>")[-1]
        if "</think>" in output:
            final_answer = output.split('</think>')[-1]
        else:
            continue
    elif "cot" in model_name:
        # cot output
        think = output.split('Answer:')[0]
        if "Answer:" in output:
            final_answer = output.split('Answer:')[-1]
        else:
            continue
    elif "chat" in model_name or "direct" in model_name:
        # direct chat output
        think = output
        final_answer = output

    result.append({
        "index": i,
        "input": prompt,
        "question": question,
        "answer": answer,
        "output": output,
        "think": think,
        "final_answer": final_answer
    })

    outputs, sample_prompt = generate_response(question, greedy=False)
    answers = []
    thinks = []
    for output in outputs:
        if is_gpt_oss:
            think, final_answer = split_response(output, sample_prompt)
        elif model_name.endswith("b") or "thinking" in model_name:
            think = output.split('</think>')[0].split("<think>")[-1]
            if "</think>" in output:
                final_answer = output.split('</think>')[-1]
            else:
                final_answer = ""
        elif "cot" in model_name:
            think = output.split('Answer:')[0]
            if "Answer:" in output:
                final_answer = output.split('Answer:')[-1]
            else:
                final_answer = ""
        elif "chat" in model_name:
            think = output
            final_answer = output
        answers.append(final_answer)
        thinks.append(think)

    sample_result.append({
        "index": i,
        "input": sample_prompt,
        "question": question,
        "answer": answer,
        "output": outputs,
        "think": thinks,
        "final_answer": answers
    })

with open(f"{output_path}/result.json", "w+") as f:
    json.dump(result, f, indent=2)
    
with open(f"{output_path}/sample_result.json", "w+") as f:
    json.dump(sample_result, f, indent=2)

with open(os.path.join(output_path, "generation_metadata.json"), "w") as f:
    json.dump(vars(args), f, indent=2)
