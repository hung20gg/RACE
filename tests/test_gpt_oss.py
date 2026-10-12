"""Offline CPU smoke tests: real cached tokenizer, no model weights or GPU."""
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import AutoConfig, AutoTokenizer
from gpt_oss import render_prompt, split_response, THINK_START, THINK_END


class HarmonyTests(unittest.TestCase):
    def test_analysis_and_final(self):
        self.assertEqual(split_response(THINK_START + "Reasoning." + THINK_END + "4<|fim_suffix|>"), ("Reasoning.", "4"))

    def test_prefilled_analysis_header(self):
        self.assertEqual(split_response("Reasoning." + THINK_END + "4<|fim_suffix|>", "<|start|>assistant" + THINK_START), ("Reasoning.", "4"))

    def test_truncated_analysis(self):
        self.assertEqual(split_response(THINK_START + "Still thinking"), ("Still thinking", ""))

    def test_final_only(self):
        self.assertEqual(split_response("<|channel|>final<|message|>4<|fim_suffix|><|end|>"), ("", "4"))


class CachedTokenizerSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", local_files_only=True)
        cls.config = AutoConfig.from_pretrained("openai/gpt-oss-20b", local_files_only=True)

    def test_config_and_fixed_date_all_efforts(self):
        self.assertEqual(self.config.model_type, "gpt_oss")
        messages = [{"role": "user", "content": "What is 2 + 2?"}]
        for effort in ("low", "medium", "high"):
            prompt = render_prompt(self.tokenizer, messages, effort, "2025-08-05")
            self.assertIn("Current date: 2025-08-05", prompt)
            self.assertIn(f"Reasoning: {effort}", prompt)
            self.assertEqual(prompt, render_prompt(self.tokenizer, messages, effort, "2025-08-05"))
            self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(prompt, add_special_tokens=False), skip_special_tokens=False), prompt)

    def test_date_changes_only_requested_date(self):
        messages = [{"role": "user", "content": "A question"}]
        first = render_prompt(self.tokenizer, messages, "medium", "2025-08-05")
        second = render_prompt(self.tokenizer, messages, "medium", "2026-10-12")
        self.assertEqual(first.replace("2025-08-05", "2026-10-12"), second)

    def test_invalid_date(self):
        with self.assertRaises(ValueError):
            render_prompt(self.tokenizer, [], "medium", "2026-02-30")

    def test_generation_pipeline_with_mocked_weights(self):
        tokenizer = self.tokenizer
        class FakeModel:
            device = torch.device("cpu")
            def generate(self, input_ids, **kwargs):
                assert kwargs["eos_token_id"] == tokenizer.convert_tokens_to_ids("<|fim_suffix|>")
                self_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                assert "Current date: 2025-08-05" in self_prompt
                # Complete the header left open by the real cached HF template.
                completion = THINK_START + "Two plus two is four." + THINK_END + "4<|fim_suffix|>"
                ids = torch.tensor([tokenizer.encode(completion, add_special_tokens=False)])
                return torch.cat([input_ids, ids], dim=1).repeat(kwargs.get("num_return_sequences", 1), 1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dataset").mkdir()
            (root / "dataset/NQ-open.efficientqa.dev.1.1.jsonl").write_text(json.dumps({"question": "What is 2 + 2?", "answer": ["4"]}) + "\n")
            (root / "cot_prompt.txt").write_text("{}")
            argv = ["generate.py", "--model_name", "gpt-oss-20b", "--model_path", "openai/gpt-oss-20b", "--dataset_name", "NQ", "--output_dir", str(root / "output"), "--local_files_only", "--current_date", "2025-08-05", "--gpu_id", "", "--num_samples", "2", "--max_examples", "1"]
            script = Path(__file__).resolve().parents[1] / "generate.py"
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch("sys.argv", argv), patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=FakeModel()) as loader:
                    runpy.run_path(str(script), run_name="__main__")
                self.assertEqual(loader.call_args.kwargs["torch_dtype"], "auto")
                self.assertEqual(loader.call_args.kwargs["attn_implementation"], "sdpa")
                output = root / "output/NQ/gpt-oss-20b"
                main = json.loads((output / "result.json").read_text())[0]
                samples = json.loads((output / "sample_result.json").read_text())[0]
                self.assertEqual(main["input"], samples["input"])
                self.assertEqual(main["final_answer"], "4")
                self.assertEqual(samples["final_answer"], ["4", "4"])
                self.assertEqual(main["think"], "Two plus two is four.")
                self.assertIn("<|fim_suffix|>", main["output"])
                self.assertEqual(json.loads((output / "generation_metadata.json").read_text())["current_date"], "2025-08-05")
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
