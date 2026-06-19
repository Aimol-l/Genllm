import yaml
import struct
from gguf import GGUFTorchLoader
from transformers import AutoTokenizer
from models.Qwen3 import Qwen3Config, Qwen3Model


if __name__ == "__main__":

    config_path = "weights/qwen3-0.6b-bf16.yaml"


    config = Qwen3Config(config_path)
    print(f"Config: hidden={config.hidden_size}, layers={config.num_layers}, "f"heads={config.num_heads}/{config.num_kv_heads}, vocab={config.vocab_size}")

    loader = GGUFTorchLoader(config.model_path)

    model = Qwen3Model(config)
    model.load_weights(loader)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    prompt = "1+1=?"

    chat_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        enable_thinking = False,
        add_generation_prompt=True
    )
    chat_ids = chat_ids["input_ids"] if hasattr(chat_ids, "input_ids") else chat_ids

    input_ids = list(chat_ids)

    EOS_TOKEN_IDS = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}

    for step in range(128):
        logits = model.forward(input_ids)
        next_token = logits[0, -1].argmax().item()

        if next_token in EOS_TOKEN_IDS:
            break

        input_ids.append(next_token)
        print(tokenizer.decode([next_token]), end="", flush=True)

    print("\n")
    print("=========================")
