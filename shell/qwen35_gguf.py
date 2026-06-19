import torch
from gguf import GGUFTorchLoader
from transformers import AutoTokenizer
from models.Qwen35 import Qwen35Config, Qwen35Model

GGUF_PATH = "weights/Qwen3.5-4B-BF16.gguf"

if __name__ == "__main__":
    loader = GGUFTorchLoader(GGUF_PATH)
    config = Qwen35Config(loader)
    print(f"Config: embedding={config.embedding_length}, blocks={config.block_count}, "
          f"heads={config.head_count}/{config.head_count_kv}, vocab={config.vocab_size}")

    model = Qwen35Model(config)
    model.load_weights(loader)

    tokenizer = AutoTokenizer.from_pretrained("weights/qwen35")

    prompt = "1+1=?"
    chat_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        enable_thinking=False,
        add_generation_prompt=True,
    )
    chat_ids = chat_ids["input_ids"] if hasattr(chat_ids, "input_ids") else chat_ids
    input_ids = list(chat_ids)

    EOS_TOKEN_IDS = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}

    model.reset_states()
    with torch.inference_mode():
        # Prefill: 跑完整个 chat_ids，建立 KV/SSM cache
        logits = model.forward(input_ids)
        next_token = logits[0, -1].argmax().item()

        # Decode: 每步只喂上一个生成的 token
        for step in range(128):
            if next_token in EOS_TOKEN_IDS:
                break
            print(tokenizer.decode([next_token]), end="", flush=True)
            input_ids.append(next_token)
            logits = model.forward([next_token])
            next_token = logits[0, -1].argmax().item()

    print("\n")
    print("=========================")
