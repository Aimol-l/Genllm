import yaml
import struct
from transformers import AutoTokenizer
from model import Qwen3Config, Qwen3Model
from gguf import GGUFTorchLoader


def _build_tensor_list(yaml_path: str) -> list:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    weight = raw["weight"]

    config = raw["config"]
    with open(config["model_path"], "rb") as f:
        f.read(4)
        f.read(4)
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        meta_kv_count = struct.unpack('<Q', f.read(8))[0]

        for _ in range(meta_kv_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            f.read(key_len)
            val_type = struct.unpack('<I', f.read(4))[0]
            if val_type == 8:
                strlen = struct.unpack('<Q', f.read(8))[0]
                f.read(strlen)
            elif val_type == 9:
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                for _ in range(arr_len):
                    if arr_type == 8:
                        slen = struct.unpack('<Q', f.read(8))[0]
                        f.read(slen)
                    elif arr_type in (0, 1):
                        f.read(1)
                    elif arr_type in (2, 3, 12, 13):
                        f.read(2)
                    elif arr_type in (4, 5, 10):
                        f.read(4)
                    elif arr_type == 6:
                        f.read(4)
                    elif arr_type == 7:
                        f.read(1)
                    elif arr_type == 11:
                        f.read(8)
                    else:
                        f.read(4)
            elif val_type in (0, 1):
                f.read(1)
            elif val_type in (2, 3, 12, 13):
                f.read(2)
            elif val_type in (4, 5, 10):
                f.read(4)
            elif val_type == 6:
                f.read(4)
            elif val_type == 7:
                f.read(1)
            elif val_type == 11:
                f.read(8)
            else:
                f.read(4)

        tensor_list = []
        for _ in range(tensor_count):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode("utf-8")
            n_dims = struct.unpack('<I', f.read(4))[0]
            f.read(8 * n_dims)
            f.read(4)
            f.read(8)
            tensor_list.append((name, weight[name][0], weight[name][1]))

    return tensor_list


if __name__ == "__main__":

    config_path = "models/qwen3-0.6b-bf16.yaml"


    config = Qwen3Config(config_path)
    print(f"Config: hidden={config.hidden_size}, layers={config.num_layers}, "f"heads={config.num_heads}/{config.num_kv_heads}, vocab={config.vocab_size}")

    tensor_list = _build_tensor_list(config_path)

    loader = GGUFTorchLoader(config.model_path, config.base_offset, tensor_list)

    model = Qwen3Model(config)
    model.load_weights(loader)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    prompt = "1+1="
    chat_result = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    chat_ids = chat_result["input_ids"] if hasattr(chat_result, "input_ids") else chat_result

    print(tokenizer.decode(chat_ids))

    print(f"Input:  {prompt!r}")
    print(f"Token ids: {chat_ids}")

    input_ids = list(chat_ids)
    max_new_tokens = 64
    EOS_TOKEN_IDS = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    print(EOS_TOKEN_IDS)

    print(f"Generating (max {max_new_tokens} tokens) ...")
    for step in range(max_new_tokens):
        logits = model.forward(input_ids)
        next_token = logits[0, -1].argmax().item()

        if next_token in EOS_TOKEN_IDS:
            break

        input_ids.append(next_token)

    print("\n")

    print(input_ids)
    print(f"Full output:\n{tokenizer.decode(input_ids)}")
