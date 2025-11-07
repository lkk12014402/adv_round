class QuantConfig:
    model_name = "openai/whisper-small"
    adaptive_max_json = "adaptive_max.json"
    init_mode = "mixed"
    block_size_w = 32
    block_size_act = 32
    lr = 5e-4
    weight_decay = 0.0
    epochs = 1
    output_ckpt = "whisper_quant_epsilon.pt"

def get_config():
    return QuantConfig()
