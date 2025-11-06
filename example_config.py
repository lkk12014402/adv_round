class Config:
    seed = 42
    in_dim = 512
    out_dim = 512
    depth = 3
    num_classes = 10
    num_samples = 4096
    batch_size = 32

    # Weight quantization groups
    group_size_weight = 32
    # Activation quantization groups
    group_size_act = 32

    # Dynamic activation quantization switches
    act_quant = True
    act_use_ema = True
    act_ema_momentum = 0.95
    act_recompute_interval = 1  # All other forward indices are recalculated; a value >1 can be set to reduce expenses.

    # Adversarial ε setup
    lr = 5e-4
    weight_decay = 0.0
    epochs = 5
    warmup_ratio = 0.1
    rho = 0.25
    pgd_steps = 1  # 1=FGSM
    alpha = 0.05   # PGD step

    use_distill = True
    lambda_distill = 0.5
    distill_temp = 1.5

    epsilon_reg = 5e-4
    log_interval = 50
    output_path = "checkpoint_mxfp4_dynamic_act.pt"

def get_config():
    return Config()
