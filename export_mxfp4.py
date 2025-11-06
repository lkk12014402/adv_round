import torch
import argparse
from modules.quantized_linear import QuantizedLinear

def export_mxfp4(checkpoint_path: str, output_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state"]

    # extract QuantizedLinear final quantized weight
    exported = {}
    for k, v in state.items():
        exported[k] = v

    torch.save(exported, output_path)
    print(f"Exported quantized weights saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="mxfp4_exported.pt")
    args = parser.parse_args()
    export_mxfp4(args.checkpoint, args.out)
