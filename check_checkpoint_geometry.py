import argparse
import os
import torch


def _load_state_dict(checkpoint_path: str):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


def main():
    parser = argparse.ArgumentParser(
        description="Check if a SAM2 checkpoint contains geometry fusion weights."
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the checkpoint (.pt) to inspect.",
    )
    args = parser.parse_args()

    state = _load_state_dict(args.checkpoint)
    fm_keys = [k for k in state.keys() if k.startswith("feature_merger.")]
    geom_keys = [k for k in state.keys() if k.startswith("geometry_extractor.")]

    print(f"Checkpoint: {args.checkpoint}")
    print(f"feature_merger keys: {len(fm_keys)}")
    print(f"geometry_extractor keys: {len(geom_keys)}")
    if fm_keys:
        print("GEOMETRY FUSION WEIGHTS: PRESENT")
    else:
        print("GEOMETRY FUSION WEIGHTS: MISSING")


if __name__ == "__main__":
    main()

