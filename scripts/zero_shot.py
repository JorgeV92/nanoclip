"""Rank candidate labels using text prompts and a trained image encoder."""

import argparse
import json

import torch
from PIL import Image

from nanoclip.checkpoint import load_model
from nanoclip.dataset import preprocess_image
from nanoclip.runtime import select_device
from nanoclip.tokenizer import ByteTokenizer


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", nargs="+", required=True, help='e.g. --labels "a cat" "a dog"')
    parser.add_argument("--template", default="a photo of {}", help="exactly one {} placeholder")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args(argv)
    if args.template.count("{}") != 1:
        parser.error("--template must contain exactly one {}")
    device = select_device(args.device)
    model = load_model(args.checkpoint, device)
    tokenizer = ByteTokenizer(model.config.context_length)
    with Image.open(args.image) as source:
        image = preprocess_image(source, model.config.image_size).unsqueeze(0).to(device)
    text = tokenizer([args.template.format(label) for label in args.labels]).to(device)
    logits = model.logit_scale.clamp(max=4.605170186).exp() * (
        model.encode_image(image) @ model.encode_text(text).T
    )
    scores = logits.softmax(dim=-1)[0].cpu().tolist()
    ranking = sorted(zip(args.labels, scores, strict=True), key=lambda pair: -pair[1])
    print(json.dumps([{"label": label, "score": score} for label, score in ranking], indent=2))


if __name__ == "__main__":
    main()
