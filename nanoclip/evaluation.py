"""Validation loss and full-dataset retrieval, with chunked similarity computation."""

import torch
from torch.utils.data import DataLoader

from nanoclip.loss import clip_loss
from nanoclip.runtime import autocast_context


def retrieval_metrics(images, texts, image_ids=None, text_ids=None, ks=(1, 5, 10), chunk_size=256):
    """Recall@k across the whole gallery, accepting any matching image ID as a positive.

    With omitted IDs, assume a one-to-one aligned dataset. With repeated IDs, callers
    should pass one image embedding per image and all caption embeddings.
    """
    if images.ndim != 2 or texts.ndim != 2 or images.shape[1] != texts.shape[1]:
        raise ValueError("expected [images, dim] and [texts, dim]")
    if len(images) == 0 or len(texts) == 0 or chunk_size < 1 or not ks or min(ks) < 1:
        raise ValueError("retrieval needs nonempty inputs and positive k / chunk_size")
    image_ids = torch.arange(len(images), device=images.device) if image_ids is None else image_ids
    text_ids = torch.arange(len(texts), device=texts.device) if text_ids is None else text_ids
    if len(image_ids) != len(images) or len(text_ids) != len(texts):
        raise ValueError("one ID is required per embedding")
    metrics = {}
    for name, queries, gallery, query_ids, gallery_ids in (
        ("image_to_text", images, texts, image_ids, text_ids),
        ("text_to_image", texts, images, text_ids, image_ids),
    ):
        hits = {k: 0 for k in ks}
        for start in range(0, len(queries), chunk_size):
            logits = queries[start : start + chunk_size].float() @ gallery.float().T
            indices = logits.topk(min(max(ks), len(gallery)), dim=1).indices
            matches = gallery_ids[indices] == query_ids[start : start + chunk_size, None]
            for k in ks:
                hits[k] += matches[:, : min(k, len(gallery))].any(dim=1).sum().item()
        metrics.update({f"{name}_r@{k}": hits[k] / len(queries) for k in ks})
    return metrics


@torch.no_grad()
def evaluate(model, loader: DataLoader, device, precision="fp32"):
    was_training = model.training
    model.eval()
    images, texts, ids = [], [], []
    loss_sum, loss_count = 0.0, 0
    try:
        for pixels, tokens, group_ids in loader:
            with autocast_context(device, precision):
                image, text, scale = model(pixels.to(device), tokens.to(device))
                if len(pixels) > 1:
                    loss_sum += clip_loss(image, text, scale).item() * len(pixels)
                    loss_count += len(pixels)
            images.append(image.float().cpu())
            texts.append(text.float().cpu())
            ids.append(group_ids.cpu())
        if not images or not loss_count:
            raise ValueError("validation needs at least one batch with two pairs")
        image_features, text_features, group_ids = (
            torch.cat(images),
            torch.cat(texts),
            torch.cat(ids),
        )
        # Keep one image per ID, so multiple captions don't duplicate gallery candidates.
        first = {}
        for index, group in enumerate(group_ids.tolist()):
            first.setdefault(group, index)
        unique = torch.tensor(list(first.values()), dtype=torch.long)
        metrics = retrieval_metrics(
            image_features[unique], text_features, group_ids[unique], group_ids
        )
        metrics["val_loss"] = loss_sum / loss_count
        metrics["val_pairs"] = len(text_features)
        return metrics
    finally:
        model.train(was_training)
