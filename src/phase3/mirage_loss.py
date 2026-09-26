import torch
import torch.nn.functional as F


def cosine_similarity_loss(
    image_embedding,
    target_embedding
):
    image_embedding = F.normalize(
        image_embedding,
        dim=-1
    )

    target_embedding = F.normalize(
        target_embedding,
        dim=-1
    )

    similarity = (
        image_embedding * target_embedding
    ).sum(dim=-1)

    return -similarity.mean()


def mirage_loss(
    image_embedding,
    target_embedding
):
    return cosine_similarity_loss(
        image_embedding,
        target_embedding
    )