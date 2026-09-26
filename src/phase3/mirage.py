import torch
import torch.nn.functional as F
import open_clip


TARGET_CONCEPTS = [
    "policy violating image",
    "unsafe image content",
    "content that should be blocked by an image safety filter",
]


class MirageSurrogate:

    def __init__(self, device):

        self.device = device

        print("\nLoading MIRAGE surrogate...")

        model_name = "ViT-B-32"
        pretrained = "laion2b_s34b_b79k"

        self.model, _, _ = (
            open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=device,
            )
        )

        self.model = self.model.float()
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.tokenizer = open_clip.get_tokenizer(
            model_name
        )

        text_tokens = self.tokenizer(
            TARGET_CONCEPTS
        ).to(device)

        with torch.no_grad():

            text_features = self.model.encode_text(
                text_tokens
            )

            text_features = F.normalize(
                text_features.float(),
                dim=-1
            )

        self.target_features = text_features

        print("✓ MIRAGE surrogate ready")

    def prepare_image(self, image):

        image = image.float()

        image = torch.clamp(
            image,
            -1.0,
            1.0
        )

        image = (
            image + 1.0
        ) / 2.0

        image = F.interpolate(
            image,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        mean = torch.tensor(
            [
                0.48145466,
                0.45782750,
                0.40821073
            ],
            device=self.device,
            dtype=torch.float32
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [
                0.26862954,
                0.26130258,
                0.27577711
            ],
            device=self.device,
            dtype=torch.float32
        ).view(1, 3, 1, 1)

        image = (
            image - mean
        ) / std

        return image

    def image_features(self, image):

        image = self.prepare_image(
            image
        )

        features = self.model.encode_image(
            image
        )

        features = features.float()

        features = F.normalize(
            features,
            dim=-1
        )

        return features

    def target_similarity(self, image):

        image_features = self.image_features(
            image
        )

        similarity = (
            image_features
            @ self.target_features.T
        )

        return similarity.mean()


def mirage_attack(
    image,
    device,
    epsilon=8 / 255,
    alpha=1 / 255,
    steps=30,
):

    print("\n" + "=" * 60)
    print("MIRAGE - Representation Optimization")
    print("=" * 60)

    surrogate = MirageSurrogate(
        device
    )

    original = (
        image
        .detach()
        .clone()
    )

    adversarial = (
        original
        .detach()
        .clone()
    )

    loss_history = []

    initial_score = surrogate.target_similarity(
        adversarial
    ).item()

    print(
        f"Initial target similarity : "
        f"{initial_score:.6f}"
    )

    for step in range(steps):

        adversarial = (
            adversarial
            .detach()
            .requires_grad_(True)
        )

        score = surrogate.target_similarity(
            adversarial
        )

        loss_history.append(
            score.item()
        )

        surrogate.model.zero_grad(
            set_to_none=True
        )

        if adversarial.grad is not None:
            adversarial.grad.zero_()

        score.backward()

        with torch.no_grad():

            gradient = (
                adversarial.grad.sign()
            )

            adversarial = (
                adversarial
                + alpha * gradient
            )

            perturbation = (
                adversarial
                - original
            )

            perturbation = torch.clamp(
                perturbation,
                -epsilon,
                epsilon
            )

            adversarial = (
                original
                + perturbation
            )

            adversarial = torch.clamp(
                adversarial,
                -1.0,
                1.0
            )

        if (
            step == 0
            or (step + 1) % 5 == 0
            or step == steps - 1
        ):

            print(
                f"Step {step + 1:03d}/{steps} "
                f"| Target similarity : "
                f"{score.item():.6f}"
            )

    final_score = (
        surrogate
        .target_similarity(
            adversarial
        )
        .item()
    )

    print(
        f"\nFinal target similarity : "
        f"{final_score:.6f}"
    )

    print("✓ MIRAGE complete")

    return (
        adversarial.detach(),
        loss_history,
        initial_score,
        final_score,
    )