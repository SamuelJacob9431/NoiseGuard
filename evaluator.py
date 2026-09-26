import math
import torch


def evaluate_protection(
    original_image,
    protected_image,
    original_latent,
    protected_latent,
    original_reconstruction,
    protected_reconstruction,
    mirage_model=None,
    pgd_loss_history=None,
    mirage_loss_history=None,
    epsilon=8 / 255,
):
    """Compute backend-only evaluation metrics for one NoiseGuard run."""
    with torch.no_grad():
        original = original_image.float()
        protected = protected_image.float()
        protected_latent = protected_latent.float()
        original_latent = original_latent.float()
        protected_reconstruction = protected_reconstruction.float()
        original_reconstruction = original_reconstruction.float()

        perturbation = protected - original
        abs_perturbation = torch.abs(perturbation)

        protected_reconstruction_mse = torch.mean(
            (original - protected_reconstruction) ** 2
        ).item()
        baseline_reconstruction_mse = torch.mean(
            (original - original_reconstruction) ** 2
        ).item()
        latent_distance = torch.mean(
            (original_latent - protected_latent) ** 2
        ).item()

        perturbation_linf = abs_perturbation.max().item()
        perturbation_mean_abs = abs_perturbation.mean().item()
        perturbation_l2 = torch.linalg.vector_norm(
            perturbation.reshape(perturbation.shape[0], -1), dim=1
        ).mean().item()

        mse_for_psnr = max(protected_reconstruction_mse, 1e-12)
        psnr = 10.0 * math.log10(4.0 / mse_for_psnr)

        baseline_psnr = 10.0 * math.log10(
            4.0 / max(baseline_reconstruction_mse, 1e-12)
        )

    result = {
        "baselineReconstructionMse": round(baseline_reconstruction_mse, 8),
        "protectedReconstructionMse": round(protected_reconstruction_mse, 8),
        "reconstructionMseChange": round(
            protected_reconstruction_mse - baseline_reconstruction_mse, 8
        ),
        "baselinePsnrDb": round(baseline_psnr, 4),
        "protectedPsnrDb": round(psnr, 4),
        "latentDistance": round(latent_distance, 8),
        "perturbationLinf": round(perturbation_linf, 8),
        "perturbationMeanAbs": round(perturbation_mean_abs, 8),
        "perturbationL2": round(perturbation_l2, 8),
        "epsilon": round(float(epsilon), 8),
        "budgetUtilization": round(
            perturbation_linf / max(float(epsilon), 1e-12), 4
        ),
    }

    if pgd_loss_history:
        result["pgdInitialLoss"] = round(float(pgd_loss_history[0]), 8)
        result["pgdFinalLoss"] = round(float(pgd_loss_history[-1]), 8)

    if mirage_loss_history:
        result["mirageInitialScore"] = round(float(mirage_loss_history[0]), 8)
        result["mirageFinalScore"] = round(float(mirage_loss_history[-1]), 8)

    if mirage_model is not None:
        # Evaluate the surrogate before and after the combined attack.
        original_score = mirage_model.target_similarity(original_image).item()
        protected_score = mirage_model.target_similarity(protected_image).item()
        result["mirageOriginalScore"] = round(float(original_score), 8)
        result["mirageProtectedScore"] = round(float(protected_score), 8)
        result["mirageScoreChange"] = round(
            float(protected_score - original_score), 8
        )

    return result
