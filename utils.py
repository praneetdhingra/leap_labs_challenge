import torch


def generate_fgsm_noise(model, image, target_class, epsilon):
    """Generates adversarial noise using the Fast Gradient Sign Method (FGSM).

    Args:
        model: The image classification model.
        image: The input image (torch.Tensor).
        target_class: The desired misclassification target class.
        epsilon: The strength of the adversarial perturbation.

    Returns:
        torch.Tensor: The adversarial noise.
    """

    image.requires_grad = True
    output = model(image)
    loss = torch.nn.CrossEntropyLoss()(output, target_class)
    model.zero_grad()
    loss.backward()

    return epsilon * image.grad.data.sign()


def apply_adversarial_noise(image, noise):
    """Applies adversarial noise to an image.

    Args:
        image: The input image (torch.Tensor).
        noise: The adversarial noise (torch.Tensor).

    Returns:
        torch.Tensor: The image with adversarial noise added.
    """

    perturbed_image = image + noise
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure values are within valid range
    return perturbed_image
