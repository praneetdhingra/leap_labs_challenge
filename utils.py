import torch
import torchvision.transforms as transforms
from PIL import Image


def preprocess_image(image_path):
    """Preprocesses the input image for the model.

    Args:
        image_path: Path to the input image.

    Returns:
        torch.Tensor: Preprocessed image.
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    return image


def image_class(image, model):
    """Determine and print the label of the class.

    Args:
        image: Image tensor.
        model: Model used for classification.

    Returns:
    """

    # Load ImageNet class labels
    with open("imagenet_classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Classify the image
    with torch.no_grad():
        outputs = model(image)
        _, predicted_class_idx = torch.max(outputs, 1)
        predicted_class_label = classes[predicted_class_idx.item()]

    print(f"Predicted class label: {predicted_class_label}", '\n')


def generate_fgsm_noise(model, image, target_class, epsilon):
    """Generates adversarial noise using the Fast Gradient Sign Method (FGSM).

    Args:
        model: The image classification model.
        image: The batched input image.
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


def save_image(image, output_path):
    """Saves the image to the specified path.

    Args:
        image: The adversarial image.
        output_path: Path to save the image.
    """

    # Convert adversarial_image tensor to PIL image
    image = transforms.ToPILImage()(image.squeeze(0))
    image.save(output_path)
    print(f"Adversarial image saved at {output_path}")
