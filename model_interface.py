import torchvision


def load_model(model_name="resnet18"):
    """Loads a pre-trained image classification model.

    Args:
        model_name: The name of the model to load (default="resnet18").

    Returns:
        torch.nn.Module: The loaded model.
    """

    model = getattr(torchvision.models, model_name)(pretrained=True)
    model.eval()  # Set the model to evaluation mode

    return model
