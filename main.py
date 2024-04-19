import argparse
from PIL import Image

from utils import generate_fgsm_noise, apply_adversarial_noise
from model_interface import load_model

def main():
    parser = argparse.ArgumentParser(description='Generate adversarial images.')
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('target_class', type=int, help='Target class for misclassification.')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Strength of perturbation.')
    args = parser.parse_args()

    model = load_model()
    image = Image.open(args.image_path).convert('RGB')
    # ... (Preprocess the image as needed for the model) ...

    noise = generate_fgsm_noise(model, image, args.target_class, args.epsilon)
    adversarial_image = apply_adversarial_noise(image, noise)

    # ... (Convert adversarial_image to PIL image and save) ...

if __name__ == "__main__":
    main()
