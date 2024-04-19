import argparse
import torch

from utils import preprocess_image, image_class, generate_fgsm_noise, apply_adversarial_noise, save_image
from model_interface import load_model

def main():
    parser = argparse.ArgumentParser(description='Generate adversarial images.')
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('target_class', type=int, help='Target class for misclassification.')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Strength of perturbation.')
    parser.add_argument('--output_path', type=str, default='adversarial_image.jpg',
                        help='Path to save the adversarial image.')
    args = parser.parse_args()

    try:
        model = load_model()
        image = preprocess_image(args.image_path)
        print('Loaded Image:')
        image_class(image, model)

        noise = generate_fgsm_noise(model, image, torch.tensor([args.target_class]), args.epsilon)
        adversarial_image = apply_adversarial_noise(image, noise)
        print('Adverserial Image:')
        image_class(adversarial_image, model)

        save_image(adversarial_image, args.output_path)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
