import unittest
import torch

from utils import generate_fgsm_noise, apply_adversarial_noise
from model_interface import load_model


class TestAdversarialUtils(unittest.TestCase):

    def setUp(self):
        self.model = load_model()
        self.image = torch.rand((1, 3, 224, 224))  # Random image tensor
        self.target_class = torch.tensor([3])  # Target class index
        self.epsilon = 0.05

    def test_generate_fgsm_noise(self):
        noise = generate_fgsm_noise(self.model, self.image, self.target_class, self.epsilon)
        self.assertIsNotNone(noise)

    def test_apply_adversarial_noise(self):
        noise = generate_fgsm_noise(self.model, self.image, self.target_class, self.epsilon)
        perturbed_image = apply_adversarial_noise(self.image, noise)
        self.assertIsNotNone(perturbed_image)


if __name__ == '__main__':
    unittest.main()
