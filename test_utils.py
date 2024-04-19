import unittest
import torch
import torchvision.models as models

from utils import preprocess_image, generate_adversarial_noise

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.image_path = "./test/panda.png"
        self.model = models.resnet18(pretrained=True).eval()
        self.image = preprocess_image(self.image_path)

    def test_preprocess_image(self):
        self.assertEqual(self.image.shape, torch.Size([1, 3, 224, 224]))

    def test_generate_adversarial_noise(self):
        target_class = 242
        epsilon = 0.05
        noise = generate_adversarial_noise(self.model, self.image, target_class, epsilon)
        self.assertEqual(noise.shape, torch.Size([1, 3, 224, 224]))


if __name__ == "__main__":
    unittest.main()
