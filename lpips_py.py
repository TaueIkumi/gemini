import lpips
import numpy as np
import torchvision.models as models
import torchvision.transforms.functional as TF

# Initialize LPIPS model
loss_fn_alex = lpips.LPIPS(net="alex")
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)


def lpips_function(img1, img2):
    # Preprocess images: convert to tensor and scale to [-1, 1]
    img1 = (TF.to_tensor(img1) - 0.5) * 2
    img1 = img1.unsqueeze(0)  # Add batch dimension

    img2 = (TF.to_tensor(img2) - 0.5) * 2
    img2 = img2.unsqueeze(0)  # Add batch dimension

    # Calculate LPIPS perceptual similarity
    d = loss_fn_alex(img1, img2)

    # Convert LPIPS distance to percentage similarity (lower distance means higher similarity)
    lpips_score = d.item()
    lpips_percentage = (1 - lpips_score) * 100
    
    return np.clip(lpips_percentage, 0, 100)
