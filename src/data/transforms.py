"""ViT preprocessing transforms."""

from torchvision import transforms


def get_vit_transform(image_size: int = 224):
    """Return the standard ViT preprocessing transform.

    Uses ImageNet normalization values as expected by google/vit-base-patch16-224-in21k.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
