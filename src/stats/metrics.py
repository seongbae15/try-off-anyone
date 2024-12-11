from config import device
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.kid import KernelInceptionDistance
from DISTS_pytorch import DISTS
from PIL import Image
from torchvision import transforms
from src.preprocessing import background_removal, background_whitening


def ssim(path1: str, path2: str):
    img1, img2 = \
        Image.open(path1).resize((384, 512)), \
        Image.open(path2).convert('RGB')

    img1 = background_whitening(background_removal(img1), 384, 512).convert("RGB")

    img1, img2 = \
        transforms.ToTensor()(img1).unsqueeze(0), transforms.ToTensor()(img2).unsqueeze(0)
    score = StructuralSimilarityIndexMeasure(data_range=1.0)
    return score(img1, img2)


def fid_kid(real_loader, generated_loader):
    fid_metric = FrechetInceptionDistance(feature=64)
    fid_metric.eval()
    kid_metric = KernelInceptionDistance(subset_size=50)
    kid_metric.eval()

    for _, real_images in enumerate(real_loader):
        fid_metric.update(real_images, real=True)
        kid_metric.update(real_images, real=True)

    for _, gen_images in enumerate(generated_loader):
        fid_metric.update(gen_images, real=False)
        kid_metric.update(gen_images, real=False)

    return fid_metric.compute(), kid_metric.compute()


def lpips(path1: str, path2: str):
    score = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
    img1, img2 = \
        Image.open(path1).resize((384, 512)), \
        Image.open(path2).convert('RGB')

    img1 = background_whitening(background_removal(img1), 384, 512).convert("RGB")

    img1, img2 = \
        transforms.ToTensor()(img1).unsqueeze(0), transforms.ToTensor()(img2).unsqueeze(0)
    return score(img1, img2)


def dists(paired_dataloader):
    dists_metric = DISTS().to(device)
    total_score = 0.0
    total_pairs = 0

    for _, (real_images, generated_images) in enumerate(paired_dataloader):
        real_images = real_images.to(device)
        generated_images = generated_images.to(device)

        scores = dists_metric(real_images, generated_images)
        batch_sum = scores.sum().item()
        batch_size = scores.size(0)

        total_score += batch_sum
        total_pairs += batch_size

    average_dists = total_score / total_pairs
    return average_dists
