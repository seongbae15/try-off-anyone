from config import TEST_DATA_PATH
from src.stats.metrics import ssim, fid_kid, lpips, dists
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import os
import glob
import numpy as np


def statistics():
    ground_truth_path = os.path.join(TEST_DATA_PATH, 'cloth')
    results_path = os.path.join('data', 'results')

    ssim_scores, lpips_scores = [], []
    for gt_image in tqdm(glob.glob(f'{ground_truth_path}*.jpg')):
        result_image = os.path.join(results_path, gt_image.split('/')[-1].split('.')[0] + '.png')
        ssim_scores.append(ssim(gt_image, result_image))
        lpips_scores.append(lpips(gt_image, result_image))

    # Compute the average SSIM over the dataset
    average_ssim = np.mean(ssim_scores)
    average_lpips = np.mean(lpips_scores)
    print("Average SSIM:", average_ssim)
    print("Average LPIPS:", average_lpips)

    class ImageDataset(Dataset):
        def __init__(self, image_dir, transformation=None):
            self.image_dir = image_dir
            self.image_files = [f for f in os.listdir(image_dir) if
                                f.endswith(('jpg', 'jpeg', 'png'))]
            self.transform = transformation

        def __getitem__(self, idx):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image

        def __len__(self):
            return len(self.image_files)

    transform = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.PILToTensor()
    ])

    # Load datasets (original and generated images) using ImageFolder
    real_dataset = ImageDataset(image_dir=ground_truth_path, transformation=transform)
    generated_dataset = ImageDataset(image_dir=results_path, transformation=transform)

    # DataLoaders
    real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
    generated_loader = DataLoader(generated_dataset, batch_size=32, shuffle=False)

    fid_score, (kid_score_mean, kid_score_std) = fid_kid(real_loader, generated_loader)

    print("Average FID:", fid_score.item())
    print("Average KID:", kid_score_mean.item())
    print("KID Std:", kid_score_std.item())

    class PairedImageDataset(Dataset):
        def __init__(self, image_dir1, image_dir2, transformation=None):
            self.image_dir1 = image_dir1
            self.image_dir2 = image_dir2
            self.image_files1 = sorted([f for f in os.listdir(image_dir1) if
                                        f.endswith(('jpg', 'jpeg', 'png'))])
            self.image_files2 = sorted([f for f in os.listdir(image_dir2) if
                                        f.endswith(('jpg', 'jpeg', 'png'))])
            self.transform = transformation

        def __getitem__(self, idx):
            img_path1 = os.path.join(self.image_dir1, self.image_files1[idx])
            img_path2 = os.path.join(self.image_dir2, self.image_files2[idx])
            img1 = Image.open(img_path1).convert("RGB")
            img2 = Image.open(img_path2).convert("RGB")

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2

        def __len__(self):
            return min(len(self.image_files1), len(self.image_files2))

    transform_dists = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor()
    ])

    paired_dataset = PairedImageDataset(
        image_dir1=ground_truth_path, image_dir2=results_path, transformation=transform_dists
    )
    paired_dataloader = DataLoader(paired_dataset, batch_size=16, shuffle=False)

    average_dists = dists(paired_dataloader)
    print("Average DISTS:", average_dists)
