import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class StainDataset(Dataset):
    """
    Класс, возвращающий пары изображений: (HE_img, Ki67_img).
    Предполагается, что в папках data/dataset_HE/tiles и data/dataset_Ki67/tiles
    лежат файлы в формате .jpg (размер 512*512).
    """
    def __init__(self,
                 he_dir: str,
                 ki67_dir: str,
                 transform=None):
        super().__init__()
        self.he_dir = he_dir
        self.ki67_dir = ki67_dir

        if transform is None:
            self.transform = transforms.Compose([
                                transforms.Resize((512, 512)),  # на всякий случай
                                transforms.ToTensor()
                            ])
        else:
            self.transform = transform

        self.he_files = sorted([
            f for f in os.listdir(self.he_dir)
            if f.lower().endswith('.jpg')
        ])
        self.ki67_files = sorted([
            f for f in os.listdir(self.ki67_dir)
            if f.lower().endswith('.jpg')
        ])

        # Минимальная длина двух списков (один 1749, другой 1760)
        self.length = min(len(self.he_files), len(self.ki67_files))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        he_path = os.path.join(self.he_dir, self.he_files[idx])
        ki67_path = os.path.join(self.ki67_dir, self.ki67_files[idx])

        he_img = Image.open(he_path).convert('RGB')
        ki67_img = Image.open(ki67_path).convert('RGB')

        if self.transform is not None:
            he_img = self.transform(he_img)
            ki67_img = self.transform(ki67_img)

        return he_img, ki67_img


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # на всякий случай
        transforms.ToTensor()
    ])

    he_dir = "data/dataset_HE/tiles"
    ki67_dir = "data/dataset_Ki67/tiles"

    dataset = StainDataset(he_dir=he_dir,
                           ki67_dir=ki67_dir,
                           transform=transform)

    # DataLoader: batch_size=1
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for i, (he_batch, ki67_batch) in enumerate(dataloader):
        print("  HE batch shape:", he_batch.shape)      # [1, 3, 512, 512]
        print("  Ki67 batch shape:", ki67_batch.shape)  # [1, 3, 512, 512]
        break