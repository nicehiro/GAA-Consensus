import torch
import torchvision
from torchvision import transforms


class DataLoader:
    datasets = {
        "MNIST": torchvision.datasets.MNIST,
        "CIFAR10": torchvision.datasets.CIFAR10,
        "CIFAR100": torchvision.datasets.CIFAR100,
    }

    transforms = {
        "MNIST": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
        "CIFAR10": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        "CIFAR100": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    }

    def __init__(self, path, name, batch_size) -> None:
        train_set = self.datasets[name](
            root=path, train=True, download=True, transform=self.transforms[name]
        )
        test_set = self.datasets[name](
            root=path, train=False, download=True, transform=self.transforms[name]
        )
        self._train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
        self._test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=2
        )

    @property
    def train_loader(self):
        """
        Get trainning loader.
        """
        return self._train_loader

    @property
    def test_loader(self):
        """
        Get testing loader.
        """
        return self._test_loader

    def next(self):
        train_iter = iter(self._train_loader)
        return train_iter.next()
