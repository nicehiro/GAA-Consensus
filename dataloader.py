import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

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


class DataDistributor:
    """
    Generate train_loader for each worker.
    Generate test_loader for final test. only one!
    """

    datasets = {
        "MNIST": torchvision.datasets.MNIST,
        "CIFAR10": torchvision.datasets.CIFAR10,
        "CIFAR100": torchvision.datasets.CIFAR100,
    }

    def __init__(self, path, name, batch_size, workers_n) -> None:
        # TODO Can we implement this method?
        # Only need to load train set data and don't use other function here.
        self.train_set = self.datasets[name](
            root=path, train=True, download=True, transform=transforms[name]
        )
        test_set = self.datasets[name](
            root=path, train=False, download=False, transform=transforms[name]
        )
        self._test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=2
        )
        self.name = name
        self.workers_n = workers_n
        self.batch_size = batch_size
        self._train_loaders = []

    def distribute(self):
        datas = self.train_set.data.detach().clone()
        labels = self.train_set.targets.detach().clone()
        # shuffle
        rand_index = np.random.randint(0, len(datas), len(datas))
        datas = datas[rand_index]
        labels = labels[rand_index]
        set_size_of_worker = len(datas) // self.workers_n
        for i in range(self.workers_n):
            a = i * set_size_of_worker
            b = (i + 1) * set_size_of_worker if i < (self.workers_n - 1) else len(datas)
            data = datas[a:b]
            label = labels[a:b]
            train_loader = DataLoader(self.name, data, label, self.batch_size)
            self._train_loaders.append(train_loader)

    @property
    def train_loaders(self):
        return self._train_loaders

    @property
    def test_loader(self):
        return self._test_loader


class DataLoader:
    def __init__(self, name, data_set, label_set, batch_size) -> None:
        """
        Data loader for eacher worker.
        """
        self.name = name
        self.data_set = data_set
        self.label_set = label_set
        self.batch_size = batch_size

    def next(self):
        data_index = np.random.randint(
            self.batch_size, len(self.data_set), self.batch_size
        )
        imgs, targets = self.data_set[data_index], self.label_set[data_index]
        imgs = self._transform(imgs)
        return imgs, targets

    def _transform(self, imgs):
        # below are copied from offical pytorch codes
        img_trans = []
        for img in imgs:
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img.numpy(), mode="L")
            if transforms[self.name] is not None:
                img = transforms[self.name](img)
            img_trans.append(img)
        imgs = torch.stack(img_trans)
        return imgs

    def val_set(self):
        return (
            self._transform(self.data_set[: self.batch_size]),
            self.label_set[: self.batch_size],
        )
