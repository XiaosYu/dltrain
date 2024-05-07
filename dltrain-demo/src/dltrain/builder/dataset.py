from ..dataset import (PyTorchNativeDataset, ImageCategoricalDataset, VectorCategoricalDataset)


class DatasetWizard:
    @classmethod
    def use_mnist(cls, root, train, download=False):
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor

        dataset = MNIST(root, train=train, transform=ToTensor(), download=download)
        return PyTorchNativeDataset(dataset)

    @classmethod
    def use_iris(cls):
        from sklearn.datasets import load_iris

        iris = load_iris()
        data = iris.data
        target = iris.target
        return VectorCategoricalDataset(data, target)

    @classmethod
    def use_image_folder(cls, root):
        from torchvision.datasets import ImageFolder
        from torchvision.transforms import ToTensor

        dataset = ImageFolder(root, transform=ToTensor())
        return PyTorchNativeDataset(dataset)

    @classmethod
    def use_lfw(cls):
        from sklearn.datasets import fetch_lfw_people

        lfw = fetch_lfw_people()
        features, labels = lfw.images, lfw.target

        return ImageCategoricalDataset(features, labels)