from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from models.model import BaselineModel, EmbeddingModel, MlpModel, Cnn, EmbeddingCnn, AbpCnn
from models.vgg import Vgg16, Vgg16_bn, EmbeddingVgg16, EmbeddingVgg16_bn, AbpVgg16, AbpVgg16bn


class Flatten:
    def __call__(self, tensor):
        return tensor.flatten(0)

    def __repr__(self):
        return self.__class__.__name__ + '()'


# Flatten
def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,)), Flatten()])

    train_loader = DataLoader(
        MNIST(download=True, root="~/Datasets", transform=data_transform, train=True),
        batch_size=train_batch_size, shuffle=True
    )

    val_loader = DataLoader(
        MNIST(download=False, root="~/Datasets", transform=data_transform, train=False),
        batch_size=val_batch_size, shuffle=False
    )
    return train_loader, val_loader


# no Flatten
def get_data_loaders1(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST(download=True, root="~/Datasets", transform=data_transform, train=True),
        batch_size=train_batch_size, shuffle=True
    )

    val_loader = DataLoader(
        MNIST(download=False, root="~/Datasets", transform=data_transform, train=False),
        batch_size=val_batch_size, shuffle=False
    )
    return train_loader, val_loader


def get_data_loaders2(train_batch_size, val_batch_size):
    train_transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_loader = DataLoader(
        CIFAR10(download=True, root="~/Datasets", transform=train_transform, train=True),
        batch_size=train_batch_size, shuffle=True, num_workers=2
    )

    val_loader = DataLoader(
        CIFAR10(download=False, root="~/Datasets", transform=test_transform, train=False),
        batch_size=val_batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader


def name2model_and_loader(modelname):
    model_loader = None
    if modelname == BaselineModel.__name__:
        model_loader = BaselineModel(), get_data_loaders, 0
    elif modelname == EmbeddingModel.__name__:
        model_loader = EmbeddingModel(), get_data_loaders, 0
    elif modelname == MlpModel.__name__:
        model_loader = MlpModel(), get_data_loaders, 1
    elif modelname == Cnn.__name__:
        model_loader = Cnn(), get_data_loaders1, 0
    elif modelname == EmbeddingCnn.__name__:
        model_loader = EmbeddingCnn(), get_data_loaders1, 0
    elif modelname == AbpCnn.__name__:
        model_loader = AbpCnn(), get_data_loaders1, 1
    elif modelname == Vgg16.__name__:
        model_loader = Vgg16(), get_data_loaders2, 0
    elif modelname == Vgg16_bn.__name__:
        model_loader = Vgg16_bn(), get_data_loaders2, 0
    elif modelname == EmbeddingVgg16.__name__:
        model_loader = EmbeddingVgg16(), get_data_loaders2, 0
    elif modelname == EmbeddingVgg16_bn.__name__:
        model_loader = EmbeddingVgg16_bn(), get_data_loaders2, 0
    elif modelname == AbpVgg16.__name__:
        model_loader = AbpVgg16(), get_data_loaders2, 1
    elif modelname == AbpVgg16bn.__name__:
        model_loader = AbpVgg16bn(), get_data_loaders2, 1
    # else:
    #     raise ValueError(f'model [{modelname}] not exists.')

    if model_loader:
        return model_loader
    else:
        print(f'model [{modelname}] not exists.')
        print(
                f'Options: ['
                f'{BaselineModel.__name__}, '
                f'{EmbeddingModel.__name__}, '
                f'{MlpModel.__name__}, '
                f'{Cnn.__name__}, '
                f'{EmbeddingCnn.__name__}, '
                f'{AbpCnn.__name__}, '
                f'{Vgg16.__name__}, '
                f'{Vgg16_bn.__name__}, '
                f'{EmbeddingVgg16.__name__}, '
                f'{EmbeddingVgg16_bn.__name__}, '
                f'{AbpVgg16.__name__}, '
                f'{AbpVgg16bn.__name__}'
                f']'
        )
        exit(1)


def correct(pred, label):
    return pred.argmax(dim=1).eq(label).sum()
