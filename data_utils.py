import numpy as np
from torchvision import datasets, transforms

def load_cifar10(num_train=5000, num_test=500):
    """
    CIFAR-10 데이터를 로드하고 numpy array (float64)로 반환
    """
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # numpy 변환 + float64 캐스팅
    X_train = trainset.data[:num_train].astype(np.float64).reshape(num_train, -1)
    y_train = np.array(trainset.targets[:num_train])

    X_test = testset.data[:num_test].astype(np.float64).reshape(num_test, -1)
    y_test = np.array(testset.targets[:num_test])

    return X_train, y_train, X_test, y_test
