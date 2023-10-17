import torch


if __name__ == '__main__':
    A: torch.Tensor = torch.tensor([[1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 9]])

    assert torch.equal(A, A.T.T)

    B = torch.eye(3)

    assert torch.equal(A.T + B.T, (A + B).T)

    C: torch.Tensor = torch.arange(24).reshape(2, 3, 4)

    print(len(C))

    print(A.sum(axis=1))
    print(A / A.sum(axis=1))

    print(8)
    print(C)
    print(C.sum(axis=0))
    print(C.sum(axis=1))
    print(C.sum(axis=2))

    print(9)

    X = torch.randn(2, 3)
    print(X)
    print(torch.linalg.norm(X), ((X ** 2).sum()) ** 0.5)
