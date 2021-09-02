import sys


def calc(a):

    return [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]


if __name__ == '__main__':
    X = calc(sys.argv[1])
    print(X)
