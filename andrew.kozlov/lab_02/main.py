from lab_01.main import divide, percent, load_data

__author__ = 'adkozlov'


def train(data):
    return 0


def test(data):
    return 0


def main():
    train_set, test_set = divide(load_data())
    c = train(train_set)
    e = test(test_set)
    print('C = %f\nerror = %6.2f' % (c, percent(e)))


if __name__ == "__main__":
    main()