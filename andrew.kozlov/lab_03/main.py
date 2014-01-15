from lab_01.main import divide, load_data


__author__ = 'adkozlov'


def main():
    train_set, test_set = divide(load_data())


if __name__ == "__main__":
    main()