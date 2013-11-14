__author__ = 'max'


def get_input(data_file, train_input_percent):
    file = open(data_file, "rU")
    instances = []
    for line in file:
        split = line.split(",")
        instance_id = int(split[0])
        if split[1] == 'M':
            diagnosis = 1
        else:
            diagnosis = -1
        features = [float(f) for f in split[2:]]
        instance = (instance_id, diagnosis, features)
        instances.append(instance)
    file.close()

    import random

    random.seed(23)
    random.shuffle(instances)
    train_input_num = int((train_input_percent / 100.) * len(instances))
    return instances[:train_input_num], instances[train_input_num + 1:]


def main():
    training, test = get_input("data/wdbc.data", 80)
    #====Starting lab 1 (Perceptron)
    #print "Perceptron lab"
    #import lab1.main as Lab1
    #Lab1.start(training, test)
    #===Starting lab 2 (SVM)
    print "SVM"
    import lab2.main as Lab2
    Lab2.start(training, test)


if __name__ == "__main__":
    main()