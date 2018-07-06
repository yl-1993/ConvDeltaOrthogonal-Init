import os
import sys
import matplotlib.pyplot as plt


def parse(log):
    train_x, train_y, test_x, test_y = [], [], [], []
    with open(log) as f:
        train_prec = []
        epoch = 0
        for x in f.readlines():
            if x.startswith('Epoch'):
                train_prec += [float(x.split(' ')[-1][1:-2])]
            elif x.startswith(' * Prec@1'):
                test_prec = float(x.split(' ')[-1])
                train_prec = sum(train_prec) / len(train_prec)
                train_x += [epoch]
                test_x += [epoch]
                train_y += [train_prec]
                test_y += [test_prec]
                epoch += 1
                train_prec = []
            else:
                continue
    return train_x, train_y, test_x, test_y
    

def plot(log_files):
    fig, ax = plt.subplots()
    for log in log_files:
        if not os.path.isfile(log):
            print('No file found at {}'.format(log))
            continue
        train_x, train_y, test_x, test_y = parse(log)
        ax.errorbar(train_x, train_y, linestyle='dashed', label='train_'+log)
        ax.errorbar(test_x, test_y, linestyle='solid', label='test_'+log)

    ax.set_xlabel('epoches')
    ax.set_ylabel('accuracy')
    plt.legend()
    plt.show()
    plt.savefig('logs.png')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[Usage] log files')
        exit()
    plot(sys.argv[1:])
