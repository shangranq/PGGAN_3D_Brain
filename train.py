from config import config

if __name__ == '__main__':
    ## perform training.
    print('----------------- configuration -----------------')
    for k, v in vars(config).items():
        print('  {}: {}'.format(k, v))
    print('-------------------------------------------------')







