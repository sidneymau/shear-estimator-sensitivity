import mcObjects
import sys


def generate():
    # DESCRIPTION OF WHAT'S BEING GENERATED
    pass

def plot():
    # DESCRIPTION OF WHAT'S BEING PLOTTED
    pass


def main():
    args = sys.argv[1:]

    if len(args) != 1:
        raise Exception('-generate or -plot')
    
    if args[0] == '-generate':
        generate()

    if args[0] == '-plot':
        plot()




if __name__ == '__main__':
    main()


