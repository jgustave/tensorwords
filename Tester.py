import numpy as np
import os
import psutil


def foo(sval):
    print(sval)
    sval = [4,5,6]


def main():
    ss =[1,2,3]
    foo(ss)
    print(ss)
    pass



if __name__ == '__main__':
    main()
