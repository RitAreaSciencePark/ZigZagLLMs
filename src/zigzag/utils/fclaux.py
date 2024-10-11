import numpy as np
import itertools


def get_intersection(list1,list2):
    list1=np.array(list1)
    list2=np.array(list2)
    nrows, ncols = list1.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
            'formats':ncols * [list1.dtype]}

    C = np.intersect1d(list1.view(dtype), list2.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(list1.dtype).reshape(-1, ncols)
    return C

def get_isin(list1,list2):
       list1=np.array(list1)
       list2=np.array(list2)
       nrows, ncols = list1.shape
       dtype={'names':['f{}'.format(i) for i in range(ncols)],
              'formats':ncols * [list1.dtype]}

       C = np.isin(list1.view(dtype), list2.view(dtype))

       # This last bit is optional if you're okay with "C" being a structured array...
       #C = C.view(list1.dtype).reshape(-1, ncols)
       return C

def ranges(i):
    for a, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
        b = list(b)
        yield b[0][1]+1, b[-1][1]+2