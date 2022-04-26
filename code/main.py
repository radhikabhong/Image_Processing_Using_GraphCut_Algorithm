import numpy as np
import imageio
from copy import deepcopy
import sys
import networkx as nx
import random
from graphcut import GraphCut

if __name__ == '__main__':
    a = np.array(imageio.imread(sys.argv[1]),dtype=np.int32)[:,:,0:3]
    graphcut = GraphCut(a,int(sys.argv[3]),int(sys.argv[4]))

    result = graphcut.patch()
    rst = result.astype('uint8')
    imageio.imwrite(sys.argv[2],rst)