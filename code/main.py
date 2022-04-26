if __name__ == '__main__':
    a = np.array(imageio.imread(sys.argv[1]),dtype=np.int)[:,:,0:3]
    graphcut = GraphCut(a,int(sys.argv[3]),int(sys.argv[4]))

    result = graphcut.patch()
    rst = result.astype('uint8')
    imageio.imwrite(sys.argv[2],rst)