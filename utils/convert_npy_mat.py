import scipy.io
import numpy as np
import argparse

def main():
    # get path from first command line arg
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to npy file")
    args = parser.parse_args()
    path = args.path

    # load npy file
    data = np.load(path)

    order = [0,1,2]
    ViewId = [i for i in range(len(data))]
    Location = [i[order,3] for i in data]
    Rotation = [i[0:3,order] for i in data]

    # create dictionary
    data = {'ViewId': ViewId, 'Location': Location, 'Orientation': Rotation}


    # write as mat file
    scipy.io.savemat(path[:-4] + ".mat", mdict={'data': data})


if __name__ == "__main__":
    main()