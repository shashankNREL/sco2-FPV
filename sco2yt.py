import yt
import numpy as np
import scipy
import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser("Arguments for sco2 turbo expo paper analysis")
    parser.add_argument("plotfile", type=str, dest='pltfile', 
                        help="Pele plotfile to read")
    parser.add_argument("outfile", type=str, 
                        dest='outfile', help="results folder")
    parser.add_argument("analysis", dest="analysis", type=int,
                        help="PDF shape=1, subgrid manifold=2")
    parser.add_argument("numvol", type=int, dest='numvol')
    args = parser.parse_args()

    # test if results folder exists or else create it
    if not os.path.exists(args.outfile):
        os.makedirs(args.outfile)


def dirchletPDF(z1, z2, z1var, z2var):

    return


def readplotfile(pltfile, zcenter, height, fields_load, odir):
    ds = yt.load(pltfile, unit_system="mks")  # load amrex plot file
    max_level = ds.index.max_level
    dxmin = ds.index.get_smallest_dx()
    for i, zc in enumerate(zcenter):
        low = np.array([ds.domain_left_edge.d[0], ds.domain_left_edge.d[1],
                        zc - 0.5 * height])
        high = np.array([ds.domain_right_edge.d[0], ds.domain_right_edge.d[1],
                        zc + 0.5 * height])

        dims = (high - low) / dxmin

        dice = ds.covering_grid(
            max_level, left_edge=low, dims=dims.astype(int), fields=fields_load
        )

    fname = os.path.join(odir, "dice_{0:04d}".format(i))
    fields_to_save = dict(zip(fields_load,
                              [dice[field].d for field in fields_load]))
    np.savez_compressed(
            fname, z=zc, dx=dxmin, low=low, high=high, **fields_to_save)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Time taken: {0}", end - start)
