import yt
import numpy as np
import argparse
import os
import time
from utilities import *
from py_chemistry_table import *
import itertools
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser("Arguments for sco2 turbo expo paper analysis")
    parser.add_argument("-pf", "--plotfile", type=str, dest='pltfile',
                        help="Pele plotfile to read")
    parser.add_argument("-of", "--outfile", type=str,
                        dest='outfile', help="results folder")
    parser.add_argument("--analysis", dest="analysis", type=int,
                        help="PDF shape=1, subgrid manifold=2")
    parser.add_argument("--numvol", type=int, dest='numvol')
    parser.add_argument("--planes", type=bool, dest='outPlane')
    args = parser.parse_args()

    # test if results folder exists or else create it
    if not os.path.exists(args.outfile):
        os.makedirs(args.outfile)

    if args.analysis == 1:
        fields_load = ["adv_0", "adv_1", "vfrac"]
        readplotfile(args.pltfile, args.numvol, fields_load, args.outfile)
        outputPDFs(args.outfile, args.numvol)
    elif args.analysis == 2:
        fields_load = ["rho_omega_H2O", "Y(H2O)", "Temp", "Y(OH)",
                       "Y(CO)", "vfrac", "adv_0", "adv_1"]
        readplotfile(args.pltfile, args.numvol, fields_load,
                     args.outfile, args.outPlane)


def readplotfile(pltfile, numvol, fields_load, odir, outplane=False):
    ds = yt.load(pltfile)  # load amrex plot file
    max_level = ds.index.max_level
    dxmin = ds.index.get_smallest_dx()

    width = (ds.domain_right_edge.d[0]-ds.domain_left_edge.d[0])/numvol

    xcenter = np.zeros(numvol)
    planeCenter = np.zeros(numvol)
    for i in range(numvol):
        xcenter[i] = ds.domain_left_edge.d[0] + width*float(i) + width*0.5
        planeCenter[i] = 0.05 + (ds.domain_right_edge.d[0] -
                                 ds.domain_left_edge.d[0])*i/numvol

    for i, xc in enumerate(xcenter):
        low = np.array([xc - 0.5 * width, ds.domain_left_edge.d[1],
                        ds.domain_left_edge.d[2]])
        high = np.array([xc + 0.5 * width, ds.domain_right_edge.d[1],
                         ds.domain_right_edge.d[2]])

        dims = (high - low) / dxmin

        dice = ds.covering_grid(
            max_level, left_edge=low, dims=dims.astype(int),
            fields=fields_load)

        fname = os.path.join(odir, "dice_{0:04d}".format(i))
        fields_to_save = dict(zip(fields_load,
                                  [dice[field].d for field in fields_load]))
        np.savez_compressed(
            fname, x=xc, dx=dxmin, low=low, high=high, **fields_to_save)

        if outplane is True:

            low = np.array([planeCenter[i], ds.domain_left_edge.d[1],
                            ds.domain_left_edge.d[2]])
            high = np.array([planeCenter[i], ds.domain_right_edge.d[1],
                             ds.domain_right_edge.d[2]])

            dims = (high - low) / dxmin

            dice = ds.covering_grid(
                max_level, left_edge=low, dims=dims.astype(int),
                fields=fields_load)

            fname = os.path.join(odir, "xplane_{0:04d}".format(i))
            fields_to_save = dict(zip(fields_load,
                                      [dice[field].d for field in fields_load]))
            np.savez_compressed(
                fname, x=planeCenter[i], dx=dxmin,
                low=low, high=high, **fields_to_save)

    low = np.array([ds.domain_left_edge.d[0], ds.domain_left_edge.d[1],
                    0.0])
    high = np.array([ds.domain_right_edge.d[0], ds.domain_right_edge.d[1],
                     0.0])

    dims = (high - low) / dxmin

    dice = ds.covering_grid(
        max_level, left_edge=low, dims=dims.astype(int),
        fields=fields_load)

    fname = os.path.join(odir, "zplane")
    fields_to_save = dict(zip(fields_load,
                              [dice[field].d for field in fields_load]))
    np.savez_compressed(
        fname, x=planeCenter[i], dx=dxmin,
        low=low, high=high, **fields_to_save)


def outputPDFs(odir, numvol):
    for i in range(numvol):
        fname = os.path.join(odir, "dice_{0:04d}".format(i))
        Anpz = np.load(fname + ".npz")

        Z1 = Anpz["adv_0"][Anpz["vfrac"] > 0.99]
        Z2 = Anpz["adv_1"][Anpz["vfrac"] > 0.99]

        binsA, binsB, dzAB, dz1D = genNonUniformBins(64, 64, 0.1, True)

        MeanA = np.mean(Z1)
        VarA = np.mean((Z1 - MeanA)**2)
        MeanB = np.mean(Z2)
        VarB = np.mean((Z2 - MeanB)**2)
        CovAB = np.mean((Z1 - MeanA)*(Z2 - MeanB))

        pdf, _, _ = np.histogram2d(np.ravel(Z1),
                                   np.ravel(Z2),
                                   bins=[binsA, binsB],
                                   density=True)
        PDF = np.reshape(pdf.flatten(order="F"), (len(binsA)-1,
                                                  len(binsB)-1), order="F")
        PDF = np.multiply(PDF, dzAB)

        CM23pdf = CM23(MeanA, MeanB, VarA, VarB, binsA, dz1D)

        outname = os.path.join(odir, "PDF_{0:04d}".format(i))
        np.savez_compressed(outname + ".npz", PDF=PDF, bins1=binsA, bins2=binsB, dz=dzAB, cm23=CM23pdf)


def srcProgApriori(outdir, ctable, numvol,
                   test_xplane=False, test_zplane=False):
    tempFunc, ohFunc, srcprogFunc = readChemInterpolate(ctable)

    for i in range(numvol):
        fname = os.path.join(outdir, "dice_{0:04d}.npz".format(i))
        dat = np.load(fname)
        zmix2 = np.clip(dat['adv_1'], 0.0, 1.0)
        wmix = np.zeros_like(zmix2)
        zmix = np.clip(dat['adv_0'], 0.0, 1.0)
        ni, nj, nk = zmix2.shape
        for i, j, k in itertools.product(range(ni), range(nj), range(nk)):
            wmix[i, j, k] = (zmix2[i, j, k]+1e-14)/(1.0-zmix[i, j, k]+1e-10)

        prog = np.clip(dat['Y(H2O)'], 0.0, 1.0)
        zvar = np.zeros_like(prog)

        srcDNS = dat['rho_omega_H2O']
        ohDNS = dat['Y(OH)']
        tempDNS = dat['Temp']

        vfrac = dat['vfrac']

        temptab = np.zeros_like(tempDNS)
        ohtab = np.zeros_like(ohDNS)
        srctab = np.zeros_like(srcDNS)

        input = np.vstack((prog.flatten(order='C'), zvar.flatten(order='C'),
                           zmix.flatten(order='C'), wmix.flatten(order='C')))

        tmp = tempFunc(input.T)
        temptab = tmp.reshape(tempDNS.shape, order='C')

        tmp = ohFunc(input.T)
        ohtab = tmp.reshape(ohDNS.shape, order='C')

        tmp = srcprogFunc(input.T)
        srctab = tmp.reshape(srcDNS.shape, order='C')

        outname = os.path.join(outdir, "chem_comp_{0:04d}".format(i))
        np.savez_compressed(outname + ".npz", srcDNS=srcDNS, ohDNS=ohDNS,
                            tempDNS=tempDNS, temptab=temptab, ohtab=ohtab,
                            srctab=srctab, vfrac=vfrac)

        dns_fl = tempDNS[vfrac>0.99]
        tab_fl = temptab[vfrac>0.99]

        plt.figure()
        plt.scatter(dns_fl[::100], tab_fl[::100])
        plt.show()
        plt.savefig("temperature.png")



# if __name__ == "__main__":
#     start = time.time()
#     main()
#     end = time.time()
#     print("Time taken: {0}", end - start)
