import yt
import numpy as np
import argparse
import os
import time
import math
from scipy import optimize
from scipy.stats import dirichlet
import struct
from scipy.interpolate import RegularGridInterpolator
import itertools


def dirchletPDF(Z1mean, Z2mean, Z1var, nbins):
    Z1 = np.linspace(0.0, 1.0, nbins)
    Z2 = np.linspace(0.0, 1.0, nbins)

    rhs = np.array([Z1mean, Z2mean, Z1var])
    a = optimize.root(fun, [Z1mean, Z2mean, Z1var], args=(rhs,),
                      jac=None, method='broyden1')

    print(a.x)
    PDF = np.zeros((nbins, nbins))
    for i in range(nbins):
        for j in range(nbins):
            amid = 0.5*Z1[i]+0.5*Z1[i+1]
            bmid = 0.5*Z2[j]+0.5*Z2[j+1]
            PDF[i, j] = dirichlet.pdf(np.array([amid, bmid, 1.0-amid-bmid]),
                                      np.array(a.x))

    return PDF


def fun(x, rhs):
    sumAll = x[0]+x[1]+x[2]
    return [x[0]*(1.0 - rhs[0])-rhs[0]*x[1]-rhs[0]*x[2],
            x[1]*(1.0 - rhs[1])-rhs[1]*x[0]-rhs[1]*x[2],
            x[0]*(x[1]+x[2]) - rhs[2]*(sumAll*sumAll*(sumAll+1.0))]


def jac(x, rhs):
    sumAll = x[0]+x[1]+x[2]
    df1da1 = 1.0-rhs[0]
    df1da2 = -rhs[0]
    df1da3 = -rhs[0]
    df2da1 = -rhs[1]
    df2da2 = 1.0-rhs[1]
    df2da3 = -rhs[1]
    df3da1 = ((x[1]+x[2]) -
              2.0*rhs[2]*sumAll*(sumAll+1) -
              rhs[2]*sumAll*sumAll)

    df3da2 = (x[0] -
              2.0*rhs[2]*sumAll*(sumAll+1) -
              rhs[2]*sumAll*sumAll)

    df3da3 = (x[0] -
              2.0*rhs[2]*sumAll*(sumAll+1) -
              rhs[2]*sumAll*sumAll)

    return np.array([[df1da1, df1da2, df1da3],
                     [df2da1, df2da2, df2da3],
                     [df3da1, df3da2, df3da3]])


def CM23(Z1m, Z2m, Z1var, Z2var, Z1cents, Z2cents, Z1width, 
         Z2width, center_move_frac=0.5, eps=1e-10):
    # get meshes
    # Z1cents, Z2cents = np.meshgrid(meshZ1, meshZ1)
    # Z1width, Z2width = np.meshgrid(dZ1, dZ1)

    # move cell centers on Z1=0 boundary away from boundary
    Z1cents[:,0] += Z1width[:,0] * center_move_frac
    # move cell centers on Z2=0 boundary away from boundary
    Z2cents[0,:] += Z2width[0,:] * center_move_frac
    # move cell centers on diagonal boundary away from boundary
    diagpoints = np.diag_indices_from(Z1cents)
    diagpoints = (diagpoints[0], diagpoints[1][::-1])
    Z1cents[diagpoints] = Z1cents[diagpoints] - Z1width[diagpoints] * center_move_frac * 0.5
    Z2cents[diagpoints] = Z2cents[diagpoints] - Z2width[diagpoints] * center_move_frac * 0.5
    # move cell centers at (0,1) and (1,0) away from the boundary
    Z1cents[-1,0] = 0.5 * Z1width[-1,0] * center_move_frac 
    Z2cents[-1,0] = 1.0 - Z2width[-1,0] * center_move_frac
    Z1cents[0,-1] = 1.0 - Z1width[0,-1] * center_move_frac
    Z2cents[0,-1] = 0.5 * Z2width[0,-1] * center_move_frac    

    # Compute parameters of CM distribution from moments    
    K = 1 - Z1m + Z1var/(1-Z1m)
    a0 = Z1m * (Z1m*(1-Z1m)/Z1var - 1) #a1
    print(K, a0, Z1var/(1-Z1m), (1-Z1m)/Z2m*(Z2m+Z2var/Z2m))
    a1 = (K - Z2m - Z2var/Z2m) / ( (1-Z1m)/Z2m*(Z2m+Z2var/Z2m) - K) #b1
    a2 = a1 * ( (1-Z1m)/Z2m -1) #a1
    a3 = a0 * (1-Z1m) / Z1m #b2
    logC = (math.lgamma(a0 + a3) - math.lgamma(a0) - math.lgamma(a3) ) + \
                             (math.lgamma(a1 + a2) - math.lgamma(a1) - math.lgamma(a2) )  
    a3 = a3 - a1 - a2 + 1.0   

    # Calculate one value from CM
    # value = math.exp( min(logC + (a0-1)*math.log(x  ) + (a1-1)*math.log(y  ) + (a2-1)*math.log(1-x-y) \
    #                               + (a3-1)*math.log(1-x), 500) )

    # Calculate the Raw PDF
    # This does twice as much work as necessary but should vectorize better than doing only the triangle
    PDF = np.exp( logC
                  + (a0-1)*np.log(Z1cents+eps)
                  + (a1-1)*np.log(Z2cents+eps)
                  + (a2-1)*np.log(1-Z1cents-Z2cents+eps)
                  + (a3-1)*np.log(1-Z1cents+eps) )

    # Multiply PDF by the cell area (P*dZ1*dZ2), except on diagonal boundary where there is an additional factor of 0.5
    PDF = PDF * Z1width * Z2width
    PDF[diagpoints] = PDF[diagpoints] * 0.5

    # Replace nans with 0s
    PDF[np.where(np.isnan(PDF))] = 0
    
    # Rescale to sum to unity
    scale_fact = 1/np.sum(PDF)
    PDF = PDF * scale_fact
    print( 'Rescaled PDF by factor of ' + str(scale_fact) + ' to ensure sum is 1' )
    
    return PDF


def readplotfile(pltfile, numvol, fields_load, odir):
    ds = yt.load(pltfile)  # load amrex plot file
    max_level = ds.index.max_level
    dxmin = ds.index.get_smallest_dx()

    width = (ds.domain_right_edge.d[0]-ds.domain_left_edge.d[0])/numvol

    xcenter = np.zeros(numvol)
    for i in range(numvol):
        xcenter[i] = np.min(ds.domain_left_edge.d[0] + width*i + width*0.5, 
                            ds.domain_right_edge.d[0])

    for i, xc in enumerate(xcenter):
        low = np.array([xc - 0.5 * width, ds.domain_left_edge.d[1], ds.domain_left_edge.d[2]])
        high = np.array([xc + 0.5 * width, ds.domain_right_edge.d[1], ds.domain_right_edge.d[2]])

        dims = (high - low) / dxmin

        dice = ds.covering_grid(
            max_level, left_edge=low, dims=dims.astype(int), fields=fields_load)

    fname = os.path.join(odir, "dice_{0:04d}".format(i))
    fields_to_save = dict(zip(fields_load,
                              [dice[field].d for field in fields_load]))
    np.savez_compressed(
            fname, x=xc, dx=dxmin, low=low, high=high, **fields_to_save)
    

def outputPDFs(odir, numvol):
    for i in range(numvol):
        fname = os.path.join(odir, "dice_{0:04d}".format(i))
        Anpz = np.load(fname + ".npz")

        Z1 = Anpz["adv_0"][Anpz["vfrac"] > 1e-2]
        Z2 = Anpz["adv_1"][Anpz["vfrac"] > 1e-2]

        binsA, binsB, dzAB, dz1D = genNonUniformBins(64, 64, 0.1, True)

        MeanA = np.mean(Z1)
        VarA  = np.mean((Z1 - MeanA)**2)
        MeanB = np.mean(Z2)
        VarB  = np.mean((Z2 - MeanB)**2)
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




def genNonUniformBins(nbinsA, nbinsB, Zst, nonUni=True):

    binsA = np.zeros(nbinsA)
    binsB = np.zeros(nbinsB)

    if(nonUni):
        zcutA = int(nbinsA/2)
        zcutB = int(nbinsB/2)

        dzA = Zst / float(zcutA-1)
        dzB = Zst / float(zcutB-1)

        for i in range(0, zcutA):
            binsA[i] = float(i) * dzA

        for i in range(0, zcutB):
            binsB[i] = float(i) * dzB

        m11 = float((nbinsA-1)**2-(zcutA-1)**2)
        m12 = float(nbinsA-zcutA)
        m21 = float(2*(zcutA-1)+1)
        m22 = 1.0
        r1 = 1.0 - Zst
        r2 = dzA
        delta = m11*m22-m12*m21
        a = (+ m22*r1 - m12*r2)/delta
        b = (- m21*r1 + m11*r2)/delta
        c = Zst - a*(zcutA-1)**2-b*(zcutA-1)
        for i in range(zcutA, nbinsA):
            binsA[i] = a*float(i)**2 + b*float(i) + c

        m11 = float((nbinsB-1)**2-(zcutB-1)**2)
        m12 = float(nbinsB-zcutB)
        m21 = float(2*(zcutB-1)+1)
        m22 = 1.0
        r1 = 1.0 - Zst
        r2 = dzB
        delta = m11*m22-m12*m21
        a = (+ m22*r1 - m12*r2)/delta
        b = (- m21*r1 + m11*r2)/delta
        c = Zst - a*(zcutB-1)**2-b*(zcutB-1)
        for i in range(zcutB, nbinsB):
            binsB[i] = a*float(i)**2 + b*float(i) + c
    else:
        binsA = np.linspace(0.0, 1.0, nbinsA)
        binsB = np.linspace(0.0, 1.0, nbinsB)

    Za = np.zeros(len(binsA)+1)
    Zb = np.zeros(len(binsB)+1)
    dzAB = np.zeros((nbinsA, nbinsB))

    Za[0] = binsA[0]-(0.5*binsA[1]+0.5*binsA[0]-binsA[0])
    Zb[0] = binsB[0]-(0.5*binsB[1]+0.5*binsB[0]-binsB[0])

    Za[-1] = binsA[-1]+(binsA[-1]-0.5*binsA[len(binsA)-1]
                        - 0.5*binsA[len(binsA)-2])
    Zb[-1] = binsB[-1]+(binsB[-1]-0.5*binsB[len(binsB)-1]
                        - 0.5*binsB[len(binsB)-2])

    for j in range(1, len(binsA)):
        Za[j] = 0.5*binsA[j]+0.5*binsA[j-1]

    for j in range(1, len(binsB)):
        Zb[j] = 0.5*binsB[j]+0.5*binsB[j-1]

    for i in range(len(binsA)):
        for j in range(len(binsB)):
            dzAB[i, j] = (Za[i+1]-Za[i])*(Zb[j+1]-Zb[j])

    dz1D = np.zeros(len(binsB))
    for j in range(len(binsB)):
        dz1D[j] = (Zb[j+1]-Zb[j])

    return Za, Zb, dzAB, dz1D
    
        
def getBinCenter(nbinsA, nbinsB, Zst, nonUni=True):
    binsA = np.zeros(nbinsA)
    binsB = np.zeros(nbinsB)

    if(nonUni):
        zcutA = int(nbinsA/2)
        zcutB = int(nbinsB/2)

        dzA = Zst / float(zcutA-1)
        dzB = Zst / float(zcutB-1)

        for i in range(0, zcutA):
            binsA[i] = float(i) * dzA

        for i in range(0, zcutB):
            binsB[i] = float(i) * dzB

        m11 = float((nbinsA-1)**2-(zcutA-1)**2)
        m12 = float(nbinsA-zcutA)
        m21 = float(2*(zcutA-1)+1)
        m22 = 1.0
        r1 = 1.0 - Zst
        r2 = dzA
        delta = m11*m22-m12*m21
        a = (+ m22*r1 - m12*r2)/delta
        b = (- m21*r1 + m11*r2)/delta
        c = Zst - a*(zcutA-1)**2-b*(zcutA-1)
        for i in range(zcutA, nbinsA):
            binsA[i] = a*float(i)**2 + b*float(i) + c

        m11 = float((nbinsB-1)**2-(zcutB-1)**2)
        m12 = float(nbinsB-zcutB)
        m21 = float(2*(zcutB-1)+1)
        m22 = 1.0
        r1 = 1.0 - Zst
        r2 = dzB
        delta = m11*m22-m12*m21
        a = (+ m22*r1 - m12*r2)/delta
        b = (- m21*r1 + m11*r2)/delta
        c = Zst - a*(zcutB-1)**2-b*(zcutB-1)
        for i in range(zcutB, nbinsB):
            binsB[i] = a*float(i)**2 + b*float(i) + c
    else:
        binsA = np.linspace(0.0, 1.0, nbinsA)
        binsB = np.linspace(0.0, 1.0, nbinsB)

    Z1Z1 = np.zeros((nbinsA, nbinsB))
    Z2Z2 = np.zeros((nbinsA, nbinsB))
    Z1Z2 = np.zeros((nbinsA, nbinsB))
    Z1M = np.zeros((nbinsA, nbinsB))
    Z2M = np.zeros((nbinsA, nbinsB))

    for i in range(nbinsA):
        for j in range(nbinsB):
            Z1M[i, j] = binsA[i]
            Z2M[i, j] = binsB[j]
            Z1Z1[i, j] = binsA[i]*binsA[i]
            Z2Z2[i, j] = binsB[j]*binsB[j]
            Z1Z2[i, j] = binsA[i]*binsB[j]

    return Z1M, Z2M, Z1Z1, Z2Z2, Z1Z2


def readChemInterpolate(chemtable, convertUnits=True):
    fileChemtable = open(chemtable, "rb")
    nprog = struct.unpack('i', fileChemtable.read(4))[0]
    nzvar = struct.unpack('i', fileChemtable.read(4))[0]
    nzmix = struct.unpack('i', fileChemtable.read(4))[0]
    nwmix = struct.unpack('i', fileChemtable.read(4))[0]
    nvar = struct.unpack('i', fileChemtable.read(4))[0]
    prog = np.array(struct.unpack(str(nprog)+'d',
                                  fileChemtable.read(nprog*8)))
    zvar = np.array(struct.unpack(str(nzvar)+'d',
                                  fileChemtable.read(nzvar*8)))
    zmix = np.array(struct.unpack(str(nzmix)+'d',
                                  fileChemtable.read(nzmix*8)))
    wmix = np.array(struct.unpack(str(nwmix)+'d',
                                  fileChemtable.read(nwmix*8)))

    combModel = struct.unpack(
        '64s', fileChemtable.read(64))[0].decode("utf-8").strip()

    varName = []
    for i in range(nvar):
        var = struct.unpack('64s',
                            fileChemtable.read(64))[0].decode("utf-8").strip()
        varName.append(var)

    for i in range(nvar):
        tmp = np.array(struct.unpack(str(nprog*nzvar*nzmix*nwmix)+'d',
                                     fileChemtable.read(nprog*nzvar*nzmix*nwmix*8)))
        if (varName[i] == 'T'):
            temp = np.reshape(tmp, (nprog, nzvar, nzmix, nwmix), order='F')
        elif (varName[i] == 'OH'):
            oh = np.reshape(tmp, (nprog, nzvar, nzmix, nwmix), order='F')
        elif (varName[i] == 'SRC_PROG'):
            srcprog = np.reshape(tmp, (nprog, nzvar, nzmix, nwmix), order='F')
            if convertUnits is True:
                srcprog = 1e-3*srcprog

    fileChemtable.close()

    print(np.max(temp), np.min(temp))

    print(prog.shape, zvar.shape, zmix.shape, wmix.shape, temp.shape)

    temp_function = RegularGridInterpolator((prog, zvar, zmix, wmix), temp,
                                            bounds_error=False, fill_value=None)
    oh_function = RegularGridInterpolator((prog, zvar, zmix, wmix), oh,
                                            bounds_error=False, fill_value=None)
    srcprog_function = RegularGridInterpolator((prog, zvar, zmix, wmix),
                                               srcprog,
                                            bounds_error=False, fill_value=None)

    return temp_function, oh_function, srcprog_function
