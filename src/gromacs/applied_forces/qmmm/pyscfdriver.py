import unittest
import numpy
import pyscf
from pyscf import lib, gto, scf, grad, dft
from pyscf.qmmm import itrf
import csv

def QMcalculation(pPDBname, pQMBasis, pQMMult, pQMCharge):

    # pPDBname = "_cp2k.pdb"
    # pQMBasis = "ccpvdz"
    # pQMMult = 1
    # pQMCharge = 0
    qmatoms = []
    qmcoords = []
    qmkinds = []
    mmcoords = []
    mmcharges = []
    with open(pPDBname, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=" ", skipinitialspace=True)
        for line in csv_reader:
            if int(line[4]) == 1:
                qmatoms.append([line[-2], (float(line[5]), float(line[6]), float(line[7]))])
                qmkinds.append(line[-2])
                qmcoords.append((float(line[5]), float(line[6]), float(line[7])))
            else:
                mmcoords.append((float(line[5]), float(line[6]), float(line[7])))
                mmcharges.append(float(line[-1]))
        print(qmcoords)
        print(qmkinds)
        print(qmatoms)
        print(mmcoords)
        print(mmcharges)

    mol = gto.M(atom=qmatoms, unit='ANG', basis=pQMBasis)
    mf_qm = dft.RKS(mol)
    mf_qm.xc = 'PBE'
    mf = itrf.mm_charge(mf_qm, mmcoords, mmcharges)

    energy = mf.kernel()

    dm = mf.make_rdm1()
    print("density matrix", dm, type(dm))

    j3c = mol.intor('int1e_grids', hermi=1, grids=mmcoords[0])
    # hcore_mm = numpy.einsum('kpq,k->pq', j3c, -mmcharges[0])

    # for i0, i1 in lib.prange(1, mmcharges.size, 1):
    #     j3c = mol.intor('int1e_grids', hermi=1, grids=mmcoords[i0:i1])
    #     hcore_mm += numpy.einsum('kpq,k->pq', j3c, -mmcharges[i0:i1])

    # energy_qme_mm = numpy.einsum('pq,qp->',hcore_mm,dm)

    energy_qmnuc_mm = mf.energy_nuc()

    print("energy = ", energy, type(energy))
    # print("h1e_mm_energy = ", energy_qme_mm, type(energy_qme_mm))
    print("qmnuc_mm_energy = ", energy_qmnuc_mm, type(energy_qmnuc_mm))
    # print("qm_mm_energy = ", energy_qm_mm, type(energy_qm_mm))

    mf_grad = itrf.mm_charge_grad(grad.RKS(mf), mmcoords, mmcharges)
    qmforce = -mf_grad.kernel()
    print("qmforce", type(qmforce))
    print(qmforce)

    mmforce_qmnuc = -mf_grad.grad_nuc_mm()
    mmforce_qme = -mf_grad.grad_hcore_mm(dm)

    print("mmforce_qmnuc", type(mmforce_qmnuc))
    print(mmforce_qmnuc)
    print("mmforce_e", type(mmforce_qme))
    print(mmforce_qme)

    mmforce = numpy.add(mmforce_qmnuc,mmforce_qme)
    print("mmforce", type(mmforce))
    print(mmforce)

    return energy, qmforce, mmforce


# QMcalculation("_cp2k.pdb", "ccpvdz", 1, 0)