import unittest
import numpy

# import pyscf
import time, copy
from pyscf import lib, gto, scf, grad, dft, neo
from pyscf.qmmm import itrf
from pyscf.data import elements, radii, nist
from pyscf.qmmm.mm_mole import create_mm_mol

# DEBUG = True
SYSTEM_CHARGE = -1
QM_CHARGE = -1
QM_MULT = 1
QM_NUC_BASIS = "pb4d"
QM_E_BASIS = "631g"
QM_E_BASIS_AUX = "aug-cc-pvdz-ri"
QM_METHOD = "cneo"
DFT_DF_FIT = False  # need to enable for each qmmmCalc and qmCalc function
DFT_ELE_XC = "BLYP"
LINK_MMHOST_NEIGHBOR_RANGE = 1.7
MM_CHARGE_MODEL = "point"

LINK_CHARGE_CORR_METHOD = "dist"
LINK_COORD_CORR_METHOD = "scale"
LINK_CORR_SCALE = 1.38
LINK_COORD_LINKCH = 1.09
LINK_PRINT_COORD_CORR = True
LINK_PRINT_FORCE_CORR = False
LINK_PRINT_CHARGE_CORR = True
LINK_PRINT_MMHOST_NEIGHBOR = False
QMMM_PRINT = True


def qmmmCalc(
    qmbasis,
    qmmult,
    qmcharge,
    qmindex,
    qmkinds,
    qmcoords,
    mmindex,
    mmkinds,
    mmcharges,
    mmcoords,
    links,
):

    t0 = time.time()

    [qmkinds_link, qmcoords_link, qmindex_link] = link_coord_corr(
        links, qmindex, qmkinds, qmcoords, mmindex, mmkinds, mmcoords
    )

    qmnucindex = []  # marks the
    qmatoms = []
    qmlocalindex = 0
    for kind, coord in zip(qmkinds_link, qmcoords_link):
        qmatom = [kind] + coord
        qmatoms.append(qmatom)
        if kind.strip().upper() == "H":
            qmnucindex.append(qmlocalindex)
        qmlocalindex = qmlocalindex + 1

    mmcharges = link_charge_corr(mmcharges, mmindex, links)

    mmradii = []
    for kind in mmkinds:
        charge = elements.charge(kind)
        if MM_CHARGE_MODEL.lower()[0:5] == "gauss":
            mmradii.append(radii.COVALENT[charge] * nist.BOHR)
        if MM_CHARGE_MODEL.lower() == "point":
            mmradii.append(1e-8 * nist.BOHR)

    numlink = len(links)
    if QMMM_PRINT == True:
        print("qm nuc index:", qmnucindex[:-numlink])
        print("mmcharges\n", mmcharges)
        print("mmcoords\n", mmcoords)
        print("qmatomds\n", qmatoms)
        print(f"method we used in this calc is {QM_METHOD}")
        print(mmradii)

    if QM_METHOD.upper() == "CNEO":
        [energy, qmforces, mmforces] = qmmmCalc_cneo(
            qmatoms, mmcoords, mmcharges, mmradii, qmnucindex[:-numlink]
        )
    elif QM_METHOD.upper() == "DFT":
        [energy, qmforces, mmforces] = qmmmCalc_dft(
            qmatoms, mmcoords, mmcharges, mmradii
        )

    [qmindex_force_corr, qmforces_link, mmforces_link] = link_force_corr(
        links, qmindex_link, qmkinds_link, qmforces, mmindex, mmkinds, mmforces
    )

    print(f"time for this step = {time.time() - t0} seconds")

    return energy, qmforces_link, mmforces_link


def qmCalc(qmbasis, qmmult, qmcharge, qmkinds, qmcoords):

    t0 = time.time()
    qmatoms = []
    for kind, coord in zip(qmkinds, qmcoords):
        qmatom = [kind] + coord
        qmatoms.append(qmatom)
    if QM_METHOD.upper() == "CNEO":
        [energy, qmforces] = qmCalc_cneo(qmatoms)
    elif QM_METHOD.upper() == "DFT":
        [energy, qmforces] = qmCalc_dft(qmatoms)

    print(f"time for this step = {time.time() - t0} seconds")

    return energy, qmforces


def qmCalc_cneo(qmatoms):
    mol = neo.M(
        atom=qmatoms,
        basis=QM_E_BASIS,
        nuc_basis=QM_NUC_BASIS,
        quantum_nuc=["H"],
        charge=QM_CHARGE,
    )
    mf = neo.CDFT(mol, df_ee=True, auxbasis_e=QM_NUC_BASIS)
    mf.mf_elec.xc = DFT_ELE_XC

    energy = mf.kernel()

    g = mf.Gradients()
    g_qm = g.grad()
    qmforce = -g_qm
    return energy, qmforce


def qmCalc_dft(qmatoms):
    mol = gto.M(atom=qmatoms, unit="ANG", basis=QM_E_BASIS, charge=QM_CHARGE)
    mf = dft.RKS(mol)
    mf.xc = DFT_ELE_XC
    mf = mf.density_fit(auxbasis=QM_E_BASIS_AUX)

    energy = mf.kernel()
    # dm = mf.make_rdm1()

    qmgrad = mf.nuc_grad_method().kernel()
    # print(type(qmgrad), qmgrad)
    qmforce = -qmgrad
    return energy, qmforce


def qmmmCalc_cneo(qmatoms, mmcoords, mmcharges, mmradii, qmnucindex):

    t0 = time.time()

    mmmol = create_mm_mol(mmcoords, mmcharges, mmradii)
    mol_neo = neo.M(
        atom=qmatoms,
        basis=QM_E_BASIS,
        nuc_basis=QM_NUC_BASIS,
        quantum_nuc=qmnucindex,
        mm_mol=mmmol,
        charge=QM_CHARGE,
    )
    # energy
    print("mol_neo quantum_nuc", mol_neo.quantum_nuc)
    print(qmatoms)
    print(qmnucindex)
    mf = neo.CDFT(mol_neo, df_ee=True, auxbasis_e=QM_E_BASIS_AUX)
    mf.mf_elec.xc = DFT_ELE_XC
    energy = mf.kernel()
    te = time.time()
    print(f"time for energy = {te - t0} seconds")

    # qm gradient
    g = mf.Gradients()
    g_qm = g.grad()
    qmforces = -g_qm
    tqf = time.time()
    print(f"time for qm force = {tqf - te} seconds")

    # mm gradient
    g_mm = g.grad_mm()
    mmforces = -g_mm
    tmf = time.time()
    print(f"time for mm force = {tmf - tqf} seconds")

    return energy, qmforces, mmforces


def qmmmCalc_dft(qmatoms, mmcoords, mmcharges, mmradii):
    t0 = time.time()

    mol = gto.M(atom=qmatoms, unit="ANG", basis=QM_E_BASIS, charge=QM_CHARGE)
    mf_dft = dft.RKS(mol)
    mf_dft.xc = DFT_ELE_XC

    if DFT_DF_FIT == True:
        mf_dft = mf_dft.density_fit(auxbasis=QM_E_BASIS_AUX)
    mf = itrf.mm_charge(mf_dft, mmcoords, mmcharges, mmradii)

    energy = mf.kernel()
    te = time.time()
    print(f"time for energy = {te - t0} seconds")

    dm = mf.make_rdm1()
    mf_grad = itrf.mm_charge_grad(grad.RKS(mf), mmcoords, mmcharges, mmradii)
    qmforces = -mf_grad.kernel()
    tqf = time.time()
    print(f"time for qm force = {tqf - te} seconds")

    mmforces_qmnuc = -mf_grad.grad_nuc_mm()
    mmforces_qme = -mf_grad.grad_hcore_mm(dm)
    mmforces = mmforces_qmnuc + mmforces_qme
    tmf = time.time()
    print(f"time for mm force = {tmf - tqf} seconds")

    print(f"time for this step = {time.time() - t0} seconds")

    return energy, qmforces, mmforces


def printProp(prop):
    print(prop)


def prop_print_xzy(xyzpropname, index, kinds, xyzprops):
    print("--------------------", xyzpropname, "--------------------")
    for i in range(len(xyzprops)):
        print(
            "%4s %4s %18.12f %18.12f %18.12f"
            % (index[i], kinds[i], xyzprops[i][0], xyzprops[i][1], xyzprops[i][2])
        )


def link_coord_corr(links, qmindex, qmkinds, qmcoords, mmindex, mmkinds, mmcoords):

    qmindex_link = qmindex[:]
    qmkinds_link = qmkinds[:]
    qmcoords_link = qmcoords[:]

    if LINK_PRINT_COORD_CORR == True:
        print("qm index", qmindex)
        print("mm index", mmindex)
        print("there are", len(links), "links to be processed: ", links)

    for i in range((len(links))):
        link_coord = [0.000, 0.000, 0.000]
        qmindex_link.append("L" + str(i))
        qm_c_index = qmindex.index(links[i][0])
        mm_c_index = mmindex.index(links[i][1])
        qm_host_coord = numpy.array(qmcoords[qm_c_index])
        mm_host_coord = numpy.array(mmcoords[mm_c_index])
        if LINK_COORD_CORR_METHOD.lower() == "scale":
            alpha = LINK_CORR_SCALE
            # need to enable user defined alpha(scaling factor) later
            alpha_rec = 1 / alpha
            link_coord = alpha_rec * mm_host_coord + (1 - alpha_rec) * qm_host_coord
        if LINK_COORD_CORR_METHOD.lower() == "flat":
            r_mm_qm = mm_host_coord - qm_host_coord
            d_mm_qm = numpy.linalg.norm(r_mm_qm)
            link_coord = qm_host_coord + LINK_COORD_LINKCH * r_mm_qm / d_mm_qm
        link_coord = list(link_coord)
        qmkinds_link.append("H  ")
        # need to enable user defined link atom kind later
        qmcoords_link.append(link_coord)
        if LINK_PRINT_COORD_CORR == True:
            print(i + 1, "th link is between [QM, MM] =", links[i])
            print("link" "s qm host in-group index", qmindex.index(links[i][0]))
            print("link" "s mm host in-group index", mmindex.index(links[i][1]))
            print(
                f"qm host atom is {qmkinds[qm_c_index]}, "
                f"coordinate: {qm_host_coord}"
            )
            print(
                f"mm host atom is {mmkinds[mm_c_index]}, "
                f"coordinate: {mm_host_coord}"
            )
            print(f"link atom is H , coordinate: {link_coord}")

    if LINK_PRINT_COORD_CORR == True:
        prop_print_xzy("qm coords original", qmindex, qmkinds, qmcoords)
        prop_print_xzy("qm coords modified", qmindex_link, qmkinds_link, qmcoords_link)
        prop_print_xzy("mm coords", mmindex, mmkinds, mmcoords)

    return qmkinds_link, qmcoords_link, qmindex_link


def link_force_corr(links, qmindex, qmkinds, qmforces, mmindex, mmkinds, mmforces):
    if LINK_PRINT_FORCE_CORR == True:
        prop_print_xzy("qm forces origin", qmindex, qmkinds, qmforces)
        prop_print_xzy("mm forces origin", mmindex, mmkinds, mmforces)

    linkcount = 0
    for i in range((len(links))):
        linkcount = linkcount + 1
        linkindex = "L" + str(i)

        qm_c_index = qmindex.index(links[i][0])
        mm_c_index = mmindex.index(links[i][1])
        link_c_index = qmindex.index(linkindex)
        if LINK_COORD_CORR_METHOD.lower() == "scale":
            alpha = LINK_CORR_SCALE
            alpha_rec = 1 / alpha
            link_force = qmforces[link_c_index]
            qm_link_force_partition = link_force * (1 - alpha_rec)
            mm_link_force_partition = link_force * alpha_rec
            qmforces[qm_c_index] += qm_link_force_partition
            mmforces[mm_c_index] += mm_link_force_partition
        # if LINK_COORD_CORR_METHOD.lower() == "scale":
        #     # empty()

        if LINK_PRINT_FORCE_CORR == True:
            print(linkcount, "th link is between [QM, MM] =", links[i])
            print("link" "s qm host in-group index", qm_c_index)
            print("link" "s mm host in-group index", mm_c_index)
            print("link" "s in-group index", link_c_index)
            print(
                "link atom: kind",
                qmkinds[link_c_index],
                ", force: ",
                link_force,
                ", paritions: qm:",
                qm_link_force_partition,
                ", mm:",
                mm_link_force_partition,
            )

    for i in range((len(links))):
        linkindex = "L" + str(i)
        link_c_index = qmindex.index(linkindex)
        qmindex.remove(linkindex)
        qmforces = numpy.delete(qmforces, link_c_index, 0)

    if LINK_PRINT_FORCE_CORR == True:
        prop_print_xzy("qm forces corrected", qmindex, qmkinds, qmforces)
        prop_print_xzy("mm forces corrected", mmindex, mmkinds, mmforces)

    return qmindex, qmforces, mmforces


def neighborlist_gen(hostcoord, coords, index, bondthreshold=1.7, mode="radius"):

    neighbor_index = []

    if len(coords) != len(index):
        raise Exception("neighbor coords and index do not match")
    if len(coords) == 0:
        raise Exception("there is no neighbor to search for host coordinate")
    if mode.lower()[0:3] == "rad":
        for coord in coords:
            dist = numpy.linalg.norm(numpy.array(coord) - numpy.array(hostcoord))
            if dist < bondthreshold and dist > 0.1:
                index[coords.index(coord)]
                neighbor_index.append(index[coords.index(coord)])

    if mode.lower()[0:4] == "near":
        nearest_index = index[0]
        nearest_coord = coord[nearest_index]
        nearest_dist = numpy.linalg.norm(
            numpy.array(nearest_coord) - numpy.array(hostcoord)
        )
        for i in range(len(coords)):
            dist = numpy.linalg.norm(numpy.array(coords[i]) - numpy.array(hostcoord))
            if dist < nearest_dist and dist > 0.1:
                nearest_index = index[i]
                nearest_coord = coords[i]
                dist0 = dist
        neighbor_index = nearest_index

    return neighbor_index


def link_charge_corr(mmcharges, mmindex, links, spread="all"):

    mmcharges_redist = copy.deepcopy(mmcharges)

    mmhostindex_global = [link[1] for link in links]
    mmhostindex_group = [mmindex.index(i) for i in mmhostindex_global]
    mmhostcharges = [mmcharges[i] for i in mmhostindex_group]

    if LINK_PRINT_CHARGE_CORR == True:
        print("mmhost global index", mmhostindex_global)
        print("mmhost in-group index", mmhostindex_group)
        print("mmhost charges", mmhostcharges)

    charge_total_qm_classical = SYSTEM_CHARGE - sum(mmcharges)
    # total_qm_classical_charge + total_mm_classical_charge = system_charge (interger)
    charge_total_mm_host = sum(mmhostcharges)
    # mm hosts classical charges are also spread out to the rest of mm atoms
    charge_zeroed_out = charge_total_mm_host + charge_total_qm_classical
    #
    charge_corr_total = charge_zeroed_out - QM_CHARGE

    if (len(mmcharges) - len(mmhostcharges)) < 1:
        raise RuntimeError("there is no mm host to spread the charge correction")
    chargecorr = charge_corr_total / (len(mmcharges) - len(mmhostcharges))

    for i in range(len(mmcharges_redist)):
        charge = mmcharges_redist[i]
        if LINK_PRINT_CHARGE_CORR == True:
            print("mmcharge in-group index", i, "charge of mmhost", charge)
        if i in mmhostindex_group:
            if LINK_PRINT_CHARGE_CORR == True:
                print("mmhost in-group index", i, "charge of mmhost", charge)
            mmcharges_redist[i] = 0.000
        else:
            mmcharges_redist[i] += chargecorr

    if LINK_PRINT_CHARGE_CORR == True:
        print("total qm classical charge", charge_total_qm_classical)
        print("total mm host charge", charge_total_mm_host)
        print("total zeroed-out classical charge", charge_zeroed_out)
        print("total charge to spread over the remaining mm atoms", charge_corr_total)
        print("remaining mm atom number", (len(mmcharges) - len(mmhostcharges)))
        print("updated mm charge list", mmcharges_redist)

    return mmcharges_redist


if __name__ == "__main__":
    # qmbasis = '631G'
    # qmmult = 1
    # qmcharge = 0
    # qmkinds = ['O','H', 'H']
    # qmcoords = [[-1.464, 0.099, 0.300],
    #            [-1.956, 0.624, -0.340],
    #            [-1.797, -0.799, 0.206]]
    # mmkinds = ['O','H','H']
    # mmcharges = [-1.040, 0.520, 0.520]
    # # mmradii = [0.63, 0.32, 0.32]
    # mmcoords = [(1.369, 0.146,-0.395),
    #              (1.894, 0.486, 0.335),
    #              (0.451, 0.165,-0.083)]

    qmbasis = "631G"
    qmmult = 1

    # qmcharge = 0
    # qmindex = [4, 5, 13, 14, 15]
    # qmkinds = ['C  ', 'O  ', 'H  ', 'H  ', 'H  ']
    # qmcoords = [[46.97, 49.78, 50.5], [45.55, 49.81, 50.5], [47.33, 48.94, 51.04], [47.23, 49.73, 49.459999999999994], [45.21, 49.86, 51.8]]
    # mmindex = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12]
    # mmkinds = ['C  ', 'O  ', 'C  ', 'C  ', 'H  ', 'H  ', 'H  ', 'H  ', 'H  ', 'H  ', 'H  ']
    # mmcharges = [-0.0521, -0.3809, 0.0113, -0.2261, 0.0824, 0.0824, 0.0824, 0.0767, 0.0767, 0.1073, 0.1073]
    # mmcoords = [[51.0, 50.98, 50.010000000000005], [49.6, 51.01, 49.97], [49.059999999999995, 50.99, 51.31999999999999], [47.53, 51.0, 51.22], [51.28999999999999, 51.06, 48.96], [51.44, 50.15, 50.58], [51.33, 51.849999999999994, 50.53], [49.3, 51.900000000000006, 51.92], [49.400000000000006, 50.12, 51.849999999999994], [47.199999999999996, 51.91, 50.739999999999995], [47.04, 51.07, 52.26]]
    # links = [[4, 3]]

    qmcharge = 0
    qmindex = [0, 5, 6, 7, 8, 15]
    qmkinds = ["C  ", "O  ", "H  ", "H  ", "H  ", "H  "]
    qmcoords = [
        [51.0, 50.98, 50.010000000000005],
        [45.55, 49.81, 50.5],
        [51.28999999999999, 51.06, 48.96],
        [51.44, 50.15, 50.58],
        [51.33, 51.849999999999994, 50.53],
        [45.21, 49.86, 51.8],
    ]
    mmindex = [1, 2, 3, 4, 9, 10, 11, 12, 13, 14]
    mmkinds = ["O  ", "C  ", "C  ", "C  ", "H  ", "H  ", "H  ", "H  ", "H  ", "H  "]
    mmcharges = [
        -0.3809,
        0.0113,
        -0.2261,
        0.0124,
        0.0767,
        0.0767,
        0.1073,
        0.1073,
        0.1034,
        0.1034,
    ]
    mmcoords = [
        [49.6, 51.01, 49.97],
        [49.059999999999995, 50.99, 51.31999999999999],
        [47.53, 51.0, 51.22],
        [46.97, 49.78, 50.5],
        [49.3, 51.900000000000006, 51.92],
        [49.400000000000006, 50.12, 51.849999999999994],
        [47.199999999999996, 51.91, 50.739999999999995],
        [47.04, 51.07, 52.26],
        [47.33, 48.94, 51.04],
        [47.23, 49.73, 49.459999999999994],
    ]
    links = [[0, 1], [5, 4]]

    qmkinds_link, qmcoords_link, qmindex_link = link_coord_corr(
        links, qmindex, qmkinds, qmcoords, mmindex, mmkinds, mmcoords
    )
    atoms = []
    coords = mmcoords + qmcoords_link
    indeces = mmindex + qmindex_link
    kinds = mmkinds + qmkinds_link
    for kind, coord, index in zip(kinds, coords, indeces):
        atom = [kind] + coord + [index]
        atoms.append(atom)
    prop_print_xzy("xyz file", [" " for x in range(len(kinds))], kinds, coords)
    prop_print_xzy(
        "coord_link_flat",
        [" " for x in range(len(qmkinds_link))],
        qmkinds_link,
        qmcoords_link,
    )
    # [energy, qmforce, mmforce] = qmmmCalc(
    #     qmbasis,
    #     qmmult,
    #     qmcharge,
    #     qmindex,
    #     qmkinds,
    #     qmcoords,
    #     mmindex,
    #     mmkinds,
    #     mmcharges,
    #     mmcoords,
    #     links,
    # )

    # atoms = []
    # coords = mmcoords + qmcoords
    # indeces = mmindex + qmindex
    # kinds = mmkinds + qmkinds
    # for kind, coord, index in zip(kinds, coords, indeces):
    #     atom = [kind] + coord + [index]
    #     atoms.append(atom)

    # prop_print_xzy("xyz file", indeces, kinds, coords)
    # print(energy)
    # print(type(qmforce), qmforce)
    # print(type(mmforce), mmforce)