/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2021- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * Implements force provider for QMMM
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_applied_forces
 */

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "gmxpre.h"

#include "numpy/arrayobject.h"

#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/math/units.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/filestream.h"
#include "gromacs/utility/stringutil.h"

#include "qmmmforceprovider.h"
#include "qmmminputgenerator.h"

// debug, delete later TODO
#include <cstdio>

#include <fstream>
#include <iostream>

namespace gmx
{

QMMMForceProvider::QMMMForceProvider(const QMMMParameters& parameters,
                                     const LocalAtomSet&   localQMAtomSet,
                                     const LocalAtomSet&   localMMAtomSet,
                                     PbcType               pbcType,
                                     const MDLogger&       logger) :
    parameters_(parameters),
    qmAtoms_(localQMAtomSet),
    mmAtoms_(localMMAtomSet),
    pbcType_(pbcType),
    logger_(logger),
    box_{ { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } }
{
}

QMMMForceProvider::~QMMMForceProvider()
{
    // Py_Finalize();
    if (force_env_ != -1)
    {
        // cp2k_destroy_force_env(force_env_);
        if (GMX_LIB_MPI)
        {
            // Python finalize?
            // cp2k_finalize_without_mpi();
        }
        else
        {
            // Python finalize?
            // cp2k_finalize();
        }
    }
}

bool QMMMForceProvider::isQMAtom(index globalAtomIndex)
{
    return std::find(qmAtoms_.globalIndex().begin(), qmAtoms_.globalIndex().end(), globalAtomIndex)
           != qmAtoms_.globalIndex().end();
}

void QMMMForceProvider::appendLog(const std::string& msg)
{
    GMX_LOG(logger_.info).asParagraph().appendText(msg);
}

void QMMMForceProvider::initPython(const t_commrec& cr)
{
    // Currently python initialization only works in the main function.
    // Need to find a proper place.
    // If I place them here, import_array 100% fails at longdouble_multiply().
    // Py_Initialize();
    /*
    import_array1(void(0));
    if (PyErr_Occurred()) {
        fprintf(stderr, "Failed to import numpy Python module(s).\n");
        return;
    }
    assert(PyArray_API);
    */

    // Set flag of successful initialization
    isPythonInitialized_ = true;
} // namespace gmx

void QMMMForceProvider::calculateForces(const ForceProviderInput& fInput, ForceProviderOutput* fOutput)
{
    if (!isPythonInitialized_)
    {
        try
        {
            initPython(fInput.cr_);
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
    }
    fprintf(stderr, "Importing pyscfdriverii...\n");
    PyObject* pModule = PyImport_ImportModule("pyscfdriverii");
    if (!pModule)
    {
        GMX_THROW(gmx::InternalError("pyscfdriverii load failed!\n"));
    }
    fprintf(stderr, "pyscfdriverii load successful!\n");

    // Total number of atoms in the system
    size_t numAtoms   = qmAtoms_.numAtomsGlobal() + mmAtoms_.numAtomsGlobal();
    size_t numAtomsQM = qmAtoms_.numAtomsGlobal();
    size_t numAtomsMM = mmAtoms_.numAtomsGlobal();
    // Save box

    PyObject* pFuncPrint = PyObject_GetAttrString(pModule, "printProp");
    PyObject* pFuncCalc  = nullptr;
    // determine whether to do a pure QM calculation or a QM/MM calculation
    if (numAtomsMM > 0)
    {
        fprintf(stderr, "Loading qmmmCalc...\n");
        pFuncCalc = PyObject_GetAttrString(pModule, "qmmmCalc");
        if (!pFuncCalc)
        {
            GMX_THROW(gmx::APIError("qmmmCalc load failed!\n"));
        }
        fprintf(stderr, "qmmmCalc load successful!\n");
    }
    else if (numAtomsMM == 0)
    {
        fprintf(stderr, "numAtomsMM = 0, Loading qmCalc...\n");
        pFuncCalc = PyObject_GetAttrString(pModule, "qmCalc");
        if (!pFuncCalc)
        {
            GMX_THROW(gmx::APIError("qmCalc load failed!\n"));
        }
        fprintf(stderr, "qmCalc load successful!\n");
    }
    else
    {
        GMX_THROW(gmx::InternalError("Unexpected MM atom number."));
    }

    copy_mat(fInput.box_, box_);
    // Initialize PBC
    t_pbc pbc;
    set_pbc(&pbc, pbcType_, box_);
    /*
     * 1) We need to gather fInput.x_ in case of MPI / DD setup
     */

    // x - coordinates (gathered across nodes in case of DD)
    std::vector<RVec> x(numAtoms, RVec({ 0.0, 0.0, 0.0 }));
    // Put all atoms into the central box (they might be shifted out of it because of the
    // translation) put_atoms_in_box(pbcType_, fInput.box_, ArrayRef<RVec>(x));

    // Fill cordinates of local QM atoms and add translation
    //
    const std::string qm_basis   = "631G";
    PyObject*         pyQMBasis  = PyUnicode_FromString(qm_basis.c_str());
    PyObject*         pyQMMult   = PyLong_FromLong(parameters_.qmMultiplicity_);
    PyObject*         pyQMCharge = PyLong_FromLong(parameters_.qmCharge_);

    PyObject* pyQMKinds      = PyList_New(numAtomsQM);
    PyObject* pyQMCoords     = PyList_New(numAtomsQM);
    PyObject* pyQMLocalIndex = PyList_New(numAtomsQM);
    for (size_t i = 0; i < qmAtoms_.numAtomsLocal(); i++)
    {
        int       qmLocalIndex = qmAtoms_.localIndex()[i];
        PyObject* pyqmLoclNdx  = PyLong_FromLong(qmLocalIndex);
        PyList_SetItem(pyQMLocalIndex, i, pyqmLoclNdx);
        x[qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]] = fInput.x_[qmLocalIndex];

        PyObject* pySymbol =
                PyUnicode_FromString(periodic_system[parameters_.atomNumbers_[qmLocalIndex]].c_str());
        PyList_SetItem(pyQMKinds, i, pySymbol);

        PyObject* pyCoords_row = PyList_New(3);
        // double coordTempX =  10 * (fInput.x_[qmLocalIndex][XX] + parameters_.qmTrans_[XX]);
        // double coordTempY =  10 * (fInput.x_[qmLocalIndex][YY] + parameters_.qmTrans_[YY]);
        // double coordTempZ =  10 * (fInput.x_[qmLocalIndex][ZZ] + parameters_.qmTrans_[ZZ]);

        double coordTempX = 10 * (fInput.x_[qmLocalIndex][XX]);
        double coordTempY = 10 * (fInput.x_[qmLocalIndex][YY]);
        double coordTempZ = 10 * (fInput.x_[qmLocalIndex][ZZ]);

        PyObject* pyCoordsX = PyFloat_FromDouble(coordTempX);
        PyObject* pyCoordsY = PyFloat_FromDouble(coordTempY);
        PyObject* pyCoordsZ = PyFloat_FromDouble(coordTempZ);

        PyList_SetItem(pyCoords_row, XX, pyCoordsX);
        PyList_SetItem(pyCoords_row, YY, pyCoordsY);
        PyList_SetItem(pyCoords_row, ZZ, pyCoordsZ);

        PyList_SetItem(pyQMCoords, i, pyCoords_row);
    }

    // fprintf(stderr, "qmkinds[%d] %3s \n", numAtomsQM-1, periodic_system[parameters_.atomNumbers_[qmAtoms_.localIndex()[numAtomsQM-1]]].c_str());
    // fprintf(stderr, "qmcoords[%d][0] %f \n", numAtomsQM-1, 10 * (fInput.x_[qmAtoms_.localIndex()[numAtomsQM-1]][XX] + parameters_.qmTrans_[XX]));

    // Fill cordinates of local MM atoms and add translation
    PyObject* pyMMKinds       = PyList_New(numAtomsMM);
    PyObject* pyMMCharges     = PyList_New(numAtomsMM);
    PyObject* pyMMCoords      = PyList_New(numAtomsMM);
    PyObject* pyMMLocalIndex  = PyList_New(numAtomsMM);
    PyObject* pyscfCalcReturn = nullptr;
    fprintf(stderr, "number of MM atoms: %ld\n", numAtomsMM);
    size_t    numLinks = parameters_.link_.size();
    PyObject* PyLinks  = PyList_New(numLinks);
    for (size_t i = 0; i < numLinks; i++)
    {
        PyObject* pyLinkQM = PyLong_FromLong(parameters_.link_[i].qm);
        PyObject* pyLinkMM = PyLong_FromLong(parameters_.link_[i].mm);

        PyObject* pyLinkPair = PyList_New(2);

        PyList_SetItem(pyLinkPair, 0, pyLinkQM);
        PyList_SetItem(pyLinkPair, 1, pyLinkMM);
        PyList_SetItem(PyLinks, i, pyLinkPair);
    }

    if (numAtomsMM == 0)
    {
        // call qmCalc for pure QM calculation
        pyscfCalcReturn = PyObject_CallFunctionObjArgs(
                pFuncCalc, pyQMBasis, pyQMMult, pyQMCharge, pyQMKinds, pyQMCoords, NULL);
    }
    else // first prepare PyObjects for MM atoms, then call qmmmCalc for QM/MM calculations
    {
        for (size_t i = 0; i < mmAtoms_.numAtomsLocal(); i++)
        {
            int       mmLocalIndex = mmAtoms_.localIndex()[i];
            PyObject* pymmLoclNdx  = PyLong_FromLong(mmLocalIndex);
            PyList_SetItem(pyMMLocalIndex, i, pymmLoclNdx);
            x[mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]]] = fInput.x_[mmLocalIndex];

            PyObject* pySymbol = PyUnicode_FromString(
                    periodic_system[parameters_.atomNumbers_[mmLocalIndex]].c_str());
            PyList_SetItem(pyMMKinds, i, pySymbol);

            PyObject* pyCharge = PyFloat_FromDouble(fInput.chargeA_[mmLocalIndex]);
            PyList_SetItem(pyMMCharges, i, pyCharge);

            PyObject* pyCoords_row = PyList_New(3);
            // double coordTempX =  10 * (fInput.x_[mmLocalIndex][XX] + parameters_.qmTrans_[XX]);
            // double coordTempY =  10 * (fInput.x_[mmLocalIndex][YY] + parameters_.qmTrans_[YY]);
            // double coordTempZ =  10 * (fInput.x_[mmLocalIndex][ZZ] + parameters_.qmTrans_[ZZ]);
            double coordTempX = 10 * (fInput.x_[mmLocalIndex][XX]);
            double coordTempY = 10 * (fInput.x_[mmLocalIndex][YY]);
            double coordTempZ = 10 * (fInput.x_[mmLocalIndex][ZZ]);

            PyObject* pyCoordsX = PyFloat_FromDouble(coordTempX);
            PyObject* pyCoordsY = PyFloat_FromDouble(coordTempY);
            PyObject* pyCoordsZ = PyFloat_FromDouble(coordTempZ);

            PyList_SetItem(pyCoords_row, XX, pyCoordsX);
            PyList_SetItem(pyCoords_row, YY, pyCoordsY);
            PyList_SetItem(pyCoords_row, ZZ, pyCoordsZ);

            PyList_SetItem(pyMMCoords, i, pyCoords_row);
        }
        pyscfCalcReturn = PyObject_CallFunctionObjArgs(pFuncCalc,
                                                       pyQMBasis,
                                                       pyQMMult,
                                                       pyQMCharge,
                                                       pyQMLocalIndex,
                                                       pyQMKinds,
                                                       pyQMCoords,
                                                       pyMMLocalIndex,
                                                       pyMMKinds,
                                                       pyMMCharges,
                                                       pyMMCoords,
                                                       PyLinks,
                                                       NULL);
    }

    if (!pyscfCalcReturn)
    {
        PyErr_Print();
        // this function must be called to print Traceback messges and python output up to that exception -
        // - with import_array() called in gmx.cpp and numpy imported in pyscfdriver.py.
        Py_FinalizeEx();
        fprintf(stderr, "Py_FinalizeEx() run finished\n");
        GMX_THROW(gmx::InternalError("GMX_THORW: pyscfCalcReturn is nullptr!!!"));
    }

    // If we are in MPI / DD conditions then gather coordinates over nodes
    if (havePPDomainDecomposition(&fInput.cr_))
    {
        gmx_sum(3 * numAtoms, x.data()->as_vec(), &fInput.cr_);
    }

    // TODO: fix for MPI version

    PyObject* pQMMMEnergy = PyTuple_GetItem(pyscfCalcReturn, 0);
    PyObject* pQMForce    = PyTuple_GetItem(pyscfCalcReturn, 1);
    PyObject* pMMForce    = PyTuple_GetItem(pyscfCalcReturn, 2);

    // fprintf(stdout, "Python print test QMForce\n");
    // PyObject_CallFunctionObjArgs(pFuncPrint, pQMForce, NULL);

    double qmmmEnergy(0);
    if (pQMMMEnergy)
    {
        qmmmEnergy = PyFloat_AsDouble(pQMMMEnergy);
        fprintf(stderr, "GROMACS received energy %f \n", qmmmEnergy);
    }
    else
    {
        GMX_THROW(gmx::InternalError("pointer to pyscf returned energy is nullptr\n"));
    }

    if (!pQMForce)
    {
        PyErr_Print();
        Py_FinalizeEx();
        GMX_THROW(gmx::InternalError(
                "parsing pyscfCalcReturn error, pyobject pQMForce is nullptr\n"));
    }

    if ((!pMMForce) && (numAtomsMM > 0))
    {
        PyErr_Print();
        Py_FinalizeEx();
        GMX_THROW(gmx::InternalError(
                "parsing pyscfCalcReturn error, pyobject pMMForce is nullptr\n"));
    }

    double         qmForce[numAtomsQM * 3 + 1] = {};
    PyArrayObject* npyQMForce                  = reinterpret_cast<PyArrayObject*>(pQMForce);
    GMX_ASSERT(numAtomsQM == static_cast<int>(PyArray_DIM(npyQMForce, 0)), "Check QM atom number");
    GMX_ASSERT(3 == static_cast<int>(PyArray_DIM(npyQMForce, 1)), "Check if 3 columns for QM force");

    double* npyQMForce_cast = static_cast<double*>(PyArray_DATA(npyQMForce));
    for (size_t i = 0; i < numAtomsQM; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            qmForce[i * 3 + j] = npyQMForce_cast[i * 3 + j];
        }
    }
    double mmForce[numAtomsMM * 3 + 1] = {};
    if (numAtomsMM > 0)
    {
        PyArrayObject* npyMMForce = reinterpret_cast<PyArrayObject*>(pMMForce);
        GMX_ASSERT(numAtomsMM == static_cast<int>(PyArray_DIM(npyMMForce, 0)),
                   "Check MM atom number");
        GMX_ASSERT(3 == static_cast<int>(PyArray_DIM(npyMMForce, 1)),
                   "Check if 3 columns for MM force");

        double* npyMMForce_cast = static_cast<double*>(PyArray_DATA(npyMMForce));
        for (size_t i = 0; i < numAtomsMM; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                mmForce[i * 3 + j] = npyMMForce_cast[i * 3 + j];
            }
        }
    }

    /*
     * 2) Cast data to double format of libpython
     *    update coordinates and box in PySCF and perform QM calculation
     */
    // x_d - coordinates casted to linear dobule vector for PySCF with parameters_.qmTrans_ added
    std::vector<double> x_d(3 * numAtoms, 0.0);
    for (size_t i = 0; i < numAtoms; i++)
    {
        x_d[3 * i]     = static_cast<double>((x[i][XX]) / c_bohr2Nm);
        x_d[3 * i + 1] = static_cast<double>((x[i][YY]) / c_bohr2Nm);
        x_d[3 * i + 2] = static_cast<double>((x[i][ZZ]) / c_bohr2Nm);
    }

    // box_d - box_ casted to linear dobule vector for PySCF
    std::vector<double> box_d(9);
    for (size_t i = 0; i < DIM; i++)
    {
        box_d[3 * i]     = static_cast<double>(box_[0][i] / c_bohr2Nm);
        box_d[3 * i + 1] = static_cast<double>(box_[1][i] / c_bohr2Nm);
        box_d[3 * i + 2] = static_cast<double>(box_[2][i] / c_bohr2Nm);
    }

    // NOTE: need to handle MPI
    // Update coordinates and box in PySCF
    // cp2k_set_positions(force_env_, x_d.data(), 3 * numAtoms);
    // cp2k_set_cell(force_env_, box_d.data());
    // Check if we have external MPI library
    if (GMX_LIB_MPI)
    {
        // We have an external MPI library
#if GMX_LIB_MPI
#endif
    }
    else
    {
        // If we have thread-MPI or no-MPI then we should initialize CP2P differently
    }

    // Run PySCF calculation
    // cp2k_calc_energy_force(force_env_);

    /*
     * 3) Get output data
     * We need to fill only local part into fOutput
     */

    // Only main process should add QM + QMMM energy
    if (MAIN(&fInput.cr_))
    {
        double qmEner = 0.0;
        qmEner        = qmmmEnergy;
        // cp2k_get_potential_energy(force_env_, &qmEner);
        fOutput->enerd_.term[F_EQM] += qmEner * c_hartree2Kj * c_avogadro;
    }

    // Get Forces they are in Hartree/Bohr and will be converted to kJ/mol/nm
    std::vector<double> pyscfForce(3 * numAtoms, 0.0);

    // Fill forces on QM atoms first
    for (size_t i = 0; i < qmAtoms_.numAtomsLocal(); i++)
    {
        pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]] =
                qmForce[qmAtoms_.collectiveIndex()[i] * 3 + 0];
        pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 1] =
                qmForce[qmAtoms_.collectiveIndex()[i] * 3 + 1];
        pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 2] =
                qmForce[qmAtoms_.collectiveIndex()[i] * 3 + 2];

        fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][XX] +=
                static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]])
                * c_hartreeBohr2Md;

        fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][YY] +=
                static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 1])
                * c_hartreeBohr2Md;

        fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][ZZ] +=
                static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 2])
                * c_hartreeBohr2Md;
    }

    // Fill forces on MM atoms then
    for (size_t i = 0; i < mmAtoms_.numAtomsLocal(); i++)
    {
        pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]]] =
                mmForce[mmAtoms_.collectiveIndex()[i] * 3 + 0];
        pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 1] =
                mmForce[mmAtoms_.collectiveIndex()[i] * 3 + 1];
        pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 2] =
                mmForce[mmAtoms_.collectiveIndex()[i] * 3 + 2];

        fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][XX] +=
                static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]]])
                * c_hartreeBohr2Md;

        fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][YY] +=
                static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 1])
                * c_hartreeBohr2Md;

        fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][ZZ] +=
                static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 2])
                * c_hartreeBohr2Md;
    }

    // forceRecorder(fOutput, pyscfForce, fInput);
    //
    Py_XDECREF(pFuncCalc);
    Py_XDECREF(pModule);
};


void QMMMForceProvider::forceRecorder(ForceProviderOutput*      fOutput,
                                      std::vector<double>       pyscfForce,
                                      const ForceProviderInput& fInput)
{

    std::string       QMMM_record = "";
    std::ofstream     recordFile;
    const std::string pyscfRecordName = parameters_.qmFileNameBase_ + "_record.txt";
    recordFile.open(pyscfRecordName.c_str(), std::ios::app);
    // size_t numAtoms = qmAtoms_.numAtomsLocal() + mmAtoms_.numAtomsLocal();
    // recordFile << "number of atoms" << numAtoms << std::endl;
    // recordFile << "step number = " << "need to find..." << std::endl;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double Ftotx = 0.0, Ftoty = 0.0, Ftotz = 0.0;
    double coordx = 0.0, coordy = 0.0, coordz = 0.0;
    double charge = 0.0;
    QMMM_record += formatString("step = ");
    QMMM_record += formatString("%" PRId64, fInput.step_);
    QMMM_record += formatString(", time = %f\n", fInput.t_);
    QMMM_record += formatString(
            "  %7s %7s %7s %6s %6s %6s %6s\n", "x", "y", "z", "i", "local", "global", "collec");
    // QMMM_record += formatString("  %6s %7s %7s %7s %19s %19s %19s %10s\n", "local index", "x", "y", "z", "Fqmmmm x", "Fqmmmm y", "Fqmmm z", "charge");
    for (size_t i = 0; i < qmAtoms_.numAtomsLocal(); i++)
    {

        QMMM_record += formatString(
                "%4d  %2s QM  ",
                qmAtoms_.localIndex()[i],
                periodic_system[parameters_.atomNumbers_[qmAtoms_.globalIndex()[i]]].c_str());
        Ftotx = fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][XX] / c_hartreeBohr2Md;
        Ftoty = fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][YY] / c_hartreeBohr2Md;
        Ftotz = fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][ZZ] / c_hartreeBohr2Md;
        Fx = static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]]);
        Fy = static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 1]);
        Fz = static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 2]);

        // Fx = Fx/c_bohr2ANG;
        // Fy = Fy/c_bohr2ANG;
        // Fz = Fz/c_bohr2ANG;

        coordx = (fInput.x_[qmAtoms_.localIndex()[i]][XX]) * 10;
        coordy = (fInput.x_[qmAtoms_.localIndex()[i]][YY]) * 10;
        coordz = (fInput.x_[qmAtoms_.localIndex()[i]][ZZ]) * 10;

        charge = fInput.chargeA_[qmAtoms_.localIndex()[i]];

        QMMM_record += formatString("%7.4lf %7.4lf %7.4lf  ", coordx, coordy, coordz);
        // QMMM_record += formatString("%19.15lf %19.15lf %19.15lf %7.3f ", Fx, Fy, Fz, charge);
        // QMMM_record += formatString("                                 %19.15lf %19.15lf %19.15lf \n", Ftotx, Ftoty, Ftotz);
        QMMM_record += formatString("%8d %6d %6d %6d\n",
                                    static_cast<int>(i),
                                    qmAtoms_.localIndex()[i],
                                    qmAtoms_.globalIndex()[i],
                                    qmAtoms_.collectiveIndex()[i]);
    }


    for (size_t i = 0; i < mmAtoms_.numAtomsLocal(); i++)
    {

        QMMM_record += formatString(
                "%4d  %2s MM  ",
                mmAtoms_.localIndex()[i],
                periodic_system[parameters_.atomNumbers_[mmAtoms_.globalIndex()[i]]].c_str());
        Ftotx = fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][XX] / c_hartreeBohr2Md;
        Ftoty = fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][YY] / c_hartreeBohr2Md;
        Ftotz = fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][ZZ] / c_hartreeBohr2Md;
        Fx = static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]]]);
        Fy = static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 1]);
        Fz = static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 2]);

        // Fx = Fx/c_bohr2ANG;
        // Fy = Fy/c_bohr2ANG;
        // Fz = Fz/c_bohr2ANG;

        coordx = (fInput.x_[mmAtoms_.localIndex()[i]][XX]) * 10;
        coordy = (fInput.x_[mmAtoms_.localIndex()[i]][YY]) * 10;
        coordz = (fInput.x_[mmAtoms_.localIndex()[i]][ZZ]) * 10;

        charge = fInput.chargeA_[mmAtoms_.localIndex()[i]];
        QMMM_record += formatString("%7.4lf %7.4lf %7.4lf  ", coordx, coordy, coordz);
        // QMMM_record += formatString("%19.15lf %19.15lf %19.15lf %7.3f ", Fx, Fy, Fz, charge);
        // QMMM_record += formatString("                                 %19.15lf %19.15lf %19.15lf \n", Ftotx, Ftoty, Ftotz);
        QMMM_record += formatString("%8d %6d %6d %6d\n",
                                    static_cast<int>(i),
                                    mmAtoms_.localIndex()[i],
                                    mmAtoms_.globalIndex()[i],
                                    mmAtoms_.collectiveIndex()[i]);
    }


    QMMM_record += formatString("\n");
    recordFile << QMMM_record;
    recordFile.close();

    return;
}


} // namespace gmx