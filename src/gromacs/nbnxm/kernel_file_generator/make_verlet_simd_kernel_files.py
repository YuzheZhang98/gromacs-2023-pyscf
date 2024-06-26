#!/usr/bin/env python3
#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2013- The GROMACS Authors
# and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
# Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# https://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at https://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out https://www.gromacs.org.

# This script is used by the GROMACS developers to build most of the
# files from which the nbnxn kernels are compiled. It is not called at
# CMake time, and users should never need to use it. It currently
# works for nbnxn kernel structure types 2xMM and 4xM. The generated
# files are versions of the *.pre files in this directory, customized
# for the kernel structure type and/or the detailed kernel type. These
# are:
#
#   A single header file that declares all the kernel functions for
#   this nbnxn kernel structure type and a function pointer table.
#
#   Many C kernel files, each defining a single kernel function. These
#   functions can take a noticeable time to compile, and should tend
#   to be in separate files to take advantage of make-time
#   parallelism.
#
# This script should be run from the directory in which it is
# located. The generated files are located in ../kernels_simd_<type>.
# There are three other files in those locations that are not generated.
# These contain:
#
#   setup logic peculiar to the kernel structure type but common to
#   all the kernels within that type, and
#
#   the logic for the outer and inner loops of the kernels, as
#   customized by numerous preprocessor defines to suit the hardware
#   and kernel type.
#
# Note that while functions for both nbnxn kernel structures are
# compiled and built into an mdrun executable, because that executable
# is not portable, only the functions for the useful nbnxn kernel
# structure for the hardware selected at CMake time contain real
# kernel logic. A run-time error occurs if an inappropriate kernel
# dispatcher function is called (but that is normally impossible).

import re
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import collections  # Requires Python 2.7

sys.path.append("../../../../admin")
from copyright import create_copyright_header

FileHeader = create_copyright_header("2012,2013,2014,2015,2019")
FileHeader += """/*
 * Note: this file was generated by the Verlet kernel generator for
 * kernel type {0}.
 */

"""


def read_kernel_template(filename):
    with open(filename, "r") as TemplateFile:
        TemplateText = TemplateFile.read()
    copyright_re = r"/\*\n \* This file is part of the GROMACS molecular simulation package\.\n( \*.*\n)* \*/\n"
    match = re.match(copyright_re, TemplateText)
    if match:
        TemplateText = TemplateText[match.end() :]
    return TemplateText


# The dict order must match the order of an enumeration in
# kernel_simd_template.c.pre
ElectrostaticsDict = collections.OrderedDict()
ElectrostaticsDict["ElecRF"] = {"define": "#define CALC_COUL_RF"}
ElectrostaticsDict["ElecQSTab"] = {"define": "#define CALC_COUL_TAB"}
ElectrostaticsDict["ElecQSTabTwinCut"] = {
    "define": "#define CALC_COUL_TAB\n#define VDW_CUTOFF_CHECK /* Use twin-range cut-off */"
}
ElectrostaticsDict["ElecEw"] = {"define": "#define CALC_COUL_EWALD"}
ElectrostaticsDict["ElecEwTwinCut"] = {
    "define": "#define CALC_COUL_EWALD\n#define VDW_CUTOFF_CHECK /* Use twin-range cut-off */"
}

# The dict order must match the order of a C enumeration.
VdwTreatmentDict = collections.OrderedDict()
VdwTreatmentDict["VdwLJCombGeom"] = {"define": "#define LJ_CUT\n#define LJ_COMB_GEOM"}
VdwTreatmentDict["VdwLJCombLB"] = {"define": "#define LJ_CUT\n#define LJ_COMB_LB"}
VdwTreatmentDict["VdwLJ"] = {
    "define": "#define LJ_CUT\n/* Use full LJ combination matrix */"
}
VdwTreatmentDict["VdwLJFSw"] = {
    "define": "#define LJ_FORCE_SWITCH\n/* Use full LJ combination matrix */"
}
VdwTreatmentDict["VdwLJPSw"] = {
    "define": "#define LJ_POT_SWITCH\n/* Use full LJ combination matrix */"
}
VdwTreatmentDict["VdwLJEwCombGeom"] = {
    "define": "#define LJ_CUT\n#define LJ_EWALD_GEOM\n/* Use full LJ combination matrix + geometric rule for the grid correction */"
}

# This is OK as an unordered dict
EnergiesComputationDict = {
    "F": {
        "function type": "nbk_func_noener",
        "define": "/* Will not calculate energies */",
    },
    "VF": {
        "function type": "nbk_func_ener",
        "define": "#define CALC_ENERGIES",
    },
    "VgrpF": {
        "function type": "nbk_func_ener",
        "define": "#define CALC_ENERGIES\n#define ENERGY_GROUPS",
    },
}

# This is OK as an unordered dict
VerletKernelTypeDict = {
    "2xmm": {
        "Define": "GMX_NBNXN_SIMD_2XNN",
        "WidthSetup": "/* Include the full-width SIMD macros */\n",
        "WidthCheck": (
            "#if !(GMX_SIMD_REAL_WIDTH == 8 || GMX_SIMD_REAL_WIDTH == 16)\n"
            '#error "unsupported SIMD width"\n'
            "#endif\n"
        ),
        "UnrollSize": 2,
    },
    "4xm": {
        "Define": "GMX_NBNXN_SIMD_4XN",
        "WidthSetup": (""),
        "WidthCheck": (
            "#if !(GMX_SIMD_REAL_WIDTH == 2 || GMX_SIMD_REAL_WIDTH == 4 || GMX_SIMD_REAL_WIDTH == 8)\n"
            '#error "unsupported SIMD width"\n'
            "#endif\n"
        ),
        "UnrollSize": 1,
    },
}

KernelsHeaderTemplate = read_kernel_template("kernel_simd_template.h.pre")

# For each Verlet kernel type, write two kinds of files:
#   a header file defining the functions for all the kernels and
#     the kernel function lookup table
#   for each kernel, a file defining the single C function for that kernel
for type in VerletKernelTypeDict:
    DirName = "../kernels_simd_{0}".format(type)
    KernelNamePrefix = "nbnxm_kernel"
    KernelFileNamePrefix = "kernel"
    KernelsName = "{0}_simd_{1}".format(KernelNamePrefix, type)
    KernelsFileName = "{0}_simd_{1}".format(KernelFileNamePrefix, type)
    KernelsHeaderFileName = "kernels.h"
    KernelsHeaderPathName = "{1}".format(type, KernelsHeaderFileName)
    KernelFunctionLookupTable = {}
    KernelDeclarations = ""
    KernelTemplate = read_kernel_template("{0}_kernel.cpp.pre".format(KernelsFileName))

    # Loop over all kernels
    for ener in EnergiesComputationDict:
        KernelFunctionLookupTable[ener] = "{\n"
        for elec in ElectrostaticsDict:
            KernelFunctionLookupTable[ener] += "    {\n"
            for ljtreat in VdwTreatmentDict:
                KernelName = "{0}_{1}_{2}_{3}_{4}".format(
                    KernelNamePrefix, elec, ljtreat, ener, type
                )
                KernelFileName = "{0}_{1}_{2}_{3}".format(
                    KernelFileNamePrefix, elec, ljtreat, ener, type
                )

                # Declare the kernel function
                KernelDeclarations += "{1:21} {0};\n".format(
                    KernelName, EnergiesComputationDict[ener]["function type"]
                )

                # Write the file with the kernel definition
                with open(
                    "{0}/{1}.cpp".format(DirName, KernelFileName), "w"
                ) as kernelfp:
                    kernelfp.write(FileHeader.format(type))
                    kernelfp.write(
                        KernelTemplate.format(
                            VerletKernelTypeDict[type]["Define"],
                            ElectrostaticsDict[elec]["define"],
                            VdwTreatmentDict[ljtreat]["define"],
                            EnergiesComputationDict[ener]["define"],
                            KernelsHeaderPathName,
                            KernelName,
                            " " * (len(KernelName) + 1),
                            VerletKernelTypeDict[type]["UnrollSize"],
                        )
                    )

                # Enter the kernel function in the lookup table
                KernelFunctionLookupTable[ener] += "        {0},\n".format(KernelName)

            KernelFunctionLookupTable[ener] += "    },\n"
        KernelFunctionLookupTable[ener] += "};\n"
        KernelDeclarations += "\n"

    # Write the header file that declares all the kernel
    # functions for this type
    with open("{0}/{1}".format(DirName, KernelsHeaderFileName), "w") as fp:
        fp.write(FileHeader.format(type))
        fp.write(
            KernelsHeaderTemplate.format(
                KernelDeclarations,
                type,
                KernelFunctionLookupTable["F"],
                KernelFunctionLookupTable["VF"],
                KernelFunctionLookupTable["VgrpF"],
            )
        )

sys.exit()
