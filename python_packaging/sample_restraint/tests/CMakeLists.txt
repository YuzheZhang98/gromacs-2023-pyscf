#
# Set up our tests
#


# Copy test files
if (GMXAPI_EXTENSION_MAIN_PROJECT)
    # TODO: Deduplicate and leverage CMake dependencies.
    # We also build a test file in the spc_water_box pytest test fixture, but we can
    # presumably extract both of those from the GROMACS installation, or at least
    # through the gmxapi Python package resources.
    # Ref: https://gitlab.com/gromacs/gromacs/-/issues/2961
    file(DOWNLOAD
         https://github.com/kassonlab/sample_restraint/raw/master/tests/data/topol.tpr
         ${CMAKE_CURRENT_BINARY_DIR}/topol.tpr
         STATUS _download_status
         LOG _download_log)
    list(GET _download_status 0 _status)
    if(${_status} GREATER 0)
        message(WARNING "Could not download test data: ${_download_log}")
    endif()
    unset(_status)
    add_custom_target(gmxapi_extension_spc2_water_box
                      COMMAND cmake -E echo
                      SOURCES ${CMAKE_CURRENT_BINARY_DIR}/topol.tpr
                      )
else()
    # We are in the GROMACS build tree.
    # We just need a simple TPR input file. The 6 atom spc water box will suffice.
    if (NOT TARGET gmx)
        message(FATAL_ERROR "Trying to use gmx wrapper binary, but gmx target not defined.")
    endif()
    set(_mdp ${CMAKE_CURRENT_BINARY_DIR}/grompp.mdp)
    file(WRITE ${_mdp} "integrator = md\n")
    file(APPEND ${_mdp} "nsteps = 6")
    set(_gro ${CMAKE_SOURCE_DIR}/src/testutils/simulationdatabase/spc2.gro)
    set(_top ${CMAKE_SOURCE_DIR}/src/testutils/simulationdatabase/spc2.top)
    add_custom_target(gmxapi_extension_spc2_water_box
                      BYPRODUCTS topol.tpr
                      COMMAND gmx -quiet grompp -f ${_mdp} -c ${_gro} -p ${_top}
                      DEPENDS gmx ${_mdp}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      COMMENT "Generating input file for sample_restraint tests.")
    unset(_mdp)
    unset(_gro)
    unset(_top)
endif ()

# Note: Expects topol.tpr to be in CURRENT_BINARY_DUR
configure_file(testingconfiguration.in.h testingconfiguration.h)

# Test the C++ force evaluation for the restrained-ensemble biasing potential.
add_executable(gmxapi_extension_histogram-test EXCLUDE_FROM_ALL test_histogram.cpp)
add_dependencies(gmxapi_extension_histogram-test gmxapi_extension_spc2_water_box)
target_include_directories(gmxapi_extension_histogram-test PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(gmxapi_extension_histogram-test PROPERTIES SKIP_BUILD_RPATH FALSE)
target_link_libraries(gmxapi_extension_histogram-test gmxapi_extension_ensemblepotential Gromacs::gmxapi
                      GTest::Main)
gtest_add_tests(TARGET gmxapi_extension_histogram-test
                TEST_LIST EnsembleHistogramPotentialPlugin)

# Test the flat-bottom bounding potential built in to the ensemble restraint.
add_executable(gmxapi_extension_bounding-test EXCLUDE_FROM_ALL test_bounding_restraint.cpp)
add_dependencies(gmxapi_extension_bounding-test gmxapi_extension_spc2_water_box)
target_include_directories(gmxapi_extension_bounding-test PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(gmxapi_extension_bounding-test PROPERTIES SKIP_BUILD_RPATH FALSE)
target_link_libraries(gmxapi_extension_bounding-test gmxapi_extension_ensemblepotential Gromacs::gmxapi
                      GTest::Main)
gtest_add_tests(TARGET gmxapi_extension_bounding-test
                TEST_LIST EnsembleBoundingPotentialPlugin)

# Like GROMACS, this project uses a `tests` target to avoid
# building the tests unnecessarily (such as for the `install` target).
# If built as part of the GROMACS build tree (GROMACS is the "main" project),
# the `tests` target is defined by the parent project.
# Otherwise (the MD plugin is the "main project"), we have to add
# the `tests` target in this project.
if (GMXAPI_EXTENSION_MAIN_PROJECT)
    add_custom_target(tests)
else()
    include(CMakeGROMACS.txt)
endif ()

add_dependencies(tests gmxapi_extension_histogram-test gmxapi_extension_bounding-test)
