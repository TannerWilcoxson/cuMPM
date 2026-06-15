cmake_minimum_required(VERSION 3.18)
project(cuMPM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)

# Option to build testing executables
option(BUILD_TESTING_EXECS "Build standalone test executables" OFF)

if(BUILD_TESTING_EXECS)
    add_executable(cuMPM_reader dev_testing/main.cpp src/cpp/electric_field/neighbor_list.cu src/cpp/electric_field/ewald_electric_field.cu)
    target_link_libraries(cuMPM_reader PRIVATE CUDA::cufft)

    add_executable(minimal_use_case dev_testing/minimal_use_case.cpp src/cpp/electric_field/neighbor_list.cu src/cpp/electric_field/ewald_electric_field.cu)
    target_link_libraries(minimal_use_case PRIVATE CUDA::cufft)

    add_executable(test_dipole_solver dev_testing/test_dipole_solver.cpp src/cpp/dipole_solver.cu src/cpp/electric_field/neighbor_list.cu src/cpp/electric_field/ewald_electric_field.cu src/cpp/electric_field/polydisperse_electric_field.cu src/cpp/numerical_solver/numerical_solver.cu src/cpp/numerical_solver/gmres_solver.cu src/cpp/numerical_solver/bicgstab_solver.cu)
    target_link_libraries(test_dipole_solver PRIVATE CUDA::cufft)
endif()

# Find pybind11 dynamically
if(NOT pybind11_FOUND)
    # Check if Python_EXECUTABLE is set, if not try to find it
    if(NOT DEFINED Python_EXECUTABLE)
        find_program(Python_EXECUTABLE NAMES python3 python)
    endif()
    
    if(Python_EXECUTABLE)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(PYBIND11_CMAKE_DIR)
            set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
        endif()
    endif()
    
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 REQUIRED)
endif()

# Build the python extension module _cuMPM
pybind11_add_module(_cuMPM MODULE
    src/cpp/dipole_solver_bindings.cpp
    src/cpp/near_field_bindings.cpp
    src/cpp/near_field.cu
    src/cpp/dipole_solver.cu
    src/cpp/electric_field/neighbor_list.cu
    src/cpp/electric_field/ewald_electric_field.cu
    src/cpp/electric_field/polydisperse_electric_field.cu
    src/cpp/numerical_solver/numerical_solver.cu
    src/cpp/numerical_solver/gmres_solver.cu
    src/cpp/numerical_solver/bicgstab_solver.cu
)
target_link_libraries(_cuMPM PRIVATE CUDA::cufft)

# Install the module target to the package folder (relocatable)
install(TARGETS _cuMPM DESTINATION cuMPM)


