cmake_minimum_required(VERSION 3.18)

if(DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(USER_SET_CUDA_ARCH TRUE)
else()
  set(USER_SET_CUDA_ARCH FALSE)
endif()

project(cuMPM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NEED_DETECTION FALSE)
if(NOT USER_SET_CUDA_ARCH OR CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
  set(NEED_DETECTION TRUE)
endif()

if(NEED_DETECTION)
  # Try to detect via nvidia-smi
  find_program(NVIDIA_SMI nvidia-smi)
  set(DETECTED_CUDA_ARCH "")
  if(NVIDIA_SMI)
    execute_process(
      COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE DETECTED_CAP
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(DETECTED_CAP)
      string(REPLACE "." "" DETECTED_CAP_FLAT ${DETECTED_CAP})
      if(DETECTED_CAP_FLAT MATCHES "^[0-9]+$")
        set(DETECTED_CUDA_ARCH ${DETECTED_CAP_FLAT})
      endif()
    endif()
  endif()

  # If detection failed, default based on compiler version
  if(NOT DETECTED_CUDA_ARCH)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
      set(DETECTED_CUDA_ARCH 120)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
      set(DETECTED_CUDA_ARCH 89)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
      set(DETECTED_CUDA_ARCH 89)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
      set(DETECTED_CUDA_ARCH 80)
    else()
      set(DETECTED_CUDA_ARCH 75)
    endif()
  endif()

  # Clamp the detected architecture to what the compiler actually supports
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
    if(DETECTED_CUDA_ARCH GREATER 75)
      set(DETECTED_CUDA_ARCH 75)
    endif()
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.8")
    if(DETECTED_CUDA_ARCH GREATER 86)
      set(DETECTED_CUDA_ARCH 86)
    endif()
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0")
    if(DETECTED_CUDA_ARCH GREATER 89)
      set(DETECTED_CUDA_ARCH 89)
    endif()
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.8")
    if(DETECTED_CUDA_ARCH GREATER 90)
      set(DETECTED_CUDA_ARCH 89)
    endif()
  endif()

  message(STATUS "Auto-detected and clamped CUDA architecture: ${DETECTED_CUDA_ARCH}")
  set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_CUDA_ARCH})
endif()

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
    src/cpp/eels_bindings.cpp
    src/cpp/near_field.cu
    src/cpp/eels_solver.cu
    src/cpp/dipole_solver.cu
    src/cpp/electric_field/neighbor_list.cu
    src/cpp/electric_field/ewald_electric_field.cu
    src/cpp/electric_field/polydisperse_electric_field.cu
    src/cpp/electric_field/direct_electric_field.cu
    src/cpp/numerical_solver/numerical_solver.cu
    src/cpp/numerical_solver/gmres_solver.cu
    src/cpp/numerical_solver/bicgstab_solver.cu
)
target_link_libraries(_cuMPM PRIVATE CUDA::cufft)

# Install the module target to the package folder (relocatable)
install(TARGETS _cuMPM DESTINATION cuMPM)


