# This CMakeLists cloned from original CVODE version and then
# modified to compile only one program.

project("DuffingsEq")
cmake_minimum_required(VERSION 3.16)

project(DuffingsEq LANGUAGES CXX)

#---------------------------------------------------------
# VTK stuff -- do I need all this stuff?
set(VTK_DIR /usr/local/)
find_package(VTK CONFIG)
find_package(VTK REQUIRED)
#message (STATUS "VTK_VERSION: ${VTK_VERSION}")
#message (STATUS "VTK_LIBRARIES: ${VTK_LIBRARIES}")

include_directories(
                "/usr/local/include/vtk-9.0/"
)

link_directories(
                "/usr/local/lib/"
                "/usr/lib/x86_64-linux-gnu/"
)


#---------------------------------------------------------
# Sundials stuff
set(SUNDIALS_INCLUDE_DIR
  /usr/local/include
  CACHE PATH "Location of SUNDIALS header files")

# Specify the path to SUNDIALS libraries
set(SUNDIALS_LIBRARY_DIR
  /usr/local/lib
  CACHE PATH "Location of SUNDIALS libraries")

#---------------------------------------------------------
# Libm
set(LIBM_INCLUDE_DIR
  /usr/include/
  CACHE PATH "Location of math.h")

set(LIBM_LIBRARY_DIR
  /usr/lib/x86_64-linux-gnu/
  CACHE PATH "Location of libm")

#---------------------------------------------------------
# Sciplot
# https://github.com/sciplot/sciplot
set(SCIPLOT_INCLUDE_DIR
  /usr/local/include/
  CACHE PATH "Location of sciplot")


# Find the SUNDIALS libraries
find_library(SUNDIALS_SOLVER_LIB
  sundials_cvode ${SUNDIALS_LIBRARY_DIR}
  DOC "CVODE library")

find_library(SUNDIALS_NVEC_LIB
  sundials_nvecserial ${SUNDIALS_LIBRARY_DIR}
  DOC "NVECTOR_SERIAL library")

find_library(LIBM
  m ${LIBM_LIBRARY_DIR}
  DOC "Math library")

# Create list of libs to link to
set(SUNDIALS_LIBRARIES
  -L${SUNDIALS_LIBRARY_DIR}
  ${SUNDIALS_SOLVER_LIB}
  ${SUNDIALS_NVEC_LIB}
  ${LIBM})


# Source file
set(src "runDuffingsEq")
add_executable(${src} "${src}.cpp")
target_include_directories(${src} PRIVATE ${SUNDIALS_INCLUDE_DIR}  ${LIBM_INCLUDE_DIR} ${SCIPLOT_INCLUDE_DIR} )
target_link_libraries(${src} PUBLIC ${SUNDIALS_LIBRARIES})
set_property(TARGET ${src} PROPERTY CXX_STANDARD 17)

# Source file
set(src "sampleDuffingsEq")
add_executable(${src} "${src}.cpp")
target_include_directories(${src} PRIVATE ${SUNDIALS_INCLUDE_DIR}  ${LIBM_INCLUDE_DIR} ${SCIPLOT_INCLUDE_DIR} )
target_link_libraries(${src} PUBLIC ${SUNDIALS_LIBRARIES} ${VTK_LIBRARIES})
set_property(TARGET ${src} PROPERTY CXX_STANDARD 17)

vtk_module_autoinit(
    TARGETS sampleDuffingsEq
    MODULES ${VTK_LIBRARIES} 
) 
