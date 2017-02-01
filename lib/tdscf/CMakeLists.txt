set(ARMADILLO_LIBRARIES "/usr/local/Cellar/armadillo/7.600.2/lib")
set(ARMADILLO_INCLUDE_DIRS "/usr/local/Cellar/armadillo/7.600.2/include")
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
add_library(tdscf SHARED tdscf.cpp)
set_target_properties(tdscf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(tdscf ${ARMADILLO_LIBRARIES})
target_link_libraries(tdscf dft cgto cint ${BLAS_LIBRARIES})
