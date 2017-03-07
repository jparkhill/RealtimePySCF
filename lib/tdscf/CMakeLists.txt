#Linking with Armadillo: if Armadillo is already in your $PATH, the following 2 lines are probably not needed.
#set(ARMADILLO_LIBRARIES "/usr/local/Cellar/armadillo/7.600.2/lib")
#set(ARMADILLO_INCLUDE_DIRS "/usr/local/Cellar/armadillo/7.600.2/include")
cmake_minimum_required(VERSION 3.6)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
add_library(tdscf SHARED tdscf.cpp)
set_target_properties(tdscf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_link_libraries(tdscf ${ARMADILLO_LIBRARIES})
