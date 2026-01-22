cmake_minimum_required(VERSION 2.8)

include_directories(./)
include_directories(./include)

add_executable(main_laplace main_laplace.cpp 
                    gmres.cpp
                    bicgstab.cpp
                    unsorted.cpp
                    )
target_link_libraries(main_laplace ${PETSC_LIBRARIES}  )