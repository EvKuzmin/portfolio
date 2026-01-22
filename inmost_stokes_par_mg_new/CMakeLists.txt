cmake_minimum_required(VERSION 2.8)

project(inmost-three-phase)

#add_compile_definitions("NDEBUG")

#set(CMAKE_BUILD_TYPE DEBUG)

set(CMAKE_BUILD_TYPE RELWITHDEBINFO)

set (CMAKE_CXX_STANDARD 20)

find_package(inmost REQUIRED)
if(inmost_FOUND)
	message("INMOST found: \ninclude dirs:${INMOST_INCLUDE_DIRS}\nlibrary dirs:${INMOST_LIBRARY_DIRS}")
	include_directories(${INMOST_INCLUDE_DIRS})
	link_directories(${INMOST_LIBRARY_DIRS})
else()
	message("INMOST was not found!")
endif()

set(PETSC_PATH $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})

set(PETSC_LIBRARIES ${PETSC_PATH}/lib/libpetsc.so )
set(PETSC_INCLUDES ${PETSC_PATH}/include $ENV{PETSC_DIR}/include/ )

include_directories(${PETSC_INCLUDES})

add_subdirectory(src)
