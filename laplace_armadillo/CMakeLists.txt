cmake_minimum_required(VERSION 3.10)

project(flashcount CXX)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

# Требуем C++20 (nbt.hpp использует concepts/bit_cast)
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ===== ИСПРАВЛЕНИЕ: Поиск Armadillo =====
# Вариант 1: Через pkg-config (более надежно)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ARMADILLO armadillo)

if (ARMADILLO_FOUND)
    message(STATUS "Armadillo найден через pkg-config")
    message(STATUS "  Include dirs: ${ARMADILLO_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${ARMADILLO_LIBRARIES}")
else()
    # Вариант 2: Fallback через стандартный FindArmadillo
    find_package(Armadillo QUIET)
    
    if (NOT Armadillo_FOUND)
        # Вариант 3: Явное указание путей
        find_path(ARMADILLO_INCLUDE_DIR
            NAMES armadillo
            PATHS /usr/include /usr/local/include
            PATH_SUFFIXES armadillo
        )
        
        find_library(ARMADILLO_LIBRARY
            NAMES armadillo
            PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
        )
        
        if (ARMADILLO_INCLUDE_DIR AND ARMADILLO_LIBRARY)
            set(Armadillo_FOUND TRUE)
            set(ARMADILLO_INCLUDE_DIRS ${ARMADILLO_INCLUDE_DIR})
            set(ARMADILLO_LIBRARIES ${ARMADILLO_LIBRARY})
            message(STATUS "Armadillo найден вручную")
        else()
            message(FATAL_ERROR "Armadillo не найден. Установите: sudo apt-get install libarmadillo-dev")
        endif()
    endif()
endif()

# Создание исполняемого файла
add_executable(main 
    src/main_laplace.cpp
    src/unsorted.cpp
)

# Локальные заголовки (src, src/include)
target_include_directories(main PRIVATE src)

# Линковка с Armadillo
if (ARMADILLO_FOUND OR Armadillo_FOUND)
    target_include_directories(main PRIVATE ${ARMADILLO_INCLUDE_DIRS} src)
    target_link_libraries(main PRIVATE ${ARMADILLO_LIBRARIES})
else()
    # Последняя попытка: связать напрямую
    target_link_libraries(main PRIVATE armadillo openblas lapack)
endif()

# Для старых GCC (<9)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        target_link_libraries(main PRIVATE stdc++fs)
        message(STATUS "GCC < 9: добавлена линковка с stdc++fs")
    endif()
endif()

# Предупреждение для старых компиляторов
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10)
    message(WARNING "GCC ${CMAKE_CXX_COMPILER_VERSION} может требовать -fconcepts для C++20")
endif()

# Вывод информации
message(STATUS "Проект: ${PROJECT_NAME}")
message(STATUS "Компилятор: ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Стандарт C++: ${CMAKE_CXX_STANDARD}")


