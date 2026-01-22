cmake_minimum_required(VERSION 2.8)

include_directories($ENV{PETSC_DIR}/src/mat/impls/aij/seq/)
include_directories($ENV{PETSC_DIR}/src/mat/impls/aij/mpi/)
include_directories($ENV{PETSC_DIR}/include/petsc/private/)
include_directories($ENV{PETSC_DIR}/src/vec/vec/impls/)
#include_directories(/home/alchem/Sources/ValgrindVerrou/include/)

include_directories(/usr/include/eigen3/)

add_executable(main main.cpp 
                    matrix_generation.cpp 
                    rhs_generation.cpp 
                    boundary_conditions.cpp 
                    make_adjacent.cpp 
                    common.cpp
                    submatrix_manager.cpp
                    Sparse_alpha.cpp
                    SPAI_api.cpp
                    stack_matrix.cpp
                    Notay_transform.cpp
                    Notay_transform_test.cpp
                    #simple_interp_prolongation.cpp
                    mesh_coarsen.cpp
                    SPAI_prolongation.cpp
                    #mesh_coarsen_improved.cpp
                    #simple_interp_prolong_improved.cpp
                    #simple_interp_prolong_kuzmin.cpp
                    prolongx.cpp
                    make_ids.cpp
                    bounding_box.cpp
                    decompose_matrix.cpp
                    mark_mesh.cpp
                    parse_config.cpp
                    prepare_local_cell_to_face.cpp
                    prepare_solver_multigrid.cpp
                    prepare_solver_multilevel.cpp
                    solver_tree.cpp
                    prepare_solver__.cpp
                    convergence_tests.cpp
                    mesh_coarsen_global_coordinates.cpp
                    interp_prolong_distance_scaled.cpp
                    petsc_matrix_write.cpp
                    dimensions.cpp
)
target_link_libraries(main ${PETSC_LIBRARIES} ${INMOST_LIBRARIES} ${INMOST_LINK_FLAGS} stdc++fs)
