
#include<numeric>
#include "prepare_solver__.hpp"
// #include "solver_tree.hpp"
#include "prepare_solver_multilevel.hpp"
#include "prepare_solver_multigrid.hpp"

// #include "mesh_coarsen_global.hpp"
// #include "simple_interp_prolong_global.hpp"
#include "mesh_coarsen_global_coordinates.hpp"
#include "interp_prolong_distance_scaled.hpp"
#include "convergence_tests.hpp"
#include "decompose_matrix.hpp"
#include "petsc_matrix_write.hpp"
#include <variant>

using MapVariant = std::map< std::string, std::variant<int, double, bool, std::string, std::vector<double> > >;

// void prepare_solver(SolverTree *node) {
//     PetscOptionsClear(NULL);
//     PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
//     PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
//     PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
//     PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
//     for( auto [opt, val] : node->petsc_options ) {
//         PetscOptionsSetValue(NULL, opt.c_str(), val.c_str());
//     }

//     KSPCreate(PETSC_COMM_WORLD,&(node->ksp));

//     KSPSetOperators(node->ksp,node->A,node->A);
//     KSPSetType(node->ksp,node->solver_type.c_str());
//     PC pc;
//     KSPGetPC(node->ksp, &pc);
//     if(node->precond_type == "multilevel11") {
//         PCSetType(pc, PCSHELL);
//         MatNullSpace nullspace;
//         MatGetNullSpace(node->submatrices->at(
//                 std::any_cast<std::string>(node->properties.at("A22_name"))
//             ), &nullspace);
//         PrecondContextMultilevel *ctx = new PrecondContextMultilevel(
//             node->submatrices->at(
//                 std::any_cast<std::string>(node->properties.at("A11_name"))
//             ),
//             node->submatrices->at(
//                 std::any_cast<std::string>(node->properties.at("A12_name"))
//             ),
//             node->submatrices->at(
//                 std::any_cast<std::string>(node->properties.at("A21_name"))
//             ),
//             node->submatrices->at(
//                 std::any_cast<std::string>(node->properties.at("A22_name"))
//             ),
//             nullspace,
//             node->index_sequences->at(std::any_cast<std::string>(node->properties.at("IS1_name"))),
//             node->index_sequences->at(std::any_cast<std::string>(node->properties.at("IS2_name")))
//         );
//         PCShellSetApply(pc, precond_multilevel11_apply);
//         PCShellSetContext(pc, ctx);
//         ctx->kspA11 = node->child[0]->ksp;
//         ctx->kspA22 = node->child[1]->ksp;
//     } else if(node->precond_type == "multilevel22") {

//     } else if(node->precond_type == "multigrid") {

//     } else {
//         PCSetType(pc, node->precond_type.c_str());
//     }
// }

void filter_matrix(Mat A, int keep_N_elements) {
    Mat A_diag, A_off_diag;
    const int *A_garray;

    MatMPIAIJGetSeqAIJ(A, &A_diag, &A_off_diag, &A_garray);

    PetscInt N_A_diag;
    const PetscInt *ia_diag;
    const PetscInt *ja_diag;
    PetscScalar *data_diag;
    PetscBool success_diag;

    MatGetRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);
    MatSeqAIJGetArray(A_diag, &data_diag);

    PetscInt N_A_off_diag;
    const PetscInt *ia_off_diag;
    const PetscInt *ja_off_diag;
    PetscScalar *data_off_diag;
    PetscBool success_off_diag;

    MatGetRowIJ(A_off_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_off_diag, &ia_off_diag, &ja_off_diag, &success_off_diag);
    MatSeqAIJGetArray(A_off_diag, &data_off_diag);
    std::vector< std::pair<double, int> > row;
    row.reserve(30);
    for( int i = 0 ; i < N_A_diag ; i++ ) {
        for( int j = ia_diag[i] ; j < ia_diag[i+1] ; j++ ) {
            row.push_back( std::make_pair( data_diag[j], j ) );
        }
        std::sort( row.begin(), row.end(), [](const auto &a, const auto &b){ return a.first < b.first; } );
        if( row.size() > keep_N_elements ) {
            for( int j = 0 ; j < row.size() - keep_N_elements ; j++ ) {
                data_diag[ row[j].second ] = 0.0;
            }
        }
        row.resize(0);
    }

    // for( int i = 0 ; i < N_A_off_diag ; i++ ) {
    //     for( int j = ia_off_diag[i] ; j < ia_off_diag[i+1] ; j++ ) {
    //         row.push_back( std::make_pair( data_off_diag[j], j ) );
    //     }
    //     std::sort( row.begin(), row.end(), [](const auto &a, const auto &b){ return a.first < b.first; } );
    //     if( row.size() > 8 ) {
    //         for( int j = 0 ; j < row.size() - 8 ; j++ ) {
    //             data_off_diag[ row[j].second ] = 0.0;
    //         }
    //     }
    //     row.resize(0);
    // }
    MatSeqAIJRestoreArray(A_off_diag, &data_off_diag);
    MatRestoreRowIJ(A_off_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_off_diag, &ia_off_diag, &ja_off_diag, &success_off_diag);

    MatSeqAIJRestoreArray(A_diag, &data_diag);
    MatRestoreRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);
}

void prepare_nested_solver(SolverTree *node, KSP parent_ksp, PC parent_pc, std::string parent_pc_type)
{
    PetscOptionsClear(NULL);
    // PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
    // PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
    // PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
    // PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
    for( auto [opt, val] : node->petsc_options ) {
        PetscOptionsSetValue(NULL, opt.c_str(), val.c_str());
    }

    KSPCreate(PETSC_COMM_WORLD,&(node->ksp));

    Mat A;

    if(node->A != nullptr) {
        A = node->A;
    } else {
        A = node->submatrices->at(
            std::any_cast<std::string>(node->properties.at("A_name"))
        );
    }

    KSPSetOperators(node->ksp,A,A);
    KSPSetType(node->ksp,node->solver_type.c_str());
    KSPSetFromOptions(node->ksp);
    ConvergenceTestParameters *params = new ConvergenceTestParameters;
    params->i_max = std::any_cast<int>(node->properties.at("ksp_i_max"));
    params->relative_residual = std::any_cast<double>(node->properties.at("ksp_relative_residual"));
    params->absolute_residual = std::any_cast<double>(node->properties.at("ksp_absolute_residual"));
    params->matrix_name = std::any_cast<std::string>(node->properties.at("matrix_name"));
    params->indent = std::any_cast<std::string>(node->properties.at("indent"));
    params->view_enabled = std::any_cast<bool>(node->properties.at("view_enabled"));
    if( node->properties.count("error_if_over_i_max") ) {
        params->error_if_over_i_max = std::any_cast<bool>(node->properties.at("error_if_over_i_max"));
    } else {
        params->error_if_over_i_max = true;
    }
    
    KSPSetConvergenceTest(node->ksp, convergence_test_full, params, [](void *ctx){ ConvergenceTestParameters *params = (ConvergenceTestParameters *)ctx; delete params; return 0;});
    PC pc;
    KSPSetErrorIfNotConverged(node->ksp, PETSC_FALSE);
    KSPGetPC(node->ksp, &pc);
    prepare_pc(node, pc);

    // KSPView(node->ksp,PETSC_VIEWER_STDOUT_WORLD);
}

void prepare_pc(SolverTree *node, PC pc)
{
    auto write_func_immediate = [](Mat A){
        Mat A_diag, A_off_diag;
        const int *garray;

        MatMPIAIJGetSeqAIJ(A, &A_diag, &A_off_diag, &garray);
        int A_off_m, A_off_n, owner_range0, owner_range1;
        MatGetSize(A_off_diag, &A_off_m, &A_off_n);
        MatGetOwnershipRange(A_diag, &owner_range0, &owner_range1);

        // Mat_SeqAIJ *A_diag_raw_data = (Mat_SeqAIJ*)A_diag->data;
        // Mat_SeqAIJ *A_off_raw_data = (Mat_SeqAIJ*)A_off_diag->data;


        PetscInt N_A_diag;
        const PetscInt *ia_diag;
        const PetscInt *ja_diag;
        PetscScalar *data_diag;
        PetscBool success_diag;

        MatGetRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);
        MatSeqAIJGetArray(A_diag, &data_diag);

        PetscInt N_A_off_diag;
        const PetscInt *ia_off_diag;
        const PetscInt *ja_off_diag;
        PetscScalar *data_off_diag;
        PetscBool success_off_diag;

        MatGetRowIJ(A_off_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_off_diag, &ia_off_diag, &ja_off_diag, &success_off_diag);
        MatSeqAIJGetArray(A_off_diag, &data_off_diag);


        const double *A_diag_data, *A_off_data;
        MatSeqAIJGetArrayRead(A_diag,&A_diag_data);
        MatSeqAIJGetArrayRead(A_off_diag,&A_off_data);
        pmw::petsc_matrix_mpi_to_seq_readable( owner_range0, 
                                owner_range1, 
                                ia_diag, 
                                ja_diag, 
                                data_diag, 
                                ia_off_diag, 
                                ja_off_diag, 
                                data_off_diag, garray, A_off_n );
        // pmw::petsc_matrix_mpi_to_seq( owner_range0, 
        //                         owner_range1, 
        //                         ia_diag, 
        //                         ja_diag, 
        //                         data_diag, 
        //                         ia_off_diag, 
        //                         ja_off_diag, 
        //                         data_off_diag, garray, A_off_n );
        MatSeqAIJRestoreArrayRead(A_diag,&A_diag_data);
        MatSeqAIJRestoreArrayRead(A_off_diag,&A_off_data);

        // MatView(A, PETSC_VIEWER_STDOUT_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        std::exit(0);
    };
    if(node->precond_type == "multilevel11") {
        PCSetType(pc, PCSHELL);
        MatNullSpace nullspace = nullptr;
        // MatGetNullSpace(node->submatrices->at(
        //         std::any_cast<std::string>(node->properties.at("A22_name"))
        //     ), &nullspace);
        PrecondContextMultilevel *ctx = new PrecondContextMultilevel(
            node->submatrices->at(
                std::any_cast<std::string>(node->properties.at("A11_name"))
            ),
            node->submatrices->at(
                std::any_cast<std::string>(node->properties.at("A12_name"))
            ),
            node->submatrices->at(
                std::any_cast<std::string>(node->properties.at("A21_name"))
            ),
            node->submatrices->at(
                std::any_cast<std::string>(node->properties.at("A22_name"))
            ),
            nullspace,
            node->index_sequences->at(std::any_cast<std::string>(node->properties.at("IS1_name"))),
            node->index_sequences->at(std::any_cast<std::string>(node->properties.at("IS2_name")))
        );
        node->ctx = ctx;
        PCShellSetApply(pc, precond_multilevel11_apply);
        PCShellSetContext(pc, ctx);
        for( auto n : node->child ) {
            prepare_nested_solver( n, node->ksp, pc, node->precond_type );
        }
        ctx->ksp_main_block = node->child[0]->ksp;
        ctx->ksp_schur = node->child[1]->ksp;
    } else if(node->precond_type == "multilevel21") {
        PCSetType(pc, PCSHELL);
        MatNullSpace nullspace = nullptr;
        // MatGetNullSpace(node->submatrices->at(
        //         std::any_cast<std::string>(node->properties.at("A12_name"))
        //     ), &nullspace);
        PrecondContextMultilevel *ctx = new PrecondContextMultilevel(
            node->submatrices->at(
                std::any_cast<std::string>(node->properties.at("A11_name"))
            ),
            node->submatrices->at(
                std::any_cast<std::string>(node->properties.at("A12_name"))
            ),
            node->submatrices->at(
                std::any_cast<std::string>(node->properties.at("A21_name"))
            ),
            node->submatrices->at(
                std::any_cast<std::string>(node->properties.at("A22_name"))
            ),
            nullspace,
            node->index_sequences->at(std::any_cast<std::string>(node->properties.at("IS1_name"))),
            node->index_sequences->at(std::any_cast<std::string>(node->properties.at("IS2_name")))
        );
        node->ctx = ctx;
        PCShellSetApply(pc, precond_multilevel21_apply);
        PCShellSetContext(pc, ctx);
        for( auto n : node->child ) {
            prepare_nested_solver( n, node->ksp, pc, node->precond_type );
        }
        ctx->ksp_main_block = node->child[0]->ksp;
        ctx->ksp_schur = node->child[1]->ksp;
    } else if(node->precond_type == "multigrid") {
        std::vector< std::tuple< double, double, double > > *cell_center_coordinates = std::any_cast<std::vector< std::tuple< double, double, double > > *>(node->properties.at("cell_center_coordinates"));
        std::vector< std::tuple< int, int, int, int, int, int > > *cell_to_face_local = std::any_cast<std::vector< std::tuple< int, int, int, int, int, int > > *>(node->properties.at("cell_to_face_local"));
        std::map< std::string, std::tuple<Mat, Mat, int, int, int, int> > *cache = std::any_cast<std::map< std::string, std::tuple<Mat, Mat, int, int, int, int> > *>(node->properties.at("cache"));
        std::map< std::string, Mat > *cache_coarse = std::any_cast<std::map< std::string, Mat > *>(node->properties.at("cache_coarse"));
        int N_levels = std::any_cast< int >( node->properties.at("N_levels") );

        Mat A;
        if(node->A != nullptr) {
            A = node->A;
        } else {
            A = node->submatrices->at(
                std::any_cast<std::string>(node->properties.at("A_name"))
            );
        }
        Mat A_lapl = std::any_cast< Mat >( node->properties.at( "A_lapl" ) );
        int face_unknowns = std::any_cast< int >( node->properties.at( "face_unknowns" ) );
        int cell_unknowns = std::any_cast< int >( node->properties.at( "cell_unknowns" ) );

        std::string multigrid_cache_tag = std::any_cast< std::string >( node->properties.at( "multigrid_cache_tag" ) );

        std::vector<VariableOnMesh> variables = std::any_cast< std::vector<VariableOnMesh> >( node->properties.at( "variables" ) );
        std::vector<int> unknowns_local = std::any_cast< std::vector<int> >( node->properties.at( "unknowns_local" ) );
        int laplace_submatrix_pos = std::any_cast< int >( node->properties.at( "laplace_submatrix_pos" ) );

        auto ctx = new PrecondContextMultigrid;

        node->ctx = ctx;

        int face_unknowns_local[N_levels-1];
        int cell_unknowns_local[N_levels-1];
        int n_coarse_ex_sum[N_levels-1];
        int n_coarse[N_levels-1];
        std::vector< std::vector<int> > coarse_unknowns_local_array(N_levels-1);
        Mat Prolong[N_levels-1], Restrict[N_levels-1];
        // Mat Ac_main_diag[N_levels-1];
        Mat Ac[N_levels-1];
        std::vector< std::map< std::string, IS > > index_sequences_sub_list;
        std::vector< std::map< std::string, Mat > > submatrices_sub_list;
        std::vector< SubmatrixManager > submatrix_map_sub_list;

        std::vector< std::tuple< double, double, double > > cell_center_coordinates_coarse[N_levels-1];
        std::vector< std::tuple< double, double, double > > all_center_coordinates_coarse[N_levels-1];
        std::vector< std::tuple< int, int, int, int, int, int > > cell_to_face_local_coarse[N_levels-1];

        {

            // auto [coarse_local_index_wo, 
            //     fine_local_index_wo, 
            //     coarse_cell_center_coordinates, 
            //     coarse_cell_to_face_local, 
            //     coarse_face_unknowns_local,
            //     coarse_cell_unknowns_local,
            //     coarse_unknowns_local] = 
            //     mesh_coarsen_global(
            //         A, 
            //         A_lapl, 
            //         face_unknowns,
            //         cell_unknowns, 
            //         *cell_center_coordinates, 
            //         *cell_to_face_local,
            //         variables,
            //         unknowns_local,
            //         2);

            int first_level_coarse_dist = 1;

            if(std::any_cast< bool >( node->properties.at("level_1_agg_coarsening") )) {
                first_level_coarse_dist = 2;
            } else {
                first_level_coarse_dist = 1;
            }

            auto [coarse_local_index_wo, 
                fine_local_index_wo, 
                coarse_cell_center_coordinates, 
                coarse_all_center_coordinates,
                coarse_cell_to_face_local, 
                coarse_face_unknowns_local,
                coarse_cell_unknowns_local,
                coarse_unknowns_local] = 
                mesh_coarsen_global_coordinates(
                    A, 
                    A_lapl, 
                    face_unknowns,
                    cell_unknowns, 
                    *cell_center_coordinates, 
                    *cell_to_face_local,
                    variables,
                    unknowns_local,
                    first_level_coarse_dist);

            auto[unknowns_local_sub, matrix_size_local_sub, matrix_size_sum_sub, matrix_size_global_sub, submatrix_map_sub] = make_submatrix_map( coarse_cell_unknowns_local, coarse_face_unknowns_local, variables );

            std::cout << "matrix_size_global_sub_l0 = " << matrix_size_global_sub << "\n";

            std::map< std::string, Mat > submatrices_sub;
            std::map< std::string, IS > index_sequences_sub;

            if( node->properties.count("index_tree_list") ) {
                for( auto i_r : std::any_cast< std::vector<IndexTreeStruct *> >( node->properties.at("index_tree_list") ) ) {
                    recombine_index( i_r, submatrix_map_sub );
                    create_IS_tree( i_r, submatrix_map_sub, index_sequences_sub );
                }
            }
            
            index_sequences_sub_list.push_back(index_sequences_sub);
            submatrix_map_sub_list.push_back(submatrix_map_sub);
            // for( auto v : fine_local_index_wo ) {

            //     std::cout << "rank = " << rank << " coarse id = " << coarse_local_index_wo << "\n";
            // }
            // MPI_Barrier(MPI_COMM_WORLD);

            face_unknowns_local[0] = coarse_face_unknowns_local;
            cell_unknowns_local[0] = coarse_cell_unknowns_local;
            cell_center_coordinates_coarse[0] = coarse_cell_center_coordinates;
            all_center_coordinates_coarse[0] = coarse_all_center_coordinates;
            coarse_unknowns_local_array[0] = coarse_unknowns_local;
            cell_to_face_local_coarse[0] = coarse_cell_to_face_local;

            // auto [Prolong_loc, Restrict_loc, n_coarse_loc, N_coarse_loc, n_coarse_ex_sum_loc, n_ex_sum_loc] = simple_interp_prolong_global_cached( multigrid_cache_tag, A, coarse_local_index_wo, unknowns_local, *cache );
            auto [Prolong_loc, Restrict_loc, n_coarse_loc, N_coarse_loc, n_coarse_ex_sum_loc, n_ex_sum_loc] = interp_prolong_distance_scaled_cached( multigrid_cache_tag, A, coarse_local_index_wo, unknowns_local, *cache, *std::any_cast< std::vector< std::tuple< double, double, double > >* >(node->properties.at("all_center_coordinates")), coarse_all_center_coordinates );
            n_coarse[0] = n_coarse_loc;
            n_coarse_ex_sum[0] = n_coarse_ex_sum_loc;
            Prolong[0] = Prolong_loc;
            Restrict[0] = Restrict_loc;
            if(cache_coarse->count(multigrid_cache_tag + "Ac0")) {
                Ac[0] = cache_coarse->at(multigrid_cache_tag + "Ac0");
                MatMatMatMult(Restrict[0], A, Prolong[0], MAT_REUSE_MATRIX, PETSC_DEFAULT, &(Ac[0]));
            } else {
                MatMatMatMult(Restrict[0], A, Prolong[0], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &(Ac[0]));
                (*cache_coarse)[multigrid_cache_tag + "Ac0"] = Ac[0];
            }
            // MatMatMatMult(Restrict[0], A, Prolong[0], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &(Ac[0]));

            // write_func_immediate(Prolong[0]);
            // write_func_immediate(Ac[0]);
            // MPI_Barrier(MPI_COMM_WORLD);
            // std::exit(0);


            {
                    int m,n;
                    MatGetSize(Restrict[0], &m, &n);
                    std::vector<double> col_sums(n);
                    MatGetColumnSumsRealPart(Restrict[0], col_sums.data());
                    MPI_Barrier(MPI_COMM_WORLD);
                    for( auto vals : col_sums ) {

                        if(std::fabs(vals-1.0)>1e-7) {
                            std::cout << vals << std::endl;
                            std::exit(0);
                        }
                    }
                    // MPI_Barrier(MPI_COMM_WORLD);
                    // std::exit(0);
                }

            if( node->properties.count("nullspace_IS_name") )
            {
                IS nsp_IS = index_sequences_sub.at( std::any_cast< std::string >( node->properties.at( "nullspace_IS_name" ) ) );
                MatNullSpace nullspace_full;

                Vec nsp_full;
                double *nsp_raw;

                MatCreateVecs(Ac[0], NULL, &nsp_full);

                VecSet(nsp_full, 0.0);

                Vec p_sub;

                VecGetSubVector(nsp_full, nsp_IS, &p_sub);

                VecSet(p_sub, 1.0);

                VecRestoreSubVector(nsp_full, nsp_IS, &p_sub);

                // VecGetArray(nsp_full, &nsp_raw);

                // for( auto idx : submatrix_map.find_nodes("Ap")[0]->map_to_parent ){
                //     nsp_raw[idx] = 1.0;
                // }

                // VecRestoreArray(nsp_full, &nsp_raw);

                VecNormalize(nsp_full, nullptr);

                MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &nsp_full, &nullspace_full);
                MatSetNullSpace(Ac[0], nullspace_full);

                MatNullSpaceDestroy(&nullspace_full);
                VecDestroy(&nsp_full);
            }
            
            // ctx->matrices["Ac_0"] = Ac[0];
            if(node->child.size() > 0){
                for( auto i_r : std::any_cast< std::vector<SubmatrixTreeStruct *> >( node->properties.at("submatrix_tree_list") ) ) {
                    create_submatrices( i_r, Ac[0], submatrices_sub, index_sequences_sub );
                }
            }
            submatrices_sub_list.push_back(submatrices_sub);

            // func(Ac[0]);

            int coarse_face_unknowns_local_ex_sum = 0;
            int coarse_face_unknowns_local_sum = 0;

            int N_size_val=0;
            int n_size_val=0;
            MatGetSize(A, &N_size_val, nullptr);
            MatGetLocalSize(A, &n_size_val, nullptr);
            std::cout << "fine global elements = " << N_size_val << " local elements = " << n_size_val << "\n";
            MatGetSize(Ac[0], &N_size_val, nullptr);
            MatGetLocalSize(Ac[0], &n_size_val, nullptr);
            std::cout << "level " << 0 << " global elements = " << N_size_val << " local elements = " << n_size_val << "\n";

            for(int i = 1 ; i < N_levels-1 ; i++) {
                std::vector<int> IS_vector(cell_unknowns_local[i-1]);
                std::iota(IS_vector.begin(), IS_vector.end(), coarse_unknowns_local_array[i-1][laplace_submatrix_pos] + n_coarse_ex_sum[i-1]);

                

                IS IS_p;
                Mat App;

                ISCreateGeneral(
                    PETSC_COMM_WORLD, 
                    IS_vector.size(), 
                    IS_vector.data(), 
                    PETSC_COPY_VALUES, 
                    &IS_p
                );

                MatCreateSubMatrix(
                    Ac[i-1],
                    IS_p,
                    IS_p,
                    MAT_INITIAL_MATRIX,
                    &App);

                

                Mat App_prolong, App_prolong_2;

                MatMatMult(Restrict[i-1], Prolong[i-1], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &App_prolong);

                MatCreateSubMatrix(
                    App_prolong,
                    IS_p,
                    IS_p,
                    MAT_INITIAL_MATRIX,
                    &App_prolong_2);

                    // write_func_immediate(App_prolong_2);
                filter_matrix(App_prolong_2, std::any_cast<int>(node->properties.at("keep_N_elements")));
                MatFilter(App_prolong_2, 1.0e-8, PETSC_TRUE, PETSC_TRUE);
                // MatFilter(App_prolong, 0.1, PETSC_TRUE, PETSC_TRUE);
                
                ISDestroy(&IS_p);

                // auto [coarse_local_index_wo, 
                // fine_local_index_wo, 
                // coarse_cell_center_coordinates,
                // coarse_cell_to_face_local, 
                // coarse_face_unknowns_local,
                // coarse_cell_unknowns_local, 
                // coarse_unknowns_local] = 
                // mesh_coarsen_global(
                //     Ac[i-1], 
                //     App_prolong_2, 
                //     face_unknowns_local[i-1], 
                //     cell_unknowns_local[i-1], 
                //     cell_center_coordinates_coarse[i-1], 
                //     cell_to_face_local_coarse[i-1],
                //     variables,
                //     coarse_unknowns_local_array[i-1],
                //     1);

                auto [coarse_local_index_wo, 
                fine_local_index_wo, 
                coarse_cell_center_coordinates, 
                coarse_all_center_coordinates,
                coarse_cell_to_face_local, 
                coarse_face_unknowns_local,
                coarse_cell_unknowns_local,
                coarse_unknowns_local] = 
                mesh_coarsen_global_coordinates(
                    Ac[i-1], 
                    App_prolong_2, 
                    face_unknowns_local[i-1], 
                    cell_unknowns_local[i-1], 
                    cell_center_coordinates_coarse[i-1], 
                    cell_to_face_local_coarse[i-1],
                    variables,
                    coarse_unknowns_local_array[i-1],
                    1);

                auto[unknowns_local_sub, matrix_size_local_sub, matrix_size_sum_sub, matrix_size_global_sub, submatrix_map_sub] = make_submatrix_map( coarse_cell_unknowns_local, coarse_face_unknowns_local, variables );

                std::map< std::string, Mat > submatrices_sub;
                std::map< std::string, IS > index_sequences_sub;

                if( node->properties.count("index_tree_list") ) {
                    for( auto i_r : std::any_cast< std::vector<IndexTreeStruct *> >( node->properties.at("index_tree_list") ) ) {
                        recombine_index( i_r, submatrix_map_sub );
                        create_IS_tree( i_r, submatrix_map_sub, index_sequences_sub );
                    }
                }
            
                index_sequences_sub_list.push_back(index_sequences_sub);
                submatrix_map_sub_list.push_back(submatrix_map_sub);

                face_unknowns_local[i] = coarse_face_unknowns_local;
                cell_unknowns_local[i] = coarse_cell_unknowns_local;
                cell_center_coordinates_coarse[i] = coarse_cell_center_coordinates;
                all_center_coordinates_coarse[i] = coarse_all_center_coordinates;
                coarse_unknowns_local_array[i] = coarse_unknowns_local;
                cell_to_face_local_coarse[i] = coarse_cell_to_face_local;
                
                // auto [Prolong_loc, Restrict_loc, n_coarse_loc, N_coarse_loc, n_coarse_ex_sum_loc, n_ex_sum_loc] = simple_interp_prolong_global_cached( multigrid_cache_tag + "_" +std::to_string(i), Ac[i-1], coarse_local_index_wo, coarse_unknowns_local_array[i-1], *cache );
                // auto [Prolong_loc, Restrict_loc, n_coarse_loc, N_coarse_loc, n_coarse_ex_sum_loc, n_ex_sum_loc] = simple_interp_prolong_global_cached( multigrid_cache_tag + "_" +std::to_string(i), App_prolong, coarse_local_index_wo, coarse_unknowns_local_array[i-1], *cache );
                auto [Prolong_loc, Restrict_loc, n_coarse_loc, N_coarse_loc, n_coarse_ex_sum_loc, n_ex_sum_loc] = interp_prolong_distance_scaled_cached( multigrid_cache_tag + "_" +std::to_string(i), Ac[i-1], coarse_local_index_wo, coarse_unknowns_local_array[i-1], *cache, all_center_coordinates_coarse[i-1], all_center_coordinates_coarse[i] );
                n_coarse[i] = n_coarse_loc;
                n_coarse_ex_sum[i] = n_coarse_ex_sum_loc;
                Prolong[i] = Prolong_loc;
                Restrict[i] = Restrict_loc;
                // if(i==1)write_func_immediate(Prolong[1]);
                if(cache_coarse->count(multigrid_cache_tag + "Ac" + std::to_string(i))) {
                    Ac[i] = cache_coarse->at(multigrid_cache_tag + "Ac" + std::to_string(i));
                    MatMatMatMult(Restrict[i], Ac[i-1], Prolong[i], MAT_REUSE_MATRIX, PETSC_DEFAULT, &(Ac[i]));
                } else {
                    MatMatMatMult(Restrict[i], Ac[i-1], Prolong[i], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &(Ac[i]));
                    (*cache_coarse)[multigrid_cache_tag + "Ac" + std::to_string(i)] = Ac[i];
                }
                // MatMatMatMult(Restrict[i], Ac[i-1], Prolong[i], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &(Ac[i]));

                {
                    int m,n;
                    MatGetSize(Restrict[i], &m, &n);
                    std::vector<double> col_sums(n);
                    MatGetColumnSumsRealPart(Restrict[i], col_sums.data());
                    MPI_Barrier(MPI_COMM_WORLD);
                    for( auto vals : col_sums ) {

                        if(std::fabs(vals-1.0)>1e-7) {
                            std::cout << vals << std::endl;
                            std::exit(0);
                        }
                    }
                    // MPI_Barrier(MPI_COMM_WORLD);
                    // std::exit(0);
                }

                if( node->properties.count("nullspace_IS_name") )
                {
                    IS nsp_IS = index_sequences_sub.at( std::any_cast< std::string >( node->properties.at( "nullspace_IS_name" ) ) );
                    MatNullSpace nullspace_full;

                    Vec nsp_full;
                    double *nsp_raw;

                    MatCreateVecs(Ac[i], NULL, &nsp_full);

                    VecSet(nsp_full, 0.0);

                    Vec p_sub;

                    VecGetSubVector(nsp_full, nsp_IS, &p_sub);

                    VecSet(p_sub, 1.0);

                    VecRestoreSubVector(nsp_full, nsp_IS, &p_sub);

                    // VecGetArray(nsp_full, &nsp_raw);

                    // for( auto idx : submatrix_map.find_nodes("Ap")[0]->map_to_parent ){
                    //     nsp_raw[idx] = 1.0;
                    // }

                    // VecRestoreArray(nsp_full, &nsp_raw);

                    VecNormalize(nsp_full, nullptr);

                    // if( i == 3 ) {
                    //     VecView(nsp_full, PETSC_VIEWER_STDOUT_WORLD);
                    //     MPI_Barrier(MPI_COMM_WORLD);
                    //     std::exit(0);
                    // }

                    MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &nsp_full, &nullspace_full);
                    MatSetNullSpace(Ac[i], nullspace_full);

                    MatNullSpaceDestroy(&nullspace_full);
                    VecDestroy(&nsp_full);
                }
                // ctx->matrices["Ac_"+std::to_string(i)] = Ac[i];
                if(node->child.size() > 0){
                    for( auto i_r : std::any_cast< std::vector<SubmatrixTreeStruct *> >( node->properties.at("submatrix_tree_list") ) ) {
                        create_submatrices( i_r, Ac[i], submatrices_sub, index_sequences_sub );
                    }
                }
                submatrices_sub_list.push_back(submatrices_sub);
                // N_size_val=0;
                // n_size_val=0;
                // MatGetSize(Ac_main_diag[i], &N_size_val, nullptr);
                // MatGetLocalSize(Ac_main_diag[i], &n_size_val, nullptr);
                // std::cout << "level " << i << " global elements = " << N_size_val << " local elements = " << n_size_val << "\n";
                MatDestroy(&App);
                MatDestroy(&App_prolong);
                MatDestroy(&App_prolong_2);
                N_size_val=0;
                n_size_val=0;
                MatGetSize(Ac[i], &N_size_val, nullptr);
                MatGetLocalSize(Ac[i], &n_size_val, nullptr);
                std::cout << "level " << i << " global elements = " << N_size_val << " local elements = " << n_size_val << "\n";
            }
        }

        {
            

            PCSetType(pc, "mg");
            
            PCMGSetLevels(pc, N_levels, NULL);
            if( std::any_cast<std::string>(node->properties.at("cycle_type")) == "V" ) {
                PCMGSetCycleType(pc, PC_MG_CYCLE_V);
            } else if( std::any_cast<std::string>(node->properties.at("cycle_type")) == "W" ) {
                PCMGSetCycleType(pc, PC_MG_CYCLE_W);
            } else {
                std::cerr << "unknown cycle type " << std::any_cast<std::string>(node->properties.at("cycle_type")) << std::endl;

                std::exit(1);
            }

            for( int l = 0 ; l < N_levels-1 ; l++ ){
                PCMGSetInterpolation(pc, l+1, Prolong[N_levels-2-l]);
            }

            Vec temp;
            MatCreateVecs(Prolong[N_levels-2], &temp, nullptr);
            ctx->vectors["x_0"] = temp;
            PCMGSetX(pc, 0, temp);
            MatCreateVecs(Prolong[N_levels-2], &temp, nullptr);
            ctx->vectors["rhs_0"] = temp;
            PCMGSetRhs(pc, 0, temp);
            // MatCreateVecs(Prolong[0], &temp, nullptr);
            // PCMGSetR(pc_lapl2, 0, temp);

            for( int l = 1 ; l < N_levels-1 ; l++ ){
                MatCreateVecs(Prolong[N_levels-2-l], &temp, nullptr);
                ctx->vectors["x_" + std::to_string(l)] = temp;
                PCMGSetX(pc, l, temp);
                MatCreateVecs(Prolong[N_levels-2-l], &temp, nullptr);
                ctx->vectors["rhs_" + std::to_string(l)] = temp;
                PCMGSetRhs(pc, l, temp);
                MatCreateVecs(Prolong[N_levels-2-l], &temp, nullptr);
                ctx->vectors["r_" + std::to_string(l)] = temp;
                PCMGSetR(pc, l, temp);
            }

            // MatCreateVecs(Prolong[4], nullptr, &temp);
            // PCMGSetX(pc, 5, temp);
            // MatCreateVecs(Prolong[4], nullptr, &temp);
            // PCMGSetRhs(pc, 5, temp);
            MatCreateVecs(Prolong[0], nullptr, &temp);
            ctx->vectors["r_" + std::to_string(N_levels-1)] = temp;
            PCMGSetR(pc, N_levels-1, temp);


            KSP ksp_level;
            PC pc_level;
            PCMGGetSmoother(pc,N_levels-1,&ksp_level);
            KSPSetOperators(ksp_level, A, A);
            KSPSetInitialGuessNonzero(ksp_level,PETSC_TRUE);
            KSPSetType(ksp_level,std::any_cast< std::string >( node->properties.at( "coarse_ksp" ) ).c_str());
            KSPGMRESSetOrthogonalization(ksp_level,KSPGMRESModifiedGramSchmidtOrthogonalization);
            KSPSetTolerances(ksp_level, 0.4, 1e-5, 1000.0, 100);
            ConvergenceTestParameters *params = new ConvergenceTestParameters;
            // params->i_max = std::any_cast< int >( node->properties.at( "fine_smoother_i_max" ) );
            // KSPSetConvergenceTest(ksp_level, convergence_test_smoother, params, [](void *ctx){ ConvergenceTestParameters *params = (ConvergenceTestParameters *)ctx; delete params; return 0;});
            params->i_max = std::any_cast<int>(node->properties.at("fine_smoother_i_max"));
            params->relative_residual = 1e-5;
            params->absolute_residual = 1e-5;
            params->matrix_name = "smoother fine";
            params->indent = "   ";
            params->view_enabled = std::any_cast<bool>(node->properties.at("coarse_view_enabled"));
            params->error_if_over_i_max = false;
            //KSPSetConvergenceTest(ksp_level, convergence_test_full, params, [](void *ctx){ ConvergenceTestParameters *params = (ConvergenceTestParameters *)ctx; delete params; return 0;});
            if(params->view_enabled) {
                KSPSetConvergenceTest(ksp_level, convergence_test_full, params, [](void *ctx){ ConvergenceTestParameters *params = (ConvergenceTestParameters *)ctx; delete params; return 0;});
            } else {
                KSPSetConvergenceTest(ksp_level, convergence_test_smoother, params, [](void *ctx){ ConvergenceTestParameters *params = (ConvergenceTestParameters *)ctx; delete params; return 0;});
            }
            KSPGetPC(ksp_level, &pc_level);
            KSPSetErrorIfNotConverged(ksp_level, PETSC_FALSE);
            PCSetType(pc_level, PCBJACOBI);
            // PCSetType(pc_level, PCGASM);
            // PCSetType(pc_level, PCNONE);

            auto recursive_submatrix_apply = [](auto recursive_submatrix_apply, auto node, auto submatrix_map, auto submatrices_sub, auto index_sequences_sub, auto add_string) -> void {
                node->submatrix_map = submatrix_map;
                node->submatrices = submatrices_sub;
                node->index_sequences = index_sequences_sub;
                node->properties.at("matrix_name") = std::any_cast<std::string>(node->properties.at("matrix_name")) + add_string;
                for(auto c : node->child) {
                    recursive_submatrix_apply(recursive_submatrix_apply, c, submatrix_map, submatrices_sub, index_sequences_sub, add_string);
                }
            };

            for( auto n : node->child ) {
                n->A = A;
                // n->submatrix_map = &(submatrix_map_sub_list[0]);
                // n->submatrices = &(submatrices_sub_list[0]);
                // n->index_sequences = &(index_sequences_sub_list[0]);
                recursive_submatrix_apply(recursive_submatrix_apply, n, node->submatrix_map, node->submatrices, node->index_sequences, "_s_level_fine");
                prepare_pc( n, pc_level );
            }
            
            if(node->child.size() == 0){
                KSPSetUp(ksp_level);
                {
                    PetscInt n_local;
                    PetscInt first_local;
                    KSP *ksp;
                    PCBJacobiGetSubKSP(pc_level, &n_local, &first_local, &ksp);
                    for(int i = 0 ; i < n_local ; i++) {
                        PC pc_sub_level;
                        KSPGetPC(ksp[i], &pc_sub_level);
                        // PCSetType(pc_sub_level, PCSOR);
                        // PCSetUp(pc_sub_level);
                        // PCSetType(pc_sub_level, PCILU);
                        // PCSetType(pc_sub_level, PCSPAI);
                        PCSetType(pc_sub_level, PCEISENSTAT);
                        // PCSetUp(pc_sub_level);
                        // PCSORSetSymmetric( pc_sub_level, SOR_EISENSTAT );
                        // PCSetUp(pc_sub_level);
                        // PCEisenstatSetNoDiagonalScaling(pc_sub_level, PETSC_FALSE);
                        PCEisenstatSetOmega(pc_sub_level, std::any_cast<double>(node->properties.at("SOR_omega")));
                        // PCSetType(pc_sub_level, PCSOR);
                        // PCSORSetOmega(pc_sub_level, 1.0);
                        // PCSetType(pc_sub_level, PCILU);


                        // PC  pcksp,pccheb;
                        // KSP kspcheb;
                        // PCSetType(pc_sub_level, PCCOMPOSITE);
                        // PCCompositeSetType(pc_sub_level, PC_COMPOSITE_ADDITIVE);
                        // PCCompositeAddPCType(pc_sub_level, PCEISENSTAT);
                        // PCCompositeAddPCType(pc_sub_level, PCKSP);
                        // PCCompositeGetPC(pc_sub_level, 1, &pcksp);
                        // PCKSPGetKSP(pcksp, &kspcheb);
                        // KSPSetType(kspcheb, KSPFGMRES);
                        // KSPSetTolerances(kspcheb, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT, 1);
                        // KSPSetNormType(kspcheb, KSP_NORM_NONE);
                        // KSPSetConvergenceTest(kspcheb, KSPConvergedSkip, NULL, NULL);
                        // KSPGetPC(kspcheb, &pccheb);
                        // PCSetType(pccheb, PCILU);
                    }
                    // KSPSetUp(ksp[0]);
                }
            }
            
            for( int l = N_levels - 2 ; l >= 0 ; l-- ){
                PCMGGetSmoother(pc,l,&ksp_level);
                // write_func_immediate(mat_level2);
                // MPI_Barrier(MPI_COMM_WORLD);
                // std::exit(0);
                KSPSetOperators(ksp_level, Ac[N_levels - 2 - l], Ac[N_levels - 2 - l]);
                KSPSetInitialGuessNonzero( ksp_level, PETSC_TRUE );
                // if(l == 0) {
                    KSPSetType(ksp_level,std::any_cast< std::string >( node->properties.at( "coarse_ksp" ) ).c_str());
                // } else {
                    // KSPSetType(ksp_level,KSPFGMRES);
                // }
                // KSPSetType(ksp_level,KSPCHEBYSHEV);
                KSPGMRESSetOrthogonalization(ksp_level,KSPGMRESModifiedGramSchmidtOrthogonalization);
                KSPSetTolerances(ksp_level, 0.4, 1e-5, 1000.0, 100);
                params = new ConvergenceTestParameters;
                // params->i_max = std::any_cast< int >( node->properties.at( "coarse_smoother_i_max" ) );
                // KSPSetConvergenceTest(ksp_level, convergence_test_smoother, params, [](void *ctx){ ConvergenceTestParameters *params = (ConvergenceTestParameters *)ctx; delete params; return 0;});
                // if(N_levels - 2 - l == 1) {
                //     write_func_immediate(Ac[N_levels - 2 - l]);
                // }
                params->i_max = std::any_cast<int>(node->properties.at("coarse_smoother_i_max"));
                params->relative_residual = 1e-5;
                params->absolute_residual = 1e-5;
                params->matrix_name = "smoother coarse level " + std::to_string(N_levels - 2 - l);
                params->indent = "      ";
                for( int b = 0 ; b < N_levels - 2 - l ; b++) {
                    params->indent += "   ";
                }
                params->view_enabled = std::any_cast<bool>(node->properties.at("coarse_view_enabled"));
                params->error_if_over_i_max = false;
                if(params->view_enabled) {
                    KSPSetConvergenceTest(ksp_level, convergence_test_full, params, [](void *ctx){ ConvergenceTestParameters *params = (ConvergenceTestParameters *)ctx; delete params; return 0;});
                } else {
                    KSPSetConvergenceTest(ksp_level, convergence_test_smoother, params, [](void *ctx){ ConvergenceTestParameters *params = (ConvergenceTestParameters *)ctx; delete params; return 0;});
                }
                KSPGetPC(ksp_level, &pc_level);
                KSPSetErrorIfNotConverged(ksp_level, PETSC_FALSE);
                PCSetType(pc_level, PCBJACOBI);
                // PCSetType(pc_level, PCGASM);
                // PCSetType(pc_level, PCNONE);
                if(node->child.size() == 0){
                    if(N_levels - 2 - l < std::any_cast<int>(node->properties.at("first_N_smoothers_SOR"))) {
                        PCSetType(pc_level, PCBJACOBI);
                            KSPSetUp(ksp_level);
                            {
                                PetscInt n_local;
                                PetscInt first_local;
                                KSP *ksp;
                                PCBJacobiGetSubKSP(pc_level, &n_local, &first_local, &ksp);
                                for(int i = 0 ; i < n_local ; i++) {
                                    PC pc_sub_level;
                                    KSPGetPC(ksp[i], &pc_sub_level);
                                    
                                    // if( l < 0) {
                                    //     PCSetType(pc_sub_level, PCLU);
                                    //     PCFactorSetMatSolverType(pc_sub_level,MATSOLVERMUMPS);
                                    // } else {
                                        // PCSetType(pc_sub_level, PCSOR);
                                        // PCSetUp(pc_sub_level);
                                        PCSetType(pc_sub_level, PCEISENSTAT);
                                        // PCSetType(pc_sub_level, PCILU);
                                        // PCSetType(pc_sub_level, PCSPAI);
                                        // PCSetUp(pc_sub_level);
                                        // PCSORSetSymmetric( pc_sub_level, SOR_EISENSTAT );
                                        // PCSORSetSymmetric( pc_sub_level, SOR_FORWARD_SWEEP );
                                        // PCSORSetSymmetric( pc_sub_level, SOR_BACKWARD_SWEEP );
                                        // PCEisenstatSetNoDiagonalScaling(pc_sub_level, PETSC_FALSE);
                                        // PCSetUp(pc_sub_level);
                                        PCEisenstatSetOmega(pc_sub_level, std::any_cast<double>(node->properties.at("SOR_omega")));
                                        // PCSetType(pc_sub_level, PCSOR);
                                        // PCSORSetOmega(pc_sub_level, 1.0);
                                        // PCSetType(pc_sub_level, PCILU);
                                    // }
                                }
                                // KSPSetUp(ksp[0]);
                            }
                        
                    } else if( l == 0 ) {
                        PCSetType(pc_level, std::any_cast< std::string >( node->properties.at( "coarsest_pc" ) ).c_str());
                        params->i_max = std::any_cast<int>(node->properties.at("coarsest_pc_iters"));
                        // PCSetType(pc_level, PCLU);
                        // PCFactorSetMatSolverType(pc_level,MATSOLVERMUMPS);
                    } else {
                        PCSetType(pc_level, PCNONE);
                    }
                    if( l == 0 ) {
                        PCSetType(pc_level, std::any_cast< std::string >( node->properties.at( "coarsest_pc" ) ).c_str());
                        params->i_max = std::any_cast<int>(node->properties.at("coarsest_pc_iters"));
                        // PCSetType(pc_level, PCLU);
                        // PCFactorSetMatSolverType(pc_level,MATSOLVERMUMPS);
                    }
                }

                for( auto n : node->child ) {
                    n->A = Ac[N_levels - 2 - l];
                    // n->submatrix_map = &(submatrix_map_sub_list[N_levels - 2 - l]);
                    // n->submatrices = &(submatrices_sub_list[N_levels - 2 - l]);
                    // n->index_sequences = &(index_sequences_sub_list[N_levels - 2 - l]);
                    recursive_submatrix_apply(recursive_submatrix_apply, n, &(submatrix_map_sub_list[N_levels - 2 - l]), &(submatrices_sub_list[N_levels - 2 - l]), &(index_sequences_sub_list[N_levels - 2 - l]), "_s_level" + std::to_string(N_levels - 2 - l));
                    prepare_pc( n, pc_level );
                }
                
            }

            ctx->index_sequences_sub_list = index_sequences_sub_list;
            ctx->submatrices_sub_list = submatrices_sub_list;

            // for(int i = 0 ; i < N_levels-1 ; i++) {
            //     MatDestroy(&(Prolong[i]));
            // }
            // for(int i = 0 ; i < N_levels-1 ; i++) {
            //     MatDestroy(&(Restrict[i]));
            // }
        }
    } else {
        PCSetType(pc, node->precond_type.c_str());
    }
}