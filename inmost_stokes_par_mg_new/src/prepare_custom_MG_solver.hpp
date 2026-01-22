#pragma once

#include "petsc.h"

#include <map>
#include <string>
#include <iostream>

#include "mesh_coarsen_improved.hpp"
#include "simple_interp_prolong_improved.hpp"
// #include "simple_interp_prolong_kuzmin.hpp"
#include"stack_matrix.hpp"

PetscErrorCode convergence_test_MG(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx)
{
    *reason = KSP_CONVERGED_ITERATING;

    if (it > 3) {
        *reason = KSP_CONVERGED_ITS;
    }

    return 0;
}

PetscErrorCode convergence_test_full(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx)
{
    *reason = KSP_CONVERGED_ITERATING;
    Mat A, P;
    KSPGetOperators(ksp, &A, &P);
    Vec rhs;
    KSPGetRhs(ksp, &rhs);
    Vec x;
    // KSPGetSolution(ksp, &x);
    KSPBuildSolution(ksp, NULL, &x);
    Vec Ax;
    VecDuplicate(rhs, &Ax);
    VecSet(Ax, 0.0);
    MatMult(A, x, Ax);
    
    // VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    VecAXPY(Ax, -1.0, rhs);

    double residual_Ax = 0.0;

    VecNorm(Ax, NORM_2, &residual_Ax);

    double f_norm = 0.0;
    VecNorm(rhs, NORM_2, &f_norm);

    double relative_residual = 0.0;

    if( f_norm != 0.0 ) {
        relative_residual = residual_Ax / f_norm;
        std::cout << "it = " << it << " solve_matrix convergence_test_full f_norm = " << f_norm << "\n";
        std::cout << "it = " << it << " solve_matrix convergence_test_full relative = " << relative_residual << "\n";
        if (relative_residual < 1e-8)
        {
            *reason = KSP_CONVERGED_ITS;
        }
    }

    // if (residual_Ax < 1e-1)
    // {
    //     *reason = KSP_CONVERGED_ITS;
    // }

    if (it > 25000) {
        *reason = KSP_DIVERGED_ITS;
    }

    PC             pc;
    KSPGetPC(ksp, &pc);

    PCFailedReason pcreason;
    PCGetFailedReason(pc,&pcreason);

    if(pcreason != PC_NOERROR) {
        *reason = KSP_DIVERGED_ITS;
    }

    std::cout << "it = " << it << " solve_matrix convergence_test_full = " << residual_Ax << "\n";
    VecDestroy(&Ax);
    return 0;
}

auto prepare_solver_custom_MG(auto &submatrix_map, 
                       std::map< std::string, Mat > &submatrices, 
                       std::map< std::string, IS > &index_sequences, 
                       auto write_func_immediate, 
                       std::vector< std::tuple< double, double, double > > &cell_center_coordinates, 
                       std::vector< std::tuple< int, int, int, int, int, int > > &cell_to_face_local) {
    std::cout << "prepare_solver_multigrid\n";

    

    Vec x;

    MatCreateVecs(submatrices.at("without_dirichlet"), NULL, &x);

    // VecCreate(PETSC_COMM_WORLD, &x);
    // VecSetSizes(x, N, N);
    VecSetFromOptions(x);
    VecSet(x,0.0);

    Mat A_main_diag;

    A_main_diag = petsc_stack_main_diag(submatrices.at("Auu"), submatrices.at("App"), 9);

    int N_levels = 3;

    int face_unknowns_local[N_levels-1];
    int n_coarse_ex_sum[N_levels-1];
    int n_coarse[N_levels-1];
    Mat Prolong[N_levels-1], Restrict[N_levels-1];
    Mat Ac_main_diag[N_levels-1], Ac[N_levels-1];
    std::vector< std::tuple< double, double, double > > cell_center_coordinates_coarse[N_levels-1];
    std::vector< std::tuple< int, int, int, int, int, int > > cell_to_face_local_coarse[N_levels-1];

    int face_unknowns_local_wo;

    MatGetLocalSize(submatrices.at("Auu"), &face_unknowns_local_wo, nullptr);

    auto [coarse_local_index_wo, 
          fine_local_index_wo, 
          coarse_cell_center_coordinates, 
          coarse_cell_to_face_local,
          coarse_face_unknowns_local] = 
          mesh_coarsen_i(
            A_main_diag, 
            submatrices.at("App"), 
            face_unknowns_local_wo, 
            cell_center_coordinates, 
            cell_to_face_local,
            2);

    face_unknowns_local[0] = coarse_face_unknowns_local;
    cell_center_coordinates_coarse[0] = coarse_cell_center_coordinates;
    cell_to_face_local_coarse[0] = coarse_cell_to_face_local;

    auto [Prolong_loc, Restrict_loc, n_coarse_loc, N_coarse_loc, n_coarse_ex_sum_loc, n_ex_sum_loc] = simple_interp_prolong_improved( A_main_diag, face_unknowns_local_wo, coarse_local_index_wo );
    // auto [Prolong_loc, Restrict_loc, n_coarse_loc, N_coarse_loc, n_coarse_ex_sum_loc, n_ex_sum_loc] = simple_interp_prolong_kuzmin( A_main_diag, face_unknowns_local_wo, coarse_local_index_wo );
    n_coarse[0] = n_coarse_loc;
    n_coarse_ex_sum[0] = n_coarse_ex_sum_loc;
    Prolong[0] = Prolong_loc;
    Restrict[0] = Restrict_loc;
    MatMatMatMult(Restrict[0], A_main_diag, Prolong[0], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &(Ac_main_diag[0]));
    // int N_size_val=0;
    // MatGetSize(Ac_main_diag[0], &N_size_val, nullptr);
    // std::cout << N_size_val << "\n";
    // std::exit(0);

    // write_func_immediate(Restrict[0]);
    // write_func_immediate(Prolong[0]);
    
    int N_size_val=0;
    int n_size_val=0;
    MatGetSize(A_main_diag, &N_size_val, nullptr);
    MatGetLocalSize(A_main_diag, &n_size_val, nullptr);
    std::cout << "fine global elements = " << N_size_val << " local elements = " << n_size_val << "\n";
    MatGetSize(Ac_main_diag[0], &N_size_val, nullptr);
    MatGetLocalSize(Ac_main_diag[0], &n_size_val, nullptr);
    std::cout << "level " << 0 << " global elements = " << N_size_val << " local elements = " << n_size_val << "\n";

    int coarse_face_unknowns_local_ex_sum = 0;
    int coarse_face_unknowns_local_sum = 0;

    for(int i = 1 ; i < N_levels-1 ; i++) {
        std::vector<int> IS_vector(n_coarse[i-1] - face_unknowns_local[i-1]);
        std::iota(IS_vector.begin(), IS_vector.end(), face_unknowns_local[i-1]+n_coarse_ex_sum[i-1]);

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
            Ac_main_diag[i-1],
            IS_p,
            IS_p,
            MAT_INITIAL_MATRIX,
            &App);

        auto [coarse_local_index_wo, 
          fine_local_index_wo, 
          coarse_cell_center_coordinates, 
          coarse_cell_to_face_local,
          coarse_face_unknowns_local] = 
          mesh_coarsen_i(
            Ac_main_diag[i-1], 
            App, 
            face_unknowns_local[i-1], 
            cell_center_coordinates_coarse[i-1], 
            cell_to_face_local_coarse[i-1],
            1);

        face_unknowns_local[i] = coarse_face_unknowns_local;
        cell_center_coordinates_coarse[i] = coarse_cell_center_coordinates;
        cell_to_face_local_coarse[i] = coarse_cell_to_face_local;

        auto [Prolong_loc, Restrict_loc, n_coarse_loc, N_coarse_loc, n_coarse_ex_sum_loc, n_ex_sum_loc] = simple_interp_prolong_improved( Ac_main_diag[i-1], face_unknowns_local[i-1], coarse_local_index_wo );
        // auto [Prolong_loc, Restrict_loc, n_coarse_loc, N_coarse_loc, n_coarse_ex_sum_loc, n_ex_sum_loc] = simple_interp_prolong_kuzmin( Ac_main_diag[i-1], face_unknowns_local[i-1], coarse_local_index_wo );
        n_coarse[i] = n_coarse_loc;
        n_coarse_ex_sum[i] = n_coarse_ex_sum_loc;
        Prolong[i] = Prolong_loc;
        Restrict[i] = Restrict_loc;
        MatMatMatMult(Restrict[i], Ac_main_diag[i-1], Prolong[i], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &(Ac_main_diag[i]));
        N_size_val=0;
        n_size_val=0;
        MatGetSize(Ac_main_diag[i], &N_size_val, nullptr);
        MatGetLocalSize(Ac_main_diag[i], &n_size_val, nullptr);
        std::cout << "level " << i << " global elements = " << N_size_val << " local elements = " << n_size_val << "\n";
    }
    // std::exit(0);
    // MatMatMatMult(Restrict[0], submatrices.at("without_dirichlet"), Prolong[0], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &(Ac[0]));
    // for(int i = 1 ; i < N_levels-1 ; i++) {
    //     MatMatMatMult(Restrict[i], Ac[i-1], Prolong[i], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &(Ac[i]));
    // }


    KSP            ksp_lapl2;
    KSPCreate(PETSC_COMM_WORLD,&ksp_lapl2);

    PetscOptionsClear(NULL);
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
    PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-8");
    PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
    // PetscOptionsSetValue(NULL, "-pc_mg_levels", "6");
    // PetscOptionsSetValue(NULL, "-mat_mumps_cntl_2", "1.0e-15");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
    PetscOptionsSetValue(NULL, "-pc_type", "mg");
    // PetscOptionsSetValue(NULL, "-pc_gamg_agg_nsmooths", "3");
    // PetscOptionsSetValue(NULL, "-pc_gamg_type", "agg");
    PetscOptionsSetValue(NULL, "-pc_gamg_aggressive_coarsening", "0");
    // PetscOptionsSetValue(NULL, "-pc_gamg_aggressive_mis_k", "1");
    // PetscOptionsSetValue(NULL, "-pc_gamg_aggressive_square_graph", "false");
    // PetscOptionsSetValue(NULL, "-mat_coarsen_view","");
    PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
    PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "150");

    // PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "250");
    PetscOptionsSetValue(NULL, "-ksp_monitor", "");
    // PetscOptionsSetValue(NULL, "-ksp_view", "");

    KSPSetFromOptions(ksp_lapl2);
    
    KSPSetOperators(ksp_lapl2,submatrices.at("without_dirichlet"),submatrices.at("without_dirichlet"));
    KSPSetType(ksp_lapl2,KSPFGMRES);

    KSPSetConvergenceTest(ksp_lapl2, convergence_test_full, nullptr, nullptr);

    PC pc_lapl2;
    KSPGetPC(ksp_lapl2, &pc_lapl2);
    PCMGSetLevels(pc_lapl2, N_levels, NULL);

    for( int l = 0 ; l < N_levels-1 ; l++ ){
        PCMGSetInterpolation(pc_lapl2, l+1, Prolong[N_levels-2-l]);
    }

    Vec temp;
    MatCreateVecs(Prolong[N_levels-2], &temp, nullptr);
    PCMGSetX(pc_lapl2, 0, temp);
    MatCreateVecs(Prolong[N_levels-2], &temp, nullptr);
    PCMGSetRhs(pc_lapl2, 0, temp);
    // MatCreateVecs(Prolong[0], &temp, nullptr);
    // PCMGSetR(pc_lapl2, 0, temp);

    for( int l = 1 ; l < N_levels-1 ; l++ ){
        MatCreateVecs(Prolong[N_levels-2-l], &temp, nullptr);
        PCMGSetX(pc_lapl2, l, temp);
        MatCreateVecs(Prolong[N_levels-2-l], &temp, nullptr);
        PCMGSetRhs(pc_lapl2, l, temp);
        MatCreateVecs(Prolong[N_levels-2-l], &temp, nullptr);
        PCMGSetR(pc_lapl2, l, temp);
    }

    // MatCreateVecs(Prolong[4], nullptr, &temp);
    // PCMGSetX(pc_lapl2, 5, temp);
    // MatCreateVecs(Prolong[4], nullptr, &temp);
    // PCMGSetRhs(pc_lapl2, 5, temp);
    MatCreateVecs(Prolong[0], nullptr, &temp);
    PCMGSetR(pc_lapl2, N_levels-1, temp);



    KSP ksp_level;
    PC pc_level;
    PCMGGetSmoother(pc_lapl2,N_levels-1,&ksp_level);
    KSPSetOperators(ksp_level, submatrices.at("without_dirichlet"), submatrices.at("without_dirichlet"));
    KSPSetType(ksp_level,KSPFGMRES);
    // KSPSetTolerances(ksp_level, 0.4, 1e-5, 1000.0, 10);
    KSPSetConvergenceTest(ksp_level, convergence_test_MG, nullptr, nullptr);
    KSPGetPC(ksp_level, &pc_level);
    PCSetType(pc_level, PCBJACOBI);
    
    Mat mat_level, mat_level2;
    mat_level = submatrices.at("without_dirichlet");
    
    for( int l = N_levels - 2 ; l >= 0 ; l-- ){
        PCMGGetSmoother(pc_lapl2,l,&ksp_level);
        MatMatMatMult(Restrict[N_levels - 2 - l], mat_level, Prolong[N_levels - 2 - l], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &mat_level2);
        // write_func_immediate(mat_level2);
        // MPI_Barrier(MPI_COMM_WORLD);
        // std::exit(0);
        KSPSetOperators(ksp_level, mat_level2, mat_level2);
        KSPSetType(ksp_level,KSPFGMRES);
        KSPSetTolerances(ksp_level, 0.4, 1e-5, 1000.0, 10);
        KSPSetConvergenceTest(ksp_level, convergence_test_MG, nullptr, nullptr);
        KSPGetPC(ksp_level, &pc_level);
        PCSetType(pc_level, PCBJACOBI);
        mat_level = mat_level2;
    }
    
    return std::make_tuple( ksp_lapl2, x );
}