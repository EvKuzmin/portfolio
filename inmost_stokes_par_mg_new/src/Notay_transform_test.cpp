#include"Notay_transform_test.hpp"
#include<iostream>

PetscErrorCode convergence_test_notay_transform_test(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx)
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

void notay_transform_test(Mat A_transformed, 
                          Vec b_wo_test, 
                          Mat Apu_Auu_SPAI, 
                          Mat Auu_SPAI_Aup, 
                          IS Au_IS,
                          IS Ap_IS,
                          Mat Awo, 
                          Vec b_wo_)
{
    Vec b_wo_test_u, b_wo_test_p;

    VecGetSubVector(b_wo_test, Au_IS, &b_wo_test_u);
    VecGetSubVector(b_wo_test, Ap_IS, &b_wo_test_p);

    Vec Apu_Auu_SPAI_b_wo_test_u;

    MatCreateVecs(Apu_Auu_SPAI, nullptr, &Apu_Auu_SPAI_b_wo_test_u);

    MatMult( Apu_Auu_SPAI, b_wo_test_u, Apu_Auu_SPAI_b_wo_test_u );
    VecAXPY( b_wo_test_p, -1.0, Apu_Auu_SPAI_b_wo_test_u );

    VecRestoreSubVector(b_wo_test, Au_IS, &b_wo_test_u);
    VecRestoreSubVector(b_wo_test, Ap_IS, &b_wo_test_p);



    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
    PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-19");
    PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
    // PetscOptionsSetValue(NULL, "-mat_mumps_cntl_2", "1.0e-15");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
    // PetscOptionsSetValue(NULL, "-pc_mg_levels", "2");
    // PetscOptionsSetValue(NULL, "-pc_type", "none");
    PetscOptionsSetValue(NULL, "-pc_type", "bjacobi");
    PetscOptionsSetValue(NULL, "-sub_pc_type", "eisenstat");
    // PetscOptionsSetValue(NULL, "-pc_type", "hypre");
    // PetscOptionsSetValue(NULL, "-pc_hypre_type", "boomeramg");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_stong_threshold", "0.2");
    // PetscOptionsSetValue(NULL, "-pc_type", "parms");
    // PetscOptionsSetValue(NULL, "-pc_parms_global", "schur");
    // PetscOptionsSetValue(NULL, "-pc_parms_local", "ilut");//"ilu0");
    // PetscOptionsSetValue(NULL, "-pc_parms_levels", "25");
    // PetscOptionsSetValue(NULL, "-pc_parms_blocksize", "100");
    // PetscOptionsSetValue(NULL, "-pc_parms_ind_tol", "1.0");
    // PetscOptionsSetValue(NULL, "-pc_parms_solve_tol", "1e-6");
    // PetscOptionsSetValue(NULL, "-pc_parms_max_dim", "10000");
    // PetscOptionsSetValue(NULL, "-pc_parms_max_it", "10000");
    // PetscOptionsSetValue(NULL, "-pc_parms_lfil_ilu_arms", "100");
    // PetscOptionsSetValue(NULL, "-pc_parms_lfil_schur", "100");
    // PetscOptionsSetValue(NULL, "-pc_parms_lfil_ilut_L_U", "100");
    // PetscOptionsSetValue(NULL, "-pc_parms_inter_nonsymmetric_perm", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_inter_column_perm", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_inter_row_scaling", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_inter_column_scaling", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_last_nonsymmetric_perm", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_last_column_perm", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_last_row_scaling", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_last_column_scaling", "True");
    // PetscOptionsSetValue(NULL, "-pc_parms_droptol_factors", "0.001");
    // PetscOptionsSetValue(NULL, "-pc_parms_droptol_schur_compl", "0.001");
    // PetscOptionsSetValue(NULL, "-pc_parms_droptol_last_schur", "0.001");
    PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_print_statistics", "");
    PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_coarsen_type", "modifiedRuge-Stueben");
    PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "2500");
    PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+e-mm");
    PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_nl", "1");
    PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_truncfactor", "0.5");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+i-mm");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_relax_type_down", "l1-Gauss-Seidel");//l1-Gauss-Seidel");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_relax_type_up", "l1-Gauss-Seidel");//l1-Gauss-Seidel");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_min_coarse_size", "50");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_relax_type_coarse", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_cycle_type", "W");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_grid_sweeps_down", "2");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_grid_sweeps_up", "2");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_grid_sweeps_coarse", "2");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_nl", "2");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_num_paths", "2");
    // PetscOptionsSetValue(NULL, "-ksp_constant_null_space", "");
    PetscOptionsSetValue(NULL, "-ksp_monitor", "");



    Vec x_wo_test;
    MatCreateVecs(A_transformed, &x_wo_test, nullptr);

    KSP ksp_wo_test;

    KSPCreate( MPI_COMM_WORLD, &ksp_wo_test );
    KSPSetOperators(ksp_wo_test,A_transformed,A_transformed);
    KSPSetType(ksp_wo_test,KSPGMRES);

    // KSPGetPC(ksp, &pc);
    // PCSetType(pc, PCSHELL);
    KSPSetConvergenceTest(ksp_wo_test, convergence_test_notay_transform_test, nullptr, nullptr);
    // PCShellSetApply(pc, precond_Full_apply);
    // PCShellSetApply(pc, precond_Aup_apply);
    // PCShellSetContext(pc, ctx_full);
    // PCSetUp(pc);
    KSPSetFromOptions(ksp_wo_test);
    // KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);
    KSPSetErrorIfNotConverged(ksp_wo_test, PETSC_TRUE);

    // MatNullSpaceRemove(nullspace_wo_test, b_wo_test);
    KSPSolve(ksp_wo_test,b_wo_test,x_wo_test);


    

    Vec Auu_SPAI_Aup_x;

    MatCreateVecs(Auu_SPAI_Aup, nullptr, &Auu_SPAI_Aup_x);
    

    Vec x_wo_u, x_wo_p;
    VecGetSubVector(x_wo_test, Au_IS, &x_wo_u);
    VecGetSubVector(x_wo_test, Ap_IS, &x_wo_p);

    MatMult( Auu_SPAI_Aup, x_wo_p, Auu_SPAI_Aup_x );
    VecAXPY( x_wo_u, -1.0, Auu_SPAI_Aup_x );

    VecRestoreSubVector(x_wo_test, Ap_IS, &x_wo_p);
    VecRestoreSubVector(x_wo_test, Au_IS, &x_wo_u);
    
    double x_wo_test_residual = 0.0;

    Vec wo_x;

    MatCreateVecs(Awo, nullptr, &wo_x);

    MatMult( Awo, x_wo_test, wo_x );

    VecAXPY( wo_x, -1.0, b_wo_ );

    VecNorm( wo_x, NORM_2, &x_wo_test_residual );
    std::cout << "residual = " << x_wo_test_residual << "\n";

    // Mat A_full = petsc_stack_v(Aff_Afc, Acf_Acc, 17);

    // MatView( A_transformed, PETSC_VIEWER_STDOUT_WORLD );
    // write_func_immediate(A_full);
    MPI_Barrier(MPI_COMM_WORLD);
    std::exit(0);
}

void execute_notay_transform_test(Mat A_transformed, Mat Apu_Auu_SPAI, Mat Auu_SPAI_Aup, std::map< std::string, Mat > &submatrices, std::map< std::string, IS > &index_sequences, SubmatrixManager &submatrix_map, Vec rhs)
{

    Vec b_wo_test;
    
    MatCreateVecs(A_transformed, &b_wo_test, nullptr);
    Vec b_wo_sub;
    VecGetSubVector(rhs, index_sequences.at("without_dirichlet"), &b_wo_sub);

    VecCopy(b_wo_sub, b_wo_test);

    VecRestoreSubVector(rhs, index_sequences.at("without_dirichlet"), &b_wo_sub);

    MatNullSpace nullspace_wo_test;
    MatGetNullSpace(A_transformed, &nullspace_wo_test);
    MatNullSpaceRemove(nullspace_wo_test, b_wo_test);

    Vec b_wo_;
    VecGetSubVector(rhs, index_sequences.at("without_dirichlet"), &b_wo_);

    notay_transform_test(A_transformed, 
                         b_wo_test, 
                         Apu_Auu_SPAI, 
                         Auu_SPAI_Aup, 
                         index_sequences.at("Au"),
                         index_sequences.at("Ap"),
                         submatrices.at("without_dirichlet"), 
                         b_wo_);

    VecRestoreSubVector(rhs, index_sequences.at("without_dirichlet"), &b_wo_);

    Vec b_wo_test_u, b_wo_test_p;

    VecGetSubVector(b_wo_test, index_sequences.at("Au"), &b_wo_test_u);
    VecGetSubVector(b_wo_test, index_sequences.at("Ap"), &b_wo_test_p);

    Vec Apu_Auu_SPAI_b_wo_test_u;

    MatCreateVecs(Apu_Auu_SPAI, nullptr, &Apu_Auu_SPAI_b_wo_test_u);

    MatMult( Apu_Auu_SPAI, b_wo_test_u, Apu_Auu_SPAI_b_wo_test_u );
    VecAXPY( b_wo_test_p, -1.0, Apu_Auu_SPAI_b_wo_test_u );

    VecRestoreSubVector(b_wo_test, index_sequences.at("Au"), &b_wo_test_u);
    VecRestoreSubVector(b_wo_test, index_sequences.at("Ap"), &b_wo_test_p);
}