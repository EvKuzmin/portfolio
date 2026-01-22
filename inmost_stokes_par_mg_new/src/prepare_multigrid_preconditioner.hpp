#pragma once

#include "inmost.h"
#include "petsc.h"
#include <map>
#include <string>

#include "submatrix_manager.hpp"

#include "../src/ksp/pc/impls/gamg/gamg.h"

// #include "Aup_multigrid.hpp"
#include "Aup_multigrid_sequence.hpp"

PetscErrorCode convergence_test_smooth_full(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx)
{
    *reason = KSP_CONVERGED_ITERATING;
    // if(it > 1) {
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

        VecAXPY(Ax, -1.0, rhs);

        double residual_Ax = 0.0;

        VecNorm(Ax, NORM_2, &residual_Ax);

        double f_norm = 0.0;
        VecNorm(rhs, NORM_2, &f_norm);

        double relative_residual = 0.0;

        if( f_norm != 0.0 ) {
            relative_residual = residual_Ax / f_norm;
            // std::cout << "   it = " << it << " convergence_test_Aff f_norm = " << f_norm << "\n";
            // std::cout << "   it = " << it << " convergence_test_Aff relative = " << relative_residual << "\n";
            if (relative_residual < 1e-2)
            {
                std::cout << "   it = " << it << " convergence_test_smooth_full = " << residual_Ax << "\n";
                *reason = KSP_CONVERGED_ITS;
            }
        }

        if (it > 1) {
            std::cout << "   it = " << it << " convergence_test_smooth_full = " << residual_Ax << "\n";
            *reason = KSP_CONVERGED_ITS;
        }

        // if(res_history.size() > 10) {
        //     auto iter = res_history.rbegin();
        //     if(*iter != 0.0){
        //         if(std::fabs(*iter - *(iter+1)) / *iter < 1e-8 && std::fabs(*(iter+1) - *(iter+2)) / *(iter+1) < 1e-8 ) {
        //             if(relative_residual < 1e-5) {
        //                 *reason = KSP_CONVERGED_ITS;
        //             }
        //         }
        //     }
        // }
        // if (residual_Ax < 1e-8)
        // {
        //     *reason = KSP_CONVERGED_ITS;
        // }
        if (it > 10000) {
            *reason = KSP_DIVERGED_ITS;
        }

        // std::cout << "   it = " << it << " convergence_test_Aff = " << residual_Ax << "\n";
        VecDestroy(&Ax);
    // }
    return 0;
}

PetscErrorCode convergence_test_Aff(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx)
{
    *reason = KSP_CONVERGED_ITERATING;
    // if(it > 1) {
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

        VecAXPY(Ax, -1.0, rhs);

        double residual_Ax = 0.0;

        VecNorm(Ax, NORM_2, &residual_Ax);

        double f_norm = 0.0;
        VecNorm(rhs, NORM_2, &f_norm);

        double relative_residual = 0.0;

        if( f_norm != 0.0 ) {
            relative_residual = residual_Ax / f_norm;
            // std::cout << "   it = " << it << " convergence_test_Aff f_norm = " << f_norm << "\n";
            // std::cout << "   it = " << it << " convergence_test_Aff relative = " << relative_residual << "\n";
            if (relative_residual < 1e-7)
            {
                std::cout << "   it = " << it << " convergence_test_Aff = " << residual_Ax << "\n";
                *reason = KSP_CONVERGED_ITS;
            }
        }

        if (it > 100) {
            std::cout << "   it = " << it << " convergence_test_Aff = " << residual_Ax << "\n";
            *reason = KSP_CONVERGED_ITS;
        }

        // if(res_history.size() > 10) {
        //     auto iter = res_history.rbegin();
        //     if(*iter != 0.0){
        //         if(std::fabs(*iter - *(iter+1)) / *iter < 1e-8 && std::fabs(*(iter+1) - *(iter+2)) / *(iter+1) < 1e-8 ) {
        //             if(relative_residual < 1e-5) {
        //                 *reason = KSP_CONVERGED_ITS;
        //             }
        //         }
        //     }
        // }
        // if (residual_Ax < 1e-8)
        // {
        //     *reason = KSP_CONVERGED_ITS;
        // }
        if (it > 10000) {
            *reason = KSP_DIVERGED_ITS;
        }

        // std::cout << "   it = " << it << " convergence_test_Aff = " << residual_Ax << "\n";
        VecDestroy(&Ax);
    // }
    return 0;
}

PetscErrorCode convergence_test_Ac(KSP ksp, int it, double rnorm, KSPConvergedReason *reason, void *mctx)
{
    *reason = KSP_CONVERGED_ITERATING;
    // if(it > 1) {
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

        VecAXPY(Ax, -1.0, rhs);

        double residual_Ax = 0.0;

        VecNorm(Ax, NORM_2, &residual_Ax);

        double f_norm = 0.0;
        VecNorm(rhs, NORM_2, &f_norm);

        double relative_residual = 0.0;

        if( f_norm != 0.0 ) {
            relative_residual = residual_Ax / f_norm;
            // std::cout << "   it = " << it << " convergence_test_Ac f_norm = " << f_norm << "\n";
            // std::cout << "   it = " << it << " convergence_test_Ac relative = " << relative_residual << "\n";
            if (relative_residual < 1e-7)
            {
                // VecView(x, PETSC_VIEWER_STDOUT_WORLD);
                std::cout << "   it = " << it << " convergence_test_Ac = " << residual_Ax << "\n";
                *reason = KSP_CONVERGED_ITS;
            }
        }

        if (it > 1000) {
            std::cout << "   it = " << it << " convergence_test_Ac = " << residual_Ax << "\n";
            *reason = KSP_CONVERGED_ITS;
        }

        // if(res_history.size() > 10) {
        //     auto iter = res_history.rbegin();
        //     if(*iter != 0.0){
        //         if(std::fabs(*iter - *(iter+1)) / *iter < 1e-8 && std::fabs(*(iter+1) - *(iter+2)) / *(iter+1) < 1e-8 ) {
        //             if(relative_residual < 1e-5) {
        //                 *reason = KSP_CONVERGED_ITS;
        //             }
        //         }
        //     }
        // }
        // if (residual_Ax < 1e-8)
        // {
        //     *reason = KSP_CONVERGED_ITS;
        // }
        if (it > 10000) {
            *reason = KSP_DIVERGED_ITS;
        }

        // std::cout << "   it = " << it << " convergence_test_Ac = " << residual_Ax << "\n";
        VecDestroy(&Ax);
    // }
    return 0;
}

// auto prepare_multigrid_preconditioner_context(
//     std::map<std::string, Mat> &submatrices,
//     std::map<std::string, IS> &index_sequences,
//     SubmatrixManager& submatrix_manager)
// {
//     Mat A_lapl;

//     MatMatMult( submatrices.at("Apu"), submatrices.at("Aup"), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A_lapl );

//     int A_lapl_N1, A_lapl_N2, A_full_N1, A_full_N2;

//     MatGetOwnershipRange(A_lapl, &A_lapl_N1, &A_lapl_N2);
//     MatGetOwnershipRange(submatrices.at("initial"), &A_full_N1, &A_full_N2);

//     MatCoarsen A_coarsen;
//     Mat a_Gmat;

//     MatCoarsenCreate(MPI_COMM_WORLD, &A_coarsen);
//     MatCoarsenSetType(A_coarsen, MATCOARSENMISK);
//     MatCoarsenSetMaximumIterations(A_coarsen, 100);
//     // MatCoarsenSetThreshold(A_coarsen, vfilter);   
//     MatCoarsenMISKSetDistance(A_coarsen, 2);
//     MatCoarsenSetStrictAggs(A_coarsen, PETSC_TRUE);
    

//     MatCreateGraph(A_lapl, PETSC_FALSE, PETSC_TRUE, -1, 0, nullptr, &a_Gmat);
//     MatCoarsenSetAdjacency(A_coarsen, a_Gmat);

//     MatCoarsenApply(A_coarsen);
//     PetscCoarsenData *agg_lists;
//     MatCoarsenGetData(A_coarsen, &agg_lists);
//     MatCoarsenDestroy(&A_coarsen);

//     int agg_lists_N = agg_lists->size;


//     std::vector<int> coarse_local_index;
//     std::vector<int> fine_local_index;

//     std::vector<char> all_index(A_full_N2 - A_full_N1, 0);

//     std::vector<int> gid_list;
//     // SubmatrixManager::Node *wo_node = submatrix_map.find_nodes("without_dirichlet")[0];
//     // int full_count = 0;
//     for( int i = 0 ; i < agg_lists_N ; i++ ) {
//         // full_count++;
//         PetscCDIntNd *item = agg_lists->array[i];
//         double x_avg = 0.0;
//         double y_avg = 0.0;
//         double z_avg = 0.0;
//         int count = 0;
//         bool near_interprocess = false;
//         if(item) {
//             while( item != NULL ) {
//                 Storage::real cnt1[3];
//                 if(item->gid < A_lapl_N2 & item->gid >= A_lapl_N1) {
//                     Cell c = Cell( m, cell_id_local_to_handle[ item->gid - A_lapl_N1] );
//                     // c.Integer(coarse_level) = full_count;
//                     c.Barycenter(cnt1);
//                     x_avg += cnt1[0];
//                     y_avg += cnt1[1];
//                     z_avg += cnt1[2];
//                     count++;
//                 } else {
//                     near_interprocess = true;
//                 }
//                 item = item->next;
//             }
//             x_avg /= count;
//             y_avg /= count;
//             z_avg /= count;
//         }
//         item = agg_lists->array[i];
//         double dist = 1e100;
//         int index = -1;
//         if(item && (count > 2 || near_interprocess)) {
//             while( item != NULL ) {
//                 Storage::real cnt1[3];
//                 if(item->gid < A_lapl_N2 & item->gid >= A_lapl_N1) {
//                     Cell c = Cell( m, cell_id_local_to_handle[ item->gid - A_lapl_N1] );
//                     c.Barycenter(cnt1);
//                     double dist_new = std::sqrt( std::pow(x_avg - cnt1[0], 2.0) + std::pow(y_avg - cnt1[1], 2.0) + std::pow(z_avg - cnt1[2], 2.0) );
//                     if( dist_new < dist ) {
//                         dist = dist_new;
//                         index = item->gid;
//                     }
//                 }
//                 item = item->next;
//             }
//             if(index != -1) {
//                 gid_list.push_back(index);
//             }
//         }
//     }

//     for( int i = 0 ; i < gid_list.size() ; i++ ) {
//         int gid = gid_list[i];
//         if(gid >= A_lapl_N2 || gid < A_lapl_N1) {
//             std::cerr << "rank = " << rank << " gid = " << gid << "\n";
//             std::abort();
//         }
//         Cell c = Cell( m, cell_id_local_to_handle[ gid - A_lapl_N1] );
//         c.Integer(coarse_level) = 1;
//         ElementArray<Face> faces = c.getFaces();
//         Face back = faces[0].getAsFace();
//         Face front = faces[1].getAsFace();
//         Face left = faces[2].getAsFace();
//         Face right = faces[3].getAsFace();
//         Face bottom = faces[4].getAsFace();
//         Face top = faces[5].getAsFace();
//         if( back.GetStatus() != Element::Ghost ) {
//             coarse_local_index.push_back( back.Integer( face_id_local ) );
//             all_index[ back.Integer( face_id_local ) ] = 1;
//         } else if (front.GetStatus() != Element::Ghost) {
//             coarse_local_index.push_back( front.Integer( face_id_local ) );
//             all_index[ front.Integer( face_id_local ) ] = 1;
//         }
//         if( left.GetStatus() != Element::Ghost ) {
//             coarse_local_index.push_back( left.Integer( face_id_local ) );
//             all_index[ left.Integer( face_id_local ) ] = 1;
//         } else if (right.GetStatus() != Element::Ghost) {
//             coarse_local_index.push_back( right.Integer( face_id_local ) );
//             all_index[ right.Integer( face_id_local ) ] = 1;
//         }
//         if( bottom.GetStatus() != Element::Ghost ) {
//             coarse_local_index.push_back( bottom.Integer( face_id_local ) );
//             all_index[ bottom.Integer( face_id_local ) ] = 1;
//         } else if (top.GetStatus() != Element::Ghost) {
//             coarse_local_index.push_back( top.Integer( face_id_local ) );
//             all_index[ top.Integer( face_id_local ) ] = 1;
//         }
//     }

    

//     int coarse_matrix_local_size_U = coarse_local_index.size();

//     for( int i = 0 ; i < gid_list.size() ; i++ ) {
//         int gid = gid_list[i];
        
//         if(gid >= A_lapl_N2 || gid < A_lapl_N1) {
//             std::cerr << "rank = " << rank << " gid = " << gid << "\n";
//             std::abort();
//         }
//         Cell c = Cell( m, cell_id_local_to_handle[ gid - A_lapl_N1] );
//         coarse_local_index.push_back( c.Integer( cell_id_local ) + face_unknowns_local );
//         all_index[ c.Integer( cell_id_local ) + face_unknowns_local ] = 1;
//     }

//     std::sort(coarse_local_index.begin(), coarse_local_index.end());

//     int coarse_matrix_local_size_P = coarse_local_index.size() - coarse_matrix_local_size_U;

//     for( int i = 0 ; i < all_index.size() ; i++ ) {
//         if(!all_index[i]) {
//             fine_local_index.push_back( i );
//         }
//     }
// }

auto prepare_multigrid_preconditioner_context(
    std::map<std::string, Mat> &submatrices,
    std::map<std::string, IS> &index_sequences)
{
    Mat Aff = submatrices.at("Aff");
    Mat Afc_P = submatrices.at("Afc_P");
    Mat Acf_C = submatrices.at("Acf_C");
    Mat Ac = submatrices.at("Ac");

    PrecondContextAup *ctx;
    ctx = new PrecondContextAup;

    ctx->Aup_ = submatrices.at("without_dirichlet");
    ctx->Aff = Aff;
    ctx->Afc_P = Afc_P;
    ctx->Acf_C = Acf_C;
    ctx->Ac = Ac;

    ctx->IS_U = index_sequences.at("Au");
    ctx->IS_P = index_sequences.at("Ap");

    ctx->IS_C_U = index_sequences.at("without_dirichlet_coarse_u");
    ctx->IS_C_P = index_sequences.at("without_dirichlet_coarse_p");

    ctx->IS_F = index_sequences.at("without_dirichlet_fine");
    ctx->IS_C = index_sequences.at("without_dirichlet_coarse");

    MatCreateVecs(Aff, NULL, &(ctx->Vf1));
    MatCreateVecs(Aff, NULL, &(ctx->Vf2));
    MatCreateVecs(Aff, NULL, &(ctx->Vf3));
    MatCreateVecs(Ac, NULL, &(ctx->Vc1));
    MatCreateVecs(Ac, NULL, &(ctx->Vc2));
    MatCreateVecs(Ac, NULL, &(ctx->Vc3));

    VecSetFromOptions(ctx->Vf1);
    VecSetFromOptions(ctx->Vf2);
    VecSetFromOptions(ctx->Vf3);
    VecSetFromOptions(ctx->Vc1);
    VecSetFromOptions(ctx->Vc2);
    VecSetFromOptions(ctx->Vc3);

    PetscOptionsClear(NULL);

    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
    PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-19");
    PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
    // PetscOptionsSetValue(NULL, "-pc_type", "none");
    PetscOptionsSetValue(NULL, "-pc_type", "bjacobi");
    // PetscOptionsSetValue(NULL, "-pc_type", "hypre");
    // PetscOptionsSetValue(NULL, "-pc_hypre_type", "boomeramg");
    PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_print_statistics", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_coarsen_type", "modifiedRuge-Stueben");
    PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "250");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+e-mm");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_nl", "1");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_truncfactor", "0.5");
    // PetscOptionsSetValue(NULL, "-ksp_monitor", "");

    KSPCreate(PETSC_COMM_WORLD, &(ctx->kspAff));
    KSPSetOperators(ctx->kspAff, ctx->Aff, ctx->Aff);
    KSPSetType(ctx->kspAff, KSPFGMRES);
    KSPSetErrorIfNotConverged(ctx->kspAff, PETSC_TRUE);
    KSPSetConvergenceTest(ctx->kspAff, convergence_test_Aff, nullptr, nullptr);
    KSPSetFromOptions(ctx->kspAff);
    KSPSetUp(ctx->kspAff);



    PetscOptionsClear(NULL);

    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
    PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-19");
    PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
    // PetscOptionsSetValue(NULL, "-pc_type", "lu");
    PetscOptionsSetValue(NULL, "-pc_type", "bjacobi");
    PetscOptionsSetValue(NULL, "-sub_pc_type", "eisenstat");
    // PetscOptionsSetValue(NULL, "-pc_type", "hypre");
    // PetscOptionsSetValue(NULL, "-pc_hypre_type", "boomeramg");
    PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_print_statistics", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_coarsen_type", "modifiedRuge-Stueben");
    PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "250");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+e-mm");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_nl", "1");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_truncfactor", "0.2");
    // PetscOptionsSetValue(NULL, "-ksp_monitor", "");
    // PetscOptionsSetValue(NULL, "-ksp_view", "");

    KSPCreate(PETSC_COMM_WORLD, &(ctx->kspAc));
    KSPSetOperators(ctx->kspAc, ctx->Ac, ctx->Ac);
    KSPSetType(ctx->kspAc, KSPFGMRES);
    KSPSetErrorIfNotConverged(ctx->kspAc, PETSC_TRUE);
    KSPSetConvergenceTest(ctx->kspAc, convergence_test_Ac, nullptr, nullptr);
    KSPSetFromOptions(ctx->kspAc);
    KSPSetUp(ctx->kspAc);

    

    PetscOptionsClear(NULL);

    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
    PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-19");
    PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
    PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
    PetscOptionsSetValue(NULL, "-pc_type", "bjacobi");
    PetscOptionsSetValue(NULL, "-sub_pc_type", "eisenstat");
    // PetscOptionsSetValue(NULL, "-pc_type", "lu");
    // PetscOptionsSetValue(NULL, "-pc_type", "hypre");
    // PetscOptionsSetValue(NULL, "-pc_hypre_type", "boomeramg");
    PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_print_statistics", "");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_coarsen_type", "modifiedRuge-Stueben");
    PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "250");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_interp_type", "ext+e-mm");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_agg_nl", "1");
    // PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_truncfactor", "0.2");
    // PetscOptionsSetValue(NULL, "-ksp_monitor", "");
    // PetscOptionsSetValue(NULL, "-ksp_view", "");

    KSPCreate(PETSC_COMM_WORLD, &(ctx->kspFull));
    KSPSetOperators(ctx->kspFull, ctx->Aup_, ctx->Aup_);
    KSPSetType(ctx->kspFull, KSPGMRES);
    KSPSetErrorIfNotConverged(ctx->kspFull, PETSC_TRUE);
    KSPSetConvergenceTest(ctx->kspFull, convergence_test_smooth_full, nullptr, nullptr);
    KSPSetFromOptions(ctx->kspFull);
    KSPSetUp(ctx->kspFull);

    return std::make_tuple(ctx);
}