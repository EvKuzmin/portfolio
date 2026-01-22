
#include"prepare_solver_multigrid.hpp"

#include<map>
#include<string>
#include<vector>
#include<numeric>

// #include "mesh_coarsen_global.hpp"
// #include "simple_interp_prolong_global.hpp"
#include "convergence_tests.hpp"

PrecondContextMultigrid::~PrecondContextMultigrid() {
    for(auto& [n, m] : matrices) {
        MatDestroy(&m);
    }
    for(auto& [n, v] : vectors) {
        VecDestroy(&v);
    }
    for(auto &m:index_sequences_sub_list ){
        for( auto &[name, is] : m ) {
            ISDestroy(&is);
        }
    }
    for(auto &m:submatrices_sub_list ){
        for( auto &[name, sub_m] : m ) {
            MatDestroy(&sub_m);
        }
    }
    matrices.clear();
    // std::cout << "PrecondContext memory free\n";
}

// std::tuple< Mat, Vec, Vec > prepare_matrix(int matrix_size_local, int matrix_size_sum, int matrix_size_global, int prealloc_diag, int prealloc_off_diag)
// {
//     Mat A;
//     MatCreate(PETSC_COMM_WORLD, &A);
//     MatSetSizes(A, matrix_size_local, matrix_size_local, matrix_size_global, matrix_size_global);
    
//     MatSetFromOptions(A);

//     MatMPIAIJSetPreallocation(A, prealloc_diag, NULL, prealloc_off_diag, NULL);
//     MatSetUp(A);

//     Vec rhs;

//     VecCreate(PETSC_COMM_WORLD, &rhs);
//     VecSetSizes(rhs, matrix_size_local, matrix_size_global);
//     VecSetFromOptions(rhs);

//     Vec            x;
//     VecDuplicate(rhs, &x);
//     VecSet(x,0.0);
//     return std::make_tuple( A, rhs, x );
// }

// void prepare_multigrid_preconditioner( int N_levels,
//                                        Mat A,
//                                        Mat A_lapl,
//                                        PC main_pc,
//                                        int face_unknowns,
//                                        int cell_unknowns,
//                                        std::vector< std::tuple< double, double, double > > &cell_center_coordinates, 
//                                        std::vector< std::tuple< int, int, int, int, int, int > > &cell_to_face_local,
//                                        std::vector<VariableOnMesh> &variables, 
//                                        std::vector<int> unknowns_local,
//                                        int laplace_submatrix_pos, // position of close to laplace submatrix block in unknowns_local
//                                        std::string multigrid_cache_tag,
//                                        std::map< std::string, std::tuple<Mat, Mat, int, int, int, int> > &cache ) {
//     int rank;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//     auto ctx = new PrecondContextMultigrid;

//     int face_unknowns_local[N_levels-1];
//     int cell_unknowns_local[N_levels-1];
//     int n_coarse_ex_sum[N_levels-1];
//     int n_coarse[N_levels-1];
//     std::vector< std::vector<int> > coarse_unknowns_local_array(N_levels-1);
//     Mat Prolong[N_levels-1], Restrict[N_levels-1];
//     // Mat Ac_main_diag[N_levels-1];
//     Mat Ac[N_levels-1];

//     std::vector< std::tuple< double, double, double > > cell_center_coordinates_coarse[N_levels-1];
//     std::vector< std::tuple< int, int, int, int, int, int > > cell_to_face_local_coarse[N_levels-1];

//     {

//         auto [coarse_local_index_wo, 
//             fine_local_index_wo, 
//             coarse_cell_center_coordinates, 
//             coarse_cell_to_face_local, 
//             coarse_face_unknowns_local,
//             coarse_cell_unknowns_local,
//             coarse_unknowns_local] = 
//             mesh_coarsen_global(
//                 A, 
//                 A_lapl, 
//                 face_unknowns,
//                 cell_unknowns, 
//                 cell_center_coordinates, 
//                 cell_to_face_local,
//                 variables,
//                 unknowns_local,
//                 2);

//         // for( auto v : fine_local_index_wo ) {

//         //     std::cout << "rank = " << rank << " coarse id = " << coarse_local_index_wo << "\n";
//         // }
//         // MPI_Barrier(MPI_COMM_WORLD);

//         face_unknowns_local[0] = coarse_face_unknowns_local;
//         cell_unknowns_local[0] = coarse_cell_unknowns_local;
//         cell_center_coordinates_coarse[0] = coarse_cell_center_coordinates;
//         coarse_unknowns_local_array[0] = coarse_unknowns_local;
//         cell_to_face_local_coarse[0] = coarse_cell_to_face_local;

//         auto [Prolong_loc, Restrict_loc, n_coarse_loc, N_coarse_loc, n_coarse_ex_sum_loc, n_ex_sum_loc] = simple_interp_prolong_global_cached( multigrid_cache_tag, A, coarse_local_index_wo, unknowns_local, cache );
//         n_coarse[0] = n_coarse_loc;
//         n_coarse_ex_sum[0] = n_coarse_ex_sum_loc;
//         Prolong[0] = Prolong_loc;
//         Restrict[0] = Restrict_loc;
//         MatMatMatMult(Restrict[0], A, Prolong[0], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &(Ac[0]));
//         ctx->matrices["Ac_0"] = Ac[0];

//         // func(Ac[0]);

//         int coarse_face_unknowns_local_ex_sum = 0;
//         int coarse_face_unknowns_local_sum = 0;

//         for(int i = 1 ; i < N_levels-1 ; i++) {
//             std::vector<int> IS_vector(cell_unknowns_local[i-1]);
//             std::iota(IS_vector.begin(), IS_vector.end(), coarse_unknowns_local_array[i-1][laplace_submatrix_pos] + n_coarse_ex_sum[i-1]);

//             IS IS_p;
//             Mat App;

//             ISCreateGeneral(
//                 PETSC_COMM_WORLD, 
//                 IS_vector.size(), 
//                 IS_vector.data(), 
//                 PETSC_COPY_VALUES, 
//                 &IS_p
//             );

//             MatCreateSubMatrix(
//                 Ac[i-1],
//                 IS_p,
//                 IS_p,
//                 MAT_INITIAL_MATRIX,
//                 &App);

//             ISDestroy(&IS_p);

//             auto [coarse_local_index_wo, 
//             fine_local_index_wo, 
//             coarse_cell_center_coordinates,
//             coarse_cell_to_face_local, 
//             coarse_face_unknowns_local,
//             coarse_cell_unknowns_local, 
//             coarse_unknowns_local] = 
//             mesh_coarsen_global(
//                 Ac[i-1], 
//                 App, 
//                 face_unknowns_local[i-1], 
//                 cell_unknowns_local[i-1], 
//                 cell_center_coordinates_coarse[i-1], 
//                 cell_to_face_local_coarse[i-1],
//                 variables,
//                 coarse_unknowns_local_array[i-1],
//                 1);

//             face_unknowns_local[i] = coarse_face_unknowns_local;
//             cell_unknowns_local[i] = coarse_cell_unknowns_local;
//             cell_center_coordinates_coarse[i] = coarse_cell_center_coordinates;
//             coarse_unknowns_local_array[i] = coarse_unknowns_local;
//             cell_to_face_local_coarse[i] = coarse_cell_to_face_local;
            

//             auto [Prolong_loc, Restrict_loc, n_coarse_loc, N_coarse_loc, n_coarse_ex_sum_loc, n_ex_sum_loc] = simple_interp_prolong_global_cached( multigrid_cache_tag + "_" +std::to_string(i), Ac[i-1], coarse_local_index_wo, coarse_unknowns_local_array[i-1], cache );
//             n_coarse[i] = n_coarse_loc;
//             n_coarse_ex_sum[i] = n_coarse_ex_sum_loc;
//             Prolong[i] = Prolong_loc;
//             Restrict[i] = Restrict_loc;
//             MatMatMatMult(Restrict[i], Ac[i-1], Prolong[i], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &(Ac[i]));
//             ctx->matrices["Ac_"+std::to_string(i)] = Ac[i];
//             // N_size_val=0;
//             // n_size_val=0;
//             // MatGetSize(Ac_main_diag[i], &N_size_val, nullptr);
//             // MatGetLocalSize(Ac_main_diag[i], &n_size_val, nullptr);
//             // std::cout << "level " << i << " global elements = " << N_size_val << " local elements = " << n_size_val << "\n";
//             MatDestroy(&App);
//         }
//     }

//     {
//         PCSetType(main_pc, "mg");
//         PCMGSetCycleType(main_pc, PC_MG_CYCLE_V);
//         PCMGSetLevels(main_pc, N_levels, NULL);

//         for( int l = 0 ; l < N_levels-1 ; l++ ){
//             PCMGSetInterpolation(main_pc, l+1, Prolong[N_levels-2-l]);
//         }

//         Vec temp;
//         MatCreateVecs(Prolong[N_levels-2], &temp, nullptr);
//         ctx->vectors["x_0"] = temp;
//         PCMGSetX(main_pc, 0, temp);
//         MatCreateVecs(Prolong[N_levels-2], &temp, nullptr);
//         ctx->vectors["rhs_0"] = temp;
//         PCMGSetRhs(main_pc, 0, temp);
//         // MatCreateVecs(Prolong[0], &temp, nullptr);
//         // PCMGSetR(pc_lapl2, 0, temp);

//         for( int l = 1 ; l < N_levels-1 ; l++ ){
//             MatCreateVecs(Prolong[N_levels-2-l], &temp, nullptr);
//             ctx->vectors["x_" + std::to_string(l)] = temp;
//             PCMGSetX(main_pc, l, temp);
//             MatCreateVecs(Prolong[N_levels-2-l], &temp, nullptr);
//             ctx->vectors["rhs_" + std::to_string(l)] = temp;
//             PCMGSetRhs(main_pc, l, temp);
//             MatCreateVecs(Prolong[N_levels-2-l], &temp, nullptr);
//             ctx->vectors["r_" + std::to_string(l)] = temp;
//             PCMGSetR(main_pc, l, temp);
//         }

//         // MatCreateVecs(Prolong[4], nullptr, &temp);
//         // PCMGSetX(main_pc, 5, temp);
//         // MatCreateVecs(Prolong[4], nullptr, &temp);
//         // PCMGSetRhs(main_pc, 5, temp);
//         MatCreateVecs(Prolong[0], nullptr, &temp);
//         ctx->vectors["r_" + std::to_string(N_levels-1)] = temp;
//         PCMGSetR(main_pc, N_levels-1, temp);


//         KSP ksp_level;
//         PC pc_level;
//         PCMGGetSmoother(main_pc,N_levels-1,&ksp_level);
//         KSPSetOperators(ksp_level, A, A);
//         KSPSetInitialGuessNonzero(ksp_level,PETSC_TRUE);
//         KSPSetType(ksp_level,KSPGMRES);
//         KSPGMRESSetOrthogonalization(ksp_level,KSPGMRESModifiedGramSchmidtOrthogonalization);
//         KSPSetTolerances(ksp_level, 0.4, 1e-5, 1000.0, 100);
//         ConvergenceTestParameters *params = new ConvergenceTestParameters;
//         params->i_max = 7;
//         KSPSetConvergenceTest(ksp_level, convergence_test_smoother, params, [](void *ctx){ ConvergenceTestParameters *params = (ConvergenceTestParameters *)ctx; delete params; return 0;});
//         KSPGetPC(ksp_level, &pc_level);
//         KSPSetErrorIfNotConverged(ksp_level, PETSC_FALSE);
//         PCSetType(pc_level, PCBJACOBI);
//         // PCSetType(pc_level, PCGASM);
//         // PCSetType(pc_level, PCNONE);
        
//         KSPSetUp(ksp_level);
//         {
//             PetscInt n_local;
//             PetscInt first_local;
//             KSP *ksp;
//             PCBJacobiGetSubKSP(pc_level, &n_local, &first_local, &ksp);
//             for(int i = 0 ; i < n_local ; i++) {
//                 PC pc_sub_level;
//                 KSPGetPC(ksp[i], &pc_sub_level);
//                 PCSetType(pc_sub_level, PCEISENSTAT);
//                 // PCSetType(pc_sub_level, PCILU);


//                 // PC  pcksp,pccheb;
//                 // KSP kspcheb;
//                 // PCSetType(pc_sub_level, PCCOMPOSITE);
//                 // PCCompositeSetType(pc_sub_level, PC_COMPOSITE_ADDITIVE);
//                 // PCCompositeAddPCType(pc_sub_level, PCEISENSTAT);
//                 // PCCompositeAddPCType(pc_sub_level, PCKSP);
//                 // PCCompositeGetPC(pc_sub_level, 1, &pcksp);
//                 // PCKSPGetKSP(pcksp, &kspcheb);
//                 // KSPSetType(kspcheb, KSPFGMRES);
//                 // KSPSetTolerances(kspcheb, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT, 1);
//                 // KSPSetNormType(kspcheb, KSP_NORM_NONE);
//                 // KSPSetConvergenceTest(kspcheb, KSPConvergedSkip, NULL, NULL);
//                 // KSPGetPC(kspcheb, &pccheb);
//                 // PCSetType(pccheb, PCILU);
//             }
//         }
        
//         for( int l = N_levels - 2 ; l >= 0 ; l-- ){
//             PCMGGetSmoother(main_pc,l,&ksp_level);
//             // write_func_immediate(mat_level2);
//             // MPI_Barrier(MPI_COMM_WORLD);
//             // std::exit(0);
//             KSPSetOperators(ksp_level, Ac[N_levels - 2 - l], Ac[N_levels - 2 - l]);
//             KSPSetInitialGuessNonzero(ksp_level,PETSC_TRUE);
//             KSPSetType(ksp_level,KSPGMRES);
//             KSPGMRESSetOrthogonalization(ksp_level,KSPGMRESModifiedGramSchmidtOrthogonalization);
//             KSPSetTolerances(ksp_level, 0.4, 1e-5, 1000.0, 100);
//             params = new ConvergenceTestParameters;
//             params->i_max = 7;
//             KSPSetConvergenceTest(ksp_level, convergence_test_smoother, params, [](void *ctx){ ConvergenceTestParameters *params = (ConvergenceTestParameters *)ctx; delete params; return 0;});
//             KSPGetPC(ksp_level, &pc_level);
//             KSPSetErrorIfNotConverged(ksp_level, PETSC_FALSE);
//             PCSetType(pc_level, PCBJACOBI);
//             // PCSetType(pc_level, PCGASM);
//             // PCSetType(pc_level, PCNONE);
            
//             KSPSetUp(ksp_level);
//             {
//                 PetscInt n_local;
//                 PetscInt first_local;
//                 KSP *ksp;
//                 PCBJacobiGetSubKSP(pc_level, &n_local, &first_local, &ksp);
//                 for(int i = 0 ; i < n_local ; i++) {
//                     PC pc_sub_level;
//                     KSPGetPC(ksp[i], &pc_sub_level);
//                     PCSetType(pc_sub_level, PCEISENSTAT);
//                     // PCSetType(pc_sub_level, PCILU);
//                 }
//             }
//         }

//         // for(int i = 0 ; i < N_levels-1 ; i++) {
//         //     MatDestroy(&(Prolong[i]));
//         // }
//         // for(int i = 0 ; i < N_levels-1 ; i++) {
//         //     MatDestroy(&(Restrict[i]));
//         // }
//     }
// }

// std::tuple< KSP, PrecondContextFull* > prepare_multigrid_solver(Mat A, 
//                          Mat A_lapl, 
//                          int face_unknowns, 
//                          int cell_unknowns, 
//                          std::vector< std::tuple< double, double, double > > &cell_center_coordinates, 
//                          std::vector< std::tuple< int, int, int, int, int, int > > &cell_to_face_local,
//                          std::vector<VariableOnMesh> &variables, 
//                          std::vector<int> unknowns_local,
//                          std::map< std::string, std::tuple<Mat, Mat, int, int, int, int> > &cache) {
//     int rank;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//     auto ctx_full = new PrecondContextFull;

//     int N_levels = 3;

//     int face_unknowns_local[N_levels-1];
//     int cell_unknowns_local[N_levels-1];
//     int n_coarse_ex_sum[N_levels-1];
//     int n_coarse[N_levels-1];
//     std::vector< std::vector<int> > coarse_unknowns_local_array(N_levels-1);
//     Mat Prolong[N_levels-1], Restrict[N_levels-1];
//     // Mat Ac_main_diag[N_levels-1];
//     Mat Ac[N_levels-1];

//     std::vector< std::tuple< double, double, double > > cell_center_coordinates_coarse[N_levels-1];
//     std::vector< std::tuple< int, int, int, int, int, int > > cell_to_face_local_coarse[N_levels-1];

//     KSP            ksp;

//     {

//         auto [coarse_local_index_wo, 
//             fine_local_index_wo, 
//             coarse_cell_center_coordinates, 
//             coarse_cell_to_face_local, 
//             coarse_face_unknowns_local,
//             coarse_cell_unknowns_local,
//             coarse_unknowns_local] = 
//             mesh_coarsen_global(
//                 A, 
//                 A_lapl, 
//                 face_unknowns,
//                 cell_unknowns, 
//                 cell_center_coordinates, 
//                 cell_to_face_local,
//                 variables,
//                 unknowns_local,
//                 2);

//         // for( auto v : fine_local_index_wo ) {

//         //     std::cout << "rank = " << rank << " coarse id = " << coarse_local_index_wo << "\n";
//         // }
//         // MPI_Barrier(MPI_COMM_WORLD);

//         face_unknowns_local[0] = coarse_face_unknowns_local;
//         cell_unknowns_local[0] = coarse_cell_unknowns_local;
//         cell_center_coordinates_coarse[0] = coarse_cell_center_coordinates;
//         coarse_unknowns_local_array[0] = coarse_unknowns_local;
//         cell_to_face_local_coarse[0] = coarse_cell_to_face_local;

//         auto [Prolong_loc, Restrict_loc, n_coarse_loc, N_coarse_loc, n_coarse_ex_sum_loc, n_ex_sum_loc] = simple_interp_prolong_global_cached( "full_solver_prolong", A, coarse_local_index_wo, unknowns_local, cache );
//         n_coarse[0] = n_coarse_loc;
//         n_coarse_ex_sum[0] = n_coarse_ex_sum_loc;
//         Prolong[0] = Prolong_loc;
//         Restrict[0] = Restrict_loc;
//         MatMatMatMult(Restrict[0], A, Prolong[0], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &(Ac[0]));
//         ctx_full->matrices["Ac_0"] = Ac[0];

//         // func(Ac[0]);

//         int coarse_face_unknowns_local_ex_sum = 0;
//         int coarse_face_unknowns_local_sum = 0;

//         for(int i = 1 ; i < N_levels-1 ; i++) {
//             std::vector<int> IS_vector(cell_unknowns_local[i-1]);
//             std::iota(IS_vector.begin(), IS_vector.end(), face_unknowns_local[i-1] + 4*cell_unknowns_local[i-1] + n_coarse_ex_sum[i-1]);

//             IS IS_p;
//             Mat App;

//             ISCreateGeneral(
//                 PETSC_COMM_WORLD, 
//                 IS_vector.size(), 
//                 IS_vector.data(), 
//                 PETSC_COPY_VALUES, 
//                 &IS_p
//             );

//             MatCreateSubMatrix(
//                 Ac[i-1],
//                 IS_p,
//                 IS_p,
//                 MAT_INITIAL_MATRIX,
//                 &App);

//             ISDestroy(&IS_p);

//             auto [coarse_local_index_wo, 
//             fine_local_index_wo, 
//             coarse_cell_center_coordinates,
//             coarse_cell_to_face_local, 
//             coarse_face_unknowns_local,
//             coarse_cell_unknowns_local, 
//             coarse_unknowns_local] = 
//             mesh_coarsen_global(
//                 Ac[i-1], 
//                 App, 
//                 face_unknowns_local[i-1], 
//                 cell_unknowns_local[i-1], 
//                 cell_center_coordinates_coarse[i-1], 
//                 cell_to_face_local_coarse[i-1],
//                 variables,
//                 coarse_unknowns_local_array[i-1],
//                 1);

//             face_unknowns_local[i] = coarse_face_unknowns_local;
//             cell_unknowns_local[i] = coarse_cell_unknowns_local;
//             cell_center_coordinates_coarse[i] = coarse_cell_center_coordinates;
//             coarse_unknowns_local_array[i] = coarse_unknowns_local;
//             cell_to_face_local_coarse[i] = coarse_cell_to_face_local;
            

//             auto [Prolong_loc, Restrict_loc, n_coarse_loc, N_coarse_loc, n_coarse_ex_sum_loc, n_ex_sum_loc] = simple_interp_prolong_global_cached( "full_solver_level_"+std::to_string(i), Ac[i-1], coarse_local_index_wo, coarse_unknowns_local_array[i-1], cache );
//             n_coarse[i] = n_coarse_loc;
//             n_coarse_ex_sum[i] = n_coarse_ex_sum_loc;
//             Prolong[i] = Prolong_loc;
//             Restrict[i] = Restrict_loc;
//             MatMatMatMult(Restrict[i], Ac[i-1], Prolong[i], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &(Ac[i]));
//             ctx_full->matrices["Ac_"+std::to_string(i)] = Ac[i];
//             // N_size_val=0;
//             // n_size_val=0;
//             // MatGetSize(Ac_main_diag[i], &N_size_val, nullptr);
//             // MatGetLocalSize(Ac_main_diag[i], &n_size_val, nullptr);
//             // std::cout << "level " << i << " global elements = " << N_size_val << " local elements = " << n_size_val << "\n";
//             MatDestroy(&App);
//         }
//     }

//     {
//         KSP            ksp_lapl2;
//         KSPCreate(PETSC_COMM_WORLD,&ksp_lapl2);

//         PetscOptionsClear(NULL);
//         PetscOptionsSetValue(NULL, "-mat_mumps_icntl_7", "3");
//         PetscOptionsSetValue(NULL, "-ksp_rtol", "1.0e-8");
//         PetscOptionsSetValue(NULL, "-mat_mumps_cntl_1", "1.0");
//         // PetscOptionsSetValue(NULL, "-pc_mg_levels", "6");
//         // PetscOptionsSetValue(NULL, "-mat_mumps_cntl_2", "1.0e-15");
//         PetscOptionsSetValue(NULL, "-mat_mumps_icntl_14", "1000");
//         PetscOptionsSetValue(NULL, "-mat_mumps_icntl_38", "1000");
//         PetscOptionsSetValue(NULL, "-pc_type", "mg");
//         // PetscOptionsSetValue(NULL, "-pc_gamg_agg_nsmooths", "3");
//         // PetscOptionsSetValue(NULL, "-pc_gamg_type", "agg");
//         PetscOptionsSetValue(NULL, "-pc_gamg_aggressive_coarsening", "0");
//         // PetscOptionsSetValue(NULL, "-pc_gamg_aggressive_mis_k", "1");
//         // PetscOptionsSetValue(NULL, "-pc_gamg_aggressive_square_graph", "false");
//         // PetscOptionsSetValue(NULL, "-mat_coarsen_view","");
//         PetscOptionsSetValue(NULL, "-ksp_gmres_modifiedgramschmidt", "");
//         PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "150");

//         // PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "250");
//         // PetscOptionsSetValue(NULL, "-ksp_monitor", "");
//         // PetscOptionsSetValue(NULL, "-ksp_view", "");

//         KSPSetFromOptions(ksp_lapl2);
        
//         KSPSetOperators(ksp_lapl2, A, A);
//         KSPSetType(ksp_lapl2,KSPFGMRES);

//         ConvergenceTestParameters *params = new ConvergenceTestParameters;
//         params->i_max = 170;
//         params->residual = 1e-5;

//         KSPSetConvergenceTest(ksp_lapl2, convergence_test_full, params, [](void *ctx){ ConvergenceTestParameters *params = (ConvergenceTestParameters *) ctx; delete params; return 0;});
//         KSPSetErrorIfNotConverged(ksp_lapl2, PETSC_FALSE);
//         KSPGMRESSetOrthogonalization(ksp_lapl2,KSPGMRESModifiedGramSchmidtOrthogonalization);
//         KSPGMRESSetRestart(ksp_lapl2, 150);

//         PC pc_lapl2;
//         KSPGetPC(ksp_lapl2, &pc_lapl2);
//         PCMGSetCycleType(pc_lapl2, PC_MG_CYCLE_V);
//         PCMGSetLevels(pc_lapl2, N_levels, NULL);

//         for( int l = 0 ; l < N_levels-1 ; l++ ){
//             PCMGSetInterpolation(pc_lapl2, l+1, Prolong[N_levels-2-l]);
//         }

//         Vec temp;
//         MatCreateVecs(Prolong[N_levels-2], &temp, nullptr);
//         ctx_full->vectors["x_0"] = temp;
//         PCMGSetX(pc_lapl2, 0, temp);
//         MatCreateVecs(Prolong[N_levels-2], &temp, nullptr);
//         ctx_full->vectors["rhs_0"] = temp;
//         PCMGSetRhs(pc_lapl2, 0, temp);
//         // MatCreateVecs(Prolong[0], &temp, nullptr);
//         // PCMGSetR(pc_lapl2, 0, temp);

//         for( int l = 1 ; l < N_levels-1 ; l++ ){
//             MatCreateVecs(Prolong[N_levels-2-l], &temp, nullptr);
//             ctx_full->vectors["x_" + std::to_string(l)] = temp;
//             PCMGSetX(pc_lapl2, l, temp);
//             MatCreateVecs(Prolong[N_levels-2-l], &temp, nullptr);
//             ctx_full->vectors["rhs_" + std::to_string(l)] = temp;
//             PCMGSetRhs(pc_lapl2, l, temp);
//             MatCreateVecs(Prolong[N_levels-2-l], &temp, nullptr);
//             ctx_full->vectors["r_" + std::to_string(l)] = temp;
//             PCMGSetR(pc_lapl2, l, temp);
//         }

//         // MatCreateVecs(Prolong[4], nullptr, &temp);
//         // PCMGSetX(pc_lapl2, 5, temp);
//         // MatCreateVecs(Prolong[4], nullptr, &temp);
//         // PCMGSetRhs(pc_lapl2, 5, temp);
//         MatCreateVecs(Prolong[0], nullptr, &temp);
//         ctx_full->vectors["r_" + std::to_string(N_levels-1)] = temp;
//         PCMGSetR(pc_lapl2, N_levels-1, temp);


//         KSP ksp_level;
//         PC pc_level;
//         PCMGGetSmoother(pc_lapl2,N_levels-1,&ksp_level);
//         KSPSetOperators(ksp_level, A, A);
//         KSPSetInitialGuessNonzero(ksp_level,PETSC_TRUE);
//         KSPSetType(ksp_level,KSPGMRES);
//         KSPGMRESSetOrthogonalization(ksp_level,KSPGMRESModifiedGramSchmidtOrthogonalization);
//         KSPSetTolerances(ksp_level, 0.4, 1e-5, 1000.0, 100);
//         ConvergenceTestParameters *params = new ConvergenceTestParameters;
//         params->i_max = 7;
//         KSPSetConvergenceTest(ksp_level, convergence_test_smoother, params, [](void *ctx){ ConvergenceTestParameters *params = (ConvergenceTestParameters *)ctx; delete params; return 0;});
//         KSPGetPC(ksp_level, &pc_level);
//         KSPSetErrorIfNotConverged(ksp_level, PETSC_FALSE);
//         PCSetType(pc_level, PCBJACOBI);
//         // PCSetType(pc_level, PCGASM);
//         // PCSetType(pc_level, PCNONE);
        
//         KSPSetUp(ksp_level);
//         {
//             PetscInt n_local;
//             PetscInt first_local;
//             KSP *ksp;
//             PCBJacobiGetSubKSP(pc_level, &n_local, &first_local, &ksp);
//             for(int i = 0 ; i < n_local ; i++) {
//                 PC pc_sub_level;
//                 KSPGetPC(ksp[i], &pc_sub_level);
//                 PCSetType(pc_sub_level, PCEISENSTAT);
//                 // PCSetType(pc_sub_level, PCILU);


//                 // PC  pcksp,pccheb;
//                 // KSP kspcheb;
//                 // PCSetType(pc_sub_level, PCCOMPOSITE);
//                 // PCCompositeSetType(pc_sub_level, PC_COMPOSITE_ADDITIVE);
//                 // PCCompositeAddPCType(pc_sub_level, PCEISENSTAT);
//                 // PCCompositeAddPCType(pc_sub_level, PCKSP);
//                 // PCCompositeGetPC(pc_sub_level, 1, &pcksp);
//                 // PCKSPGetKSP(pcksp, &kspcheb);
//                 // KSPSetType(kspcheb, KSPFGMRES);
//                 // KSPSetTolerances(kspcheb, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT, 1);
//                 // KSPSetNormType(kspcheb, KSP_NORM_NONE);
//                 // KSPSetConvergenceTest(kspcheb, KSPConvergedSkip, NULL, NULL);
//                 // KSPGetPC(kspcheb, &pccheb);
//                 // PCSetType(pccheb, PCILU);
//             }
//         }
        
//         for( int l = N_levels - 2 ; l >= 0 ; l-- ){
//             PCMGGetSmoother(pc_lapl2,l,&ksp_level);
//             // write_func_immediate(mat_level2);
//             // MPI_Barrier(MPI_COMM_WORLD);
//             // std::exit(0);
//             KSPSetOperators(ksp_level, Ac[N_levels - 2 - l], Ac[N_levels - 2 - l]);
//             KSPSetInitialGuessNonzero(ksp_level,PETSC_TRUE);
//             KSPSetType(ksp_level,KSPGMRES);
//             KSPGMRESSetOrthogonalization(ksp_level,KSPGMRESModifiedGramSchmidtOrthogonalization);
//             KSPSetTolerances(ksp_level, 0.4, 1e-5, 1000.0, 100);
//             params = new ConvergenceTestParameters;
//             params->i_max = 7;
//             KSPSetConvergenceTest(ksp_level, convergence_test_smoother, params, [](void *ctx){ ConvergenceTestParameters *params = (ConvergenceTestParameters *)ctx; delete params; return 0;});
//             KSPGetPC(ksp_level, &pc_level);
//             KSPSetErrorIfNotConverged(ksp_level, PETSC_FALSE);
//             PCSetType(pc_level, PCBJACOBI);
//             // PCSetType(pc_level, PCGASM);
//             // PCSetType(pc_level, PCNONE);
            
//             KSPSetUp(ksp_level);
//             {
//                 PetscInt n_local;
//                 PetscInt first_local;
//                 KSP *ksp;
//                 PCBJacobiGetSubKSP(pc_level, &n_local, &first_local, &ksp);
//                 for(int i = 0 ; i < n_local ; i++) {
//                     PC pc_sub_level;
//                     KSPGetPC(ksp[i], &pc_sub_level);
//                     PCSetType(pc_sub_level, PCEISENSTAT);
//                     // PCSetType(pc_sub_level, PCILU);
//                 }
//             }
//         }

//         // for(int i = 0 ; i < N_levels-1 ; i++) {
//         //     MatDestroy(&(Prolong[i]));
//         // }
//         // for(int i = 0 ; i < N_levels-1 ; i++) {
//         //     MatDestroy(&(Restrict[i]));
//         // }

//         ksp = ksp_lapl2;
//     }

//     return std::make_tuple( ksp, ctx_full );
// }