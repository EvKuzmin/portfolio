#include"Notay_transform.hpp"
#include"stack_matrix.hpp"

std::tuple<Mat, Mat, Mat, Mat, Mat, Mat> notay_transform(Mat Auu, Mat Aup, Mat Apu, Mat App, SubmatrixManager &submatrix_map)
{

    Mat Auu_SPAI, Auu_SPAI_cd;

    // build_SPAI(submatrices.at("Auu"), &Auu_SPAI);
    Vec Auu_diagonal;
    MatCreateVecs(Auu, &Auu_diagonal, nullptr);

    MatGetDiagonal(Auu, Auu_diagonal);
    VecReciprocal(Auu_diagonal);

    int Auu_m, Auu_n, Auu_M, Auu_N;

    MatGetSize(Auu, &Auu_M, &Auu_N);
    MatGetLocalSize(Auu, &Auu_m, &Auu_n);

    MatCreateConstantDiagonal(MPI_COMM_WORLD, Auu_m, Auu_n, Auu_M, Auu_N, 0.0, &Auu_SPAI_cd);
    MatConvert(Auu_SPAI_cd, MATMPIAIJ, MAT_INITIAL_MATRIX, &Auu_SPAI);
    MatDestroy(&Auu_SPAI_cd);
    MatDiagonalSet(Auu_SPAI, Auu_diagonal, INSERT_VALUES);
    VecDestroy(&Auu_diagonal);

    Mat Apu_Auu_SPAI, Auu_SPAI_Aup;

    MatMatMult(Apu, Auu_SPAI, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Apu_Auu_SPAI);
    MatMatMult(Auu_SPAI, Aup, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Auu_SPAI_Aup);

    Mat Auu_Auu_SPAI_Aup;

    MatMatMult(Auu, Auu_SPAI_Aup, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Auu_Auu_SPAI_Aup);

    Mat Aup_m_Auu_Auu_SPAI_Aup;

    MatConvert(Aup, MATSAME, MAT_INITIAL_MATRIX, &Aup_m_Auu_Auu_SPAI_Aup);

    MatAXPY(Aup_m_Auu_Auu_SPAI_Aup, -1.0, Auu_Auu_SPAI_Aup, DIFFERENT_NONZERO_PATTERN);

    MatDestroy(&Auu_Auu_SPAI_Aup);

    Mat Apu_Auu_Auu_SPAI;

    MatMatMult(Apu_Auu_SPAI, Auu, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Apu_Auu_Auu_SPAI);

    Mat Apu_m_Apu_Auu_Auu_SPAI;

    MatConvert(Apu, MATSAME, MAT_INITIAL_MATRIX, &Apu_m_Apu_Auu_Auu_SPAI);

    MatAXPY(Apu_m_Apu_Auu_Auu_SPAI, -1.0, Apu_Auu_Auu_SPAI, DIFFERENT_NONZERO_PATTERN);

    MatDestroy(&Apu_Auu_Auu_SPAI);

    Mat SchurApprox;

    MatMatMatMult(Apu_Auu_SPAI, Auu, Auu_SPAI_Aup, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &SchurApprox);

    Mat Apu_Auu_SPAI_Aup;

    MatMatMatMult(Apu, Auu_SPAI, Aup, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Apu_Auu_SPAI_Aup);

    MatDestroy(&Auu_SPAI);

    MatAXPY(SchurApprox, -2.0, Apu_Auu_SPAI_Aup, DIFFERENT_NONZERO_PATTERN);

    MatAXPY(SchurApprox, 1.0, App, DIFFERENT_NONZERO_PATTERN);

    MatDestroy(&Apu_Auu_SPAI_Aup);

    Mat A_row_1 = petsc_stack_h(Auu, Aup_m_Auu_Auu_SPAI_Aup, 17);
    Mat A_row_2 = petsc_stack_h(Apu_m_Apu_Auu_Auu_SPAI, SchurApprox, 17);
    Mat A_transformed = petsc_stack_v(A_row_1, A_row_2, 17);
    MatDestroy(&A_row_1);
    MatDestroy(&A_row_2);

    MatNullSpace nullspace_wo_test;

    Vec nsp_wo_test;
    double *nsp_wo_test_raw;

    MatCreateVecs(A_transformed, NULL, &nsp_wo_test);

    VecSet(nsp_wo_test, 0.0);

    VecGetArray(nsp_wo_test, &nsp_wo_test_raw);

    for( auto idx : submatrix_map.find_nodes("Ap")[0]->map_to_parent ){
        nsp_wo_test_raw[idx] = 1.0;
    }

    VecRestoreArray(nsp_wo_test, &nsp_wo_test_raw);

    MPI_Barrier(MPI_COMM_WORLD);

    VecNormalize(nsp_wo_test, nullptr);

    MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &nsp_wo_test, &nullspace_wo_test);
    MatSetNullSpace(A_transformed, nullspace_wo_test);

    MatNullSpaceDestroy(&nullspace_wo_test);
    VecDestroy(&nsp_wo_test);

    return std::make_tuple(A_transformed, Aup_m_Auu_Auu_SPAI_Aup, Apu_m_Apu_Auu_Auu_SPAI, SchurApprox, Apu_Auu_SPAI, Auu_SPAI_Aup);
}

std::tuple<Mat, Mat, Mat, Mat, Mat, Mat> notay_transform_right(Mat Auu, Mat Aup, Mat Apu, Mat App, SubmatrixManager &submatrix_map)
{

    Mat Auu_SPAI, Auu_SPAI_cd;

    // build_SPAI(submatrices.at("Auu"), &Auu_SPAI);
    Vec Auu_diagonal;
    MatCreateVecs(Auu, &Auu_diagonal, nullptr);

    MatGetDiagonal(Auu, Auu_diagonal);
    VecReciprocal(Auu_diagonal);

    int Auu_m, Auu_n, Auu_M, Auu_N;

    MatGetSize(Auu, &Auu_M, &Auu_N);
    MatGetLocalSize(Auu, &Auu_m, &Auu_n);

    MatCreateConstantDiagonal(MPI_COMM_WORLD, Auu_m, Auu_n, Auu_M, Auu_N, 0.0, &Auu_SPAI_cd);
    MatConvert(Auu_SPAI_cd, MATMPIAIJ, MAT_INITIAL_MATRIX, &Auu_SPAI);
    MatDestroy(&Auu_SPAI_cd);
    MatDiagonalSet(Auu_SPAI, Auu_diagonal, INSERT_VALUES);
    VecDestroy(&Auu_diagonal);

//     Mat Apu_Auu_SPAI, Auu_SPAI_Aup;
    Mat Auu_SPAI_Aup;

//     MatMatMult(Apu, Auu_SPAI, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Apu_Auu_SPAI);
    MatMatMult(Auu_SPAI, Aup, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Auu_SPAI_Aup);

    Mat Auu_Auu_SPAI_Aup;

    MatMatMult(Auu, Auu_SPAI_Aup, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Auu_Auu_SPAI_Aup);

    Mat Aup_m_Auu_Auu_SPAI_Aup;

    MatConvert(Aup, MATSAME, MAT_INITIAL_MATRIX, &Aup_m_Auu_Auu_SPAI_Aup);

    MatAXPY(Aup_m_Auu_Auu_SPAI_Aup, -1.0, Auu_Auu_SPAI_Aup, DIFFERENT_NONZERO_PATTERN);

    MatDestroy(&Auu_Auu_SPAI_Aup);

//     Mat Apu_Auu_Auu_SPAI;

//     MatMatMult(Apu_Auu_SPAI, Auu, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Apu_Auu_Auu_SPAI);

//     Mat Apu_m_Apu_Auu_Auu_SPAI;

//     MatConvert(Apu, MATSAME, MAT_INITIAL_MATRIX, &Apu_m_Apu_Auu_Auu_SPAI);

//     MatAXPY(Apu_m_Apu_Auu_Auu_SPAI, -1.0, Apu_Auu_Auu_SPAI, DIFFERENT_NONZERO_PATTERN);

//     MatDestroy(&Apu_Auu_Auu_SPAI);

    Mat SchurApprox;

//     MatMatMatMult(Apu_Auu_SPAI, Auu, Auu_SPAI_Aup, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &SchurApprox);

//     Mat Apu_Auu_SPAI_Aup;

    MatMatMatMult(Apu, Auu_SPAI, Aup, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &SchurApprox);

    MatDestroy(&Auu_SPAI);
    MatScale(SchurApprox, -1.0);

//     MatAXPY(SchurApprox, -2.0, Apu_Auu_SPAI_Aup, DIFFERENT_NONZERO_PATTERN);

    MatAXPY(SchurApprox, 1.0, App, DIFFERENT_NONZERO_PATTERN);

//     MatDestroy(&Apu_Auu_SPAI_Aup);

    Mat A_row_1 = petsc_stack_h(Auu, Aup_m_Auu_Auu_SPAI_Aup, 17);
    Mat A_row_2 = petsc_stack_h(Apu, SchurApprox, 17);
    Mat A_transformed = petsc_stack_v(A_row_1, A_row_2, 17);
    MatDestroy(&A_row_1);
    MatDestroy(&A_row_2);

    MatNullSpace nullspace_wo_test;

    Vec nsp_wo_test;
    double *nsp_wo_test_raw;

    MatCreateVecs(A_transformed, NULL, &nsp_wo_test);

    VecSet(nsp_wo_test, 0.0);

    VecGetArray(nsp_wo_test, &nsp_wo_test_raw);

    for( auto idx : submatrix_map.find_nodes("Ap")[0]->map_to_parent ){
        nsp_wo_test_raw[idx] = 1.0;
    }

    VecRestoreArray(nsp_wo_test, &nsp_wo_test_raw);

    MPI_Barrier(MPI_COMM_WORLD);

    VecNormalize(nsp_wo_test, nullptr);

    MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &nsp_wo_test, &nullspace_wo_test);
    MatSetNullSpace(A_transformed, nullspace_wo_test);

    MatNullSpaceDestroy(&nullspace_wo_test);
    VecDestroy(&nsp_wo_test);

    return std::make_tuple(A_transformed, Aup_m_Auu_Auu_SPAI_Aup, nullptr, SchurApprox, nullptr, Auu_SPAI_Aup);
}

void notay_transform_submatrices(std::map< std::string, Mat > &submatrices, std::map< std::string, IS > &index_sequences, SubmatrixManager &submatrix_map ) {
    auto [A_transformed, 
            Aup_m_Auu_Auu_SPAI_Aup, 
            Apu_m_Apu_Auu_Auu_SPAI, 
            SchurApprox, 
            Apu_Auu_SPAI, 
            Auu_SPAI_Aup] = notay_transform_right(submatrices.at("Auu"), 
                                            submatrices.at("Aup"), 
                                            submatrices.at("Apu"), 
                                            submatrices.at("App"), 
                                            submatrix_map);

//     MatDestroy( &(submatrices.at("without_dirichlet")) );
//     MatDestroy( &(submatrices.at("Aup")) );
//     MatDestroy( &(submatrices.at("Apu")) );
//     MatDestroy( &(submatrices.at("App")) );

//     submatrices.at("without_dirichlet") = A_transformed;
//     submatrices.at("Aup") = Aup_m_Auu_Auu_SPAI_Aup;
//     submatrices.at("Apu") = Apu_m_Apu_Auu_Auu_SPAI;
//     submatrices.at("App") = SchurApprox;

//     submatrices["Apu_Auu_SPAI"] = Apu_Auu_SPAI;
//     submatrices["Auu_SPAI_Aup"] = Auu_SPAI_Aup;

    submatrices["A_Notay_"] = A_transformed;
    submatrices["App_Notay"] = SchurApprox;

    submatrices["Apu_Auu_SPAI"] = Apu_Auu_SPAI;
    submatrices["Auu_SPAI_Aup"] = Auu_SPAI_Aup;

//     Mat Aff, Afc, Acf, Acc;

//     MatCreateSubMatrix(
//             submatrices.at("without_dirichlet"),
//             index_sequences.at("without_dirichlet_fine"),
//             index_sequences.at("without_dirichlet_fine"),
//             MAT_INITIAL_MATRIX,
//             &Aff);
    
//     MatCreateSubMatrix(
//             submatrices.at("without_dirichlet"),
//             index_sequences.at("without_dirichlet_fine"),
//             index_sequences.at("without_dirichlet_coarse"),
//             MAT_INITIAL_MATRIX,
//             &Afc);
    
//     MatCreateSubMatrix(
//             submatrices.at("without_dirichlet"),
//             index_sequences.at("without_dirichlet_coarse"),
//             index_sequences.at("without_dirichlet_fine"),
//             MAT_INITIAL_MATRIX,
//             &Acf);
    
//     MatCreateSubMatrix(
//             submatrices.at("without_dirichlet"),
//             index_sequences.at("without_dirichlet_coarse"),
//             index_sequences.at("without_dirichlet_coarse"),
//             MAT_INITIAL_MATRIX,
//             &Acc);

//     submatrices["Aff"] = Aff;
//     submatrices["Afc"] = Afc;
//     submatrices["Acf"] = Acf;
//     submatrices["Acc"] = Acc;
}

void prepare_main_block_diag(std::map< std::string, Mat > &submatrices, std::map< std::string, IS > &index_sequences) {
    Mat Aff = submatrices.at("Aff");
    Mat Afc = submatrices.at("Afc");
    Mat Acf = submatrices.at("Acf");
    Mat Aff_uu, Aff_pp, Aff_main_diag;

    MatCreateSubMatrix(
            Aff,
            index_sequences.at("without_dirichlet_fine_u"),
            index_sequences.at("without_dirichlet_fine_u"),
            MAT_INITIAL_MATRIX,
            &Aff_uu);
    
    MatCreateSubMatrix(
            Aff,
            index_sequences.at("without_dirichlet_fine_p"),
            index_sequences.at("without_dirichlet_fine_p"),
            MAT_INITIAL_MATRIX,
            &Aff_pp);

    Aff_main_diag = petsc_stack_main_diag(Aff_uu, Aff_pp, 17);

    Mat Afc_uu, Afc_pp, Afc_main_diag;

    MatCreateSubMatrix(
            Afc,
            index_sequences.at("without_dirichlet_fine_u"),
            index_sequences.at("without_dirichlet_coarse_u"),
            MAT_INITIAL_MATRIX,
            &Afc_uu);
    
    MatCreateSubMatrix(
            Afc,
            index_sequences.at("without_dirichlet_fine_p"),
            index_sequences.at("without_dirichlet_coarse_p"),
            MAT_INITIAL_MATRIX,
            &Afc_pp);

    Afc_main_diag = petsc_stack_main_diag(Afc_uu, Afc_pp, 17);

    Mat Acf_uu, Acf_pp, Acf_main_diag;

    MatCreateSubMatrix(
            Acf,
            index_sequences.at("without_dirichlet_coarse_u"),
            index_sequences.at("without_dirichlet_fine_u"),
            MAT_INITIAL_MATRIX,
            &Acf_uu);
    
    MatCreateSubMatrix(
            Acf,
            index_sequences.at("without_dirichlet_coarse_p"),
            index_sequences.at("without_dirichlet_fine_p"),
            MAT_INITIAL_MATRIX,
            &Acf_pp);

    Acf_main_diag = petsc_stack_main_diag(Acf_uu, Acf_pp, 17);

    submatrices["Aff_main_diag"] = Aff_main_diag;
    submatrices["Afc_main_diag"] = Afc_main_diag;
    submatrices["Acf_main_diag"] = Acf_main_diag;
}