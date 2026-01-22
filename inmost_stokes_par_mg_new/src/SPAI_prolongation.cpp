#include "SPAI_prolongation.hpp"
#include <iostream>

std::tuple< Mat, Mat > SPAI_prolongation( std::map< std::string, Mat > &submatrices, std::map< std::string, IS > &index_sequences, SubmatrixManager &submatrix_map )
{
    Mat Aff = submatrices.at("Aff");
    Mat Afc = submatrices.at("Afc");
    Mat Acf = submatrices.at("Acf");

    Vec Aff_diagonal;
    MatCreateVecs(Aff, &Aff_diagonal, nullptr);

    MatGetDiagonal(Aff, Aff_diagonal);
    VecReciprocal(Aff_diagonal);

    int Aff_m, Aff_n, Aff_M, Aff_N;

    Mat Aff_SPAI, Aff_SPAI_cd;

    MatGetSize(Aff, &Aff_M, &Aff_N);
    MatGetLocalSize(Aff, &Aff_m, &Aff_n);

    MatCreateConstantDiagonal(MPI_COMM_WORLD, Aff_m, Aff_n, Aff_M, Aff_N, 0.0, &Aff_SPAI_cd);
    MatConvert(Aff_SPAI_cd, MATMPIAIJ, MAT_INITIAL_MATRIX, &Aff_SPAI);
    MatDestroy(&Aff_SPAI_cd);
    MatDiagonalSet(Aff_SPAI, Aff_diagonal, INSERT_VALUES);
    VecDestroy(&Aff_diagonal);



    // Mat Aff_SPAI;

    // build_SPAI(Aff, &Aff_SPAI);
    // build_SPAI(Aff_main_diag, &Aff_SPAI);

    // write_func_immediate(Aff);

    // BARRIER;
    // std::exit(0);

    // Mat Aff_Aff, Aff_Aff_Aff, Aff_Aff_Aff_Aff;
    // MatMatMult(Aff, Aff, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Aff_Aff);
    // MatMatMatMult(Aff, Aff, Aff, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Aff_Aff);
    // MatMatMatMult(Aff_Aff, Aff_Aff, Aff_Aff, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Aff_Aff_Aff);
    // MatMatMatMult(Aff_Aff_Aff, Aff_Aff_Aff, Aff_Aff_Aff, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Aff_Aff_Aff_Aff);
    // MatTranspose(Aff, MAT_INPLACE_MATRIX, &Aff);
    // build_SPAI_pattern(Aff, Aff_Aff, &Aff_SPAI);
    // MatTranspose(Aff, MAT_INPLACE_MATRIX, &Aff);
    // build_SPAI_pattern(Aff, Aff_Aff_Aff, &Aff_SPAI);

    // MatMatMult(Aff_main_diag, Aff_main_diag, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Aff_Aff);
    // build_SPAI_pattern(Aff, Aff_Aff, &Aff_SPAI);

    // write_func_immediate(Aff_SPAI);

    // BARRIER;
    // std::exit(0);

    // Mat Aff_Aff_SPAI;
    // MatTranspose(Aff_SPAI, MAT_INPLACE_MATRIX, &Aff_SPAI);
    // MatMatMult(Aff_SPAI, Aff, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Aff_Aff_SPAI);
    // MatTranspose(Aff_SPAI, MAT_INPLACE_MATRIX, &Aff_SPAI);

    // write_func_immediate(Aff_Aff_SPAI);

    // BARRIER;
    // std::exit(0);

    // Vec diag_Aff_SPAI;
    // int diag_size;
    // MatCreateVecs( Aff_SPAI, NULL, &diag_Aff_SPAI );
    // MatGetDiagonal( Aff_SPAI, diag_Aff_SPAI );
    // MatGetDiagonal( Aff, diag_Aff_SPAI );

    // VecScale(diag_Aff_SPAI, -1.0);

    // Mat Aff_SPAI_Afc;

    // MatDuplicate(Afc, MAT_COPY_VALUES, &Aff_SPAI_Afc);

    // MatDiagonalScale(Aff_SPAI_Afc, diag_Aff_SPAI, NULL);

    Mat Afc_P, Acf_C;

    // MatDuplicate(Afc, MAT_COPY_VALUES, &Afc_P);

    // MatDuplicate(Acf, MAT_COPY_VALUES, &Acf_C);

    // MatDiagonalScale(Afc_P, diag_Aff_SPAI, NULL);

    // MatDiagonalScale(Acf_C, NULL, diag_Aff_SPAI);

    MatMatMult(Aff_SPAI, Afc, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Afc_P);
    MatMatMult(Acf, Aff_SPAI, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Acf_C);

    // MatMatMult(Aff_SPAI, Afc_main_diag, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Afc_P);
    // MatMatMult(Acf_main_diag, Aff_SPAI, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Acf_C);
    MatScale(Afc_P, -1.0);
    MatScale(Acf_C, -1.0);

    // write_func_immediate(Afc_P);

    return std::make_tuple( Afc_P, Acf_C );
}

void check_prolongation_operator(std::map< std::string, Mat > &submatrices)
{
    Mat Aff = submatrices.at("Aff");
    Mat Afc = submatrices.at("Afc");
    Mat Acf = submatrices.at("Acf");

    Mat Afc_P = submatrices.at("Afc_P");
    Mat Acf_C = submatrices.at("Acf_C");

    Mat Aff_AffSPAI_Afc, Afc_Aff_AffSPAI_Afc;
    MatMatMult(Aff, Afc_P, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Aff_AffSPAI_Afc);
    MatConvert(Afc, MATMPIAIJ, MAT_INITIAL_MATRIX, &Afc_Aff_AffSPAI_Afc);
    MatAXPY( Afc_Aff_AffSPAI_Afc, 1.0, Aff_AffSPAI_Afc, DIFFERENT_NONZERO_PATTERN );
    double multigrid_fc_norm = 0.0;
    MatNorm(Afc_Aff_AffSPAI_Afc, NORM_FROBENIUS, &multigrid_fc_norm);
    // MatView(Afc_Aff_AffSPAI_Afc, PETSC_VIEWER_STDOUT_WORLD);
    std::cout << "multigrid_fc_norm = " << multigrid_fc_norm << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    Mat Acf_AffSPAI_Aff, Acf_Acf_AffSPAI_Aff;
    MatMatMult( Acf_C, Aff, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Acf_AffSPAI_Aff);
    MatConvert(Acf, MATMPIAIJ, MAT_INITIAL_MATRIX, &Acf_Acf_AffSPAI_Aff);
    MatAXPY( Acf_Acf_AffSPAI_Aff, 1.0, Acf_AffSPAI_Aff, DIFFERENT_NONZERO_PATTERN );
    double multigrid_cf_norm = 0.0;
    MatNorm(Acf_Acf_AffSPAI_Aff, NORM_FROBENIUS, &multigrid_cf_norm);
    // MatView(Acf_Acf_AffSPAI_Aff, PETSC_VIEWER_STDOUT_WORLD);
    std::cout << "multigrid_cf_norm = " << multigrid_cf_norm << "\n";
    MPI_Barrier(MPI_COMM_WORLD);
}