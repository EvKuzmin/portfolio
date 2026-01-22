#pragma once

#include"petsc.h"

void diagonal_compensation(Mat A, Mat A_inv);
void schur_diagonal_compensation(Mat Auu, Mat Aup, Mat Apu, Mat Schur);
void build_SPAI( Mat A, Mat *A_inv );
void build_SPAI_pattern( Mat A, Mat pattern, Mat *A_inv_p );