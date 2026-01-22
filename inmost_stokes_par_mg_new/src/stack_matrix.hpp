#pragma once

#include"petsc.h"
#include<tuple>

Mat petsc_stack_h(Mat A, Mat B, int nnz);
Mat petsc_stack_v(Mat A, Mat B, int nnz);
Mat petsc_stack_main_diag(Mat A, Mat B, int nnz);