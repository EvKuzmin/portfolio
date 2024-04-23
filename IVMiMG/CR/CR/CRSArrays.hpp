#pragma once
#include <math.h>
#include <stdio.h>
#include "mkl.h"
using namespace std;
struct CRSArrays {
	MKL_INT n;  //< the number of rows
	MKL_INT m;  //< the number of columns
	MKL_INT nnz;//< the number of nnz (== ia[n])
	double* a;  //< the values (of size NNZ)
	MKL_INT* ia;//< the usual rowptr (of size n+1)
	MKL_INT* ja;//< the colidx of each NNZ (of size nnz)

	CRSArrays() {
		a = NULL;
		ia = NULL;
		ja = NULL;
	}

	~CRSArrays() {
	}

	void print() const;
};

// cols must be in ascending order
CRSArrays get_csr_submatrix(const CRSArrays& A, MKL_INT* rows, MKL_INT* cols, MKL_INT N, MKL_INT M);