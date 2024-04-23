#pragma once
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <Sparse>
#include <Eigen>
#include <Dense>
#include <iterator>


using namespace std;
class COOMatrix;
class CSCMatrix;
class CRSMatrix;

class COOMatrix {
private:

public:
	int n;
	int m;      //< the dimension of the matrix
	int nnz;    //< the number of nnz inside the matrix
	std::vector<double> a;    //< the values (size = nnz)
	std::vector<int> ia;//< the row indexes (size = nnz)
	std::vector<int> ja;//< the col indexes (size = nnz)
	void print() const;
	CRSMatrix coo_to_crs();
	CSCMatrix coo_to_csc();
	COOMatrix(int n, int m, double* a, int* ia, int* ja, int nnz);
	CSCMatrix SPAI_precond(CSCMatrix pattern);
	CSCMatrix SPAI0();
	CSCMatrix SPAI1();
	CSCMatrix SPAI2();
	int _sort_type; // 0 - ia, 1 - ja, -1 - isn't sort
	void sort_ia();
	void sort_ja();
	Eigen::SparseMatrix<double> get_eigen_sparse();
};

class CRSMatrix {
private:
public:
	int n;  //< the number of rows
	int m;  //< the number of columns
	int nnz;//< the number of nnz (== ia[n])
	std::vector<double> a;  //< the values (of size NNZ)
	std::vector<int> ia;//< the usual rowptr (of size n+1)
	std::vector<int> ja;//< the colidx of each NNZ (of size nnz)
	void print() const;
	CRSMatrix(int n, int m, double* a, int* ia, int* ja);
	COOMatrix crs_to_coo();
	CSCMatrix crs_to_csc();
	CSCMatrix SPAI_precond(CSCMatrix pattern);
	CSCMatrix SPAI0();
	CSCMatrix SPAI1();
	CSCMatrix SPAI2();
	CRSMatrix get_csr_submatrix(int* rows, int* cols, int N, int M);

};
class CSCMatrix {
private:
public:
	int n;  //< the number of rows
	int m;  //< the number of columns
	int nnz;//< the number of nnz (== ia[n])
	std::vector<double> a;  //< the values (of size NNZ)
	std::vector<int> ia;//< the usual rowptr (of size nnz)
	std::vector<int> ja;//< the colidx of each NNZ (of size m+1)
	void print() const;
	COOMatrix csc_to_coo();
	CRSMatrix csc_to_crs();
	CSCMatrix(int n, int m, double* a, int* ia, int* ja);
	CSCMatrix SPAI_precond(CSCMatrix pattern);
	CSCMatrix SPAI0();
	CSCMatrix SPAI1();
	CSCMatrix SPAI2();

};