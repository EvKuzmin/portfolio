#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <mkl.h>
#include <iomanip>
#include <sstream>
#include "COO_2_CSR.h"
#include <time.h>
#include<array>

//заменить линейные пробеги в void CR_precond
//read_linear(string filename,MKL_INT* array)
//read_linear(string filename,double* array)
//read_mtx(string filename, COOArrays* coo)
//report(string filename, string abstr, MKL_INT n_of_oper, double* u, double* f, sparse_matrix_t& A, double eps, double time)
using namespace std;

int main() {
	COOArrays coo;

	stringstream x;
	char word[256] = {};
	string line;
	string split_line[3];
	//File.mtx -> COO
	fstream mtx("matrix_without_zeros.mtx");
	if (mtx)
	{
		while (getline(mtx, line))
		{
			if ((int(line[0]) == int('%'))) {
				//cout << line << endl;
			}
			else break;
		}
		x << line;
		int i = 0;
		while (x >> word) {
			switch (i)
			{
			case 0:
				coo.m = stoi(word);
			case 2:
				coo.nnz = stoi(word);
			default:
				break;
			}

			i++;
		}
		double* val = new double[coo.nnz];    //< the values (size = nnz)

		MKL_INT* rowind = new MKL_INT[coo.nnz];//< the row indexes (size = nnz)

		MKL_INT* colind = new MKL_INT[coo.nnz];

		int j = 0;
		while (getline(mtx, line))
		{
			i = 0;
			x.clear();
			x << line;
			while (x >> word) {
				switch (i)
				{
				case 0:
					rowind[j] = stoi(word) - 1;


				case 1:
					colind[j] = stoi(word) - 1;

				case 2:
					val[j] = stod(word);


				default:
					break;
				}
				i++;
			}
			j++;
			coo.rowind = rowind;
			coo.colind = colind;
			coo.val = val;


		}

		//cout << line << endl;


		//закрытие потока
		mtx.close();

		cout << "Reading done" << endl;
	}
	else cout << "Файл не существует" << endl;
	x.clear();

	
	
	//p ind read
	int len_p = len_file("p_dof_ids4.txt") - 1 ;
	MKL_INT* p_ind = new MKL_INT[len_p];
	fstream p_ind_srt("p_dof_ids4.txt");
	if (p_ind_srt)
	{
		int i = 0;
		while (getline(p_ind_srt, line))
		{
			p_ind[i] = stod(line);
			i++;

		}

		//закрытие потока
		p_ind_srt.close();

		cout << "Reading p_ind done" << endl;

	}
	else cout << "File don't exist" << endl;


	//u ind read
	int len_u = len_file("u_dof_ids4.txt")-1;
	MKL_INT* u_ind = new MKL_INT[len_u];
	fstream u_ind_srt("u_dof_ids4.txt");
	if (u_ind_srt)
	{
		int i = 0;
		while (getline(u_ind_srt, line))
		{
			u_ind[i] = stod(line);
			i++;

		}

		//закрытие потока
		u_ind_srt.close();

		cout << "Reading u_ind done" << endl;
	}
	else cout << "File don't exist" << endl;


	//COO -> CRS

	CRSArrays crs;
	ex_convert_COO_2_CSR(coo.m, coo.m, coo.nnz, coo.rowind, coo.colind, coo.val, &(crs.ia), &(crs.ja), &(crs.a));
	crs.m = coo.m;
	crs.n = coo.m;
	crs.nnz = coo.nnz;
	cout << "Convert done" << endl;


	//f_G read
	double* f_G = new double[crs.m];
	fstream f_G_srt("vector_values4.txt");
	if (f_G_srt)
	{
		int i = 0;
		while (getline(f_G_srt, line))
		{
			f_G[i] = stod(line);
			i++;

		}

		//закрытие потока
		f_G_srt.close();

		cout << "Reading f done" << endl;
	}
	else cout << "File don't exist" << endl;


	double* u = new double[crs.n];
	for (int i = 0; i < crs.n; i++) u[i] = 1;
	double eps = 1e-7;
	int itermax = 1000;
	int n_of_oper;
	double gamma = 0;
	CRSSystem A;
	A.f = f_G;
	A.M = crs;


	//sparse_matrix_t crsA;
	//mkl_sparse_d_create_csr(&crsA, SPARSE_INDEX_BASE_ZERO, crs.n, crs.m, crs.ia, (crs.ia + 1), crs.ja, crs.a);
	//CR(crs.m, crsA, u, f_G, eps, itermax, &n_of_oper);



	CRSSystem CRSA_ = precondCCT(A, gamma, p_ind, u_ind, len_p, len_u);
	sparse_matrix_t A_;
	mkl_sparse_d_create_csr(&A_, SPARSE_INDEX_BASE_ZERO, CRSA_.M.n, CRSA_.M.m, CRSA_.M.ia, (CRSA_.M.ia + 1), CRSA_.M.ja, CRSA_.M.a);
	CR(crs.m, A_, u, CRSA_.f, eps, itermax, &n_of_oper);

	///////////////////////////тестим
	//CRSArrays X;
	//X.m = 3;
	//X.n = 3;
	//X.nnz = 5;
	//double* Xa = new double[X.nnz] {5, 5, 5, 5, 5};
	//MKL_INT* Xja = new MKL_INT[X.nnz]{ 0,2,1,0,2 };
	//MKL_INT* Xia = new MKL_INT[X.n + 1]{ 0,2,3,5 };
	//X.a = Xa;
	//X.ia = Xia;
	//X.ja = Xja;
	//X.print();

	//CRSArrays Y;
	//Y.m = 3;
	//Y.n = 5;
	//Y.nnz = 9;
	//double* Ya = new double[Y.nnz] {7, 7, 7, 7, 7, 7, 7, 7, 7};
	//MKL_INT* Yja = new MKL_INT[Y.nnz]{ 0,1,2,0,1,2,0,1,2 };
	//MKL_INT* Yia = new MKL_INT[Y.n + 1]{ 0,3,3,6,6,9 };
	//Y.a = Ya;
	//Y.ia = Yia;
	//Y.ja = Yja;
	//Y.print();

	//CRSArrays XY = vertical_concatinate(X, Y);
	//XY.print();
	/////////////////////////////


	return 0;
}


