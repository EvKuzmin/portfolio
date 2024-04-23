#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <mkl.h>
#include <iomanip>
#include "Support_functions.h"
#include <sstream>
#include "COO_2_CSR.h"
#include <time.h>

using namespace std;




void maian() {

	COOArrays coo;
	COOArrays coo_C;

	stringstream x;
	char word[256] = {}; 

	string line;
	string split_line[3];


	//File.mtx -> COO
	fstream mtx("T-H_4x4x4.mtx");
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
				coo_C.m = stoi(word);
			case 2:
				coo.nnz = stoi(word);
				coo_C.nnz = stoi(word);
			default:
				break;
			}

			i++;
		}
		double* val = new double[coo.nnz];    //< the values (size = nnz)
		double* valC = new double[coo.nnz];    //< the values (size = nnz)
		MKL_INT* rowind = new MKL_INT[coo.nnz];//< the row indexes (size = nnz)
		MKL_INT* rowindC = new MKL_INT[coo.nnz];//< the row indexes (size = nnz)
		MKL_INT* colind = new MKL_INT[coo.nnz];
		MKL_INT* colindC = new MKL_INT[coo.nnz];
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
					rowindC[j] = stoi(word) - 1;

				case 1:
					colind[j] = stoi(word) - 1;
					colindC[j] = stoi(word) - 1;
				case 2:
					val[j] = stod(word);
					valC[j] = stod(word);

				default:
					break;
				}
				i++;
			}
			j++;
			coo.rowind = rowind;
			coo.colind = colind;
			coo.val = val;

			coo_C.rowind = rowindC;
			coo_C.colind = colindC;
			coo_C.val = valC;

		}
		
		//cout << line << endl;
		

		//закрытие потока
		mtx.close();

		cout << "Reading done" << endl;
	}
	else cout << "Файл не существует" << endl;
	x.clear();


	//упорядочить


	//COO -> CRS

	CRSArrays crs;
	//ex_convert_COO_2_CSR(coo.m, coo.m, coo.nnz, coo.rowind, coo.colind, coo.val, &(crs.ia), &(crs.ja), &(crs.a));
	crs.m = coo.m;
	crs.nnz = coo.nnz;

	cout << "Convert done" << endl;



	sparse_matrix_t A;
	mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, crs.m, crs.m, crs.ia, (crs.ia + 1), crs.ja, crs.a);
	double* u = new double[crs.m];
	double* u_fG = new double[crs.m];
	double* u_G = new double[crs.m];
	double* f_G = new double[crs.m];
	double* f = new double[crs.m];
	double* u_exact = new double[crs.m];
	double* res = new double[crs.m];
	double* Au = new double[crs.m];
	//ofstream report("Report_4x4x4.txt");
	matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_UNIT };
	for (int i = 0; i < crs.m; i++) u_exact[i] = 45.; //точное решение
	for (int i = 0; i < crs.m; i++)	u[i] = -10.; //начальное приближение
	for (int i = 0; i < crs.m; i++)	u_fG[i] = -10.; //начальное приближение
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, { SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_UNIT }, u_exact, 0, f);


	//f_G read
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
	//u_G read
	fstream u_G_srt("solution_values4.txt");
	if (u_G_srt)
	{
		int i = 0;
		while (getline(u_G_srt, line))
		{
			u_G[i] = stod(line);
			i++;

		}

		//закрытие потока
		u_G_srt.close();

		cout << "Reading u_G done" << endl;
	}
	else cout << "File don't exist" << endl;






	int number_of_operations = 0;
	double eps = 1e-8;
	int itermax = 100000;
	CR(crs.m, A, u, f_G, eps, itermax, &number_of_operations);
	//number_of_operations = 0;
	clock_t t = clock();
	//CR(crs.m, A, u, f, eps, itermax, &number_of_operations);
	////CR(crs.m, A, u, f, eps, itermax, &number_of_operations);
	double t_alg = (double)(clock() - t) / CLOCKS_PER_SEC;


	//p ind read
	int len_p = 125;
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

	double* p_value = new double[len_p];
	for (int i = 0; i < len_p; i++) {
		p_value[i] = u[p_ind[i]];
	}
	//записываем p
	ofstream report("p_value.txt");
	report << "p value" << endl;
	int k = 0;
	for (int i = 0; i < 25; i++) {
		for (int j = 0; j < 5; j++) {
			report << p_value[5 * i + j] << '\t';
		}
		report << '\n';
		k++;
		if (k % 5 == 0) report << endl;

	}




	////CR(crs.m, A, u_fG, f_G, eps, itermax, &number_of_operations);

	
	/*
	report << "Iterations carried out (N_iter): " << number_of_operations << "\n";
	report << "\n";
	mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1., A, descr, u, 0, Au);
	copy(f, f + crs.m, res);
	cblas_daxpy(crs.m, -1, Au, 1, res, 1);
	report << "Norm of total residual ||Au - f|| = " << cheb_norm(res ,crs.m) << endl;
	cblas_daxpy(crs.m, -1, u_G, 1, u_fG, 1);
	report << "Norm of difference ||u_G - u|| = " << cheb_norm(u_fG, crs.m) << endl;
	cblas_daxpy(crs.m, -1, u, 1, u_exact, 1);
	report << "Error ||u_exact - u|| = " << cheb_norm(u_exact, crs.m) << endl;
	report << "eps = " << eps << endl;
	report << "N of rows = " << crs.m << endl;
	report << "time,sec = " << t_alg << endl;
	report << "\n"; report << "\n"; report << "\n"; report << "\n"; report << "\n"; report << "\n"; report << "\n"; report << "\n";
	cout << "Computing 1st done" << endl;*/





	//eps = 1e-8;
	//for (int i = 0; i < crs.m; i++)	u[i] = -10.; //начальное приближение
	//for (int i = 0; i < crs.m; i++)	u_fG[i] = -10.; //начальное приближение
	//number_of_operations = 0;
	////CR(crs.m, A, u, f, eps, itermax, &number_of_operations);
	//number_of_operations = 0;
	//t = clock();
	//CR(crs.m, A, u_G, f_G, eps, itermax, &number_of_operations);
	//t_alg = (double)(clock() - t) / CLOCKS_PER_SEC;
	//report << "Iterations carried out (N_iter): " << number_of_operations << "\n";
	//report << "\n";
	//mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1., A, descr, u_G, 0, Au);
	//copy(f_G, f_G + crs.m, res);
	//cblas_daxpy(crs.m, -1, Au, 1, res, 1);
	//report << "Norm of total residual ||Au - f|| = " << eucl_norm(res, crs.m) << endl;
	//cblas_daxpy(crs.m, -1, u_G, 1, u_fG, 1);
	//report << "Norm of difference ||u_G - u|| = " << cheb_norm(u_fG, crs.m) << endl;
	//cblas_daxpy(crs.m, -1, u, 1, u_exact, 1);
	//report << "Error ||u_exact - u|| = " << cheb_norm(u_exact, crs.m) << endl;
	//report << "eps = " << eps << endl;
	//report << "N of rows = " << crs.m << endl;
	//report << "time,sec = " << t_alg << endl;
	//report << "\n"; report << "\n"; report << "\n"; report << "\n"; report << "\n"; report << "\n"; report << "\n"; report << "\n";

	//cout << "Computing 2nd done" << endl;


	//eps = 1e-6;
	//for (int i = 0; i < crs.m; i++)	u[i] = -10.; //начальное приближение
	//for (int i = 0; i < crs.m; i++)	u_fG[i] = -10.; //начальное приближение
	//number_of_operations = 0;
	//CR(crs.m, A, u, f, eps, itermax, &number_of_operations);
	//number_of_operations = 0;
	//t = clock();
	//CR(crs.m, A, u_G, f_G, eps, itermax, &number_of_operations);
	//t_alg = (double)(clock() - t) / CLOCKS_PER_SEC;
	//report << "Iterations carried out (N_iter): " << number_of_operations << "\n";
	//report << "\n";
	//mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1., A, descr, u_G, 0, Au);
	//copy(f, f + crs.m, res);
	//cblas_daxpy(crs.m, -1, Au, 1, res, 1);
	//report << "Norm of total residual ||Au - f|| = " << cheb_norm(res, crs.m) << endl;
	//cblas_daxpy(crs.m, -1, u_G, 1, u_fG, 1);
	//report << "Norm of difference ||u_G - u|| = " << cheb_norm(u_fG, crs.m) << endl;
	//cblas_daxpy(crs.m, -1, u, 1, u_exact, 1);
	//report << "Error ||u_exact - u|| = " << cheb_norm(u_exact, crs.m) << endl;
	//report << "eps = " << eps << endl;
	//report << "N of rows = " << crs.m << endl;
	//report << "time,sec = " << t_alg << endl;
	//report << "\n"; report << "\n"; report << "\n"; report << "\n"; report << "\n"; report << "\n"; report << "\n"; report << "\n";


	//report.close();






}