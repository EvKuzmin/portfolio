#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "SPAI_func.h"
#include <iomanip>
#include <sstream>
#include <Sparse>
#include <Eigen>
#include <Dense>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <set>
using namespace std;

typedef Eigen::Triplet<double> T;

COOArrays::COOArrays() {
	val = NULL;
	rowind = NULL;
	colind = NULL;
}
CRSArrays::CRSArrays() {
	a = NULL;
	ia = NULL;
	ja = NULL;
}
CSCArrays::CSCArrays() {
	a = NULL;
	ia = NULL;
	ja = NULL;
}
void CRSArrays::print_crs() const{
	for (int i = 0; i < n; i++) {
		std::cout << "        ";
		for (int j = ia[i]; j < ia[i + 1]; j++) {
			std::cout << " " << std::setw(8) << ja[j];
		}
		std::cout << "\n";
		std::cout << std::setw(8) << i;
		for (int j = ia[i]; j < ia[i + 1]; j++) {
			std::cout << " " << std::setw(8) << std::setprecision(3) << a[j];
		}
		std::cout << "\n\n";
	}
}
SystemCOO::SystemCOO(string matrix_name, string rp_name) {
	read_system(matrix_name, rp_name);
	_sort_type = -1;
}
vector<double> System::load_rp(string filename) {
	string line;
	fstream g(filename);
	if (!g) {
		std::cerr << "нет файла" << endl;
		std::exit(1);
	}
	int count = 0;
	while (getline(g, line)) {
		count++;
	}
	g.close();
	vector<double> f_c(count);
	fstream f(filename);
	int i = 0;
	while (getline(f, line))
	{
		f_c[i] = stod(line);
		i++;
	}
	f.close();
	rp = f_c;
	return rp;
}
COOArrays SystemCOO::load_coo_mtx(string filename) {
	fstream mtx(filename);
	stringstream x;
	char word[256] = {};
	string line;
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
				coo_matrix.n = stoi(word);
			case 1:
				coo_matrix.m = stoi(word);
			case 2:
				coo_matrix.nnz = stoi(word);
			default:
				break;
			}

			i++;
		}
		double* val = new double[coo_matrix.nnz];    //< the values (size = nnz)

		int* rowind = new int[coo_matrix.nnz];//< the row indexes (size = nnz)

		int* colind = new int[coo_matrix.nnz];

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
			coo_matrix.rowind = rowind;
			coo_matrix.colind = colind;
			coo_matrix.val = val;


		}

		//cout << line << endl;


		//закрытие потока
		mtx.close();

		cout << "Reading done" << endl;
	}
	else cout << "‘айл не существует" << endl;
	x.clear();
	return coo_matrix;
}
void SystemCOO::read_system(string matrix, string right_part) {
	load_coo_mtx(matrix);
	load_rp(right_part);

}
void SystemCOO::sort_ia() {
	std::vector<int> indexes(coo_matrix.nnz);
	std::iota(indexes.begin(), indexes.end(), 0);
	auto compare = [this](int a, int b) {
		if (coo_matrix.rowind[a] < coo_matrix.rowind[b]) {
			return true;
		}
		else if (coo_matrix.rowind[a] == coo_matrix.rowind[b]) {
			if (coo_matrix.colind[a] < coo_matrix.colind[b]) return true;
			else return false;
		}
		else return false;
	};
	std::sort(indexes.begin(), indexes.end(), compare);
	std::vector<double> a(coo_matrix.nnz);
	std::vector<int> ia(coo_matrix.nnz);
	std::vector<int> ja(coo_matrix.nnz);

	for (int i = 0; i < coo_matrix.nnz; i++) {
		a[i] = coo_matrix.val[indexes[i]];
		ia[i] = coo_matrix.rowind[indexes[i]];
		ja[i] = coo_matrix.colind[indexes[i]];
	}
	std::copy(a.begin(), a.end(), coo_matrix.val);
	std::copy(ia.begin(), ia.end(), coo_matrix.rowind);
	std::copy(ja.begin(), ja.end(), coo_matrix.colind);
	_sort_type = 0;

}
void SystemCOO::sort_ja() {
	std::vector<int> indexes(coo_matrix.nnz);
	std::iota(indexes.begin(), indexes.end(), 0);
	auto compare = [this](int a, int b) {
		if (coo_matrix.colind[a] < coo_matrix.colind[b]) {
			return true;
		}
		else if (coo_matrix.colind[a] == coo_matrix.colind[b]) {
			if (coo_matrix.rowind[a] < coo_matrix.rowind[b]) return true;
			else return false;
		}
		else return false;
	};
	std::sort(indexes.begin(), indexes.end(), compare);
	std::vector<double> a(coo_matrix.nnz);
	std::vector<int> ia(coo_matrix.nnz);
	std::vector<int> ja(coo_matrix.nnz);

	for (int i = 0; i < coo_matrix.nnz; i++) {
		a[i] = coo_matrix.val[indexes[i]];
		ia[i] = coo_matrix.rowind[indexes[i]];
		ja[i] = coo_matrix.colind[indexes[i]];
	}
	std::copy(a.begin(), a.end(), coo_matrix.val);
	std::copy(ia.begin(), ia.end(), coo_matrix.rowind);
	std::copy(ja.begin(), ja.end(), coo_matrix.colind);
	_sort_type = 1;
}
SystemCSC SystemCOO::coo_to_csc() {
	sort_ja();
	vector<int> ja(coo_matrix.m + 1);
	int h = 1;
	ja[0] = 0;
	int k = 1;
	for (int i = 0; i < coo_matrix.nnz - 1; i++) {
		if (coo_matrix.colind[i] == coo_matrix.colind[i + 1]) h++;
		else {
			ja[k] = h;
			h++;
			k++;
		}
	}
	ja[coo_matrix.m] = coo_matrix.nnz;
	std::vector<double> a(coo_matrix.nnz);
	std::vector<int> ia(coo_matrix.nnz);
	std::copy(coo_matrix.val, coo_matrix.val + coo_matrix.nnz, a.begin());
	std::copy(coo_matrix.rowind, coo_matrix.rowind + coo_matrix.nnz, ia.begin());
	SystemCSC csc(coo_matrix.n, coo_matrix.m, coo_matrix.nnz, 
		rp, a, ia, ja);
	return csc;
}
SystemCOO SystemCRS::crs_to_coo(){
	std::vector<int> ia;
	ia.reserve(crs_matrix.nnz);
	std::vector<double> a(crs_matrix.nnz);
	std::vector<int> ja(crs_matrix.nnz);
	std::copy(crs_matrix.a, crs_matrix.a + crs_matrix.nnz, a.begin());
	std::copy(crs_matrix.ja, crs_matrix.ja + crs_matrix.nnz, ja.begin());
	for (int i = 0; i < crs_matrix.n; i++) {
		for (int j = crs_matrix.ia[i]; j < crs_matrix.ia[i + 1]; j++) {
			ia.push_back(i);
		}
	}
	SystemCOO coo(crs_matrix.n, crs_matrix.m, crs_matrix.nnz,
		rp, a, ia, ja);
	coo._sort_type = 0;
	return coo;
}
SystemCSC SystemCRS::crs_to_csc(){
	SystemCOO coo = this->crs_to_coo();
	return coo.coo_to_csc();
}
SystemCRS SystemCSC::csc_to_crs() {
	SystemCOO coo = this->csc_to_coo();
	return coo.coo_to_crs();
}
SystemCOO SystemCSC::csc_to_coo(){
	std::vector<int> ia(csc_matrix.nnz);
	std::vector<double> a(csc_matrix.nnz);
	std::vector<int> ja;
	ja.reserve(csc_matrix.nnz);
	std::copy(csc_matrix.a, csc_matrix.a + csc_matrix.nnz, a.begin());
	std::copy(csc_matrix.ia, csc_matrix.ia + csc_matrix.nnz, ia.begin());

	for (int i = 0; i < csc_matrix.n; i++) {
		for (int j = csc_matrix.ja[i]; j < csc_matrix.ja[i + 1]; j++) {
			ja.push_back(i);
		}
	}
	SystemCOO coo(csc_matrix.n, csc_matrix.m, csc_matrix.nnz,
		rp, a, ia, ja);
	coo._sort_type = 1;
	return coo;
}
SystemCRS SystemCOO::coo_to_crs() {
	sort_ia();
	vector<int> ia(coo_matrix.n + 1);
	int h = 1;
	ia[0] = 0;
	int k = 1;
	for (int i = 0; i < coo_matrix.nnz - 1; i++) {
		if (coo_matrix.rowind[i] == coo_matrix.rowind[i + 1]) h++;
		else {
			ia[k] = h;
			h++;
			k++;
		}
	}
	ia[coo_matrix.n] = coo_matrix.nnz;
	std::vector<double> a(coo_matrix.nnz);
	std::vector<int> ja(coo_matrix.nnz);
	std::copy(coo_matrix.val, coo_matrix.val + coo_matrix.nnz, a.begin());
	std::copy(coo_matrix.colind, coo_matrix.colind + coo_matrix.nnz, ja.begin());
	SystemCRS crs(coo_matrix.n, coo_matrix.m, coo_matrix.nnz,
		rp, a, ia, ja);
	return crs;
}
SystemCRS::SystemCRS(int n, int m, int nnz, vector<double> rprt,
	vector<double> a, vector<int> ia, vector<int> ja) {
	crs_matrix.n = n;
	crs_matrix.m = m;
	crs_matrix.nnz = nnz;
	rp = rprt;
	crs_matrix.a = new double[nnz];
	crs_matrix.ia = new int[n + 1];
	crs_matrix.ja = new int[nnz];
	std::copy(a.begin(), a.end(), crs_matrix.a);
	std::copy(ia.begin(), ia.end(), crs_matrix.ia);
	std::copy(ja.begin(), ja.end(), crs_matrix.ja);

}
SystemCSC::SystemCSC(int n, int m, int nnz,
	vector<double> rprt, vector<double> a, vector<int> ia, vector<int> ja) {
	csc_matrix.n = n;
	csc_matrix.m = m;
	csc_matrix.nnz = nnz;
	rp = rprt;
	csc_matrix.a = new double[nnz];
	csc_matrix.ia = new int[nnz];
	csc_matrix.ja = new int[m+1];
	std::copy(a.begin(), a.end(), csc_matrix.a);
	std::copy(ia.begin(), ia.end(), csc_matrix.ia);
	std::copy(ja.begin(), ja.end(), csc_matrix.ja);
}
SystemCOO::SystemCOO(int n, int m, int nnz, vector<double> rprt,
	vector<double> a, vector<int> ia, vector<int> ja) {
	_sort_type = -1;
	coo_matrix.n = n;
	coo_matrix.m = m;
	coo_matrix.nnz = nnz;
	rp = rprt;
	coo_matrix.val = new double[nnz];
	coo_matrix.rowind = new int[nnz];
	coo_matrix.colind = new int[nnz];
	std::copy(a.begin(), a.end(), coo_matrix.val);
	std::copy(ia.begin(), ia.end(), coo_matrix.rowind);
	std::copy(ja.begin(), ja.end(), coo_matrix.colind);
}
CRSArrays CRSArrays::get_csr_submatrix(int* rows, int* cols, int N, int M) {
	CRSArrays res;
	std::vector<int> ia_new(N + 1, 0);
	std::vector<int> ja_new;
	std::vector<double> a_new;
	int n = 0;
	for (int i = 0; i < N; i++) {
		int row = rows[i];
		for (int j = ia[row]; j < ia[row + 1]; j++) {
			int col = ja[j];
			auto it = std::lower_bound(cols, cols + M, col);
			if (it != cols + M)
				if ((*it) == col) {
					n++;
					ja_new.push_back(it - cols);
					a_new.push_back(a[j]);
				}
		}
		ia_new[i + 1] = n;
	}
	res.n = N;
	res.m = M;
	res.nnz = ja_new.size();
	res.ia = new int[N + 1];
	res.ja = new int[res.nnz];
	res.a = new double[res.nnz];
	std::copy(ia_new.begin(), ia_new.end(), res.ia);
	std::copy(ja_new.begin(), ja_new.end(), res.ja);
	std::copy(a_new.begin(), a_new.end(), res.a);
	return res;
}
CSCArrays SystemCOO::SPAI_precond(CSCArrays pattern) {
	return this->coo_to_csc().SPAI_precond(pattern);
}
CSCArrays SystemCRS::SPAI_precond(CSCArrays pattern) {
	return this->crs_to_csc().SPAI_precond(pattern);
};
CSCArrays SystemCSC::SPAI_precond(CSCArrays pattern) {
	CSCArrays M;
	M.n = csc_matrix.n;
	M.m = csc_matrix.m;
	M.nnz = 0;
	vector<double> a;
	vector<int> ia;
	vector<int> ja;
	int count = 0;
	ja.push_back(count);

	// get AT in CRS
	CRSArrays AT;
	AT.n = csc_matrix.m;
	AT.m = csc_matrix.n;
	AT.nnz = csc_matrix.nnz;
	AT.a = new double[csc_matrix.nnz];
	AT.ia = new int[csc_matrix.n + 1];
	AT.ja = new int[csc_matrix.nnz];
	std::copy(csc_matrix.a, csc_matrix.a + csc_matrix.nnz, AT.a);
	std::copy(csc_matrix.ia, csc_matrix.ia + csc_matrix.nnz, AT.ja);
	std::copy(csc_matrix.ja, csc_matrix.ja + csc_matrix.n + 1, AT.ia);

	for (int j = 1; j <= pattern.m; j++) {
		//J_j
		vector<int> J;
		for (int m = pattern.ja[j - 1]; m < pattern.ja[j]; m++) {
			J.push_back(pattern.ia[m]);
		}
		//I_j
		set<int> I;																	
		for (auto it = J.begin(); it != J.end(); it++) {
			for (int m = csc_matrix.ja[*it]; m < csc_matrix.ja[*it + 1]; m++) {
				I.insert(csc_matrix.ia[m]);
			}
		}
		
		// get AT_sub
		int* rows = new int[J.size()];
		std::copy(J.begin(), J.end(), rows);
		int* cols = new int[I.size()];
		std::copy(I.begin(), I.end(), cols);
		CRSArrays AT_sub = AT.get_csr_submatrix(rows, cols, J.size(), I.size());
		//получить матрицу eigen, умножить на ее же транспонированную и решить систему
		Eigen::MatrixXd AT_sub_mat = Eigen::MatrixXd::Zero(J.size(), I.size());
		for (int i = 0; i < AT_sub.n; i++) {
			for (int j = AT_sub.ia[i]; j < AT_sub.ia[i + 1]; j++) {
				AT_sub_mat(i, AT_sub.ja[j]) = AT_sub.a[j];
			}
		}
		Eigen::MatrixXd D = 2 * AT_sub_mat * AT_sub_mat.transpose();
		Eigen::VectorXd e = Eigen::VectorXd::Zero(AT_sub.m);
		{																	//окружение?
			int counter = 0;
			for (auto it = I.begin(); it != I.end(); it++) {
				if (*it == j - 1) {
					e(counter) = 1;
				}
				counter++;
			}
		}
		Eigen::VectorXd b = 2 * AT_sub_mat * e;
		Eigen::VectorXd m = D.householderQr().solve(b);
		for (int i = 0; i < m.size(); i++) {
			a.push_back(m(i));
			ia.push_back(J[i]);
		}
		count += m.size();
		ja.push_back(count);
	}

	M.ia = new int[ia.size()]; //убрать нули
	M.ja = new int[ja.size()];
	M.a = new double[a.size()];
	M.nnz = a.size();
	std::copy(ia.begin(), ia.end(), M.ia);
	std::copy(ja.begin(), ja.end(), M.ja);
	std::copy(a.begin(), a.end(), M.a);
	return M;
}
Eigen::SparseMatrix<double> get_eigen_sparse(vector<double> coo_a, vector<int> coo_ia, vector<int> coo_ja, int n, int m) {
	std::vector<T> triplets;
	triplets.reserve(coo_ia.size());
	for (int i = 0; i < coo_ia.size(); i++) {
		triplets.push_back(T(coo_ia[i], coo_ja[i], coo_a[i]));
	}
	Eigen::SparseMatrix<double> M(n, m);
	M.setFromTriplets(triplets.begin(), triplets.end());
	return M;
}

COOArrays CSCArrays::csc_to_coo() {
	std::vector<int> ja;
	ja.reserve(nnz);

	for (int i = 0; i < this->n; i++) {
		for (int j = this->ja[i]; j < this->ja[i + 1]; j++) {
			ja.push_back(i);
		}
	}
	COOArrays coo;
	coo.m = this->m;
	coo.n = this->n;
	coo.nnz = this->nnz;
	coo.val = new double[coo.nnz];
	coo.rowind = new int[coo.nnz];
	coo.colind = new int[coo.nnz];
	std::copy(this->a, this->a + nnz, coo.val);
	std::copy(this->ia, this->ia + nnz, coo.rowind);
	std::copy(ja.begin(), ja.end(), coo.colind);
	return coo;
}