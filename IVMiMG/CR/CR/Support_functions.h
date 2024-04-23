#pragma once
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <mkl.h>
#include <iomanip>
// "CEisP.h"

using namespace std;

typedef struct MatStruct
{
	int n;

	int* ia;
	int* ja;
	double* a;

}MatStruct;

int box_check(
	int x,
	int y,
	int z,
	int nx,
	int ny,
	int nz
);

//��������� �����
void diffconv3d_mx2(
	int nx,
	int ny,
	int nz,
	double p,
	double q,
	double r,
	int* n,
	int** ia,
	int** ja,
	double** a
);

//������ �������
double* exact(
	int nx,
	int ny,
	int nz
);

void print(
	double* vec,
	int size,
	string
	vec_name
);

void print(
	int* vec,
	int size,
	string vec_name
);

//������ � ���� ������� � ������� Excel
void print_vect_to_Excel(
	double* vec,
	int size,
	string vec_name
);

//������ � ���� ������� � ������� Wolfram
void print_to_Wolfram(
	double* vec,
	int size, 
	string vec_name
);

//����������� �����
double cheb_norm(
	double* vec,
	int size
);

//��������� �����
double eucl_norm(
	double* vec,
	int size);

double e_norm(
	double* vec,
	int size
);

//����� ���������� ���������� � �������������� ���������� MKL
void CG(
	int nrows,                  //����� ����� � �������
	sparse_matrix_t& A,         //���������� �.�.�. ������� � ������� csr
	double* u,                  //������ �������
	double* f,                  //������ �����
	double eps,                 //�������� �����������
	int itermax,                //������������ ����� ��������
	int* number_of_operations   //��������� ����� ����������� ��������
);

//����� ���������� �������
void CR(
	int nrows,
	sparse_matrix_t& A,
	double* u,
	double* f,
	double eps,
	int itermax,
	int* number_of_operations
);

//����� ���������� ������� � ������������ ����������
void CR_Eis(
	int nrows,
	double* data,
	int* irow,
	int* icol,
	double* u,
	double* f,
	double eps,
	double omega,
	int itermax,
	int* number_of_operations
);

//����� ���������� ������� � ������������������� A^T
void CR_AT(
	int nrows,
	sparse_matrix_t& A,
	double* u,
	double* f, double eps, int itermax, int* number_of_operations);

//����� ���������� ������� � ������������������� A^T � ������������ ����������
void CR_AT_Eis(
	int nrows,
	double* data,
	int* irow,
	int* icol,
	double* u,
	double* f,
	double eps,
	double omega,
	int itermax,
	int* number_of_operations
);

//������ ����� ��������������
void csr_cond(int nrows,
	int nelems,
	int* csr_pos,
	int* csr_j_index,
	double* csr_value,
	double eps
);

//�������, ������������ ����� ���������������
double csr_cond_ret(
	int nrows,
	int nelems,
	int* csr_pos,
	int* csr_j_index,
	double* csr_value,
	double eps
);

//������ ����� �������������� ��� ������� A^TA
void csr_cond_ATA(
	int nrows,
	int nelems,
	int* csr_pos,
	int* csr_j_index,
	double* csr_value,
	double eps
);

//��������� csr-������� �� �����
int mult_Ax(
	void* Matr,
	double* x, 
	double* Ax
);

int multMatrVect(
	void* Matr,
	double* v,
	double* Mv
);

int Init_MatStruct(
	int nrows,
	int* irow,
	int* icol,
	double* data,
	MatStruct* Str
);