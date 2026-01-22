#include"common.hpp"

#include<algorithm>

#include "aij.h"
#include "mpiaij.h"
#include "dvecimpl.h"
#include "vecimpl.h"

std::vector<int> find_dirichlet_csr( Mat A ) {

    PetscInt m;
    PetscInt n;

    MatGetOwnershipRange(A, &m, &n);
    std::vector<int> local_dirichlet_rows;

    void *A_raw_data = A->data;
    Mat_MPIAIJ *A_mpi_data = (Mat_MPIAIJ *)A_raw_data;

    Mat A_diag = A_mpi_data->A;
    Mat A_off_diag = A_mpi_data->B;

    void *A_diag_raw_data = A_diag->data;
    Mat_SeqAIJ *A_diag_data = (Mat_SeqAIJ *)A_diag_raw_data;
    void *A_off_diag_raw_data = A_off_diag->data;
    Mat_SeqAIJ *A_off_diag_data = (Mat_SeqAIJ *)A_off_diag_raw_data;

    PetscInt *ia_diag = A_diag_data->i;
    PetscInt *ja_diag = A_diag_data->j;
    PetscScalar *data_diag = A_diag_data->a;

    PetscInt *ia_off_diag = A_off_diag_data->i;
    PetscInt *ja_off_diag = A_off_diag_data->j;
    PetscScalar *data_off_diag = A_off_diag_data->a;
    PetscInt *garray = A_mpi_data->garray;

    PetscInt N_diag, M_diag;
    MatGetSize(A_diag,&N_diag,&M_diag);

    PetscInt N_off, M_off;
    MatGetSize(A_off_diag,&N_off,&M_off);

    for( int i = 0 ; i < N_diag ; i++ ) {
        int found_nonzero = 0;
        bool found_one_diag = false;
        for( int j = ia_diag[i] ; j < ia_diag[i+1] ; j++ ) {
            if(data_diag[j] != 0.0) {
                found_nonzero++;
                if(data_diag[j] == 1.0 && ja_diag[j] == i) {
                    found_one_diag = true;
                }
            }
        }
        for( int j = ia_off_diag[i] ; j < ia_off_diag[i+1] ; j++ ) {
            if(data_off_diag[j] != 0.0) {
                found_nonzero++;
            }
        }
        if(found_nonzero == 1 && found_one_diag) {
            local_dirichlet_rows.push_back(i + m);
        }
    }
    std::sort( local_dirichlet_rows.begin(), local_dirichlet_rows.end() );
    return local_dirichlet_rows;
}

void apply_dirichlet_csr(
    Mat A, 
    Vec rhs,
    const std::vector<int>& dirichlet_rows
    ) {    
    PetscScalar diag = 1;   
    Vec x;
    VecDuplicate(rhs, &x);
    VecCopy(rhs, x);
    MatZeroRowsColumns(A, dirichlet_rows.size(), dirichlet_rows.data(), diag, x, rhs);  
    VecDestroy(&x);
}