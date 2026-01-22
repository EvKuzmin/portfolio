#pragma once

namespace pmw {
    void petsc_matrix_mpi_to_seq( int owner_range0, int owner_range1, const int *ia_d, const int *ja_d, const double *data_d, const int *ia_o, const int *ja_o, const double *data_o, const int *garray, int garray_size );
    void petsc_seq_matrix_write(int *ia, int *ja, double *data, int N);
    void petsc_matrix_mpi_to_seq_readable( int owner_range0, int owner_range1, const int *ia_d, const int *ja_d, const double *data_d, const int *ia_o, const int *ja_o, const double *data_o, const int *garray, int garray_size );
    
};
