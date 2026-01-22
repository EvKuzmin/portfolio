
#include "mpi.h"
#include<vector>
#include<fstream>
#include "petsc_matrix_write.hpp"
#include<iomanip>

namespace pmw {

struct CSR {
    std::vector<int> ia;
    std::vector<int> ja;
    std::vector<double> data;
};

struct CSRO {
    std::vector<int> ia;
    std::vector<int> ja;
    std::vector<double> data;
    std::vector<int> garray;
};

void petsc_seq_matrix_write(int *ia, int *ja, double *data, int N){
    std::ofstream ia_file( "ia.txt");
    for( int i = 0 ; i < N+1 ; i++ ) {
        ia_file << ia[i] << "\n";
    }
    ia_file.close();
    std::ofstream ja_file( "ja.txt");
    for( int i = 0 ; i < ia[N] ; i++ ) {
        ja_file << ja[i] << "\n";
    }
    ja_file.close();
    std::ofstream data_file( "data.txt");
    for( int i = 0 ; i < ia[N] ; i++ ) {
        data_file << data[i] << "\n";
    }
    data_file.close();
}

void petsc_matrix_mpi_to_seq( int owner_range0, int owner_range1, const int *ia_d, const int *ja_d, const double *data_d, const int *ia_o, const int *ja_o, const double *data_o, const int *garray, int garray_size ) {

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_size = owner_range1 - owner_range0;
    std::vector<int> local_sizes(size);

    int global_size = 0;

    MPI_Allgather( &local_size, 1, MPI_INTEGER, local_sizes.data(), 1, MPI_INTEGER, MPI_COMM_WORLD );
    MPI_Allreduce( &local_size, &global_size, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD );

    int local_nnz_d = ia_d[local_size] - ia_d[0];
    int local_nnz_o = ia_o[local_size] - ia_o[0];

    std::vector<int> local_nnz_ds(size);
    std::vector<int> local_nnz_os(size);

    MPI_Allgather( &local_nnz_d, 1, MPI_INTEGER, local_nnz_ds.data(), 1, MPI_INTEGER, MPI_COMM_WORLD );
    MPI_Allgather( &local_nnz_o, 1, MPI_INTEGER, local_nnz_os.data(), 1, MPI_INTEGER, MPI_COMM_WORLD );

    std::vector<CSR> diag_parts(size);

    std::vector<CSRO> off_diag_parts(size);

    if(rank == 0) {
        for(int i = 0 ; i < size ; i++) {
            diag_parts[i].ia.resize(local_sizes[i]+1);
            diag_parts[i].ja.resize(local_nnz_ds[i]);
            diag_parts[i].data.resize(local_nnz_ds[i]);
            off_diag_parts[i].ia.resize(local_sizes[i]+1);
            off_diag_parts[i].ja.resize(local_nnz_os[i]);
            off_diag_parts[i].data.resize(local_nnz_os[i]);
            off_diag_parts[i].garray.resize(local_nnz_os[i]);
        }
    }
    MPI_Request si,sj,s;
    MPI_Isend(ia_d, local_size+1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &si);
    MPI_Isend(ja_d, local_nnz_d, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &sj);
    MPI_Isend(data_d, local_nnz_d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &s);
    MPI_Isend(ia_o, local_size+1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &si);
    MPI_Isend(ja_o, local_nnz_o, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &sj);
    MPI_Isend(data_o, local_nnz_o, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &s);
    MPI_Isend(garray, local_nnz_o, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &s);
    if(rank == 0) {
        std::vector<MPI_Request> r(size);
        for(int i = 0 ; i < size ; i++) {
            MPI_Irecv( diag_parts[i].ia.data(), local_sizes[i]+1, MPI_INTEGER, i, 0, MPI_COMM_WORLD, &(r[size]) );
            MPI_Irecv( diag_parts[i].ja.data(), local_nnz_ds[i], MPI_INTEGER, i, 0, MPI_COMM_WORLD, &(r[size]) );
            MPI_Irecv( diag_parts[i].data.data(), local_nnz_ds[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &(r[size]) );
            MPI_Irecv( off_diag_parts[i].ia.data(), local_sizes[i]+1, MPI_INTEGER, i, 0, MPI_COMM_WORLD, &(r[size]) );
            MPI_Irecv( off_diag_parts[i].ja.data(), local_nnz_os[i], MPI_INTEGER, i, 0, MPI_COMM_WORLD, &(r[size]) );
            MPI_Irecv( off_diag_parts[i].data.data(), local_nnz_os[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &(r[size]) );
            MPI_Irecv( off_diag_parts[i].garray.data(), local_nnz_os[i], MPI_INTEGER, i, 0, MPI_COMM_WORLD, &(r[size]) );
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) {
        std::vector<int> local_sizes_ex(size+1, 0);
        for(int i = 0 ; i < size+1 ; i++) {
            for(int j = 0 ; j < i ; j++) {
                local_sizes_ex[i]+=local_sizes[j];
            }
        }
        int nnz_sum = 0;
        for( auto v:local_nnz_ds ){
            nnz_sum += v;
        }
        for( auto v:local_nnz_os ){
            nnz_sum += v;
        }
        CSR res;
        res.ia.reserve(global_size+1);
        res.ja.reserve(nnz_sum);
        res.data.reserve(nnz_sum);

        auto check_left_off_diag = []( int r0, int r1, int size_ex_scan, const std::vector<int> &ja_o, const std::vector<double> &data_o, const std::vector<int> &garray, CSR& res )
        {
            for( int i = r0 ; i < r1 ; i++ ) {
                if( garray[ja_o[i]] < size_ex_scan ) {
                    res.ja.push_back( garray[ja_o[i]] );
                    res.data.push_back( data_o[i] );
                }
            }
        };

        auto copy_diag_part = []( int r0, int r1, int size_ex_scan, const std::vector<int> &ja_d, const std::vector<double> &data_d, CSR& res )
        {
            for( int i = r0 ; i < r1 ; i++ ) {
                res.ja.push_back( ja_d[i] + size_ex_scan );
                res.data.push_back( data_d[i] );
            }
        };

        auto check_right_off_diag = []( int r0, int r1, int size_ex_scan, const std::vector<int> &ja_o, const std::vector<double> &data_o, const std::vector<int> &garray, CSR& res )
        {
            for( int i = r0 ; i < r1 ; i++ ) {
                if( garray[ja_o[i]] >= size_ex_scan ) {
                    res.ja.push_back( garray[ja_o[i]] );
                    res.data.push_back( data_o[i] );
                }
            }
        };

        res.ia.push_back(0);

        for(int i = 0 ; i < size ; i++) {
            for( int j = 0 ; j < local_sizes[i] ; j++ ) {
                check_left_off_diag(off_diag_parts[i].ia[j], off_diag_parts[i].ia[j+1], local_sizes_ex[i], off_diag_parts[i].ja, off_diag_parts[i].data, off_diag_parts[i].garray, res);
                copy_diag_part(diag_parts[i].ia[j], diag_parts[i].ia[j+1], local_sizes_ex[i], diag_parts[i].ja, diag_parts[i].data, res);
                check_right_off_diag(off_diag_parts[i].ia[j], off_diag_parts[i].ia[j+1], local_sizes_ex[i+1], off_diag_parts[i].ja, off_diag_parts[i].data, off_diag_parts[i].garray, res);
                res.ia.push_back(res.ja.size());
            }
        }
        std::ofstream ia_file("ia.txt");
        std::ofstream ja_file("ja.txt");
        std::ofstream data_file("data.txt");
        std::ofstream row_sum_file("row_sum.txt");

        data_file << std::fixed << std::setprecision(14);
        row_sum_file << std::fixed << std::setprecision(14);
        for( auto v:res.ia ) {
            ia_file << v << "\n";
        }

        // for( int i = 0 ; i < res.ia.size()-1 ; i++ ) {
        //     double row_sum = 0.0;
        //     for( int j = res.ia[i] ; j < res.ia[i+1] ; j++ ) {
        //         ja_file << res.ja[j] << " ";
        //         data_file << res.data[j] << " ";
        //         row_sum += res.data[j];
        //     }
        //     ja_file << "\n";
        //     data_file << "\n";
        //     row_sum_file << row_sum << "\n";
        // }
        for( int i = 0 ; i < res.ia.size()-1 ; i++ ) {
            double row_sum = 0.0;
            for( int j = res.ia[i] ; j < res.ia[i+1] ; j++ ) {
                ja_file << res.ja[j] << "\n";
                data_file << res.data[j] << "\n";
                row_sum += res.data[j];
            }
            // ja_file << "\n";
            // data_file << "\n";
            row_sum_file << row_sum << "\n";
        }
        ia_file.close();
        ja_file.close();
        data_file.close();
        row_sum_file.close();
    }
}

void petsc_matrix_mpi_to_seq_readable( int owner_range0, int owner_range1, const int *ia_d, const int *ja_d, const double *data_d, const int *ia_o, const int *ja_o, const double *data_o, const int *garray, int garray_size ) {

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_size = owner_range1 - owner_range0;
    std::vector<int> local_sizes(size);

    int global_size = 0;

    MPI_Allgather( &local_size, 1, MPI_INTEGER, local_sizes.data(), 1, MPI_INTEGER, MPI_COMM_WORLD );
    MPI_Allreduce( &local_size, &global_size, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD );

    int local_nnz_d = ia_d[local_size] - ia_d[0];
    int local_nnz_o = ia_o[local_size] - ia_o[0];

    std::vector<int> local_nnz_ds(size);
    std::vector<int> local_nnz_os(size);

    MPI_Allgather( &local_nnz_d, 1, MPI_INTEGER, local_nnz_ds.data(), 1, MPI_INTEGER, MPI_COMM_WORLD );
    MPI_Allgather( &local_nnz_o, 1, MPI_INTEGER, local_nnz_os.data(), 1, MPI_INTEGER, MPI_COMM_WORLD );

    std::vector<CSR> diag_parts(size);

    std::vector<CSRO> off_diag_parts(size);

    if(rank == 0) {
        for(int i = 0 ; i < size ; i++) {
            diag_parts[i].ia.resize(local_sizes[i]+1);
            diag_parts[i].ja.resize(local_nnz_ds[i]);
            diag_parts[i].data.resize(local_nnz_ds[i]);
            off_diag_parts[i].ia.resize(local_sizes[i]+1);
            off_diag_parts[i].ja.resize(local_nnz_os[i]);
            off_diag_parts[i].data.resize(local_nnz_os[i]);
            off_diag_parts[i].garray.resize(local_nnz_os[i]);
        }
    }
    MPI_Request si,sj,s;
    MPI_Isend(ia_d, local_size+1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &si);
    MPI_Isend(ja_d, local_nnz_d, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &sj);
    MPI_Isend(data_d, local_nnz_d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &s);
    MPI_Isend(ia_o, local_size+1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &si);
    MPI_Isend(ja_o, local_nnz_o, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &sj);
    MPI_Isend(data_o, local_nnz_o, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &s);
    MPI_Isend(garray, local_nnz_o, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &s);
    if(rank == 0) {
        std::vector<MPI_Request> r(size);
        for(int i = 0 ; i < size ; i++) {
            MPI_Irecv( diag_parts[i].ia.data(), local_sizes[i]+1, MPI_INTEGER, i, 0, MPI_COMM_WORLD, &(r[size]) );
            MPI_Irecv( diag_parts[i].ja.data(), local_nnz_ds[i], MPI_INTEGER, i, 0, MPI_COMM_WORLD, &(r[size]) );
            MPI_Irecv( diag_parts[i].data.data(), local_nnz_ds[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &(r[size]) );
            MPI_Irecv( off_diag_parts[i].ia.data(), local_sizes[i]+1, MPI_INTEGER, i, 0, MPI_COMM_WORLD, &(r[size]) );
            MPI_Irecv( off_diag_parts[i].ja.data(), local_nnz_os[i], MPI_INTEGER, i, 0, MPI_COMM_WORLD, &(r[size]) );
            MPI_Irecv( off_diag_parts[i].data.data(), local_nnz_os[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &(r[size]) );
            MPI_Irecv( off_diag_parts[i].garray.data(), local_nnz_os[i], MPI_INTEGER, i, 0, MPI_COMM_WORLD, &(r[size]) );
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) {
        std::vector<int> local_sizes_ex(size+1, 0);
        for(int i = 0 ; i < size+1 ; i++) {
            for(int j = 0 ; j < i ; j++) {
                local_sizes_ex[i]+=local_sizes[j];
            }
        }
        int nnz_sum = 0;
        for( auto v:local_nnz_ds ){
            nnz_sum += v;
        }
        for( auto v:local_nnz_os ){
            nnz_sum += v;
        }
        CSR res;
        res.ia.reserve(global_size+1);
        res.ja.reserve(nnz_sum);
        res.data.reserve(nnz_sum);

        auto check_left_off_diag = []( int r0, int r1, int size_ex_scan, const std::vector<int> &ja_o, const std::vector<double> &data_o, const std::vector<int> &garray, CSR& res )
        {
            for( int i = r0 ; i < r1 ; i++ ) {
                if( garray[ja_o[i]] < size_ex_scan ) {
                    res.ja.push_back( garray[ja_o[i]] );
                    res.data.push_back( data_o[i] );
                }
            }
        };

        auto copy_diag_part = []( int r0, int r1, int size_ex_scan, const std::vector<int> &ja_d, const std::vector<double> &data_d, CSR& res )
        {
            for( int i = r0 ; i < r1 ; i++ ) {
                res.ja.push_back( ja_d[i] + size_ex_scan );
                res.data.push_back( data_d[i] );
            }
        };

        auto check_right_off_diag = []( int r0, int r1, int size_ex_scan, const std::vector<int> &ja_o, const std::vector<double> &data_o, const std::vector<int> &garray, CSR& res )
        {
            for( int i = r0 ; i < r1 ; i++ ) {
                if( garray[ja_o[i]] >= size_ex_scan ) {
                    res.ja.push_back( garray[ja_o[i]] );
                    res.data.push_back( data_o[i] );
                }
            }
        };

        res.ia.push_back(0);

        for(int i = 0 ; i < size ; i++) {
            for( int j = 0 ; j < local_sizes[i] ; j++ ) {
                check_left_off_diag(off_diag_parts[i].ia[j], off_diag_parts[i].ia[j+1], local_sizes_ex[i], off_diag_parts[i].ja, off_diag_parts[i].data, off_diag_parts[i].garray, res);
                copy_diag_part(diag_parts[i].ia[j], diag_parts[i].ia[j+1], local_sizes_ex[i], diag_parts[i].ja, diag_parts[i].data, res);
                check_right_off_diag(off_diag_parts[i].ia[j], off_diag_parts[i].ia[j+1], local_sizes_ex[i+1], off_diag_parts[i].ja, off_diag_parts[i].data, off_diag_parts[i].garray, res);
                res.ia.push_back(res.ja.size());
            }
        }
        std::ofstream ia_file("ia.txt");
        std::ofstream ja_file("ja.txt");
        std::ofstream data_file("data.txt");
        std::ofstream row_sum_file("row_sum.txt");

        data_file << std::fixed << std::setprecision(14);
        row_sum_file << std::fixed << std::setprecision(14);
        for( auto v:res.ia ) {
            ia_file << v << "\n";
        }

        for( int i = 0 ; i < res.ia.size()-1 ; i++ ) {
            double row_sum = 0.0;
            for( int j = res.ia[i] ; j < res.ia[i+1] ; j++ ) {
                ja_file << res.ja[j] << " ";
                data_file << res.data[j] << " ";
                row_sum += res.data[j];
            }
            ja_file << "\n";
            data_file << "\n";
            row_sum_file << row_sum << "\n";
        }
        // for( int i = 0 ; i < res.ia.size()-1 ; i++ ) {
        //     double row_sum = 0.0;
        //     for( int j = res.ia[i] ; j < res.ia[i+1] ; j++ ) {
        //         ja_file << res.ja[j] << "\n";
        //         data_file << res.data[j] << "\n";
        //         row_sum += res.data[j];
        //     }
        //     // ja_file << "\n";
        //     // data_file << "\n";
        //     row_sum_file << row_sum << "\n";
        // }
        ia_file.close();
        ja_file.close();
        data_file.close();
        row_sum_file.close();
    }
}

};
