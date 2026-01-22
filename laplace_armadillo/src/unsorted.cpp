#include"unsorted.hpp"


std::vector<int> coarsening_MIS(arma::sp_mat A, int dist)
{
    arma::sp_mat App_wo = A;
    App_wo = App_wo.t();

    App_wo.sync();

    std::vector<char> processed(App_wo.n_cols, 0);
    std::vector<int> coarse_elements;
    std::vector<int> coarse_elements_to_save(4*4*4);
    std::vector<int> close_neighbors;
    std::vector<int> close_neighbors2;
    // processed[0]=1;
    bool continue_process = true;

    // for(int i = 0 ; i < App_wo.n_cols ; i++) {
    for(int i = 0 ; i < App_wo.n_cols ; i++) {
        if(processed[i]==0){
            coarse_elements.push_back(i);
            // processed[i] = 1;
            close_neighbors.push_back(i);
            for( int d = 0 ; d < dist ; d++ ) {
                for( auto l : close_neighbors ) {
                    processed[l] = 1;
                    for( int j = App_wo.col_ptrs[l] ; j < App_wo.col_ptrs[l+1] ; j++ ) {
                        if(App_wo.row_indices[j] != l) {
                            if(processed[App_wo.row_indices[j]] == 0) {
                                close_neighbors2.push_back(App_wo.row_indices[j]);
                            }
                        }
                    }
                }
                std::swap(close_neighbors, close_neighbors2);
                close_neighbors2.clear();
            }
            close_neighbors.clear();
        }
    }
    
    return coarse_elements;
    // for(auto e : coarse_elements ) {
    //     coarse_elements_to_save[e] = 1;
    //     std::cout << "coarse element = " << e << "\n";
    // }
    // write_mhd("coarse_elements", 4, 4, 4, 1, 1, 1, coarse_elements_to_save.data());
    // App_wo.print();
}

arma::sp_mat SPAI(const arma::sp_mat& A_wo_d)
{
    arma::sp_mat A_t = A_wo_d.t();
    A_t.sync();
    
    arma::sp_mat A_SPAI = A_t;
    A_SPAI.sync();

    // arma::sp_mat A_SPAI(unknowns_local_wo[2],unknowns_local_wo[2]);

    // A_SPAI.transform( [](double val){ return 1.0; } );

    //SPAI
    {
        for(int i = 0 ; i < A_SPAI.n_cols ; i++) {
            int N = A_SPAI.col_ptrs[i+1] - A_SPAI.col_ptrs[i];
            arma::mat A_s(N,N);
            arma::vec b(N);
            int j_loc = 0;
            for( int j = A_SPAI.col_ptrs[i] ; j < A_SPAI.col_ptrs[i+1] ; j++ ) {
                auto row = A_t.col(A_SPAI.row_indices[j]);
                int l_loc = 0;
                for( int jj = A_SPAI.col_ptrs[i] ; jj < A_SPAI.col_ptrs[i+1] ; jj++ ) {
                    for( int l = A_t.col_ptrs[A_SPAI.row_indices[jj]] ; l < A_t.col_ptrs[A_SPAI.row_indices[jj]+1] ; l++ ) {
                        A_s(j_loc,l_loc) += 2.0 * A_t.values[l] * row[A_t.row_indices[l]];
                        if(i == A_t.row_indices[l] && b(j_loc)==0){
                            b(j_loc) = 2.0 * row[A_t.row_indices[l]];
                        }
                    }
                    l_loc++;
                }
                j_loc++;
            }
            arma::vec x_s = arma::solve( A_s, b );
            // std::cout << "res = " << arma::norm(A_s * x_s - b) << "\n";
            j_loc = 0;
            auto col = A_SPAI.col(i);
            for(auto it = col.begin() ; it != col.end() ; ++it) {
                *it = x_s[j_loc];
                j_loc++;
            }
        }
    }

    return A_SPAI.t();
}


MatrixBlocks extract_sparse_block(arma::sp_mat &A, const arma::uvec &fine_elements_b, const arma::uvec &coarse_elements_b, const arma::uvec &fine_elements_prefix_sum, const arma::uvec &coarse_elements_prefix_sum, int N_fine, int N_coarse)
{
    MatrixBlocks blocks;
    
    std::vector<unsigned long long> mat_i_ff, mat_j_ff, mat_i_fc, mat_j_fc, mat_i_cf, mat_j_cf, mat_i_cc, mat_j_cc;
    std::vector<double> values_ff, values_fc, values_cf, values_cc;

    arma::sp_mat::const_iterator it     = A.begin();
    arma::sp_mat::const_iterator it_end = A.end();

    // Извлекаем все четыре блока за один проход
    for(; it != it_end; ++it)
    {
        bool row_fine = fine_elements_b[it.row()] == 1;
        bool row_coarse = coarse_elements_b[it.row()] == 1;
        bool col_fine = fine_elements_b[it.col()] == 1;
        bool col_coarse = coarse_elements_b[it.col()] == 1;

        if(row_fine && col_fine) {
            // Aff: fine rows, fine columns
            mat_i_ff.push_back(fine_elements_prefix_sum[it.row()]);
            mat_j_ff.push_back(fine_elements_prefix_sum[it.col()]);
            values_ff.push_back(*it);
        } else if(row_fine && col_coarse) {
            // Afc: fine rows, coarse columns
            mat_i_fc.push_back(fine_elements_prefix_sum[it.row()]);
            mat_j_fc.push_back(coarse_elements_prefix_sum[it.col()]);
            values_fc.push_back(*it);
        } else if(row_coarse && col_fine) {
            // Acf: coarse rows, fine columns
            mat_i_cf.push_back(coarse_elements_prefix_sum[it.row()]);
            mat_j_cf.push_back(fine_elements_prefix_sum[it.col()]);
            values_cf.push_back(*it);
        } else if(row_coarse && col_coarse) {
            // Acc: coarse rows, coarse columns
            mat_i_cc.push_back(coarse_elements_prefix_sum[it.row()]);
            mat_j_cc.push_back(coarse_elements_prefix_sum[it.col()]);
            values_cc.push_back(*it);
        }
    }

    // Строим Aff
    if(mat_i_ff.size() > 0) {
        arma::umat locations(2, mat_i_ff.size());
        for(size_t i = 0; i < mat_i_ff.size(); i++) {
            locations(0, i) = mat_i_ff[i];
            locations(1, i) = mat_j_ff[i];
        }
        arma::vec values_arma{values_ff};
        blocks.Aff = arma::sp_mat(locations, values_arma, N_fine, N_fine);
    }

    // Строим Afc
    if(mat_i_fc.size() > 0) {
        arma::umat locations(2, mat_i_fc.size());
        for(size_t i = 0; i < mat_i_fc.size(); i++) {
            locations(0, i) = mat_i_fc[i];
            locations(1, i) = mat_j_fc[i];
        }
        arma::vec values_arma{values_fc};
        blocks.Afc = arma::sp_mat(locations, values_arma, N_fine, N_coarse);
    }

    // Строим Acf
    if(mat_i_cf.size() > 0) {
        arma::umat locations(2, mat_i_cf.size());
        for(size_t i = 0; i < mat_i_cf.size(); i++) {
            locations(0, i) = mat_i_cf[i];
            locations(1, i) = mat_j_cf[i];
        }
        arma::vec values_arma{values_cf};
        blocks.Acf = arma::sp_mat(locations, values_arma, N_coarse, N_fine);
    }

    // Строим Acc
    if(mat_i_cc.size() > 0) {
        arma::umat locations(2, mat_i_cc.size());
        for(size_t i = 0; i < mat_i_cc.size(); i++) {
            locations(0, i) = mat_i_cc[i];
            locations(1, i) = mat_j_cc[i];
        }
        arma::vec values_arma{values_cc};
        blocks.Acc = arma::sp_mat(locations, values_arma, N_coarse, N_coarse);
    }

    return blocks;
}

