#include"make_ids.hpp"
#include<numeric>

int make_face_ids(INMOST::Mesh *m)
{
    int count = 0;

    INMOST::TagInteger face_id_local = m->CreateTag("face_id_local", INMOST::DATA_INTEGER, INMOST::FACE, INMOST::NONE, 1);

    for( INMOST::Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
    {
        if( face->GetStatus() != INMOST::Element::Ghost ) {
            face->Integer(face_id_local) = count;
            count++;
        }
    }

    int face_unknowns_local = count;

    return face_unknowns_local;
}

int make_cell_ids(INMOST::Mesh *m) {
    int count = 0;

    INMOST::TagInteger cell_id_local = m->CreateTag("cell_id_local", INMOST::DATA_INTEGER, INMOST::CELL, INMOST::NONE, 1);

    for( INMOST::Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
    {
        if( cell->GetStatus() != INMOST::Element::Ghost ) {
            cell->Integer(cell_id_local) = count;
            count++;
        }
    }

    int cell_unknowns_local = count;
    return cell_unknowns_local;
}

std::tuple< std::vector<int>, int, int, int, SubmatrixManager > make_variable_ids( INMOST::Mesh *m, int cell_unknowns_local, int face_unknowns_local, std::vector<VariableOnMesh> &variables ) {

    std::vector<int> unknowns_local;

    unknowns_local.push_back(0);
    for( int i = 0 ; i < variables.size() ; i++ ) {
        if( variables[i].type == "all_cells" ) {
            variables[i].properties["local_range"] = std::pair<int, int>( unknowns_local.back(), unknowns_local.back() + cell_unknowns_local );
            unknowns_local.push_back( unknowns_local.back() + cell_unknowns_local );
        } else if( variables[i].type == "all_faces" ) {
            variables[i].properties["local_range"] = std::pair<int, int>( unknowns_local.back(), unknowns_local.back() + cell_unknowns_local );
            unknowns_local.push_back( unknowns_local.back() + face_unknowns_local );
        } else {
            std::cerr << "Wrong variable type" << std::endl;
            std::exit(1);
        }
    }

    int matrix_size_local = unknowns_local.back() - unknowns_local.front();

    int matrix_size_sum = m->ExclusiveSum(matrix_size_local);

    int matrix_size_global = m->Integrate(matrix_size_local);

    INMOST::TagInteger cell_id_local = INMOST::TagInteger( m->GetTag( "cell_id_local" ) );
    INMOST::TagInteger face_id_local = INMOST::TagInteger( m->GetTag( "face_id_local" ) );

    auto make_index_vector = [](int i1, int i2){
        std::vector< int > v( i2 - i1 );
        std::iota(v.begin(), v.end(), i1);
        return v;
    };

    std::map< std::string, std::vector< int > > submatrix_manager_vars;

    for( int i = 0 ; i < variables.size() ; i++ ) {
        if( variables[i].type == "all_cells" ) {
            INMOST::TagInteger cell_var_id_global = m->CreateTag(variables[i].name + "_id_global", INMOST::DATA_INTEGER, INMOST::CELL, INMOST::NONE, 1);
            for( INMOST::Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
            {
                if( cell->GetStatus() != INMOST::Element::Ghost ) {
                    cell->Integer(cell_var_id_global) = cell->Integer(cell_id_local) + matrix_size_sum + unknowns_local[i];
                }
            }

            submatrix_manager_vars[variables[i].name] = make_index_vector( unknowns_local[i], unknowns_local[i+1] );

            MPI_Barrier(MPI_COMM_WORLD);

            m->ExchangeData(cell_var_id_global, INMOST::CELL);

        } else if( variables[i].type == "all_faces" ) {
            INMOST::TagInteger face_var_id_global = m->CreateTag(variables[i].name + "_id_global", INMOST::DATA_INTEGER, INMOST::FACE, INMOST::NONE, 1);
            for( INMOST::Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
            {
                if( face->GetStatus() != INMOST::Element::Ghost ) {
                    face->Integer(face_var_id_global) = face->Integer(face_id_local) + matrix_size_sum + unknowns_local[i];
                }
            }

            submatrix_manager_vars[variables[i].name] = make_index_vector( unknowns_local[i], unknowns_local[i+1] );

            MPI_Barrier(MPI_COMM_WORLD);

            m->ExchangeData(face_var_id_global, INMOST::FACE);
        } else {
            std::cerr << "Wrong variable type" << std::endl;
            std::exit(1);
        }
    }

    
    SubmatrixManager submatrix_map( matrix_size_local, matrix_size_sum, submatrix_manager_vars );
    
    return std::make_tuple(unknowns_local, matrix_size_local, matrix_size_sum, matrix_size_global, submatrix_map);
}

std::tuple< std::vector<int>, int, int, int, SubmatrixManager > make_submatrix_map( int cell_unknowns_local, int face_unknowns_local, std::vector<VariableOnMesh> &variables ) {

    std::vector<int> unknowns_local;

    unknowns_local.push_back(0);
    for( int i = 0 ; i < variables.size() ; i++ ) {
        if( variables[i].type == "all_cells" ) {
            variables[i].properties["local_range"] = std::pair<int, int>( unknowns_local.back(), unknowns_local.back() + cell_unknowns_local );
            unknowns_local.push_back( unknowns_local.back() + cell_unknowns_local );
        } else if( variables[i].type == "all_faces" ) {
            variables[i].properties["local_range"] = std::pair<int, int>( unknowns_local.back(), unknowns_local.back() + face_unknowns_local );
            unknowns_local.push_back( unknowns_local.back() + face_unknowns_local );
        } else {
            std::cerr << "Wrong variable type" << std::endl;
            std::exit(1);
        }
    }

    int matrix_size_local = unknowns_local.back() - unknowns_local.front();

    
    // int matrix_size_sum = m->ExclusiveSum(matrix_size_local);
    int matrix_size_sum = 0;
    MPI_Exscan(&matrix_size_local,
                   &matrix_size_sum,
                   1,
               MPI_INT,
               MPI_SUM,
               MPI_COMM_WORLD);

    // int matrix_size_global = m->Integrate(matrix_size_local);
    int matrix_size_global = 0;
    MPI_Allreduce(&matrix_size_local, &matrix_size_global, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);

    auto make_index_vector = [](int i1, int i2){
        std::vector< int > v( i2 - i1 );
        std::iota(v.begin(), v.end(), i1);
        return v;
    };

    std::map< std::string, std::vector< int > > submatrix_manager_vars;

    for( int i = 0 ; i < variables.size() ; i++ ) {
        if( variables[i].type == "all_cells" ) {
            submatrix_manager_vars[variables[i].name] = make_index_vector( unknowns_local[i], unknowns_local[i+1] );
        } else if( variables[i].type == "all_faces" ) {
            submatrix_manager_vars[variables[i].name] = make_index_vector( unknowns_local[i], unknowns_local[i+1] );
        } else {
            std::cerr << "Wrong variable type" << std::endl;
            std::exit(1);
        }
    }
    
    SubmatrixManager submatrix_map( matrix_size_local, matrix_size_sum, submatrix_manager_vars );
    
    return std::make_tuple(unknowns_local, matrix_size_local, matrix_size_sum, matrix_size_global, submatrix_map);
}