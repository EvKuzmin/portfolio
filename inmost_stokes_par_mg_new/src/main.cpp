#include "inmost.h"
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <numeric>
#include <variant>
#include <chrono>
#include <array>
#include <thread>
#include<filesystem>
#include<regex>
#include "make_adjacent.hpp"
#include "parse_config.hpp"
#include "bounding_box.hpp"
#include "mark_mesh.hpp"
#include "prepare_local_cell_to_face.hpp"
#include "decompose_matrix.hpp"
#include "solver_tree.hpp"
#include "make_ids.hpp"
#include "prepare_solver__.hpp"
#include "dimensions.hpp"

#include "petsc.h"
#include "aij.h"
#include "dvecimpl.h"
#include "vecimpl.h"
#include "mpiaij.h"

#include "matrix_generation.hpp"
#include "rhs_generation.hpp"
#include "boundary_conditions.hpp"

#include "petsc_matrix_write.hpp"
// #include "simple_interp_prolongation.hpp"
#include "SPAI_prolongation.hpp"
#include "SPAI_api.hpp"
#include "../src/ksp/pc/impls/gamg/gamg.h"

#include "common.hpp"

#include "submatrix_manager.hpp"

#include "Sparse_alpha.hpp"

// #include "mesh_coarsen.hpp"
// #include "mesh_coarsen_improved.hpp"

// #include "prepare_solver.hpp"
// #include "prepare_diag_solver.hpp"
// #include "prepare_multigrid_solver.hpp"
// #include "prepare_MG_solver.hpp"
// #include "prepare_custom_MG_solver.hpp"

#include "stack_matrix.hpp"
#include "Notay_transform.hpp"
// #include "Notay_transform_test.hpp"
// #include "simple_interp_prolong_improved.hpp"

using namespace INMOST;

using MapVariant = std::map< std::string, std::variant<int, double, bool, std::string, std::vector<double> > >;

#define BARRIER MPI_Barrier(MPI_COMM_WORLD);

int main(int argc, char *argv[]) {
    // feenableexcept(FE_INVALID | FE_OVERFLOW);
    // Solver::Initialize(&argc,&argv,"");
    // Partitioner::Initialize(&argc,&argv);

    // MapVariant parameters{
    //     { "dt", 1.0 },
    //     { "N_t", 1},
    //     { "mu", 0.005 },
    //     { "rho", 1000.0 },
    //     { "inlet_velocity", 1.0e-3 },
    //     { "L_d", 1.0e-6 }, 
    //     { "U_d", 1.0e-3 },
    //     { "Mu_d", 5.0e-3 },
    //     { "Rho_d", 1.0e3 },
    // };

    // std::get< double >(parameters.at("dt")) = std::get< double >(parameters.at("dt")) * std::get< double >(parameters.at("U_d")) / std::get< double >(parameters.at("L_d"));
    
    // std::get< double >(parameters.at("inlet_velocity")) = std::get< double >(parameters.at("inlet_velocity")) / std::get< double >(parameters.at("U_d"));

    // std::get< double >(parameters.at("mu")) = std::get< double >(parameters.at("mu")) / std::get< double >(parameters.at("Mu_d"));

    // std::get< double >(parameters.at("rho")) = std::get< double >(parameters.at("rho")) / std::get<double>(parameters.at("Rho_d"));

    MapVariant parameters;
    MapVariant config;
    int __trace_rank = -1;
    auto get_env_rank = []() -> int {
        const char *e = std::getenv("OMPI_COMM_WORLD_RANK");
        if(e) return std::atoi(e);
        e = std::getenv("PMI_RANK");
        if(e) return std::atoi(e);
        e = std::getenv("PMI_ID");
        if(e) return std::atoi(e);
        e = std::getenv("MPI_RANK");
        if(e) return std::atoi(e);
        return -1;
    };
    __trace_rank = get_env_rank();
    auto trace_print = [&](const std::string &msg){
        std::cout << "TRACE[rk=" << __trace_rank << "] " << msg << std::endl;
        std::cout.flush();
    };

    std::vector<std::string> config_lines = read_file( "config.ini" );
    parse_config( config_lines, parameters, config );

    dimless_parameters( parameters );

    Mesh * m = new Mesh();

    m->SetCommunicator(INMOST_MPI_COMM_WORLD);



    auto mesh_time_begin = std::chrono::steady_clock::now();



    int rank = m->GetProcessorRank();

    if(rank != 0) {
        std::cout.setstate(std::ios_base::failbit); 
    }

    int size = m->GetProcessorsNumber();
    if( m->GetProcessorRank() == 0 ) {
        m->Load(std::get<std::string>(config.at("mesh_path")));
    }

    // m->Load("box_64_64_64.pvtu");

    Partitioner * p = new Partitioner(m);
    p->SetMethod(Partitioner::MetisKwayContig,Partitioner::Partition);
    p->Evaluate();
    m->Redistribute();
    m->ReorderEmpty(CELL|FACE|EDGE|NODE);

    std::cout << "mesh partitioned\n";

    m->ExchangeGhost(1,FACE);

    std::cout << "ghost exchanged\n";

    trace_print("mesh partition and ghost exchange complete");

    auto mesh_time_end = std::chrono::steady_clock::now();
    // synchronize and aggregate mesh timing across ranks
    BARRIER;
    double mesh_time_local = std::chrono::duration<double>(mesh_time_end - mesh_time_begin).count();
    double mesh_time_min = 0.0, mesh_time_max = 0.0, mesh_time_sum = 0.0;
    MPI_Reduce(&mesh_time_local, &mesh_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mesh_time_local, &mesh_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mesh_time_local, &mesh_time_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0) {
        double mesh_time_avg = mesh_time_sum / size;
        std::cout << "Mesh read+partition time (s) across ranks: min=" << mesh_time_min << " avg=" << mesh_time_avg << " max=" << mesh_time_max << std::endl;
    }
    trace_print("mesh timing reduced (min/avg/max) processed");

    Tag porosity = m->GetTag("cell_data");
    trace_print("got porosity tag");

    TagInteger face_id_local = m->CreateTag("face_id_local", DATA_INTEGER, FACE, NONE, 1);
    TagInteger cell_id_local = m->CreateTag("cell_id_local", DATA_INTEGER, CELL, NONE, 1);
    // TagInteger lapl_id_global = m->CreateTag("lapl_id_global", DATA_INTEGER, CELL, NONE, 1);
    // TagInteger U_id_global = m->CreateTag("U_id_global", DATA_INTEGER, FACE, NONE, 1);
    // TagInteger P_id_global = m->CreateTag("P_id_global", DATA_INTEGER, CELL, NONE, 1);
    TagInteger coarse_level = m->CreateTag("coarse_level", DATA_INTEGER, CELL, NONE, 1);
    
    TagReference face_adj_cells = m->CreateTag("face_adj_cells", DATA_REFERENCE, FACE, NONE);

    TagReference face_lapl_adj = m->CreateTag("face_lapl_adj", DATA_REFERENCE, FACE, NONE);

    TagReference cell_lapl_adj = m->CreateTag("cell_lapl_adj", DATA_REFERENCE, CELL, NONE);
 

    trace_print("created integer/reference tags (face_id_local, cell_id_local, coarse_level, face_adj_cells, ...)");

    TagReal U0 = m->CreateTag("U0", DATA_REAL, FACE, NONE, 1);
    TagReal P0 = m->CreateTag("P0", DATA_REAL, CELL, NONE, 1);

    trace_print("created real tags U0/P0");

    TagReal U = m->CreateTag("U", DATA_REAL, FACE, NONE, 1);
    TagReal P = m->CreateTag("P", DATA_REAL, CELL, NONE, 1);

    TagReal U_interp = m->CreateTag("U_interp", DATA_REAL, CELL, NONE, 3);

    MarkerType u_mrk = m->CreateMarker();
    MarkerType v_mrk = m->CreateMarker();
    MarkerType w_mrk = m->CreateMarker();

    trace_print("marker handles created (u_mrk,v_mrk,w_mrk,...)");

    MarkerType inner_face_mrk = m->CreateMarker();
    MarkerType pore_mrk = m->CreateMarker();

    MarkerType inlet = m->CreateMarker();
    MarkerType outlet = m->CreateMarker();
    MarkerType wall = m->CreateMarker();

    std::map< std::string, MarkerType > markers;

    markers["u_mrk"] = u_mrk;
    markers["v_mrk"] = v_mrk;
    markers["w_mrk"] = w_mrk;
    markers["inner_face_mrk"] = inner_face_mrk;
    markers["pore_mrk"] = pore_mrk;
    markers["inlet"] = inlet;
    markers["outlet"] = outlet;
    markers["wall"] = wall;

    trace_print("markers mapped into dictionary");
    trace_print("tags and markers created");

    //double h;
    //{
    //    
    //    bool found = false;
    //    for(Mesh::iteratorCell it = m->BeginCell(); it != m->EndCell(); ++it) {
    //        if(it->GetStatus() == Element::Ghost) continue;
    //        // if(!it->isValid()) continue;
    //        ElementArray<Face> faces = it->getFaces();
    //        if(faces.size() < 2) continue;
    //        Face back = faces[0].getAsFace();
    //        Face front = faces[1].getAsFace();
    //        Storage::real cnt1[3], cnt2[3];
    //        back.Barycenter(cnt1);
    //        front.Barycenter(cnt2);
    //        h = std::fabs(cnt2[0] - cnt1[0]);
    //        found = true;
    //        break;
    //    }
    //    if(!found) {
    //        // std::abort();
    //        h = 1.0; // fallback
    //    } else {
    //        
    //    }
    //}

    double h;

    {
        trace_print("about to compute base spacing h from first cell");
        Mesh::iteratorCell cell = m->BeginCell();
        ElementArray<Face> faces = cell->getFaces();
        Face back = faces[0].getAsFace();
        Face front = faces[1].getAsFace();
        Storage::real cnt1[3];
        Storage::real cnt2[3];
        back.Barycenter(cnt1);
        front.Barycenter(cnt2);
        h = cnt2[0] - cnt1[0];
        trace_print(std::string("computed base spacing h=") + std::to_string(h));
    }
    
    double hx = h;
    double hy = h;
    double hz = h;

    std::array<double, 6> bounding_box;

    bounding_box = prepare_bounding_box(m);

    trace_print("bounding box prepared");
    mark_inlet_outlet(m, bounding_box, markers);
    mark_velocities(m, markers);

    trace_print("inlet/outlet and velocity markers set");

    int count = 0;

    for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
    {
        if( face->GetStatus() != Element::Ghost ) {
            face->Integer(face_id_local) = count;
            count++;
        }
    }

    int face_unknowns_local = count;

    count = 0;

    std::vector<unsigned int> cell_id_local_to_handle;

    for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
    {
        if( cell->GetStatus() != Element::Ghost ) {
            cell->Integer(cell_id_local) = count;
            count++;
        }
    }

    int cell_unknowns_local = count;

    std::vector< std::tuple< double, double, double > > cell_center_coordinates;

    for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
    {
        if( cell->GetStatus() != Element::Ghost ) {
            Storage::real cnt[3];
            cell->Barycenter(cnt);
            cell_center_coordinates.push_back({cnt[0], cnt[1], cnt[2]});
        }
    }

    std::vector< std::tuple< int, int, int, int, int, int > > cell_to_face_local = prepare_local_cell_to_face( m );

    // for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
    // {
    //     ElementArray<Face> faces = cell->getFaces();
    //     Face back = faces[0].getAsFace();
    //     Face front = faces[1].getAsFace();
    //     Face left = faces[2].getAsFace();
    //     Face right = faces[3].getAsFace();
    //     Face bottom = faces[4].getAsFace();
    //     Face top = faces[5].getAsFace();

    //     back.SetMarker(u_mrk);
    //     front.SetMarker(u_mrk);
    //     left.SetMarker(v_mrk);
    //     right.SetMarker(v_mrk);
    //     bottom.SetMarker(w_mrk);
    //     top.SetMarker(w_mrk);
    // }

    // for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
    // {
    //     ElementArray<Cell> cells = face->getCells();
    //     if(cells.size() > 1) {
    //         if(cells[0].isValid() && cells[1].isValid()) {
    //             face->SetMarker(inner_face_mrk);
    //         }
    //     }
    // }

    make_face_adj_cells( m, u_mrk, v_mrk, w_mrk, inner_face_mrk );
    make_face_lapl_adj( m, u_mrk, v_mrk, w_mrk, inner_face_mrk );
    make_cell_lapl_adj( m );

    trace_print("face/cell adjacency and laplace adjacency created");

    for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
    {
        if( !face->GetMarker(inner_face_mrk) ) {
            if( !face->GetMarker(inlet) && !face->GetMarker(outlet) ) {
                face->SetMarker(wall);
            }
        }
    }

    std::vector< VariableOnMesh > main_mesh_variables{
        {"U", "all_faces"},
        {"P", "all_cells"}
    };

    std::vector< std::tuple< double, double, double > > all_center_coordinates;

    for( int i = 0 ; i < main_mesh_variables.size() ; i++ ) {
        if( main_mesh_variables[i].type == "all_cells" ) {
            for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
            {
                if( cell->GetStatus() != Element::Ghost ) {
                    Storage::real cnt[3];
                    cell->Barycenter(cnt);
                    all_center_coordinates.push_back({cnt[0], cnt[1], cnt[2]});
                }
            }
        } else if( main_mesh_variables[i].type == "all_faces" ) {
            for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
            {
                if( face->GetStatus() != Element::Ghost ) {
                    Storage::real cnt[3];
                    face->Barycenter(cnt);
                    all_center_coordinates.push_back({cnt[0], cnt[1], cnt[2]});
                }
            }
        } else {
            std::cerr << "Wrong variable type" << std::endl;
            std::exit(1);
        }
    }

    auto [unknowns_local, matrix_size_local, matrix_size_sum, matrix_size_global, submatrix_map] = make_variable_ids( m, cell_unknowns_local, face_unknowns_local, main_mesh_variables );

    SubmatrixTreeStruct *submatrix_tree_root = new SubmatrixTreeStruct(nullptr, "without_dirichlet", std::make_pair<std::string, std::string>("without_dirichlet","without_dirichlet"));

    submatrix_tree_root->child.push_back( new SubmatrixTreeStruct( submatrix_tree_root, "Auu", std::make_pair<std::string, std::string>("Au","Au") ) );
    submatrix_tree_root->child.push_back( new SubmatrixTreeStruct( submatrix_tree_root, "Aup", std::make_pair<std::string, std::string>("Au","Ap") ) );
    submatrix_tree_root->child.push_back( new SubmatrixTreeStruct( submatrix_tree_root, "Apu", std::make_pair<std::string, std::string>("Ap","Au") ) );
    submatrix_tree_root->child.push_back( new SubmatrixTreeStruct( submatrix_tree_root, "App", std::make_pair<std::string, std::string>("Ap","Ap") ) );

    TagInteger U_id_global = TagInteger( m->GetTag( "U_id_global" ) );
    TagInteger P_id_global = TagInteger( m->GetTag( "P_id_global" ) );

    trace_print("variable ids and submatrix tree root created");

    // std::vector<int> unknowns_local{0, 
    //                           face_unknowns_local, 
    //                           face_unknowns_local + cell_unknowns_local};

    // std::vector<IS> IS_vector(3);

    // int matrix_size_local = unknowns_local.back() - unknowns_local.front();
    // int matrix_size_sum = m->ExclusiveSum(matrix_size_local);
    // int matrix_size_global = m->Integrate(matrix_size_local);

    // int lapl_matrix_size_local = cell_unknowns_local;
    // int lapl_matrix_size_sum = m->ExclusiveSum(lapl_matrix_size_local);
    // int lapl_matrix_size_global = m->Integrate(lapl_matrix_size_local);

    // for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
    // {
    //     if( face->GetStatus() != Element::Ghost ) {
    //         face->Integer(U_id_global) = face->Integer(face_id_local) + matrix_size_sum;
    //     }
    // }

    // for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
    // {
    //     if( cell->GetStatus() != Element::Ghost ) {
    //         cell->Integer(P_id_global) = cell->Integer(cell_id_local) + matrix_size_sum + face_unknowns_local;
    //     }
    // }

    // for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
    // {
    //     if( cell->GetStatus() != Element::Ghost ) {
    //         cell->Integer(lapl_id_global) = cell->Integer(cell_id_local) + lapl_matrix_size_sum;
    //     }
    // }

    // BARRIER;
    // m->ExchangeData(U_id_global, FACE);
    // m->ExchangeData(P_id_global, CELL);
    // m->ExchangeData(lapl_id_global, CELL);

    BARRIER;

    trace_print("before restart check");

    int initial_timestep = 0;

    if( std::filesystem::exists("restart") ) {
        std::basic_regex restart_filename(R"blabla(restart_([0-9]+)_rank_([0-9]+)\.dat)blabla");
        std::set<int> n;
        for (const auto & entry : std::filesystem::directory_iterator("restart/")){
            std::smatch m;
            // std::cout << entry.path().string().substr(8) << "\n";
            std::string s = entry.path().string().substr(8);
            bool match = std::regex_match(s, m, restart_filename);
            // std::cout << match << "\n";
            // std::cout << m[1].str() << "\n";
            n.insert(std::stoi(m[1].str()));
        }
        // if(n.size() == 0) {
        //     std::cerr << "Empty\n";
        //     std::abort();
        // }
        if(n.size() > 1) {
            std::cerr << "More than one restart checkpoints\n";
            std::abort();
        }
        if(n.size() == 1) {
            std::cout << "reading restart\n";
            std::string restart_name("restart/restart_" + std::to_string( *n.begin() ) + "_rank_" + std::to_string(rank) + ".dat");
            std::ifstream restart_file(restart_name, std::ios::binary);
            int ncells;
            restart_file.read((char*)&initial_timestep, sizeof(int));
            restart_file.read((char*)&parameters.at("dt"), sizeof(double));
            restart_file.read((char*)&ncells, sizeof(int));
            for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
            {
                if( face->GetStatus() != Element::Ghost ) {
                    restart_file.read((char*)&face->Real(U), sizeof(double));
                    face->Real(U0) = face->Real(U);
                }
            }
            restart_file.read((char*)&ncells, sizeof(int));
            for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
            {
                if( cell->GetStatus() != Element::Ghost ) {
                    restart_file.read((char*)&cell->Real(P), sizeof(double));
                    cell->Real(P0) = cell->Real(P);
                }
            }
        }
    }
    BARRIER;

    m->ExchangeData(U, FACE);
    m->ExchangeData(P, CELL);
    m->ExchangeData(U0, FACE);
    m->ExchangeData(P0, CELL);

    trace_print("after restart handling and data exchange, about to initialize PETSc");
    PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, "solver");

    Mat A;
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, matrix_size_local, matrix_size_local, matrix_size_global, matrix_size_global);

    // MatSeqAIJSetPreallocation(A,32,NULL);
    // MatXAIJSetPreallocation(A, 1, diag_nonzeroes, off_diag_nonzeroes, NULL, NULL);
    MatSetFromOptions(A);
    
    MatMPIAIJSetPreallocation(A, 14, NULL, 14, NULL);
    // MatSeqAIJSetPreallocation(A, 0, diag_nonzeroes);
    // MatMPIAIJSetPreallocation(A, 25, NULL, 25, NULL);
    // MatSeqAIJSetPreallocation(A, 25, NULL);

    // MatSetFromOptions(A);
    MatSetUp(A);

    trace_print("PETSc matrix created and set up");

    // std::exit(0);

    Stokes_matrix( m, 
                A,
                hx,
                hy,
                hz,
                u_mrk, 
                v_mrk, 
                w_mrk, 
                inner_face_mrk, 
                parameters );

    MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY); //MAT_FINAL_ASSEMBLY
    MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY);

    trace_print("Stokes matrix assembled");

    Vec rhs;

    VecCreate(PETSC_COMM_WORLD,&rhs);
    VecSetSizes(rhs, matrix_size_local, matrix_size_global);
    VecSetFromOptions(rhs);

    Stokes_rhs( m,
             rhs,
             hx,
             hy,
             hz,
             u_mrk, 
             v_mrk, 
             w_mrk, 
             inner_face_mrk,
             parameters );

    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    trace_print("RHS vector created and assembled");

    boundary_conditions(m,
        A,
        rhs,
        hx,
        hy,
        hz,
        inlet,
        outlet,
        wall,
        u_mrk, 
        v_mrk, 
        w_mrk, 
        inner_face_mrk,
        parameters);

    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);

    trace_print("boundary conditions applied and assemblies completed");

    auto get_MPI_diag_abs_average = [](Mat A){
        int size = 0;

        MPI_Comm_size(MPI_COMM_WORLD, &size);

        Vec diag;
        int diag_size;
        MatCreateVecs( A, NULL, &diag );
        MatGetDiagonal( A, diag );

        VecGetLocalSize(diag, &diag_size);

        double *diag_raw;

        VecGetArray(diag, &diag_raw);

        double average_local = 0;
        int counter_local = 0;
        for (int j = 0; j < diag_size; j++) {
            average_local += std::fabs(diag_raw[j]);
            counter_local++;
        }
        double average = 0.0;
        MPI_Allreduce(&average_local, &average, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        int counter = 0;
        MPI_Allreduce(&counter_local, &counter, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);

        VecRestoreArray(diag, &diag_raw);


        VecDestroy(&diag);
        average /= counter;
        return average;
    };

    auto get_MPI_abs_average = [](Mat A){
        int size = 0;

        MPI_Comm_size(MPI_COMM_WORLD, &size);
        void *A_raw_data = A->data;
        Mat_MPIAIJ* A_mpi_data = (Mat_MPIAIJ*)A_raw_data;

        Mat A_diag = A_mpi_data->A;
        //Mat A_off_diag = A_mpi_data->B;

        PetscInt N_A;
        const PetscInt *ia;
        const PetscInt *ja;
        PetscScalar *data;
        PetscBool success;

        MatGetRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A, &ia, &ja, &success);
        MatSeqAIJGetArray(A_diag, &data);
        
        double average_local = 0.0;
        int counter_local = 0;
        for( int j = 0 ; j < N_A ; j++ ) {
            for( int k = ia[j] ; k < ia[j+1] ; k++ ) {
                if(data[k] != 0.0){
                    average_local += std::fabs(data[k]);
                    counter_local++;
                }
            }
        }
        double average = 0.0;
        MPI_Allreduce(&average_local, &average, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        int counter = 0;
        MPI_Allreduce(&counter_local, &counter, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);
        // average_local /= counter;

        MatRestoreRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A, &ia, &ja, &success);

        

        
        average /= counter;
        return average;
    };

    auto write_func_immediate = [](Mat A){
        Mat A_diag, A_off_diag;
        const int *garray;

        MatMPIAIJGetSeqAIJ(A, &A_diag, &A_off_diag, &garray);
        int A_off_m, A_off_n, owner_range0, owner_range1;
        MatGetSize(A_off_diag, &A_off_m, &A_off_n);
        MatGetOwnershipRange(A_diag, &owner_range0, &owner_range1);

        Mat_SeqAIJ *A_diag_raw_data = (Mat_SeqAIJ*)A_diag->data;
        Mat_SeqAIJ *A_off_raw_data = (Mat_SeqAIJ*)A_off_diag->data;

        const double *A_diag_data, *A_off_data;
        MatSeqAIJGetArrayRead(A_diag,&A_diag_data);
        MatSeqAIJGetArrayRead(A_off_diag,&A_off_data);
        pmw::petsc_matrix_mpi_to_seq_readable( owner_range0, 
                                owner_range1, 
                                A_diag_raw_data->i, 
                                A_diag_raw_data->j, 
                                A_diag_data, 
                                A_off_raw_data->i, 
                                A_off_raw_data->j, 
                                A_off_data, garray, A_off_n );
        // pmw::petsc_matrix_mpi_to_seq( owner_range0, 
        //                         owner_range1, 
        //                         A_diag_raw_data->i, 
        //                         A_diag_raw_data->j, 
        //                         A_diag_data, 
        //                         A_off_raw_data->i, 
        //                         A_off_raw_data->j, 
        //                         A_off_data, garray, A_off_n );
        MatSeqAIJRestoreArrayRead(A_diag,&A_diag_data);
        MatSeqAIJRestoreArrayRead(A_off_diag,&A_off_data);

        // MatView(A, PETSC_VIEWER_STDOUT_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        std::exit(0);
    };

    std::vector<int> local_dirichlet_rows = find_dirichlet_csr( A );

    apply_dirichlet_csr( A, rhs, local_dirichlet_rows);

    trace_print("dirichlet elimination applied");

    PetscInt m_own;
    PetscInt n_own;

    MatGetOwnershipRange(A, &m_own, &n_own);

    for( auto& v:local_dirichlet_rows ) {
        v -= m_own;
    }

    IndexTreeStruct *index_tree_root = new IndexTreeStruct( nullptr, "without_dirichlet", &local_dirichlet_rows );
    index_tree_root->child.push_back( new IndexTreeStruct( index_tree_root, "Au", { "U" } ) );
    index_tree_root->child.push_back( new IndexTreeStruct( index_tree_root, "Ap", { "P" } ) );

    recombine_index( index_tree_root, submatrix_map );

    std::map< std::string, Mat > submatrices;

    std::map< std::string, IS > index_sequences;

    create_IS_tree( index_tree_root, submatrix_map, index_sequences );

    create_submatrices( submatrix_tree_root, A, submatrices, index_sequences );

    trace_print("initial submatrices created");

    SubmatrixManager::Node * wo_node = submatrix_map.find_nodes("without_dirichlet")[0];
    for( int i = 0 ; i < cell_to_face_local.size() ; i++ ) {

        if( std::get<0>(cell_to_face_local[i])!=-1 ) {
            std::get<0>(cell_to_face_local[i]) = wo_node->map_from_parent[std::get<0>(cell_to_face_local[i])];
        }
        if( std::get<1>(cell_to_face_local[i])!=-1 ) {
            std::get<1>(cell_to_face_local[i]) = wo_node->map_from_parent[std::get<1>(cell_to_face_local[i])];
        }
        if( std::get<2>(cell_to_face_local[i])!=-1 ) {
            std::get<2>(cell_to_face_local[i]) = wo_node->map_from_parent[std::get<2>(cell_to_face_local[i])];
        }
        if( std::get<3>(cell_to_face_local[i])!=-1 ) {
            std::get<3>(cell_to_face_local[i]) = wo_node->map_from_parent[std::get<3>(cell_to_face_local[i])];
        }
        if( std::get<4>(cell_to_face_local[i])!=-1 ) {
            std::get<4>(cell_to_face_local[i]) = wo_node->map_from_parent[std::get<4>(cell_to_face_local[i])];
        }
        if( std::get<5>(cell_to_face_local[i])!=-1 ) {
            std::get<5>(cell_to_face_local[i]) = wo_node->map_from_parent[std::get<5>(cell_to_face_local[i])];
        }
    }

    std::vector< std::tuple< double, double, double > > all_center_coordinates_wo(wo_node->size);

    for( int i = 0 ; i < wo_node->map_from_parent.size() ; i++ ) {
        if( wo_node->map_from_parent[ i ] != -1 ) {
            all_center_coordinates_wo[ wo_node->map_from_parent[ i ] ] = all_center_coordinates[ i ];
        }
    }

    std::vector<int> unknowns_local_wo;
    unknowns_local_wo.push_back(0);
    unknowns_local_wo.push_back(unknowns_local_wo.back() + wo_node->ib.at("U").size());
    unknowns_local_wo.push_back(unknowns_local_wo.back() + wo_node->ib.at("P").size());

    Vec            x;
    PetscCall(VecDuplicate(rhs,&x));

    KSP ksp=NULL;

    Vec x_wo;

    // PetscInt N_wo;
    // ISGetSize(index_sequences.at("without_dirichlet"), &N_wo);

    MatCreateVecs(submatrices.at("without_dirichlet"), NULL, &x_wo);
    // VecCreate(PETSC_COMM_WORLD, &x_wo);
    // VecSetSizes(x_wo, N_wo, N_wo);
    // VecSetFromOptions(x_wo);
    VecSet(x_wo,0.0);

    Vec Ax;
    VecDuplicate(x_wo, &Ax);

    int N_t = std::get< int >(parameters.at("N_t"));

    PetscScalar *x_raw;
    PetscScalar *x_wo_raw;

    for( auto& [k, v] : index_sequences) {
        ISDestroy(&v);
    }
    for( auto& [k, v] : submatrices) {
        MatDestroy(&v);
    }
    index_sequences.clear();
    submatrices.clear();

    std::map< std::string, std::tuple< Mat, Mat, int, int, int, int > > prolong_restrict_cache;

    std::map< std::string, Mat > cache_coarse;

    std::vector<SubmatrixTreeStruct *> submatrix_tree_smoother_list;

    submatrix_tree_smoother_list.push_back( new SubmatrixTreeStruct( nullptr, "Auu", std::make_pair<std::string, std::string>("Au","Au") ) );
    submatrix_tree_smoother_list.push_back( new SubmatrixTreeStruct( nullptr, "Aup", std::make_pair<std::string, std::string>("Au","Ap") ) );
    submatrix_tree_smoother_list.push_back( new SubmatrixTreeStruct( nullptr, "Apu", std::make_pair<std::string, std::string>("Ap","Au") ) );
    submatrix_tree_smoother_list.push_back( new SubmatrixTreeStruct( nullptr, "App", std::make_pair<std::string, std::string>("Ap","Ap") ) );

    std::vector<IndexTreeStruct *> index_tree_smoother_list;
    index_tree_smoother_list.push_back( new IndexTreeStruct( nullptr, "Au", { "U" } ) );
    index_tree_smoother_list.push_back( new IndexTreeStruct( nullptr, "Ap", { "P" } ) );

    SolverContext *ctx_full = nullptr;

    trace_print("entering time-stepping loop");
    for( int i = initial_timestep ; i < N_t ; i++ ) {
        std::cout << "timestep " << std::get< double >(parameters.at("dt"))*std::get< double >(parameters.at("L_d")) / std::get< double >(parameters.at("U_d")) << "\n";

    auto prep_start = std::chrono::steady_clock::now();

    trace_print("iteration preparation started");

        MatZeroEntries(A);
        VecZeroEntries(rhs);
        VecZeroEntries(x);
        VecZeroEntries(Ax);
        VecZeroEntries(x_wo);

        Stokes_matrix( m, 
                A,
                hx,
                hy,
                hz,
                u_mrk, 
                v_mrk, 
                w_mrk, 
                inner_face_mrk, 
                parameters );

        MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY); //MAT_FINAL_ASSEMBLY
        MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY);

        Stokes_rhs( m,
                rhs,
                hx,
                hy,
                hz,
                u_mrk, 
                v_mrk, 
                w_mrk, 
                inner_face_mrk,
                parameters );

        VecAssemblyBegin(rhs);
        VecAssemblyEnd(rhs);

        boundary_conditions(m,
            A,
            rhs,
            hx,
            hy,
            hz,
            inlet,
            outlet,
            wall,
            u_mrk, 
            v_mrk, 
            w_mrk, 
            inner_face_mrk,
            parameters);

        MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
        VecAssemblyBegin(rhs);
        VecAssemblyEnd(rhs);

        std::vector<int> local_dirichlet_rows = find_dirichlet_csr( A );

        apply_dirichlet_csr( A, rhs, local_dirichlet_rows);

        create_IS_tree( index_tree_root, submatrix_map, index_sequences );
    create_submatrices( submatrix_tree_root, A, submatrices, index_sequences );

    trace_print("submatrices recreated for iteration");

        Vec left_scale, right_scale;
        double *left_scale_raw, *right_scale_raw;
        MatCreateVecs(A, &left_scale, &right_scale);

        VecGetArray(left_scale, &left_scale_raw);
        VecGetArray(right_scale, &right_scale_raw);

        double Auu_diag_aver = std::sqrt(get_MPI_diag_abs_average(submatrices.at("Auu")));
        double Aup_row_aver = get_MPI_abs_average(submatrices.at("Aup"));
        double Apu_row_aver = get_MPI_abs_average(submatrices.at("Apu"));

        for (int j = unknowns_local[0]; j < unknowns_local[1]; j++) {
            left_scale_raw[j] = 1.0 / Auu_diag_aver;
            right_scale_raw[j] = 1.0 / Auu_diag_aver;
        }
        for (int j = unknowns_local[1]; j < unknowns_local[2]; j++) {
            left_scale_raw[j] = Auu_diag_aver / Apu_row_aver;
            right_scale_raw[j] = Auu_diag_aver / Aup_row_aver;
        }

        VecRestoreArray(left_scale, &left_scale_raw);
        VecRestoreArray(right_scale, &right_scale_raw);

        MatDiagonalScale(A, left_scale, right_scale);
        VecPointwiseMult(rhs, rhs, left_scale);

        for( auto& [k, v] : submatrices) {
            MatDestroy(&v);
        }

        submatrices.clear();

        create_submatrices( submatrix_tree_root, A, submatrices, index_sequences );

        // create_submatrix(
        //     "without_dirichlet", 
        //     index_sequences.at("without_dirichlet"),
        //     index_sequences.at("without_dirichlet"), 
        //     A);

        MatNullSpace nullspace_full;

        Vec nsp_full;
        double *nsp_raw;

        MatCreateVecs(submatrices.at("without_dirichlet"), NULL, &nsp_full);

        VecSet(nsp_full, 0.0);

        VecGetArray(nsp_full, &nsp_raw);

        const auto& Ap_Aup_map = submatrix_map.find_nodes("Ap")[0]->map_to_parent;

        for( auto idx : submatrix_map.find_nodes("Ap")[0]->map_to_parent ){
            nsp_raw[idx] = 1.0;
        }

        VecRestoreArray(nsp_full, &nsp_raw);

        MPI_Barrier(MPI_COMM_WORLD);

        VecNormalize(nsp_full, nullptr);

        MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &nsp_full, &nullspace_full);
        MatSetNullSpace(submatrices.at("without_dirichlet"), nullspace_full);

        // create_submatrix(
        //     "Auu", 
        //     index_sequences.at("Au"),
        //     index_sequences.at("Au"), 
        //     submatrices.at("without_dirichlet"));
        // create_submatrix(
        //     "Aup", 
        //     index_sequences.at("Au"),
        //     index_sequences.at("Ap"), 
        //     submatrices.at("without_dirichlet"));
        // create_submatrix(
        //     "Apu", 
        //     index_sequences.at("Ap"),
        //     index_sequences.at("Au"), 
        //     submatrices.at("without_dirichlet"));
        // create_submatrix(
        //     "App", 
        //     index_sequences.at("Ap"),
        //     index_sequences.at("Ap"), 
        //     submatrices.at("without_dirichlet"));

        Mat A_wo_orig;

        MatDuplicate(submatrices.at("without_dirichlet"), MAT_COPY_VALUES, &A_wo_orig);

        notay_transform_submatrices( submatrices, index_sequences, submatrix_map );

        MatSetNullSpace(submatrices.at("A_Notay_"), nullspace_full);
        MatNullSpaceDestroy(&nullspace_full);
        VecDestroy(&nsp_full);

        SubmatrixTreeStruct *submatrix_tree_notay_base_parent = new SubmatrixTreeStruct( nullptr, "A_Notay_", std::make_pair<std::string, std::string>("","") );

        std::vector<SubmatrixTreeStruct *> submatrix_tree_notay_list;

        submatrix_tree_notay_list.push_back( new SubmatrixTreeStruct( submatrix_tree_notay_base_parent, "Auu", std::make_pair<std::string, std::string>("Au","Au") ) );
        submatrix_tree_notay_list.push_back( new SubmatrixTreeStruct( submatrix_tree_notay_base_parent, "Aup", std::make_pair<std::string, std::string>("Au","Ap") ) );
        submatrix_tree_notay_list.push_back( new SubmatrixTreeStruct( submatrix_tree_notay_base_parent, "Apu", std::make_pair<std::string, std::string>("Ap","Au") ) );
        submatrix_tree_notay_list.push_back( new SubmatrixTreeStruct( submatrix_tree_notay_base_parent, "App", std::make_pair<std::string, std::string>("Ap","Ap") ) );

        for(auto mat_val : submatrix_tree_notay_list) {
            create_submatrices( mat_val, A, submatrices, index_sequences );
        }

        int face_unknowns_wo;

        MatGetLocalSize(submatrices.at("Auu"), &face_unknowns_wo, nullptr);
        int cell_unknowns_wo;

        MatGetLocalSize(submatrices.at("App_Notay"), &cell_unknowns_wo, nullptr);

        std::cout << "face_unknowns_wo = " << face_unknowns_wo << "\n";
        std::cout << "cell_unknowns_wo = " << cell_unknowns_wo << "\n";

        SolverTree *root_solver_node = new SolverTree;
        root_solver_node->A = submatrices.at("A_Notay_");
        root_solver_node->submatrix_map = &submatrix_map;
        root_solver_node->submatrices = &submatrices;
        root_solver_node->index_sequences = &index_sequences;

        root_solver_node->solver_type = "fgmres";
        root_solver_node->precond_type = "multigrid";
        root_solver_node->petsc_options = std::map<std::string, std::string>{ { "-ksp_gmres_modifiedgramschmidt", "" }, { "-ksp_gmres_restart", "150" } };
        root_solver_node->properties = std::map<std::string, std::any>{ 
            {"ksp_i_max", std::get<int>(config.at("ksp_i_max"))}, 
            {"ksp_relative_residual", std::get<double>(config.at("ksp_relative_residual"))}, 
            {"ksp_absolute_residual", std::get<double>(config.at("ksp_absolute_residual"))}, 
            {"fine_smoother_i_max",std::get<int>(config.at("fine_smoother_i_max"))}, 
            {"coarse_smoother_i_max",std::get<int>(config.at("coarse_smoother_i_max"))},
            {"N_levels", std::get<int>(config.at("N_levels"))},
            {"cycle_type", std::get<std::string>(config.at("cycle_type"))},
            {"level_1_agg_coarsening", std::get<bool>(config.at("agg_coarsening"))},
            {"keep_N_elements", std::get<int>(config.at("keep_N_elements"))},
            {"first_N_smoothers_SOR", std::get<int>(config.at("first_N_smoothers_SOR"))},
            {"SOR_omega", std::get<double>(config.at("SOR_omega"))},
            {"coarsest_pc", std::get<std::string>(config.at("coarsest_pc"))},
            {"coarsest_pc_iters", std::get<int>(config.at("coarsest_pc_iters"))},
            {"cell_center_coordinates", &cell_center_coordinates},
            {"all_center_coordinates", &all_center_coordinates_wo},
            {"cell_to_face_local", &cell_to_face_local},
            {"cache", &prolong_restrict_cache},
            {"cache_coarse", &cache_coarse},
            {"A_lapl", submatrices.at("App")},
            {"face_unknowns", face_unknowns_wo},
            {"cell_unknowns", cell_unknowns_wo},
            {"multigrid_cache_tag", std::string("main_multigrid_operators")},
            {"variables",main_mesh_variables},
            {"unknowns_local", unknowns_local_wo},
            {"laplace_submatrix_pos",1},
            {"matrix_name", std::string("A_wo")},
            {"nullspace_IS_name", std::string("Ap")},
            {"indent", std::string("")},
            {"view_enabled", std::get<bool>(config.at("view_enabled"))},
            {"coarse_view_enabled", std::get<bool>(config.at("coarse_view_enabled"))},
            {"coarse_ksp", std::get<std::string>(config.at("coarse_ksp"))},
            {"index_tree_list", index_tree_smoother_list},
            {"submatrix_tree_list", submatrix_tree_smoother_list} };

        // GAMG prolongation construction test
        

        // { // simple Laplace preconditioner
        //     Mat A_R = submatrices.at("Aup");
        //     Mat A_L = submatrices.at("Apu");
        //     MatDestroy(&(submatrices.at("App")));
        //     MatMatMult(A_L, A_R, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(submatrices.at("App")));
        //     MatScale(submatrices.at("App"), -1.0);
        // }

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        // { // SPAI multilevel preconditioner
        //     Mat A_R = submatrices.at("Aup");
        //     Mat A_L = submatrices.at("Apu");
        //     Mat Auu_inv;
        //     build_SPAI(submatrices.at("Auu"), &Auu_inv);

        //     // diagonal_compensation(submatrices.at("Auu"), Auu_inv);

        //     // MPI_Barrier(MPI_COMM_WORLD);
        //     // std::exit(0);

        //     // MatDestroy(&(submatrices.at("App")));
        //     Mat Apu_Auu_inv;
        //     Mat Apu_Auu_inv_Aup;
        //     BARRIER;

        //     MatMatMult(A_L, Auu_inv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Apu_Auu_inv);
        //     MatMatMult(Apu_Auu_inv, A_R, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Apu_Auu_inv_Aup);
        //     MatAXPY(submatrices.at("App"), -1.0, Apu_Auu_inv_Aup, DIFFERENT_NONZERO_PATTERN);
        //     // MatScale(submatrices.at("App"), -1.0);
        //     BARRIER;

        //     MatDestroy(&Auu_inv);
        //     MatDestroy(&Apu_Auu_inv);
        //     MatDestroy(&Apu_Auu_inv_Aup);
        //     // MatDestroy(&Auu_inv_Auu);

        //     // MatView(submatrices.at("App"), PETSC_VIEWER_STDOUT_WORLD);
        //     // BARRIER;
        //     // std::exit(0);
        // }
        // { // gradient of divergence regularization (broken boundary conditions!!! relocate before dirichlet elimination)
        //     Vec k;
        //     MatCreateVecs(submatrices.at("Apu"), NULL, &k);
        //     VecSet(k, 0.1);
        //     Mat Apu_scaled;
        //     MatConvert(submatrices.at("Apu"), MATSAME, MAT_INITIAL_MATRIX, &Apu_scaled);

        //     MatDiagonalScale(Apu_scaled, k, NULL);

        //     Mat Aup_k_Apu;

        //     MatMatMult(submatrices.at("Aup"), Apu_scaled, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Aup_k_Apu);
        //     MatAXPY(submatrices.at("Auu"), 1.0, Aup_k_Apu, DIFFERENT_NONZERO_PATTERN);
        //     MatDestroy(&Aup_k_Apu);
        // }

        // { // gradient of Laplace of divergence regularization (broken boundary conditions!!! relocate before dirichlet elimination)
        //     Mat Lapl;
        //     Mat Aup_k_Apu;
        //     MatMatMult(submatrices.at("Apu"), submatrices.at("Aup"), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Lapl);
        //     MatMatMatMult(submatrices.at("Aup"), Lapl, submatrices.at("Apu"), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Aup_k_Apu);

        //     MatAXPY(submatrices.at("Auu"), 1.0, Aup_k_Apu, DIFFERENT_NONZERO_PATTERN);
        //     MatDestroy(&Lapl);
        //     MatDestroy(&Aup_k_Apu);
        // }

        // { // gradient of Laplace SPAI of divergence regularization (broken boundary conditions!!! relocate before dirichlet elimination)
        //     Mat Lapl;
        //     Mat Lapl_inv;
        //     Mat Aup_k_Apu;
        //     MatMatMult(submatrices.at("Apu"), submatrices.at("Aup"), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Lapl);

        //     build_SPAI(Lapl, &Lapl_inv);

        //     MatMatMatMult(submatrices.at("Aup"), Lapl_inv, submatrices.at("Apu"), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Aup_k_Apu);

        //     MatAXPY(submatrices.at("Auu"), 1.0, Aup_k_Apu, DIFFERENT_NONZERO_PATTERN);
        //     MatDestroy(&Lapl);
        //     MatDestroy(&Lapl_inv);
        //     MatDestroy(&Aup_k_Apu);
        // }

        // write_func_immediate(submatrices.at("App"));

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        PetscInt m_own;
        PetscInt n_own;

        MatGetOwnershipRange(A, &m_own, &n_own);

    trace_print("about to prepare nested solver (build/preconditioner)");
        {
            prepare_nested_solver(root_solver_node, nullptr, nullptr, "");
            // auto [ksp2, x_wo2] = prepare_solver_block_SPAI(submatrices, index_sequences);
            // auto [ksp2, x_wo2] = prepare_solver_diag(submatrices, index_sequences);
            // auto [ksp2, x_wo2] = prepare_solver_multigrid(submatrices, index_sequences);
            // auto [ksp2, x_wo2] = prepare_solver_MG(submatrix_map, submatrices, index_sequences, write_func_immediate);
            // auto [ksp2, x_wo2] = prepare_solver_custom_MG(submatrix_map, submatrices, index_sequences, write_func_immediate, cell_center_coordinates, cell_to_face_local);

            if(ksp!=NULL) {
                // PC pc;
                // KSPGetPC(ksp, &pc);
                // PrecondContextAup *ctx_full;
                // PCShellGetContext(pc, &ctx_full);
                // delete ctx_full;
                KSPDestroy(&ksp);
            }
            // if(x_wo!=NULL) {
            //     VecDestroy(&x_wo);
            // }
            // ksp = ksp2;
            // x_wo = x_wo2;
            ksp = root_solver_node->ksp;
            ctx_full = root_solver_node->ctx;

            trace_print("nested solver prepared and KSP/context obtained");
        }
        // KSPSetOperators(ksp,submatrices.at("without_dirichlet"),submatrices.at("without_dirichlet"));

        VecGetArray(x_wo, &x_wo_raw);

        auto node = submatrix_map.find_nodes("without_dirichlet")[0];
        auto& map_from_parent = node->map_from_parent;

        for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
        {
            if( face->GetStatus() != Element::Ghost ) {
                int id = face->Integer(U_id_global) - m_own;
                if( map_from_parent[ id ] != -1 ) {
                    x_wo_raw[ map_from_parent[ id ] ] = face->Real(U) * Auu_diag_aver;
                }
            }
        }

        for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
        {
            if( cell->GetStatus() != Element::Ghost ) {
                int P_id = cell->Integer(P_id_global)- m_own;

                if( map_from_parent[ P_id ] != -1 ) {
                    x_wo_raw[ map_from_parent[ P_id ] ] = cell->Real(P) / Auu_diag_aver * Apu_row_aver;
                }
            }
        }

        VecRestoreArray(x_wo, &x_wo_raw);

        { //Prepare Notay x 
            Vec Auu_SPAI_Aup_x;

            MatCreateVecs(submatrices.at("Auu_SPAI_Aup"), nullptr, &Auu_SPAI_Aup_x);
            
            Vec x_wo_u, x_wo_p;
            VecGetSubVector(x_wo, index_sequences.at("Au"), &x_wo_u);
            VecGetSubVector(x_wo, index_sequences.at("Ap"), &x_wo_p);

            MatMult( submatrices.at("Auu_SPAI_Aup"), x_wo_p, Auu_SPAI_Aup_x );
            VecAXPY( x_wo_u, 1.0, Auu_SPAI_Aup_x );

            VecRestoreSubVector(x_wo, index_sequences.at("Au"), &x_wo_u);
            VecRestoreSubVector(x_wo, index_sequences.at("Ap"), &x_wo_p);

            VecDestroy( &Auu_SPAI_Aup_x );
        }

        Vec b_wo;
        VecGetSubVector(rhs, index_sequences.at("without_dirichlet"), &b_wo);

        Vec b_wo_orig;

        VecDuplicate(b_wo, &b_wo_orig);
        VecCopy(b_wo, b_wo_orig);

        if(submatrices.at("Apu_Auu_SPAI")!=nullptr)
        { // Prepare Notay rhs
            Vec b_wo_u, b_wo_p;

            VecGetSubVector(b_wo, index_sequences.at("Au"), &b_wo_u);
            VecGetSubVector(b_wo, index_sequences.at("Ap"), &b_wo_p);

            Vec Apu_Auu_SPAI_b_wo_u;

            MatCreateVecs(submatrices.at("Apu_Auu_SPAI"), nullptr, &Apu_Auu_SPAI_b_wo_u);

            MatMult( submatrices.at("Apu_Auu_SPAI"), b_wo_u, Apu_Auu_SPAI_b_wo_u );
            VecAXPY( b_wo_p, -1.0, Apu_Auu_SPAI_b_wo_u );

            VecRestoreSubVector(b_wo, index_sequences.at("Au"), &b_wo_u);
            VecRestoreSubVector(b_wo, index_sequences.at("Ap"), &b_wo_p);

            VecDestroy( &Apu_Auu_SPAI_b_wo_u );
        }

        // { // Prepare rhs Ilin diag preconditioner
        //     PC pc;
        //     KSPGetPC(ksp, &pc);
        //     PrecondContextAup *ctx_full;
        //     PCShellGetContext(pc, &ctx_full);
            
        //     Vec b_wo_u, b_wo_p;

        //     VecGetSubVector(b_wo, ctx_full->IS_U, &b_wo_u);
        //     VecGetSubVector(b_wo, ctx_full->IS_P, &b_wo_p);

        //     Vec Auu_inv_b_wo_u, Apu_Auu_inv_b_wo_u;

        //     MatCreateVecs(submatrices.at("Auu"), nullptr, &Auu_inv_b_wo_u);
        //     MatCreateVecs(submatrices.at("App"), nullptr, &Apu_Auu_inv_b_wo_u);

        //     // MatMult( submatrices.at("Apu_Auu_SPAI"), b_wo_u, Apu_Auu_SPAI_b_wo_u );
        //     KSPSolve(ctx_full->kspAuu, b_wo_u, Auu_inv_b_wo_u);
        //     MatMult( submatrices.at("Apu"), Auu_inv_b_wo_u, Apu_Auu_inv_b_wo_u );
        //     VecAXPY( b_wo_p, -1.0, Apu_Auu_inv_b_wo_u );

        //     VecRestoreSubVector(b_wo, ctx_full->IS_U, &b_wo_u);
        //     VecRestoreSubVector(b_wo, ctx_full->IS_P, &b_wo_p);

        //     VecDestroy( &Apu_Auu_inv_b_wo_u );
        //     VecDestroy( &Auu_inv_b_wo_u );
        // }

        // write_local_PetSc_matrix(submatrices.at("without_dirichlet"));
        // write_local_PetSc_vector(b_wo);
        
        // std::exit(0);

    // synchronize before measuring end of preparation to get comparable timings
    BARRIER;
    auto prep_end = std::chrono::steady_clock::now();
    double prep_time_local = std::chrono::duration<double>(prep_end - prep_start).count();
    double prep_min = 0.0, prep_max = 0.0, prep_sum = 0.0;
    MPI_Reduce(&prep_time_local, &prep_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&prep_time_local, &prep_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&prep_time_local, &prep_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0) {
        double prep_avg = prep_sum / size;
        std::cout << "Iteration " << i << ": Preparation time (s) across ranks: min=" << prep_min << " avg=" << prep_avg << " max=" << prep_max << std::endl;
    }

    std::cout << "KSPSolve(ksp,b_wo,x_wo)\n";   
    trace_print("about to start KSPSolve (timer relocated before prepare rhs)");
    begin = std::chrono::steady_clock::now(); // Relocate before prepare rhs
        MatNullSpaceRemove(nullspace_full, b_wo);  
        KSPSolve(ksp,b_wo,x_wo); // Main Solver


        // { // Ilin diag preconditioner solver
        //     PC pc;
        //     KSPGetPC(ksp, &pc);
        //     PrecondContextAup *ctx;
        //     PCShellGetContext(pc,&ctx);
        //     VecZeroEntries(x_wo);

        //     Vec bu, bp;
        //     Vec yu, yp;
        //     VecGetSubVector(b_wo, ctx->IS_U, &bu);
        //     VecGetSubVector(b_wo, ctx->IS_P, &bp);
        //     VecGetSubVector(x_wo, ctx->IS_U, &yu);
        //     VecGetSubVector(x_wo, ctx->IS_P, &yp);

        //     KSPConvergedReason reason;

        //     KSPSolve(ctx->kspAuu, bu, yu); 
        //     KSPGetConvergedReason(ctx->kspAuu, &reason);
        //     if(reason < 0) {
        //         std::cerr << "Auu diverged\n";
        //         VecRestoreSubVector(b_wo, ctx->IS_U, &bu);
        //         VecRestoreSubVector(b_wo, ctx->IS_P, &bp);
        //         VecRestoreSubVector(x_wo, ctx->IS_U, &yu);
        //         VecRestoreSubVector(x_wo, ctx->IS_P, &yp);
        //         // throw std::runtime_error("something not converged\n");
        //         PCSetFailedReason(pc, PC_FACTOR_OTHER);
        //         return 0;
        //     }

            

        //     KSPSolve(ctx->kspApp, bp, yp); 
        //     KSPGetConvergedReason(ctx->kspApp, &reason);
        //     if(reason < 0) {
        //         std::cerr << "App diverged\n";
        //         VecRestoreSubVector(b_wo, ctx->IS_U, &bu);
        //         VecRestoreSubVector(b_wo, ctx->IS_P, &bp);
        //         VecRestoreSubVector(x_wo, ctx->IS_U, &yu);
        //         VecRestoreSubVector(x_wo, ctx->IS_P, &yp);
        //         // throw std::runtime_error("something not converged\n");
        //         PCSetFailedReason(pc, PC_FACTOR_OTHER);
        //         return 0;
        //     }

        //     MatNullSpace nullspace;
        //     MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);
        //     MatNullSpaceRemove(nullspace, yp);
        //     MatNullSpaceDestroy(&nullspace);

        //     VecRestoreSubVector(b_wo, ctx->IS_U, &bu);
        //     VecRestoreSubVector(b_wo, ctx->IS_P, &bp);
        //     VecRestoreSubVector(x_wo, ctx->IS_U, &yu);
        //     VecRestoreSubVector(x_wo, ctx->IS_P, &yp);
        // }

    end = std::chrono::steady_clock::now();

    trace_print("KSPSolve finished");
        double stokes_local = std::chrono::duration<double>(end - begin).count();
        double stokes_min = 0.0, stokes_max = 0.0, stokes_sum = 0.0;
        MPI_Reduce(&stokes_local, &stokes_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stokes_local, &stokes_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stokes_local, &stokes_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if(rank == 0) {
            double stokes_avg = stokes_sum / size;
            std::cout << "Iteration " << i << ": Stokes solution time (s) across ranks: min=" << stokes_min << " avg=" << stokes_avg << " max=" << stokes_max << std::endl;
        }

        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp, &reason);
        std::cout << "ksp convergence reason = " << reason << "\n";
        if(reason < 0) {
            std::cerr << "ksp diverged\n";
            VecRestoreSubVector(rhs, index_sequences.at("without_dirichlet"), &b_wo);
            throw std::runtime_error("something not converged\n");
        }

        if(submatrices.at("Auu_SPAI_Aup")!=nullptr)
        { // Transform back Notay X
            Vec Auu_SPAI_Aup_x;

            MatCreateVecs(submatrices.at("Auu_SPAI_Aup"), nullptr, &Auu_SPAI_Aup_x);
            
            Vec x_wo_u, x_wo_p;
            VecGetSubVector(x_wo, index_sequences.at("Au"), &x_wo_u);
            VecGetSubVector(x_wo, index_sequences.at("Ap"), &x_wo_p);

            MatMult( submatrices.at("Auu_SPAI_Aup"), x_wo_p, Auu_SPAI_Aup_x );
            VecAXPY( x_wo_u, -1.0, Auu_SPAI_Aup_x );

            VecRestoreSubVector(x_wo, index_sequences.at("Au"), &x_wo_u);
            VecRestoreSubVector(x_wo, index_sequences.at("Ap"), &x_wo_p);

            VecDestroy( &Auu_SPAI_Aup_x );
        }

        // { // Transform back Ilin diag X
        //     Vec Aup_x, Auu_inv_Aup_x;

        //     MatCreateVecs(submatrices.at("Auu"), nullptr, &Aup_x);
        //     MatCreateVecs(submatrices.at("Auu"), nullptr, &Auu_inv_Aup_x);

        //     PC pc;
        //     KSPGetPC(ksp, &pc);
        //     PrecondContextAup *ctx_full;
        //     PCShellGetContext(pc, &ctx_full);
            
        //     Vec x_wo_u, x_wo_p;
        //     VecGetSubVector(x_wo, index_sequences.at("Au"), &x_wo_u);
        //     VecGetSubVector(x_wo, index_sequences.at("Ap"), &x_wo_p);

        //     MatMult( submatrices.at("Aup"), x_wo_p, Aup_x );
        //     KSPSolve(ctx_full->kspAuu, Aup_x, Auu_inv_Aup_x);
        //     VecAXPY( x_wo_u, -1.0, Auu_inv_Aup_x );

        //     VecRestoreSubVector(x_wo, index_sequences.at("Au"), &x_wo_u);
        //     VecRestoreSubVector(x_wo, index_sequences.at("Ap"), &x_wo_p);

        //     VecDestroy( &Aup_x );
        //     VecDestroy( &Auu_inv_Aup_x );
        // }

        // { // check divergence
        //     Vec divergence;
        //     MatCreateVecs(submatrices.at("App"), nullptr, &divergence);

        //     Vec x_wo_u;
        //     VecGetSubVector(x_wo, index_sequences.at("Au"), &x_wo_u);
        //     MatMult(submatrices.at("Apu"), x_wo_u, divergence);
        //     VecRestoreSubVector(x_wo, index_sequences.at("Au"), &x_wo_u);

        //     VecView(divergence, PETSC_VIEWER_STDOUT_WORLD);

        //     MPI_Barrier( MPI_COMM_WORLD );
        //     std::exit(0);
        // }

        MatMult(A_wo_orig, x_wo, Ax);

        VecAXPY(Ax, -1.0, b_wo_orig);

        double residual_Ax = 0.0;

        VecNorm(Ax, NORM_2, &residual_Ax);

        VecRestoreSubVector(rhs, index_sequences.at("without_dirichlet"), &b_wo);

        std::cout << "|Ax - rhs| = " << residual_Ax << "\n";

        VecDestroy( &b_wo_orig );
        MatDestroy( &A_wo_orig );

        VecCopy( rhs, x );

        VecGetArray(x, &x_raw);
        VecGetArray(x_wo, &x_wo_raw);

        for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
        {
            if( face->GetStatus() != Element::Ghost ) {
                if( map_from_parent[ face->Integer(U_id_global) - m_own ] != -1 ) {
                    x_raw[ face->Integer(U_id_global) - m_own ] = x_wo_raw[ map_from_parent[ face->Integer(U_id_global) - m_own ] ];
                }
            }
        }

        for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
        {
            if( cell->GetStatus() != Element::Ghost ) {
                int P_id = cell->Integer(P_id_global);
                if( map_from_parent[ P_id - m_own ] != -1 ) {
                    x_raw[ P_id - m_own ] = x_wo_raw[ map_from_parent[ P_id - m_own ] ];
                }
            }
        }

        VecRestoreArray(x, &x_raw);
        VecRestoreArray(x_wo, &x_wo_raw);

        VecPointwiseMult(x, x, right_scale);

        VecGetArray(x, &x_raw);

        for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
        {
            if( face->GetStatus() != Element::Ghost ) {
                // residual += std::pow(x_raw[ face->Integer(face_id) ], 2.0);
                face->Real(U) = x_raw[ face->Integer(U_id_global) - m_own ];
            }
        }

        for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
        {
            if( cell->GetStatus() != Element::Ghost ) {
                // residual += std::pow(x_raw[ cell->Integer(cell_id) + face_unknowns], 2.0);
                cell->Real(P) = x_raw[ cell->Integer(P_id_global) - m_own ];

                ElementArray<Face> faces = cell->getFaces();
                Face back = faces[0].getAsFace();
                Face front = faces[1].getAsFace();
                Face left = faces[2].getAsFace();
                Face right = faces[3].getAsFace();
                Face bottom = faces[4].getAsFace();
                Face top = faces[5].getAsFace();

                cell->RealArray(U_interp)[0] = 0.5 * ( back.Real(U) + front.Real(U) ) * std::get< double >(parameters.at("U_d"));
                cell->RealArray(U_interp)[1] = 0.5 * ( left.Real(U) + right.Real(U) ) * std::get< double >(parameters.at("U_d"));
                cell->RealArray(U_interp)[2] = 0.5 * ( bottom.Real(U) + top.Real(U) ) * std::get< double >(parameters.at("U_d"));
            }
        }

        m->ExchangeData(U, FACE);
        m->ExchangeData(P, CELL);

        VecRestoreArray(x, &x_raw);

        for( auto& [k, v] : index_sequences) {
            ISDestroy(&v);
        }
        for( auto& [k, v] : submatrices) {
            MatDestroy(&v);
        }
        index_sequences.clear();
        submatrices.clear();

        VecGetArray(x, &x_raw);
        for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
        {
            if( face->GetStatus() != Element::Ghost ) {
                face->Real(U0) = x_raw[ face->Integer(U_id_global) - m_own ];
            }
        }

        for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
        {
            if( cell->GetStatus() != Element::Ghost ) {
                cell->Real(P0) = x_raw[ cell->Integer(P_id_global) - m_own];

                ElementArray<Face> faces = cell->getFaces();
                Face back = faces[0].getAsFace();
                Face front = faces[1].getAsFace();
                Face left = faces[2].getAsFace();
                Face right = faces[3].getAsFace();
                Face bottom = faces[4].getAsFace();
                Face top = faces[5].getAsFace();

                cell->RealArray(U_interp)[0] = 0.5 * ( back.Real(U) + front.Real(U) ) * std::get< double >(parameters.at("U_d"));
                cell->RealArray(U_interp)[1] = 0.5 * ( left.Real(U) + right.Real(U) ) * std::get< double >(parameters.at("U_d"));
                cell->RealArray(U_interp)[2] = 0.5 * ( bottom.Real(U) + top.Real(U) ) * std::get< double >(parameters.at("U_d"));
            }
        }
        VecRestoreArray(x, &x_raw);

        m->ExchangeData(U0, FACE);
        m->ExchangeData(P0, CELL);

    trace_print("about to write restart file and save results");
    std::ofstream restart_file("restart_" + std::to_string(i%4) + "_rank_" + std::to_string(rank) + ".dat", std::ios::out | std::ios::binary | std::ios::trunc);

        restart_file.write((char*)&i, sizeof(int));
        restart_file.write((char*)&parameters.at("dt"), sizeof(double));
        restart_file.write((char*)&face_unknowns_local, sizeof(int));
        for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
        {
            if( face->GetStatus() != Element::Ghost ) {
                restart_file.write((char*)&face->Real(U0), sizeof(double));
            }
        }

        restart_file.write((char*)&cell_unknowns_local, sizeof(int));
        for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
        {
            if( cell->GetStatus() != Element::Ghost ) {
                restart_file.write((char*)&cell->Real(P0), sizeof(double));
            }
        }

        restart_file.close();

        for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
        {
            if( cell->GetStatus() != Element::Ghost ) {
                cell->Real(P0) = cell->Real(P0) * std::get< double >(parameters.at("U_d")) * std::get< double >(parameters.at("U_d")) * std::get< double >(parameters.at("Rho_d"));
                cell->Real(P) = cell->Real(P) * std::get< double >(parameters.at("U_d")) * std::get< double >(parameters.at("U_d")) * std::get< double >(parameters.at("Rho_d"));
            }
        }

        m->Save("result" + std::to_string(i) + ".pvtu");

        for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
        {
            if( cell->GetStatus() != Element::Ghost ) {
                cell->Real(P0) = cell->Real(P0) / (std::get< double >(parameters.at("U_d")) * std::get< double >(parameters.at("U_d")) * std::get< double >(parameters.at("Rho_d")) );
                cell->Real(P) = cell->Real(P) / (std::get< double >(parameters.at("U_d")) * std::get< double >(parameters.at("U_d")) * std::get< double >(parameters.at("Rho_d")) );
            }
        }
    }
    VecDestroy(&rhs);
    MatDestroy(&A);

    trace_print("before PetscFinalize and cleanup");
    ierr = PetscFinalize();

    trace_print("starting cleanup");
    delete m;
    Partitioner::Finalize();
    Solver::Finalize();
    return 0;
}