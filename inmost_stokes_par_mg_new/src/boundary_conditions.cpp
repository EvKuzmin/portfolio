#include "boundary_conditions.hpp"
#include "petsc.h"
#include "aij.h"

void boundary_conditions_U(Mesh *m,
           Mat A,
           Vec rhs,
           double hx,
           double hy,
           double hz,
           MarkerType inlet,
           MarkerType outlet,
           MarkerType wall,
           MarkerType u_mrk, 
           MarkerType v_mrk, 
           MarkerType w_mrk, 
           MarkerType inner_face_mrk,
           MapVariant parameters)
{
    TagInteger U_id_global = TagInteger( m->GetTag( "U_id_global" ) );
    TagInteger P_id_global = TagInteger( m->GetTag( "P_id_global" ) );
    TagReference face_lapl_adj = TagReference( m->GetTag( "face_lapl_adj" ) );
    TagReference face_adj_cells = TagReference( m->GetTag( "face_adj_cells" ) );
    TagReal U = TagReal( m->GetTag( "U" ) );

    double mu = std::get< double >(parameters.at("mu"));

    for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
    {
        if( face->GetStatus() != Element::Ghost ) {
            double dt = std::get<double>(parameters.at("dt"));
            double U_d = std::get<double>(parameters.at("U_d"));
            double L_d = std::get<double>(parameters.at("L_d"));
            double Rho_d = std::get<double>(parameters.at("Rho_d"));
            double Mu_d = std::get<double>(parameters.at("Mu_d"));
            if( face->GetMarker(inner_face_mrk) ) { // viscosity
                int vel_id = face->Integer(U_id_global);

                Storage::reference_array adj_faces = face->ReferenceArray(face_lapl_adj);

                double Re = U_d * L_d * Rho_d / Mu_d;

                int col_id;
                double value;

                if( face->GetMarker(u_mrk) ) {

                    if(!adj_faces[2]->isValid()) { 
                        value = mu * 1.0 / (hy*hy) / Re;
                        MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);
                    }
                    if(!adj_faces[3]->isValid()) { 
                        value = mu * 1.0 / (hy*hy) / Re;
                        MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);
                    }
                    if(!adj_faces[4]->isValid()) { 
                        value = mu * 1.0 / (hz*hz) / Re;
                        MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);
                    }
                    if(!adj_faces[5]->isValid()) { 
                        value = mu * 1.0 / (hz*hz) / Re;
                        MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);
                    }
                }

                if( face->GetMarker(v_mrk) ) {

                    if(!adj_faces[2]->isValid()) { 
                        value = mu * 1.0 / (hx*hx) / Re;
                        MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);
                    }
                    if(!adj_faces[3]->isValid()) { 
                        value = mu * 1.0 / (hx*hx) / Re;
                        MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);
                    }
                    if(!adj_faces[4]->isValid()) { 
                        value = mu * 1.0 / (hz*hz) / Re;
                        MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);
                    }
                    if(!adj_faces[5]->isValid()) { 
                        value = mu * 1.0 / (hz*hz) / Re;
                        MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);
                    }
                }

                if( face->GetMarker(w_mrk) ) {

                    if(!adj_faces[2]->isValid()) { 
                        value = mu * 1.0 / (hx*hx) / Re;
                        MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);
                    }
                    if(!adj_faces[3]->isValid()) { 
                        value = mu * 1.0 / (hx*hx) / Re;
                        MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);
                    }
                    if(!adj_faces[4]->isValid()) { 
                        value = mu * 1.0 / (hy*hy) / Re;
                        MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);
                    }
                    if(!adj_faces[5]->isValid()) { 
                        value = mu * 1.0 / (hy*hy) / Re;
                        MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);
                    }
                }
            }

            if( face->GetMarker(inlet) ) {
                int row_id = face->Integer(U_id_global);
                double value = std::get<double>(parameters.at("inlet_velocity"));
                VecSetValues(rhs, 1, &row_id, &value, INSERT_VALUES);
            }
            if( face->GetMarker(outlet) ) {
                int row_id = face->Integer(U_id_global);
                double value = std::get<double>(parameters.at("inlet_velocity"));
                VecSetValues(rhs, 1, &row_id, &value, INSERT_VALUES);
            }
            if( face->GetMarker(wall) ) {
                int row_id = face->Integer(U_id_global);
                double value = 0.0;
                VecSetValues(rhs, 1, &row_id, &value, INSERT_VALUES);
            }
        }
    }
}

void boundary_conditions_P(Mesh *m,
           Mat A,
           Vec rhs,
           double hx,
           double hy,
           double hz,
           MarkerType inlet,
           MarkerType outlet,
           MarkerType wall,
           MarkerType u_mrk, 
           MarkerType v_mrk, 
           MarkerType w_mrk, 
           MarkerType inner_face_mrk,
           MapVariant parameters)
{
    INMOST_MPI_Comm comm = m->GetCommunicator();
    int rank = m->GetProcessorRank();
    TagInteger U_id_global = TagInteger( m->GetTag( "U_id_global" ) );
    TagInteger P_id_global = TagInteger( m->GetTag( "P_id_global" ) );
    int p_id = -1;
    if(rank == 0) {
        TagReal P = TagReal( m->GetTag( "P" ) );

        for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
        {
            if( cell->GetStatus() != Element::Ghost ) {
                p_id = cell->Integer(P_id_global);
                break;
            }
        }
    }
    
    MPI_Bcast(&p_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
    double value_rhs = 0;
    PetscScalar diag = 1;   
    Vec x;
    VecDuplicate(rhs, &x);
    VecCopy(rhs, x);
    VecSetValues(x, 1, &p_id, &value_rhs, INSERT_VALUES);
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    MatZeroRowsColumns(A, 1, &p_id, diag, x, rhs);  
    VecDestroy(&x);
}

void boundary_conditions(Mesh *m,
           Mat A,
           Vec rhs,
           double hx,
           double hy,
           double hz,
           MarkerType inlet,
           MarkerType outlet,
           MarkerType wall,
           MarkerType u_mrk, 
           MarkerType v_mrk, 
           MarkerType w_mrk, 
           MarkerType inner_face_mrk,
           MapVariant parameters) {

    boundary_conditions_U(m,
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

    VecAssemblyBegin(rhs);
    VecAssemblyEnd(rhs);
    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); //MAT_FINAL_ASSEMBLY
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

    // boundary_conditions_P(m,
    //     A,
    //     rhs,
    //     hx,
    //     hy,
    //     hz,
    //     inlet,
    //     outlet,
    //     wall,
    //     u_mrk, 
    //     v_mrk, 
    //     w_mrk, 
    //     inner_face_mrk,
    //     parameters);
}