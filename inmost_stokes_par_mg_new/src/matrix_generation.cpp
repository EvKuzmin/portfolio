
#include "matrix_generation.hpp"

void matrix_U_row( Mesh *m, 
                   Mat A,
                   double hx,
                   double hy,
                   double hz,
                   MarkerType u_mrk, 
                   MarkerType v_mrk, 
                   MarkerType w_mrk, 
                   MarkerType inner_face_mrk, 
                   MapVariant parameters )
 {
    double dt = std::get<double>(parameters.at("dt"));
    double U_d = std::get<double>(parameters.at("U_d"));
    double L_d = std::get<double>(parameters.at("L_d"));
    double Rho_d = std::get<double>(parameters.at("Rho_d"));
    double Mu_d = std::get<double>(parameters.at("Mu_d"));

    TagInteger U_id_global = TagInteger( m->GetTag( "U_id_global" ) );
    TagInteger P_id_global = TagInteger( m->GetTag( "P_id_global" ) );
    TagReference face_lapl_adj = TagReference( m->GetTag( "face_lapl_adj" ) );
    TagReference face_adj_cells = TagReference( m->GetTag( "face_adj_cells" ) );
    TagReal U = TagReal( m->GetTag( "U" ) );

    for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
    {
        if( face->GetStatus() != Element::Ghost ) {
            if( face->GetMarker(inner_face_mrk) ) {
                int vel_id = face->Integer(U_id_global);

                Storage::reference_array adj_faces = face->ReferenceArray(face_lapl_adj);

                double rho = std::get< double >(parameters.at("rho"));

                double mu = std::get< double >(parameters.at("mu"));

                double Re = U_d * L_d * Rho_d / Mu_d;

                // double We = Rho_d * U_d * U_d * L_d / Sigma_d;

                double value = rho / dt + mu * 2.0 / ( hx * hx ) / Re + mu * 2.0 / ( hy * hy ) / Re + mu * 2.0 / ( hz * hz ) / Re;

                MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);

                int col_id;

                if( face->GetMarker(u_mrk) ) {

                    if(adj_faces[0]->isValid()) { 
                        col_id = adj_faces[0]->Integer(U_id_global);
                        value = -mu * 1.0 / (hx*hx) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }

                    if(adj_faces[1]->isValid()) { 
                        col_id = adj_faces[1]->Integer(U_id_global);
                        value = -mu * 1.0 / (hx*hx) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }

                    if(adj_faces[2]->isValid()) { 
                        col_id = adj_faces[2]->Integer(U_id_global);
                        value = -mu * 1.0 / (hy*hy) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }

                    if(adj_faces[3]->isValid()) { 
                        col_id = adj_faces[3]->Integer(U_id_global);
                        value = -mu * 1.0 / (hy*hy) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }

                    if(adj_faces[4]->isValid()) { 
                        col_id = adj_faces[4]->Integer(U_id_global);
                        value = -mu * 1.0 / (hz*hz) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }

                    if(adj_faces[5]->isValid()) { 
                        col_id = adj_faces[5]->Integer(U_id_global);
                        value = -mu * 1.0 / (hz*hz) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }
                    //P
                    col_id = face->ReferenceArray(face_adj_cells)[0].Integer(P_id_global);
                    value = -1.0 / hx;
                    MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);

                    col_id = face->ReferenceArray(face_adj_cells)[1].Integer(P_id_global);
                    value = 1.0 / hx;
                    MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                }
                if( face->GetMarker(v_mrk) ) {

                    if(adj_faces[0]->isValid()) { 
                        col_id = adj_faces[0]->Integer(U_id_global);
                        value = -mu * 1.0 / (hy*hy) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }

                    if(adj_faces[1]->isValid()) { 
                        col_id = adj_faces[1]->Integer(U_id_global);
                        value = -mu * 1.0 / (hy*hy) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }

                    if(adj_faces[2]->isValid()) { 
                        col_id = adj_faces[2]->Integer(U_id_global);
                        value = -mu * 1.0 / (hx*hx) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }

                    if(adj_faces[3]->isValid()) { 
                        col_id = adj_faces[3]->Integer(U_id_global);
                        value = -mu * 1.0 / (hx*hx) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }

                    if(adj_faces[4]->isValid()) { 
                        col_id = adj_faces[4]->Integer(U_id_global);
                        value = -mu * 1.0 / (hz*hz) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }

                    if(adj_faces[5]->isValid()) { 
                        col_id = adj_faces[5]->Integer(U_id_global);
                        value = -mu * 1.0 / (hz*hz) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }
                    //P
                    col_id = face->ReferenceArray(face_adj_cells)[0].Integer(P_id_global);
                    value = -1.0 / hy;
                    MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);

                    col_id = face->ReferenceArray(face_adj_cells)[1].Integer(P_id_global);
                    value = 1.0 / hy;
                    MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                }
                if( face->GetMarker(w_mrk) ) {

                    if(adj_faces[0]->isValid()) { 
                        col_id = adj_faces[0]->Integer(U_id_global);
                        value = -mu * 1.0 / (hz*hz) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }

                    if(adj_faces[1]->isValid()) { 
                        col_id = adj_faces[1]->Integer(U_id_global);
                        value = -mu * 1.0 / (hz*hz) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }

                    if(adj_faces[2]->isValid()) { 
                        col_id = adj_faces[2]->Integer(U_id_global);
                        value = -mu * 1.0 / (hx*hx) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }

                    if(adj_faces[3]->isValid()) { 
                        col_id = adj_faces[3]->Integer(U_id_global);
                        value = -mu * 1.0 / (hx*hx) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }

                    if(adj_faces[4]->isValid()) { 
                        col_id = adj_faces[4]->Integer(U_id_global);
                        value = -mu * 1.0 / (hy*hy) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }

                    if(adj_faces[5]->isValid()) { 
                        col_id = adj_faces[5]->Integer(U_id_global);
                        value = -mu * 1.0 / (hy*hy) / Re;
                        MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                    }
                    //P
                    col_id = face->ReferenceArray(face_adj_cells)[0].Integer(P_id_global);
                    value = -1.0 / hz;
                    MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);

                    col_id = face->ReferenceArray(face_adj_cells)[1].Integer(P_id_global);
                    value = 1.0 / hz;
                    MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
                }
            } else {
                int vel_id = face->Integer(U_id_global);
                double value = 1.0;

                MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);
            }
        }
    }
}

void matrix_P_row(Mesh *m, 
                   Mat A,
                   double hx,
                   double hy,
                   double hz,
                   MapVariant parameters) {
    TagInteger U_id_global = TagInteger( m->GetTag( "U_id_global" ) );
    TagInteger P_id_global = TagInteger( m->GetTag( "P_id_global" ) );
    TagReference cell_lapl_adj = TagReference( m->GetTag( "cell_lapl_adj" ) );
    for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
    {
        if( cell->GetStatus() != Element::Ghost ) {
            double dt = std::get<double>(parameters.at("dt"));
            double U_d = std::get<double>(parameters.at("U_d"));
            double L_d = std::get<double>(parameters.at("L_d"));
            double Rho_d = std::get<double>(parameters.at("Rho_d"));
            double Mu_d = std::get<double>(parameters.at("Mu_d"));

            Cell back_cell = cell->ReferenceArray(cell_lapl_adj)[0].getAsCell();
            Cell front_cell = cell->ReferenceArray(cell_lapl_adj)[1].getAsCell();
            Cell left_cell = cell->ReferenceArray(cell_lapl_adj)[2].getAsCell();
            Cell right_cell = cell->ReferenceArray(cell_lapl_adj)[3].getAsCell();
            Cell bottom_cell = cell->ReferenceArray(cell_lapl_adj)[4].getAsCell();
            Cell top_cell = cell->ReferenceArray(cell_lapl_adj)[5].getAsCell();

            int vel_id = cell->Integer(P_id_global);
            double value = 0.0;

            int col_id;

            MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);
            if( back_cell.isValid() ) {
                col_id = back_cell.Integer(P_id_global);
                MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
            }
            if( front_cell.isValid() ) {
                col_id = front_cell.Integer(P_id_global);
                MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
            }
            if( left_cell.isValid() ) {
                col_id = left_cell.Integer(P_id_global);
                MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
            }
            if( right_cell.isValid() ) {
                col_id = right_cell.Integer(P_id_global);
                MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
            }
            if( bottom_cell.isValid() ) {
                col_id = bottom_cell.Integer(P_id_global);
                MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
            }
            if( top_cell.isValid() ) {
                col_id = top_cell.Integer(P_id_global);
                MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
            }

            ElementArray<Face> faces = cell->getFaces();

            
            col_id = faces[0]->Integer(U_id_global);
            value = -1.0 / hx * (-1.0);

            MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);

            col_id = faces[1]->Integer(U_id_global);
            value = 1.0 / hx * (-1.0);

            MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);

            col_id = faces[2]->Integer(U_id_global);
            value = -1.0 / hy * (-1.0);

            MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);

            col_id = faces[3]->Integer(U_id_global);
            value = 1.0 / hy * (-1.0);

            MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);

            col_id = faces[4]->Integer(U_id_global);
            value = -1.0 / hz * (-1.0);

            MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);

            col_id = faces[5]->Integer(U_id_global);
            value = 1.0 / hz * (-1.0);

            MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
        }
    }
}

void Stokes_matrix( Mesh *m, 
                   Mat A,
                   double hx,
                   double hy,
                   double hz,
                   MarkerType u_mrk, 
                   MarkerType v_mrk, 
                   MarkerType w_mrk, 
                   MarkerType inner_face_mrk, 
                   MapVariant parameters ) {

    matrix_U_row( m, 
                  A,
                  hx,
                  hy,
                  hz,
                  u_mrk, 
                  v_mrk, 
                  w_mrk, 
                  inner_face_mrk, 
                  parameters );
    matrix_P_row( m, 
                  A,
                  hx,
                  hy,
                  hz,
                  parameters );

}

// void matrix_Laplace_row(Mesh *m, 
//                    Mat A,
//                    double hx,
//                    double hy,
//                    double hz,
//                    MapVariant parameters) {
//     TagInteger lapl_id_global = TagInteger( m->GetTag( "lapl_id_global" ) );
    
//     TagReference cell_lapl_adj = TagReference( m->GetTag( "cell_lapl_adj" ) );
//     for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
//     {
//         if( cell->GetStatus() != Element::Ghost ) {
//             double dt = std::get<double>(parameters.at("dt"));
//             double U_d = std::get<double>(parameters.at("U_d"));
//             double L_d = std::get<double>(parameters.at("L_d"));
//             double Rho_d = std::get<double>(parameters.at("Rho_d"));
//             double Mu_d = std::get<double>(parameters.at("Mu_d"));

//             Cell back_cell = cell->ReferenceArray(cell_lapl_adj)[0].getAsCell();
//             Cell front_cell = cell->ReferenceArray(cell_lapl_adj)[1].getAsCell();
//             Cell left_cell = cell->ReferenceArray(cell_lapl_adj)[2].getAsCell();
//             Cell right_cell = cell->ReferenceArray(cell_lapl_adj)[3].getAsCell();
//             Cell bottom_cell = cell->ReferenceArray(cell_lapl_adj)[4].getAsCell();
//             Cell top_cell = cell->ReferenceArray(cell_lapl_adj)[5].getAsCell();

//             int vel_id = cell->Integer(lapl_id_global);
//             double value = 6.0;

//             int col_id;

//             MatSetValues(A,1,&vel_id,1,&vel_id,&value,ADD_VALUES);
//             value = -1.0;
//             if( back_cell.isValid() ) {
//                 col_id = back_cell.Integer(lapl_id_global);
//                 MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
//             }
//             if( front_cell.isValid() ) {
//                 col_id = front_cell.Integer(lapl_id_global);
//                 MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
//             }
//             if( left_cell.isValid() ) {
//                 col_id = left_cell.Integer(lapl_id_global);
//                 MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
//             }
//             if( right_cell.isValid() ) {
//                 col_id = right_cell.Integer(lapl_id_global);
//                 MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
//             }
//             if( bottom_cell.isValid() ) {
//                 col_id = bottom_cell.Integer(lapl_id_global);
//                 MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
//             }
//             if( top_cell.isValid() ) {
//                 col_id = top_cell.Integer(lapl_id_global);
//                 MatSetValues(A,1,&vel_id,1,&col_id,&value,ADD_VALUES);
//             }
//         }
//     }
// }