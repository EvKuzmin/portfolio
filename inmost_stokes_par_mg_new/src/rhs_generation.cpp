#include "rhs_generation.hpp"

void rhs_U(Mesh *m,
           Vec rhs,
           double hx,
           double hy,
           double hz,
           MarkerType u_mrk, 
           MarkerType v_mrk, 
           MarkerType w_mrk, 
           MarkerType inner_face_mrk,
           MapVariant parameters)
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
    TagReal P = TagReal( m->GetTag( "P" ) );
    TagReal U0 = TagReal( m->GetTag( "U0" ) );

    for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
    {
        if( face->GetStatus() != Element::Ghost ) {
            int row_id = face->Integer(U_id_global);
            double value = 0;
            if( face->GetMarker(inner_face_mrk) ) {

                Storage::reference_array adj_faces = face->ReferenceArray(face_lapl_adj);

                double rho = std::get< double >(parameters.at("rho"));
                double rho0 = std::get< double >(parameters.at("rho"));

                value = rho0 * face->Real(U0) / dt;
                // value += dudt + grad_p + lapl_u + brinkman;
            }
            VecSetValues(rhs, 1, &row_id, &value, INSERT_VALUES);
        }
    }
}

void Stokes_rhs(Mesh *m,
           Vec rhs,
           double hx,
           double hy,
           double hz,
           MarkerType u_mrk, 
           MarkerType v_mrk, 
           MarkerType w_mrk, 
           MarkerType inner_face_mrk,
           MapVariant parameters) 
{
    double dt = std::get<double>(parameters.at("dt"));
    double U_d = std::get<double>(parameters.at("U_d"));
    double L_d = std::get<double>(parameters.at("L_d"));
    double Rho_d = std::get<double>(parameters.at("Rho_d"));
    double Mu_d = std::get<double>(parameters.at("Mu_d"));

    rhs_U(m,
          rhs,
          hx,
          hy,
          hz,
          u_mrk, 
          v_mrk, 
          w_mrk, 
          inner_face_mrk,
          parameters);

    // rhs_P(m,
    //       rhs,
    //       face_unknowns,
    //       cell_unknowns,
    //       hx,
    //       hy,
    //       hz,
    //       u_mrk, 
    //       v_mrk, 
    //       w_mrk, 
    //       inner_face_mrk,
    //       parameters);
}