
#include"dimensions.hpp"

void dimless_parameters( MapVariant &parameters )
{
    double U_d = std::get<double>(parameters.at("U_d"));
    double L_d = std::get<double>(parameters.at("L_d"));
    double Rho_d = std::get<double>(parameters.at("Rho_d"));
    double Mu_d = std::get<double>(parameters.at("Mu_d"));

    std::get< double >(parameters.at("mu")) = std::get< double >(parameters.at("mu")) / Mu_d;

    std::get< double >(parameters.at("rho")) = std::get< double >(parameters.at("rho")) / Rho_d;
    std::get< double >(parameters.at("dt")) = std::get< double >(parameters.at("dt")) * std::get< double >(parameters.at("U_d")) / std::get< double >(parameters.at("L_d"));
    std::get< double >(parameters.at("dt_max")) = std::get< double >(parameters.at("dt_max")) * std::get< double >(parameters.at("U_d")) / std::get< double >(parameters.at("L_d"));
    std::get< double >(parameters.at("inlet_velocity")) = std::get< double >(parameters.at("inlet_velocity")) / std::get< double >(parameters.at("U_d"));

}