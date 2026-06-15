#ifndef NEAR_FIELD_H
#define NEAR_FIELD_H

#include <vector>
#include <complex>
#include <string>
#include <memory>
#include "electric_field/electric_field.h"

using Complex = std::complex<double>;

class Near_Field {
private:
    std::vector<double> box;
    std::vector<Complex> E0;
    std::vector<double> radius;
    double xi;
    double errortol;
    std::string field_type;

    double length_scale = 1.0;
    size_t num_particles = 0;
    size_t num_field_points = 0;
    bool use_polydisperse = false;

    std::unique_ptr<Electric_Field> EF;

    std::vector<double> dip_pos_x;
    std::vector<double> dip_pos_y;
    std::vector<double> dip_pos_z;

    std::vector<double> field_pos_x;
    std::vector<double> field_pos_y;
    std::vector<double> field_pos_z;

    std::vector<double> dips_xr;
    std::vector<double> dips_xi;
    std::vector<double> dips_yr;
    std::vector<double> dips_yi;
    std::vector<double> dips_zr;
    std::vector<double> dips_zi;

public:
    Near_Field(const std::vector<double>& box,
               const std::vector<Complex>& E0,
               const std::vector<double>& radius = {},
               double xi = 0.5,
               double errortol = 1e-3,
               const std::string& field_type = "auto");

    ~Near_Field();

    void set_dipoles(const std::vector<Complex>& dip);
    void set_dipole_positions(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z);
    void set_field_points(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z);

    std::vector<double> calculate();
};

#endif // NEAR_FIELD_H
