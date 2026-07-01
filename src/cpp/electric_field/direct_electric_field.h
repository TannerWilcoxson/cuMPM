#ifndef DIRECT_ELECTRIC_FIELD_H
#define DIRECT_ELECTRIC_FIELD_H

#include <vector>
#include <complex>
#include "electric_field.h"

using Complex = std::complex<double>;

class Direct_Electric_Field : public Base_Electric_Field {
private:
    std::vector<double> host_radius;
    double* d_radius = nullptr;

    void electricField();

public:
    Direct_Electric_Field(const std::vector<double>& radius = {},
                          FieldCalcMode mode = FieldCalcMode::SOLVER_AX,
                          bool solve_quadrupoles = false,
                          const std::vector<int>& quad_idxs = {});
    ~Direct_Electric_Field() override;

    Direct_Electric_Field(const Direct_Electric_Field&) = delete;
    Direct_Electric_Field& operator=(const Direct_Electric_Field&) = delete;

    void calculate() override;

    void updateParticleCoordinates(const std::vector<double>& x_part,
                                   const std::vector<double>& y_part,
                                   const std::vector<double>& z_part) override;
};

#endif // DIRECT_ELECTRIC_FIELD_H
