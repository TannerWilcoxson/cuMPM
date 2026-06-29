#ifndef ELECTRIC_FIELD_H
#define ELECTRIC_FIELD_H

#include <vector>

enum class FieldCalcMode {
    SOLVER_AX,
    INTERACTION_FIELD
};

class Electric_Field {
public:
    virtual ~Electric_Field() = default;

    virtual double* getDevDipoles() const = 0;
    virtual double* getDevEPoint() const = 0;
    virtual void calculate() = 0;
    virtual void updateParticleCoordinates(const std::vector<double>& x_part,
                                           const std::vector<double>& y_part,
                                           const std::vector<double>& z_part) = 0;
    virtual void setSelfCoef(const std::vector<double>& self_coef_r,
                             const std::vector<double>& self_coef_i) = 0;
};

#endif // ELECTRIC_FIELD_H
