#ifndef ELECTRIC_FIELD_H
#define ELECTRIC_FIELD_H

#include <vector>
#include <complex>

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

class Base_Electric_Field : public Electric_Field {
protected:
    size_t num_particles = 0;

    double* d_x_part = nullptr;
    double* d_y_part = nullptr;
    double* d_z_part = nullptr;

    double* d_dipoles = nullptr;
    double* d_E_point = nullptr;

    double* d_self_coef_r = nullptr;
    double* d_self_coef_i = nullptr;

    double* d_x_field = nullptr;
    double* d_y_field = nullptr;
    double* d_z_field = nullptr;
    size_t num_field_points = 0;

    bool particles_updated = false;
    bool field_points_updated = false;
    bool dipoles_updated = false;

    bool solve_quadrupoles = false;
    std::vector<int> quad_idxs;
    int* d_quad_idxs = nullptr;
    int* d_quad_map = nullptr;
    size_t num_quads = 0;

    FieldCalcMode mode = FieldCalcMode::SOLVER_AX;

public:
    Base_Electric_Field(FieldCalcMode mode = FieldCalcMode::SOLVER_AX,
                        bool solve_quadrupoles = false,
                        const std::vector<int>& quad_idxs = {});
    virtual ~Base_Electric_Field();

    Base_Electric_Field(const Base_Electric_Field&) = delete;
    Base_Electric_Field& operator=(const Base_Electric_Field&) = delete;

    // Getters and Setters
    double* getDevXPart() const { return d_x_part; }
    double* getDevYPart() const { return d_y_part; }
    double* getDevZPart() const { return d_z_part; }
    size_t getNumParticles() const { return num_particles; }

    double* getDevXField() const { return d_x_field; }
    double* getDevYField() const { return d_y_field; }
    double* getDevZField() const { return d_z_field; }
    size_t getNumFieldPoints() const { return num_field_points; }

    FieldCalcMode getCalcMode() const { return mode; }

    bool getParticlesUpdated() const { return particles_updated; }
    bool getFieldPointsUpdated() const { return field_points_updated; }
    void clearParticlesUpdated() { particles_updated = false; }
    void clearFieldPointsUpdated() { field_points_updated = false; }

    double* getDevDipoles() const override { return d_dipoles; }
    double* getDevSelfCoefReal() const { return d_self_coef_r; }
    double* getDevSelfCoefImag() const { return d_self_coef_i; }
    bool getDipolesUpdated() const { return dipoles_updated; }
    void clearDipolesUpdated() { dipoles_updated = false; }

    double* getDevEPoint() const override { return d_E_point; }

    bool getSolveQuadrupoles() const { return solve_quadrupoles; }
    size_t getNumQuads() const { return num_quads; }
    int* getDevQuadIdxs() const { return d_quad_idxs; }
    int* getDevQuadMap() const { return d_quad_map; }

    // Coordinate and dipole update methods
    void updateParticleCoordinates(const std::vector<double>& x_part,
                                   const std::vector<double>& y_part,
                                   const std::vector<double>& z_part) override;

    void updateFieldCoordinates(const std::vector<double>& x_field,
                                 const std::vector<double>& y_field,
                                 const std::vector<double>& z_field);

    void updateDipoles(const std::vector<double>& dip_x,
                       const std::vector<double>& dip_y,
                       const std::vector<double>& dip_z);

    void updateDipolesComplex(const std::vector<double>& dip_xr, const std::vector<double>& dip_xi,
                              const std::vector<double>& dip_yr, const std::vector<double>& dip_yi,
                              const std::vector<double>& dip_zr, const std::vector<double>& dip_zi);

    void updateQuadrupoles(const std::vector<double>& quad_1,
                           const std::vector<double>& quad_2,
                           const std::vector<double>& quad_3,
                           const std::vector<double>& quad_4,
                           const std::vector<double>& quad_5);

    void updateQuadrupolesComplex(const std::vector<double>& quad_1r, const std::vector<double>& quad_1i,
                                  const std::vector<double>& quad_2r, const std::vector<double>& quad_2i,
                                  const std::vector<double>& quad_3r, const std::vector<double>& quad_3i,
                                  const std::vector<double>& quad_4r, const std::vector<double>& quad_4i,
                                  const std::vector<double>& quad_5r, const std::vector<double>& quad_5i);

    // Self coefficients
    void setSelfCoef(const std::vector<double>& self_coef_r,
                     const std::vector<double>& self_coef_i) override;
    void setSelfCoef(double val_r, double val_i = 0.0);

    // Host helper getters
    void getDipolesHost(std::vector<double>& host_dip_x,
                        std::vector<double>& host_dip_y,
                        std::vector<double>& host_dip_z) const;
    void getDipolesComplexHost(std::vector<double>& host_dip_xr, std::vector<double>& host_dip_xi,
                               std::vector<double>& host_dip_yr, std::vector<double>& host_dip_yi,
                               std::vector<double>& host_dip_zr, std::vector<double>& host_dip_zi) const;
    std::vector<std::complex<double>> getEPointHost() const;
    void clearEPoint();
};

#endif // ELECTRIC_FIELD_H
