#ifndef DIRECT_ELECTRIC_FIELD_H
#define DIRECT_ELECTRIC_FIELD_H

#include <vector>
#include <complex>
#include "electric_field.h"

using Complex = std::complex<double>;

enum class PrecisionMode {
    AUTO,
    MIXED,
    DOUBLE,
    FP32 = MIXED,
    FP64 = DOUBLE
};

class Direct_Electric_Field : public Base_Electric_Field {
private:
    std::vector<double> host_radius;
    double* d_radius = nullptr;

    PrecisionMode precision_setting = PrecisionMode::AUTO;
    bool use_fp32 = false;

    // Device FP32 buffers (allocated and used when use_fp32 == true)
    float* d_float_x_part = nullptr;
    float* d_float_y_part = nullptr;
    float* d_float_z_part = nullptr;
    float* d_float_radius = nullptr;
    float2* d_float_dipoles = nullptr;
    float2* d_float_self_coef = nullptr;
    float* d_float_x_field = nullptr;
    float* d_float_y_field = nullptr;
    float* d_float_z_field = nullptr;
    float2* d_float_E_point = nullptr;

    size_t fp32_num_particles = 0;
    size_t fp32_num_field_points = 0;
    size_t fp32_num_quads = 0;

    void freeFloatBuffers();
    void allocateFloatBuffers(size_t num_p, size_t num_fp, size_t n_quads);
    bool determinePrecisionMode(PrecisionMode mode);

    void electricField();

public:
    Direct_Electric_Field(const std::vector<double>& radius = {},
                          FieldCalcMode mode = FieldCalcMode::SOLVER_AX,
                          bool solve_quadrupoles = false,
                          const std::vector<int>& quad_idxs = {},
                          PrecisionMode precision = PrecisionMode::AUTO);
    ~Direct_Electric_Field() override;

    Direct_Electric_Field(const Direct_Electric_Field&) = delete;
    Direct_Electric_Field& operator=(const Direct_Electric_Field&) = delete;

    bool isUsingFP32() const { return use_fp32; }
    PrecisionMode getPrecisionMode() const { return precision_setting; }

    void calculate() override;

    void updateParticleCoordinates(const std::vector<double>& x_part,
                                   const std::vector<double>& y_part,
                                   const std::vector<double>& z_part) override;
};

#endif // DIRECT_ELECTRIC_FIELD_H
