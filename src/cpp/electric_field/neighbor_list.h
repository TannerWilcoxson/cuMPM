#ifndef NEIGHBOR_LIST_H
#define NEIGHBOR_LIST_H

#include <vector>
#include <cstddef>

class NeighborList {
private:
    int* d_neighbor_list = nullptr;
    int* d_neighbor_counts = nullptr;
    int* d_particle_offsets = nullptr;
    int max_neighbors = 0;
    size_t num_pairs = 0;
    size_t num_particles_allocated = 0;

    // Cell List specific fields
    int* d_particle_cell_ids = nullptr;
    int* d_particle_ids = nullptr;
    int* d_cell_starts = nullptr;
    int* d_cell_ends = nullptr;
    size_t num_field_points_allocated = 0;
    size_t num_cells_allocated = 0;

    void free_buffers();

public:
    NeighborList() = default;
    ~NeighborList();

    // Disable copy constructor and copy assignment operator to prevent double-free
    NeighborList(const NeighborList&) = delete;
    NeighborList& operator=(const NeighborList&) = delete;

    // Getters
    int* get_list() const { return d_neighbor_list; }
    int* get_counts() const { return d_neighbor_counts; }
    int* get_offsets() const { return d_particle_offsets; }
    int get_max_neighbors() const { return max_neighbors; }
    size_t get_num_pairs() const { return num_pairs; }

    // Build the neighbor list and offsets
    void build(
        const double* d_x_part, const double* d_y_part, const double* d_z_part,
        const double* d_x_field, const double* d_y_field, const double* d_z_field,
        size_t num_particles, size_t num_field_points,
        double box_x, double box_y, double box_z,
        double rc, bool calc_inter_dipole,
        int initial_max_neighbors = 128
    );

    // Host helper
    void get_host(std::vector<int>& host_list, std::vector<int>& host_counts) const;
};

#endif // NEIGHBOR_LIST_H
