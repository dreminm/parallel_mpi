#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "GridSplitter.hpp"
#include "Function.hpp"

class GridManager {
public:
    int grid_size;
    double dx, dy, dz;
    double dt;
    int rank;
    int size;
    Function func;
    int steps;

    int layerSize;
    Volume volume;

    std::vector<Volume> sendNeighbours;
    std::vector<Volume> recvNeighbours;
    std::vector<int> processNeighbours;

    GridManager::GridManager(Function func, Volume volume, int grid_size, int steps, double dx, double dy, double dz, double dt, int rank, int size) {
        this->grid_size = grid_size;
        this->steps = steps;
        this->volume = volume;
        this->func = func;
        this->dt = dt;
        this->dx = dx;
        this->dy = dy;
        this->dz = dz;


        this->layerSize = std::pow(grid_size + 1, 3);

        this->rank = rank;
        this->size = size;
    }

    std::vector<double> GridManager::pack_volume(const std::vector<double> &u, Volume v) const {
        std::vector<double> packed(v.size);

        #pragma omp parallel for collapse(3)
        for (int i = v.xmin; i <= v.xmax; i++)
            for (int j = v.ymin; j <= v.ymax; j++)
                for (int k = v.zmin; k <= v.zmax; k++)
                    packed[get_index(i, j, k, v)] = u[get_local_index(i, j, k)];

        return packed;
    }

    // отправка/получение соседних значений
    std::vector<std::vector<double>> GridManager::send_recv_values(const std::vector<double> &u) const {
        std::vector<std::vector<double>> u_recv(processNeighbours.size());

        for (auto i = 0; i < processNeighbours.size(); i++) {
            std::vector<double> packed = pack_volume(u, sendNeighbours[i]);
            u_recv[i] = std::vector<double>(recvNeighbours[i].size);

            std::vector<MPI_Request> requests(2);
            std::vector<MPI_Status> statuses(2);

            MPI_Isend(packed.data(), sendNeighbours[i].size, MPI_DOUBLE, processNeighbours[i], 0, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(u_recv[i].data(), recvNeighbours[i].size, MPI_DOUBLE, processNeighbours[i], 0, MPI_COMM_WORLD, &requests[1]);
            MPI_Waitall(2, requests.data(), statuses.data());
        }

        return u_recv;
    }

    // отправка/получение общих начений
    std::vector<double> GridManager::send_recv_total(const std::vector<double> &u, const std::vector<Volume> &volumes) const {
        if (rank != 0) {
            MPI_Request request;
            MPI_Status status;

            MPI_Isend(u.data(), volume.size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
            MPI_Waitall(1, &request, &status);
            return u;
        }

        std::vector<double> u_all(layerSize);
        Volume volumeAll = make_volume(0, grid_size, 0, grid_size, 0, grid_size);

        for (int index = 0; index < size; index++) {
            std::vector<double> ui(volumes[index].size);

            if (index == rank) {
                ui = u;
            }
            else {
                std::vector<MPI_Request> requests(1);
                std::vector<MPI_Status> statuses(1);

                MPI_Irecv(ui.data(), volumes[index].size, MPI_DOUBLE, index, 0, MPI_COMM_WORLD, &requests[0]);
                MPI_Waitall(1, requests.data(), statuses.data());
            }

            for (int i = volumes[index].xmin; i <= volumes[index].xmax; i++) {
                for (int j = volumes[index].ymin; j <= volumes[index].ymax; j++) {
                    for (int k = volumes[index].zmin; k <= volumes[index].zmax; k++) {
                        u_all[get_index(i, j, k, volumeAll)] = ui[get_index(i, j, k, volumes[index])];
                    }
                }
            }
        }

        return u_all;
    }

    bool GridManager::is_inside(int xmin1, int xmax1, int ymin1, int ymax1, int xmin2, int xmax2, int ymin2, int ymax2) const {
        return xmin2 <= xmin1 && xmax1 <= xmax2 && ymin2 <= ymin1 && ymax1 <= ymax2;
    }

    bool GridManager::get_neighbours(Volume v1, Volume v2, Volume &neighbour) const {
        if (v1.xmin == v2.xmax + 1 || v2.xmin == v1.xmax + 1) {
            int x = v1.xmin == v2.xmax + 1 ? v1.xmin : v1.xmax;

            if (is_inside(v1.ymin, v1.ymax, v1.zmin, v1.zmax, v2.ymin, v2.ymax, v2.zmin, v2.zmax)) {
                neighbour = make_volume(x, x, v1.ymin, v1.ymax, v1.zmin, v1.zmax);
                return true;
            }

            if (is_inside(v2.ymin, v2.ymax, v2.zmin, v2.zmax, v1.ymin, v1.ymax, v1.zmin, v1.zmax)) {
                neighbour = make_volume(x, x, v2.ymin, v2.ymax, v2.zmin, v2.zmax);
                return true;
            }

            return false;
        }

        if (v1.ymin == v2.ymax + 1 || v2.ymin == v1.ymax + 1) {
            int y = v1.ymin == v2.ymax + 1 ? v1.ymin : v1.ymax;

            if (is_inside(v1.xmin, v1.xmax, v1.zmin, v1.zmax, v2.xmin, v2.xmax, v2.zmin, v2.zmax)) {
                neighbour = make_volume(v1.xmin, v1.xmax, y, y, v1.zmin, v1.zmax);
                return true;
            }

            if (is_inside(v2.xmin, v2.xmax, v2.zmin, v2.zmax, v1.xmin, v1.xmax, v1.zmin, v1.zmax)) {
                neighbour = make_volume(v2.xmin, v2.xmax, y, y, v2.zmin, v2.zmax);
                return true;
            }

            return false;
        }

        if (v1.zmin == v2.zmax + 1 || v2.zmin == v1.zmax + 1) {
            int z = v1.zmin == v2.zmax + 1 ? v1.zmin : v1.zmax;

            if (is_inside(v1.xmin, v1.xmax, v1.ymin, v1.ymax, v2.xmin, v2.xmax, v2.ymin, v2.ymax)) {
                neighbour = make_volume(v1.xmin, v1.xmax, v1.ymin, v1.ymax, z, z);
                return true;
            }

            if (is_inside(v2.xmin, v2.xmax, v2.ymin, v2.ymax, v1.xmin, v1.xmax, v1.ymin, v1.ymax)) {
                neighbour = make_volume(v2.xmin, v2.xmax, v2.ymin, v2.ymax, z, z);
                return true;
            }

            return false;
        }

        return false;
    }

    void GridManager::init_neighbours(const std::vector<Volume> &volumes) {
        sendNeighbours.clear();
        recvNeighbours.clear();
        processNeighbours.clear();

        for (int i = 0; i < size; i++) {
            if (i == rank)
                continue;

            Volume sendNeighbour;
            Volume recvNeighbour;

            if (!get_neighbours(volume, volumes[i], sendNeighbour))
                continue;

            get_neighbours(volumes[i], volume, recvNeighbour);
            processNeighbours.push_back(i);
            sendNeighbours.push_back(sendNeighbour);
            recvNeighbours.push_back(recvNeighbour);
        }
    }

    int GridManager::get_index(int i, int j, int k, Volume v) const {
        return (i - v.xmin) * v.dy * v.dz + (j - v.ymin) * v.dz + (k - v.zmin);
    }

    int GridManager::get_local_index(int i, int j, int k) const {
        return get_index(i, j, k, volume);
    }

    // TODO: numerical boundary values
    double GridManager::get_boundary_value(int i, int j, int k, double t) const {
        if (i == 0 || i == grid_size) {
            return func.u(i * dx, j * dy, k * dz, t);
        }

        if (j == 0 || j == grid_size) {
            return func.u(i * dx, j * dy, k * dz, t);
        }

        if (k == 0 || k == grid_size) {
            return func.u(i * dx, j * dy, k * dz, t);
        }

        throw "not boundary value";
    }

    // заполнение граничными значениями
    void GridManager::fill_boundary_values(std::vector<double> &u, double t) const {
        if (volume.xmin == 0) {
            #pragma omp parallel for collapse(2)
            for (int i = volume.ymin; i <= volume.ymax; i++)
                for (int j = volume.zmin; j <= volume.zmax; j++)
                    u[get_local_index(volume.xmin, i, j)] = get_boundary_value(volume.xmin, i, j, t);
        }

        if (volume.xmax == grid_size) {
            #pragma omp parallel for collapse(2)
            for (int i = volume.ymin; i <= volume.ymax; i++)
                for (int j = volume.zmin; j <= volume.zmax; j++)
                    u[get_local_index(volume.xmax, i, j)] = get_boundary_value(volume.xmax, i, j, t);
        }

        if (volume.ymin == 0) {
            #pragma omp parallel for collapse(2)
            for (int i = volume.xmin; i <= volume.xmax; i++)
                for (int j = volume.zmin; j <= volume.zmax; j++)
                    u[get_local_index(i, volume.ymin, j)] = get_boundary_value(i, volume.ymin, j, t);
        }

        if (volume.ymax == grid_size) {
            #pragma omp parallel for collapse(2)
            for (int i = volume.xmin; i <= volume.xmax; i++)
                for (int j = volume.zmin; j <= volume.zmax; j++)
                    u[get_local_index(i, volume.ymax, j)] = get_boundary_value(i, volume.ymax, j, t);
        }

        if (volume.zmin == 0) {
            #pragma omp parallel for collapse(2)
            for (int i = volume.xmin; i <= volume.xmax; i++)
                for (int j = volume.ymin; j <= volume.ymax; j++)
                    u[get_local_index(i, j, volume.zmin)] = get_boundary_value(i, j, volume.zmin, t);
        }

        if (volume.zmax == grid_size) {
            #pragma omp parallel for collapse(2)
            for (int i = volume.xmin; i <= volume.xmax; i++)
                for (int j = volume.ymin; j <= volume.ymax; j++)
                    u[get_local_index(i, j, volume.zmax)] = get_boundary_value(i, j, volume.zmax, t);
        }
    }

    void GridManager::initial_fill(std::vector<double> &u0, std::vector<double> &u1) const {
        fill_boundary_values(u0, 0);
        fill_boundary_values(u1, dt);

        int xmin = std::max(volume.xmin, 1);
        int xmax = std::min(volume.xmax, grid_size - 1);

        int ymin = std::max(volume.ymin, 1);
        int ymax = std::min(volume.ymax, grid_size - 1);

        int zmin = std::max(volume.zmin, 1);
        int zmax = std::min(volume.zmax, grid_size - 1);

        #pragma omp parallel for collapse(3)
        for (int i = xmin; i <= xmax; i++) {
            for (int j = ymin; j <= ymax; j++) {
                for (int k = zmin; k <= zmax; k++) {
                    u0[get_local_index(i, j, k)] = func.u(i * dx, j * dy, k * dz, 0);
                }
            }
        }

        std::vector<std::vector<double>> u0_recv = send_recv_values(u0);

        #pragma omp parallel for collapse(3)
        for (int i = xmin; i <= xmax; i++) {
            for (int j = ymin; j <= ymax; j++) {
                for (int k = zmin; k <= zmax; k++) {
                    u1[get_local_index(i, j, k)] = u0[get_local_index(i, j, k)] + dt * dt / 2 * get_diff(u0, i, j, k, u0_recv);
                }
            }
        }
    }

    void GridManager::fill_next(const std::vector<double> &u0, const std::vector<double> &u1, std::vector<double> &u, double t) {
        int xmin = std::max(volume.xmin, 1);
        int xmax = std::min(volume.xmax, grid_size - 1);

        int ymin = std::max(volume.ymin, 1);
        int ymax = std::min(volume.ymax, grid_size - 1);

        int zmin = std::max(volume.zmin, 1);
        int zmax = std::min(volume.zmax, grid_size - 1);

        std::vector<std::vector<double>> u_recv = send_recv_values(u1);

        #pragma omp parallel for collapse(3)
        for (int i = xmin; i <= xmax; i++)
            for (int j = ymin; j <= ymax; j++)
                for (int k = zmin; k <= zmax; k++)
                    u[get_local_index(i, j, k)] = 2 * u1[get_local_index(i, j, k)] - u0[get_local_index(i, j, k)] + dt * dt * get_diff(u1, i, j, k, u_recv);

        fill_boundary_values(u, t);
    }

    // заполнение аналитических значений
    void GridManager::fill_analytical_values(std::vector<double> &u, double t) const {
        #pragma omp parallel for collapse(3)
        for (int i = volume.xmin; i <= volume.xmax; i++)
            for (int j = volume.ymin; j <= volume.ymax; j++)
                for (int k = volume.zmin; k <= volume.zmax; k++)
                    u[get_local_index(i, j, k)] = func.u(i * dx, j * dy, k * dz, t);
    }

    // заполнение разности значений
    void GridManager::fill_diff_values(std::vector<double> &u, double t) const {
        #pragma omp parallel for collapse(3)
        for (int i = volume.xmin; i <= volume.xmax; i++) {
            for (int j = volume.ymin; j <= volume.ymax; j++) {
                for (int k = volume.zmin; k <= volume.zmax; k++) {
                    u[get_local_index(i, j, k)] = fabs(u[get_local_index(i, j, k)] - func.u(i * dx, j * dy, k * dz, t));
                }
            }
        }
    }

    double GridManager::get_value(const std::vector<double> &u, int i, int j, int k, const std::vector<std::vector<double>> &u_recv) const {
        for (auto it = 0; it < processNeighbours.size(); it++) {
            Volume v = recvNeighbours[it];

            if (i < v.xmin || i > v.xmax || j < v.ymin || j > v.ymax || k < v.zmin || k > v.zmax)
                continue;

            return u_recv[it][get_index(i, j, k, v)];
        }

        return u[get_local_index(i, j, k)];
    }

    // оператор Лапласа
    double GridManager::get_diff(const std::vector<double> &u, int i, int j, int k, const std::vector<std::vector<double>> &u_recv) const {
        double diff_x = (get_value(u, i - 1, j, k, u_recv) - 2 * u[get_local_index(i, j, k)] + get_value(u, i + 1, j, k, u_recv)) / (dx * dx);
        double diff_y = (get_value(u, i, j - 1, k, u_recv) - 2 * u[get_local_index(i, j, k)] + get_value(u, i, j + 1, k, u_recv)) / (dy * dy);
        double diff_z = (get_value(u, i, j, k - 1, u_recv) - 2 * u[get_local_index(i, j, k)] + get_value(u, i, j, k + 1, u_recv)) / (dz * dz);
        return diff_x + diff_y + diff_z;
    }

    // оценка погрешности на слое
    double GridManager::calc_error(const std::vector<double> &u, double t) const {
        double localError = 0;

        #pragma omp parallel for collapse(3) reduction(max: localError)
        for (int i = volume.xmin; i <= volume.xmax; i++)
            for (int j = volume.ymin; j <= volume.ymax; j++)
                for (int k = volume.zmin; k <= volume.zmax; k++)
                    localError = std::max(localError, fabs(u[get_local_index(i, j, k)] - func.u(i * dx, j * dy, k * dz, t)));

        double error;
        MPI_Reduce(&localError, &error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        return error;
    }
};

