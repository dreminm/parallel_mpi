#include <cmath>

class Function {
public:
    Function() = default;
    Function(double dx, double dy, double dz, double Lx, double Ly, double Lz)
    : dx(dx), dy(dy), dz(dz), Lx(Lx), Ly(Ly), Lz(Lz) {
        a_t = M_PI * std::sqrt(4.0 / std::pow(Lx, 2) + 16.0 / std::pow(Ly, 2) + 36.0 / std::pow(Lz, 2));
    }

    double u(double x, double y, double z, double t) const {
        return std::sin(x * 2.0 * M_PI / Lx) * std::sin(y * 4.0 * M_PI / Ly) * std::sin(z * 6.0 * M_PI / Lz) * std::cos(a_t * t);
    }

    double u_from_idx(int i, int j, int k, double t) const {
        return u(i * dx, j * dy, k * dz, t);
    }

    int get_index(int i, int j, int k, Volume v) const {
        return (i - v.xmin) * v.dy * v.dz + (j - v.ymin) * v.dz + (k - v.zmin);
    }
    
    double dx, dy, dz;
    double Lx, Ly, Lz;
    double a_t;
};