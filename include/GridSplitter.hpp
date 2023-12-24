#include <vector>
#include <algorithm>

struct Volume {
    int xmin;
    int xmax;
    int ymin;
    int ymax;
    int zmin;
    int zmax;
    int dx;
    int dy;
    int dz;
    int size;
};

enum Axis {
    X_AXIS,
    Y_AXIS,
    Z_AXIS
};

Volume make_volume(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax) {
    int dx = xmax - xmin + 1;
    int dy = ymax - ymin + 1;
    int dz = zmax - zmin + 1;
    int size = dx * dy * dz;

    return { xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz, size };
}

class GridSplitter {
public:
    GridSplitter(int n) : n(n) {}
    std::vector<Volume> split(int size);
private:
    int n;
    void split_blocks(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax, int size, Axis axis, std::vector<Volume> &volumes);
};

void GridSplitter::split_blocks(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax, int size, Axis axis, std::vector<Volume> &volumes) {
    if (size == 1) {
        volumes.push_back(make_volume(xmin, xmax, ymin, ymax, zmin, zmax));
        return;
    }
    
    if (size % 2 == 1) {
        if (axis == X_AXIS) {
            int coordinate = xmin + (xmax - xmin) / size;
            volumes.push_back(make_volume(xmin, coordinate, ymin, ymax, zmin, zmax));
            xmin = coordinate + 1;
            axis = Y_AXIS;
        }
        else if (axis == Y_AXIS) {
            int coordinate = ymin + (ymax - ymin) / size;
            volumes.push_back(make_volume(xmin, xmax, ymin, coordinate, zmin, zmax));
            ymin = coordinate + 1;
            axis = Z_AXIS;
        }
        else {
            int coordinate = zmin + (zmax - zmin) / size;
            volumes.push_back(make_volume(xmin, xmax, ymin, ymax, zmin, coordinate));
            zmin = coordinate + 1;
            axis = X_AXIS;
        }

        size--;
    }

    if (axis == X_AXIS) {
        int coordinate = (xmin + xmax) / 2;
        split_blocks(xmin, coordinate, ymin, ymax, zmin, zmax, size / 2, Y_AXIS, volumes);
        split_blocks(coordinate + 1, xmax, ymin, ymax, zmin, zmax, size / 2, Y_AXIS, volumes);
    }
    else if (axis == Y_AXIS) {
        int coordinate = (ymin + ymax) / 2;
        split_blocks(xmin, xmax, ymin, coordinate, zmin, zmax, size / 2, Z_AXIS, volumes);
        split_blocks(xmin, xmax, coordinate + 1, ymax, zmin, zmax, size / 2, Z_AXIS, volumes);
    }
    else {
        int coordinate = (zmin + zmax) / 2;
        split_blocks(xmin, xmax, ymin, ymax, zmin, coordinate, size / 2, X_AXIS, volumes);
        split_blocks(xmin, xmax, ymin, ymax, coordinate + 1, zmax, size / 2, X_AXIS, volumes);
    }
}

std::vector<Volume> GridSplitter::split(int size) {
    std::vector<Volume> volumes;
    split_blocks(0, n, 0, n, 0, n, size, X_AXIS, volumes);
    return volumes;
}