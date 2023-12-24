#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <mpi.h>

#include "include/GridManager.hpp"

using namespace std;
void InitMPI(int &rank, int &size) {
    if (MPI_Init(NULL, NULL) != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    if (MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS) {
        std::cerr << "MPI_Comm_size failed" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) {
        std::cerr << "MPI_Comm_rank failed" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }
}

int main(int argc, char **argv) {
    int rank, size;
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << "L grid_size T time_grid_size steps" << std::endl;
        return 1;
    }
    double Lx, Ly, Lz;
    Lx = Ly = Lz = atof(argv[1]);
    int grid_size = atoi(argv[2]);
    double T = atof(argv[3]);
    int time_grid_size = atoi(argv[4]);
    int steps = atoi(argv[5]);


    InitMPI(rank, size);

    double dx = Lx / static_cast<float>(grid_size), dy, dz;
    dy = dz = dx;
    double dt = T / time_grid_size;

    Function func(dx, dy, dz, Lx, Ly, Lz);
    GridSplitter splitter(grid_size);
    std::vector<Volume> volumes = splitter.split(size);
    Volume volume = volumes[rank];
    GridManager manager(func, volumes[rank], grid_size, steps, dx, dy, dz, dt, rank, size);
    
    double t0 = MPI_Wtime();
    
    // if (arguments.debug && rank == 0) {
    //     solver.PrintParams(arguments.solveParams.outputPath);
    // }

    // solver.Solve();
    manager.init_neighbours(volumes);

    std::vector<std::vector<double>> u(3, std::vector<double>(volume.size));

    manager.initial_fill(u[0], u[1]);

    double error0 = manager.calc_error(u[0], 0);
    double error1 = manager.calc_error(u[1], dt);

    // const char *outputPath = "output.txt";
    // const char *numericalPath = "numerical.json";
    // const char *analyticalPath = "analytical.json";
    // const char *differencePath = "difference.json";

    // if (rank == 0) {
    //     std::ofstream fout(outputPath, std::ios::app);
    //     fout << "Step 0 max error: " << error0 << std::endl;
    //     fout << "Step 1 max error: " << error1 << std::endl;
    //     fout.close();
    // }

    for (int step = 2; step <= steps; step++) {
        manager.fill_next(u[(step + 1) % 3], u[(step + 2) % 3], u[step % 3], step * dt);

        double error = manager.calc_error(u[step % 3], step * dt);

        // if (rank == 0) {
        //     std::ofstream fout(outputPath, std::ios::app);
        //     fout << "Step " << step << " max error: " << error << std::endl;
        //     fout.close();
        // }
    }

    // manager.save(u[steps % 3], steps * dt, volumes, numericalPath);
    
    // manager.fill_diff_values(u[steps % 3], steps * dt);
    // manager.save(u[steps % 3], steps * dt, volumes, differencePath);

    // manager.fill_analytical_values(u[0], steps * dt);
    // manager.save(u[0], steps * dt, volumes, analyticalPath);

    double final_error = manager.calc_error(u[steps % 3], steps * dt);

    double t1 = MPI_Wtime();
    double delta = t1 - t0;

    double maxTime, minTime, avgTime;

    MPI_Reduce(&delta, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&delta, &minTime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&delta, &avgTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const char* ompNumThreads = std::getenv("OMP_NUM_THREADS");
        std::ofstream fout("time.txt", std::ios::app);
        fout << "processes: " << size << ", threads: " << ompNumThreads << ", L: " << Lx << ", grid_size: " << grid_size << ", T: " << T << ", time_grid_size: " << time_grid_size << ", steps: " << steps << ", error: " << final_error << ", min: " << minTime << ", max: " << maxTime << ", avg: " << avgTime / size << std::endl;
        fout.close();
    }

    MPI_Finalize();
}