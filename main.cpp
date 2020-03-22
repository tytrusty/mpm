#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include<Eigen/StdVector>

using namespace Eigen;

constexpr int kResolution = 64;
constexpr int kDimensions = 3;
constexpr float kTimestep = 1.0f;
constexpr float kGravity = -0.05f;

const Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);

struct Particles {
	MatrixXd x; // positions
	MatrixXd v; // velocities
	//MatrixXd C; // affine matrix
	std::vector<Matrix3d, Eigen::aligned_allocator<Matrix3d>> C;

	MatrixXd mass; // masses
} particles;

struct Grid {
	MatrixXd v; // velocities
	MatrixXd mass; // masses
} grid;

void init_particles(int width, int height) {
	particles.x.resize(width*height, kDimensions);
	particles.v = MatrixXd::Random(width*height, kDimensions);
	particles.C.resize(width*height);
	particles.mass.resize(width*height, 1);

	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < height; ++j) {
			particles.x.row(i*width + j) << double(i) / kResolution, double(j) / kResolution, 0.0;
		}
	}
}

void init_grid(int width, int height) {
	grid.v.resize(width*height, kDimensions);
	grid.mass.resize(width*height, 1);
}

void simulation_step(int width, int height) {

	grid.v.setZero();
	grid.mass.setZero();

	int ncells = width * height;
	int nparticles = particles.x.rows();

	// Particles -> Grid
	for (int i = 0; i < nparticles; ++i) {
		Vector3d pos = particles.x.row(i) * kResolution;
		Vector3d vel = particles.v.row(i);
		double mass = particles.mass(i);

		// Interpolation
		Vector3i cell_idx = pos.cast<int>();
		Vector3d diff = (pos - cell_idx.cast<double>()).array() - 0.5;
		Matrix3d weights;
		weights.row(0) = 0.5 * (0.5 - diff.array()).pow(2);
		weights.row(1) = 0.75 - diff.array().pow(2); 
		weights.row(2) = 0.5 * (0.5 + diff.array()).pow(2);

		// for all surrounding 9 cells
		for (int x = 0; x < 3; ++x) {
			for (int y = 0; y < 3; ++y) {
				double weight = weights(x, 0) * weights(y, 1);

				Vector3i idx = Vector3i(cell_idx(0) + x - 1, cell_idx(0) + y - 1, 0);
				Vector3d dist = (idx.cast<double>() - pos).array() + 0.5;
				
				Vector3d Q = particles.C[i] * dist;

				// MPM course, equation 172
				double mass_contrib = weight * mass;

				// converting 2D index to 1D
				int grid_i = idx(0) * kResolution + idx(1);
				
				double cell_mass = grid.mass(grid_i);

				// scatter mass to the grid
				grid.mass(grid_i) += mass_contrib;
				grid.v.row(grid_i) += mass_contrib * (vel + Q);
				// note: currently "cell.v" refers to MOMENTUM, not velocity!
				// this gets corrected in the UpdateGrid step below.
			}
		}
	}

	// Velocity Updates.
	for (int i = 0; i < ncells; ++i) {

		if (grid.mass(i) > 0) {
			// Converting momentum to velocity and applying gravity force.
			grid.v.row(i) /= grid.mass(i);
			grid.v.row(i) += kTimestep * Vector3d(0, kGravity, 0); 

			// Enforcing boundaries.
			int x = i / kResolution;
			int y = i % kResolution;
			if (x < 2 || x > kResolution - 3) { grid.v(i, 0) = 0; }
			if (y < 2 || y > kResolution - 3) { grid.v(i, 1) = 0; }
		}
	}


	// Grid -> Particles
	for (int i = 0; i < nparticles; ++i) {

		// reset particle velocity. we calculate it from scratch each step using the grid
		particles.v.row(i).setZero();

		Vector3d pos = particles.x.row(i) * kResolution;

		// Interpolation
		Vector3i cell_idx = pos.cast<int>();
		Vector3d diff = (pos - cell_idx.cast<double>()).array() - 0.5;
		Matrix<double, 3, 3> weights;
		weights.row(0) = 0.5 * (0.5 - diff.array()).pow(2);
		weights.row(1) = 0.75 - diff.array().pow(2);
		weights.row(2) = 0.5 * (0.5 + diff.array()).pow(2);

		// constructing affine per-particle momentum matrix from APIC / MLS-MPM.
		// see APIC paper (https://web.archive.org/web/20190427165435/https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf), page 6
		// below equation 11 for clarification. this is calculating C = B * (D^-1) for APIC equation 8,
		// where B is calculated in the inner loop at (D^-1) = 4 is a constant when using quadratic interpolation functions
		Matrix<double, 3, 3> B;
		B.setZero();


		for (int x = 0; x < 3; ++x) {
			for (int y = 0; y < 3; ++y) {
				double weight = weights(x, 0) * weights(y, 1);

				Vector3i idx = Vector3i(cell_idx(0) + x - 1, cell_idx(0) + y - 1, 0);
				Vector3d dist = (idx.cast<double>() - pos).array() + 0.5;

				Vector3d weighted_vel = grid.v.row(i) * weight;

				// APIC paper equation 10, constructing inner term for B
				Matrix<double, 3, 3> b_term;
				b_term << weighted_vel * dist(0), weighted_vel*dist(1), weighted_vel*dist(2);
				B += b_term;
				particles.v.row(i) += weighted_vel;
			}
		}

		//Matrix<double, 3, 3, RowMajor> B2(B * 4);
		//particles.C.row(i) = Map<RowVectorXd>(B2.data());

		//// advect particles
		//p.x += p.v * dt;

		//// safety clamp to ensure particles don't exit simulation domain
		//p.x = math.clamp(p.x, 1, grid_res - 2);

	}


}



int main(int argc, char *argv[])
{
	init_particles(64, 64);
	init_grid(64, 64);
	simulation_step(64, 64);
	// Plot the mesh
	igl::opengl::glfw::Viewer viewer;
	viewer.data().set_points(particles.x, sea_green);
	viewer.launch();
}
