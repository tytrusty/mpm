
#include "MpmHook.h"
#include "Util.h"
#include "igl/opengl/glfw/imgui/ImGuiHelpers.h"
#include "igl/polar_dec.h"
#include <igl/readPLY.h>
#include <string>
#include <omp.h>
#include <cmath>

using namespace Eigen;

MpmHook::MpmHook() : PhysicsHook()
{
    clickedVertex = -1;
    meshFile_ = "bunny.off";
    solverIters = 40;
    solverTol = 1e-7;
    render_color = 1;
    enable_iso  = true;
    enable_heat = false;
}

void MpmHook::initParticles(int width, int height) {
	particles_.x.resize(width*height, dimensions_);
	//particles_.v = MatrixXd::Random(width*height, dimensions_);
	particles_.v = MatrixXd::Zero(width*height, dimensions_);
	particles_.C.resize(width*height);
	particles_.F.resize(width*height);
	particles_.Jp.resize(width*height, 1);
    particles_.Jp.setConstant(1.0);
	particles_.mass.resize(width*height, 1);
    particles_.mass.setConstant(1.0);

	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < height; ++j) {
            double x = (double(i) + resolution_/4.0);
            double y = (double(j) + resolution_/4.0);
            double z = resolution_/2.0;
			particles_.x.row(i*width + j) << x,y,z; 		
            particles_.C[i*width+j].setZero();
            particles_.F[i*width+j].setIdentity();
            particles_.v(i*width + j, 2) = 0; // TODO 
        }
	}
}

void MpmHook::initGrid(int width, int height) {
	grid_.v.resize(width*height, dimensions_);
	grid_.mass.resize(width*height, 1);
}

void MpmHook::drawGUI(igl::opengl::glfw::imgui::ImGuiMenu &menu)
{
	if (ImGui::CollapsingHeader("Mesh", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputText("Filename", meshFile_);
	}
	if (ImGui::CollapsingHeader("Simulation Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Enable heat flow", &enable_heat);
        ImGui::InputInt("Solver Iters", &solverIters);
        ImGui::InputFloat("Solver Tolerance", &solverTol, 0, 0, 12);
	}
    const char* listbox_items[] = { "Inferno", "Jet", "Magma", "Parula", "Plasma", "Viridis"}; if (ImGui::CollapsingHeader("Render Options"))
    {
        ImGui::ListBox("Render color", &render_color, listbox_items, IM_ARRAYSIZE(listbox_items), 4);
        ImGui::Checkbox("Isocontour", &enable_iso);
	}
}


void MpmHook::tick()
{
    mouseMutex.lock();
    for (MouseEvent me : mouseEvents)
    {
        if (me.type == MouseEvent::ME_CLICKED)
        {
            curPos = me.pos;
            clickedVertex = me.vertex;
            mouseDown = true;
        }
        if (me.type == MouseEvent::ME_RELEASED)
        {
            clickedVertex = -1;
            mouseDown = false;
        }
        if (me.type == MouseEvent::ME_DRAGGED)
        {
            if (mouseDown)
            {
                curPos = me.pos;
                clickedVertex = me.vertex;
            }
        }
    }
    mouseEvents.clear();
    mouseMutex.unlock();
}

void MpmHook::initSimulation()
{
    Eigen::initParallel();
    prevClicked = -1;

    // Reading in mesh.
    Util::readMesh(meshFile_, V, F);

    // Scaling mesh to fill bounding box.
    Eigen::Vector3d min = V.colwise().minCoeff();
    V = V.rowwise() - min.transpose();
    V = V / V.maxCoeff();
    Eigen::Vector3d max = V.colwise().maxCoeff();
    V = V.array().rowwise() / max.transpose().array();

    // Generating point cloud from mesh.
    //particles_.x = Util::meshToPoints(V, F) * resolution_;
    //TODO if using mesh also need to init other fields (resize and whatnot)
    initParticles(32, 32);
	initGrid(64, 64);
}
bool MpmHook::simulationStep()
{

    // Initial Lamé parameters
    const double mu_0 = E / (2 * (1 + nu));
    const double lambda_0 = E * nu / ((1+nu) * (1 - 2 * nu));

    std::cout << "simulationStep() " << std::endl;
	grid_.v.setZero();
	grid_.mass.setZero();

	int ncells = resolution_*resolution_;
	int nparticles = particles_.x.rows();

    //#pragma omp parallel
    //{

	// Particles -> Grid
    //#pragma omp for
	for (int i = 0; i < nparticles; ++i) {
		const Vector3d& pos = particles_.x.row(i);
		const Vector3d& vel = particles_.v.row(i);
        const Matrix3d& F = particles_.F[i];
        const double Jp = particles_.Jp(i);

		double mass = particles_.mass(i);

		// Interpolation
		Vector3i cell_idx = pos.cast<int>();
		Vector3d diff = (pos - cell_idx.cast<double>()).array() - 0.5;
		Matrix3d weights;
		weights.row(0) = 0.5 * (0.5 - diff.array()).pow(2);
		weights.row(1) = 0.75 - diff.array().pow(2); 
		weights.row(2) = 0.5 * (0.5 + diff.array()).pow(2);

        // Lame parameters
        double e = std::exp(hardening * (1.0 - Jp));
        double mu = mu_0 * e;
        double lambda = lambda_0 * e;

        // Polar decomposition.
        //Matrix3d R, S;
        //igl::polar_dec(particles_.F[i], R, S);

        //// Cauch stress
        //double Dinv = 4 *  1 * 1;
        //Matrix3d PF = (2 * mu * (particles_.F[i] - R) * particles_.F[i].transpose() 
        //              + lambda * (J-1)*J * Matrix3d::Identity());
        //Matrix3d stress = -(timestep_ * vol) * (Dinv * PF);
        //Matrix3d affine = stress + mass * particles_.C[i];

        double volume = Jp;

        // MPM course eq 48
        // mu=20 lambda=10
        Matrix3d F_T_inv = F.transpose().inverse();
        Matrix3d P = 20.0*(F - F_T_inv) + 10*std::log(Jp)*F_T_inv;
        Matrix3d stress = (1.0 / Jp) * P * F.transpose();
        stress = -volume*4*stress*timestep_; // eq 16 MLS-MPM

		// for all surrounding 9 cells
		for (int x = 0; x < 3; ++x) {
			for (int y = 0; y < 3; ++y) {
				double weight = weights(x, 0) * weights(y, 1);

				Vector3i idx = Vector3i(cell_idx(0) + x - 1, cell_idx(1) + y - 1, resolution_/2.);
				Vector3d dist = (idx.cast<double>() - pos).array() + 0.5;
                dist(2)=0;
				
				Vector3d Q = particles_.C[i] * dist;
				//Vector3d Q = affine * dist;

				// MPM course, equation 172
				double mass_contrib = weight * mass;

				// converting 2D index to 1D
				int grid_i = idx(0) * resolution_ + idx(1);
				
				// scatter mass and momentum contributions to the grid
				grid_.mass(grid_i) += mass_contrib;
				grid_.v.row(grid_i) += mass_contrib * (vel + Q);


                // Neo-hookean contribution
                grid_.v.row(grid_i) += (weight*stress) * dist;
			}
		}
	}

	// Velocity Updates.
    //#pragma omp for
	for (int i = 0; i < ncells; ++i) {

		if (grid_.mass(i) > 0) {
			// Converting momentum to velocity and applying gravity force.
			grid_.v.row(i) /= grid_.mass(i);
			grid_.v.row(i) += timestep_ * Vector3d(0, gravity_, 0); 

			// Enforcing boundaries.
			int x = i / resolution_;
			int y = i % resolution_;
			if (x < 2 || x > resolution_ - 3) { grid_.v(i, 0) = 0; }
			if (y < 2 || y > resolution_ - 3) { grid_.v(i, 1) = 0; }
		}
	}


	// Grid -> Particles
    //#pragma omp for
	for (int i = 0; i < nparticles; ++i) {
		particles_.v.row(i).setZero();
		particles_.C[i].setZero();

		Vector3d pos = particles_.x.row(i);

		// Interpolation
		Vector3i cell_idx = pos.cast<int>();
		Vector3d diff = (pos - cell_idx.cast<double>()).array() - 0.5;
		Matrix<double, 3, 3> weights;
		weights.row(0) = 0.5 * (0.5 - diff.array()).pow(2);
		weights.row(1) = 0.75 - diff.array().pow(2);
		weights.row(2) = 0.5 * (0.5 + diff.array()).pow(2);


		for (int x = 0; x < 3; ++x) {
			for (int y = 0; y < 3; ++y) {
				double weight = weights(x, 0) * weights(y, 1);

				Vector3i idx = Vector3i(cell_idx(0) + x - 1, cell_idx(1) + y - 1, resolution_/2.);
				Vector3d dist = (idx.cast<double>() - pos).array() + 0.5;
                dist(2)=0;

				// converting 2D index to 1D
				int grid_i = idx(0) * resolution_ + idx(1);

				Vector3d weighted_vel = grid_.v.row(grid_i) * weight;

				// APIC paper equation 10, constructing inner term for B
				particles_.C[i] += 4 * weighted_vel * dist.transpose();
				particles_.v.row(i) += weighted_vel;
			}
		}
		// advect particles
        pos += particles_.v.row(i) * timestep_; 
		// clamp to ensure particles don't exit simulation domain
        for (int j = 0; j < dimensions_; ++j) {
            double x = double(pos(j));
            pos(j) = std::clamp(x, 1.0, double(resolution_ - 2));
        }
        particles_.x.row(i) = pos;
        particles_.F[i] = (Matrix3d::Identity() + timestep_ * particles_.C[i]) * particles_.F[i];
        particles_.Jp(i) = particles_.F[i].determinant();
	}

    //}
    return false;
}
