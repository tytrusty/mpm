
#include "MpmHook.h"
#include "Util.h"
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/polar_dec.h>
#include <igl/readPLY.h>
#include <string>
#include <omp.h>
#include <cmath>

using namespace Eigen;

MpmHook::MpmHook() : PhysicsHook()
{
    clickedVertex = -1;
    meshFile_ = "bunny.off";

    is_3d_ = false;
    resolution_ = 32;
    dimensions_ = 3;
    point_size_ = 5;
    timestep_ = 0.1;
    gravity_ = -0.4;
    lambda_ = 10.0;
    mu_ = 20.0;
    render_color = 1;
    enable_iso  = true;
    enable_heat = false;
    box_dx_ = 0.5;
    point_color_= ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    enable_addbox_ = false;
}

void MpmHook::addParticleBox(Vector3d pos, Vector3i lengths, double dx) {
    int r0 = particles_.x.rows();
    int c0 = particles_.x.cols();

    lengths = (lengths.cast<double>().array() / dx).cast<int>();
    lengths(2) = 1; // if 2d

    int new_rows = r0 + lengths.prod();

    particles_.x.conservativeResize(new_rows, dimensions_);
    particles_.v.conservativeResize(new_rows, dimensions_);
    particles_.color.conservativeResize(new_rows, 3);
    particles_.Jp.conservativeResize(new_rows, 1);
    particles_.mass.conservativeResize(new_rows, 1);
    particles_.C.resize(new_rows);
    particles_.F.resize(new_rows);

    for (int i = 0; i < lengths(0); ++i) {
        for (int j = 0; j < lengths(1); ++j) {
            double x = pos(0) + double(i) * dx;
            double y = pos(1) + double(j) * dx;
            double z = resolution_/2.0;

            int idx = r0 + i*lengths(0) + j;
            particles_.x.row(idx) << x,y,z;         
            particles_.v.row(idx) = Vector3d::Zero();
            particles_.C[idx].setZero();
            particles_.F[idx].setIdentity();
            particles_.Jp(idx) = 1.0;
            particles_.mass(idx) = 1.0;
            particles_.color.row(idx) << point_color_.x, point_color_.y, point_color_.z;
        }
    }

}

void MpmHook::initParticles(int width, int height) {
    particles_.x.resize(width*height, dimensions_);
    particles_.v = MatrixXd::Zero(width*height, dimensions_);
    particles_.C.resize(width*height);
    particles_.F.resize(width*height);
    particles_.Jp.resize(width*height, 1);
    particles_.Jp.setConstant(1.0);
    particles_.mass.resize(width*height, 1);
    particles_.mass.setConstant(1.0);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            double x = (double(i)/2. + resolution_/4.0);
            double y = (double(j)/2. + resolution_/4.0);
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
        ImGui::Checkbox("Enable 3D", &is_3d_);
        ImGui::InputInt("Grid Resolution", &resolution_);
        ImGui::InputInt("Particle size", &point_size_);
        ImGui::InputDouble("Timestep", &timestep_);
        ImGui::InputDouble("Gravity", &gravity_);
        ImGui::InputDouble("Lambda", &lambda_);
        ImGui::InputDouble("Mu", &mu_);
        ImGui::InputFloat("Solver Tolerance", &solverTol);//, 0, 0, 12);
    }
    const char* listbox_items[] = { "Inferno", "Jet", "Magma", "Parula", "Plasma", "Viridis"};
    if (ImGui::CollapsingHeader("Render Options"))
    {
        ImGui::ListBox("Render color", &render_color, listbox_items, IM_ARRAYSIZE(listbox_items), 4);
        ImGui::Checkbox("Isocontour", &enable_iso);
    }
    if (ImGui::CollapsingHeader("Box Options"))
    {
        ImGui::Checkbox("Enable placement", &enable_addbox_);
        ImGui::ColorEdit3("Point color", (float*)&point_color_);
        ImGui::InputDouble("Spacing", &box_dx_);
    }
}

bool MpmHook::mouseClicked(igl::opengl::glfw::Viewer &viewer, int button) {
    if(button != 0)
        return false;

    render_mutex.lock();

    MouseEvent me;
    int fid;
    Eigen::Vector3f bc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.core().viewport(3) - viewer.current_mouse_y;
    if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core().view,
      viewer.core().proj, viewer.core().viewport, grid_V, grid_F, fid, bc))
    {
        me.type = MouseEvent::ME_CLICKED;
        me.vertex = grid_F(fid,0);
    }
    else
    {
        me.type = MouseEvent::ME_RELEASED;
    }
    render_mutex.unlock();

    mouseMutex.lock();
    mouseEvents.push_back(me);
    mouseMutex.unlock();
    return false;
}

bool MpmHook::mouseReleased(igl::opengl::glfw::Viewer &viewer,  int button) {
    MouseEvent me;
    me.type = MouseEvent::ME_RELEASED;
    mouseMutex.lock();
    mouseEvents.push_back(me);
    mouseMutex.unlock();
    return false;
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

void MpmHook::buildLaplacian() {
    //TODO currently assuming 2D when indexing
    L_ = SparseMatrix<double>(T_.rows(), T_.rows());
    L_.setZero();

    std::vector<Triplet<double>> triplets;

    for (int i = 0; i < T_.rows(); ++i) {
        int x = i / resolution_; // (resolution_*resolutoin_ for 3D)
        int y = i % resolution_;

        double dt = 1e-1;

        triplets.emplace_back(Triplet<double>(i, i, 1 + 4*dt));

        if ((x+1) < resolution_)
            triplets.emplace_back(Triplet<double>(i, (x+1)*resolution_ + y, -dt));

        if ((x-1) >= 0)
            triplets.emplace_back(Triplet<double>(i, (x-1)*resolution_ + y, -dt));

        if ((y+1) < resolution_)
            triplets.emplace_back(Triplet<double>(i, x*resolution_ + (y+1), -dt));

        if ((y-1) >= 0)
            triplets.emplace_back(Triplet<double>(i, x*resolution_ + (y-1), -dt));

    }
    L_.setFromTriplets(triplets.begin(), triplets.end());
}

void MpmHook::initSimulation()
{
    Eigen::initParallel();
    prevClicked = -1;

    particles_.clear();

    // Reading in mesh.
    //Util::readMesh(meshFile_, V, F);

    //// Scaling mesh to fill bounding box.
    //Eigen::Vector3d min = V.colwise().minCoeff();
    //V = V.rowwise() - min.transpose();
    //V = V / V.maxCoeff();
    //Eigen::Vector3d max = V.colwise().maxCoeff();
    //V = V.array().rowwise() / max.transpose().array();
    // Generating point cloud from mesh.
    //particles_.x = Util::meshToPoints(V, F) * resolution_;

    //TODO if using mesh also need to init other fields (resize and whatnot)
    //initParticles(resolution_/2., resolution_/2.); // create basic brick
    addParticleBox(Vector3d(resolution_/2.,resolution_/2.,16.), Vector3i(8,8,1), box_dx_);
    initGrid(resolution_, resolution_); // create domain

    T_ = MatrixXd::Zero(resolution_*resolution_, 1); // Temperature
    f_ = MatrixXd::Zero(resolution_*resolution_, 1); // Source
    f_( 0.5*resolution_*(resolution_ + 1)) = 1.0;
    buildLaplacian();
}
bool MpmHook::simulationStep() {
    // Initial Lamé parameters
    const double mu_0 = E / (2 * (1 + nu));
    const double lambda_0 = E * nu / ((1+nu) * (1 - 2 * nu));

    std::cout << "simulationStep() " << std::endl;
    grid_.v.setZero();
    grid_.mass.setZero();

    if (enable_addbox_ && clickedVertex != -1) {
        Vector3d pos = grid_V.row(clickedVertex) * resolution_;
        addParticleBox(pos, Vector3i(8,8,1), box_dx_);
    }

    int ncells = resolution_*resolution_;
    int nparticles = particles_.x.rows();

    bool failure = false;

    ////////
    ConjugateGradient<SparseMatrix<double>, Lower|Upper> solver;
    SparseMatrix<double> I(T_.rows(), T_.rows());
    I.setIdentity();
    solver.compute(L_);
    
    T_ = solver.solve(f_);
    f_ = T_; // Set new source!
    f_( 0.5*resolution_*(resolution_ + 1)) = 5.0;
    ////////




    // Particles -> Grid
    #pragma omp parallel for
    for (int i = 0; i < nparticles; ++i) {

        if (failure) continue;

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
        Matrix3d P = mu_*(F - F_T_inv) + lambda_*std::log(Jp)*F_T_inv;
        Matrix3d stress = (1.0 / Jp) * P * F.transpose();
        stress = -volume*4*stress*timestep_; // eq 16 MLS-MPM

        if (isnan(stress(0,0))) {
            std::cout << "Stress is nan for particle : " << i << std::endl;
            std::cout << "pos: \n" << pos << std::endl;
            std::cout << "vel: \n" << vel << std::endl;
            std::cout << "F: \n" << F << std::endl;
            std::cout << "Jp: \n" << Jp << std::endl;
            std::cout << "mass: \n" << mass << std::endl;
            std::cout << "log Jp: " << std::log(Jp) << std::endl;
            failure = true;
            continue;
        }

        //std::cout << "Stress: \n" << stress << std::endl;
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
                
                Vector3d vel_momentum = mass_contrib * (vel + Q);
                Vector3d neohookean_momentum = (weight*stress) * dist;
                vel_momentum += neohookean_momentum;


                // scatter mass and momentum contributions to the grid
                #pragma omp critical
                {
                    grid_.mass(grid_i) += mass_contrib;
                    grid_.v.row(grid_i) += vel_momentum;
                }
            }
        }
    }

    if (failure) return true;

    // Velocity Updates.
    #pragma omp parallel for
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
    #pragma omp parallel for
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
        particles_.Jp(i) = std::abs(particles_.F[i].determinant());
    }

    //} // omp parallel
    return false;
}
