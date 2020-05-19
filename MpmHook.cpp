
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
    button_ = -1;
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
    enable_snow_ = true;
    theta_compression_ = 2.5e-2; // values from snow paper
    theta_stretch_ = 7.5e-3;
    box_dx_ = 0.5;
    point_color_= ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    enable_addbox_ = false;
    render_particle_heat_ = true;
    render_grid_heat_ = true;
    alpha_ = 55.0;
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
    particles_.mu.conservativeResize(new_rows, 1);
    particles_.T.conservativeResize(new_rows, 1);
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
            particles_.mu(idx) = mu_;
            particles_.T(idx) = 0.0;
            particles_.color.row(idx) << point_color_.x, point_color_.y, point_color_.z;
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
        ImGui::Checkbox("Enable Snow", &enable_snow_);
        ImGui::InputInt("Grid Resolution", &resolution_);
        ImGui::InputDouble("Timestep", &timestep_);
        ImGui::InputDouble("Gravity", &gravity_);
        ImGui::InputDouble("Lambda", &lambda_);
        ImGui::InputDouble("Mu", &mu_);
        ImGui::InputDouble("Critical Compression", &theta_compression_);
        ImGui::InputDouble("Critical Stretch", &theta_stretch_);
        ImGui::InputDouble("Thermal Diffusivity", &alpha_);
    }
    const char* listbox_items[] = { "Inferno", "Jet", "Magma", "Parula", "Plasma", "Viridis"};
    if (ImGui::CollapsingHeader("Render Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::ListBox("Render color", &render_color, listbox_items, IM_ARRAYSIZE(listbox_items), 4);
        ImGui::Checkbox("Render particle temperature", &render_particle_heat_);
        ImGui::Checkbox("Render grid temperature", &render_grid_heat_);
        ImGui::InputInt("Particle size", &point_size_);
    }
    if (ImGui::CollapsingHeader("Box Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Enable placement", &enable_addbox_);
        ImGui::ColorEdit3("Point color", (float*)&point_color_);
        ImGui::InputDouble("Spacing", &box_dx_);
    }
}

bool MpmHook::mouseClicked(igl::opengl::glfw::Viewer &viewer, int button) {
    std::cout << "button: " << button << std::endl;

    render_mutex.lock();

    MouseEvent me;
    me.button = button;
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
        me.pos = grid_V.row(me.vertex) * resolution_;
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
    for (MouseEvent me : mouseEvents) {
        if (me.type == MouseEvent::ME_CLICKED) {
            curPos = me.pos;
            clickedVertex = me.vertex;
            button_ = me.button;
        }
        if (me.type == MouseEvent::ME_RELEASED) {
            clickedVertex = -1;
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

    double dt = 1e-0 * alpha_;

    for (int i = 0; i < T_.rows(); ++i) {
        int x = i / resolution_; // (resolution_*resolutoin_ for 3D)
        int y = i % resolution_;

        if (grid_.mass(i) > 1e-7) {

        triplets.emplace_back(Triplet<double>(i, i, 1 + 4*dt));

        if ((x+1) < resolution_)
            triplets.emplace_back(Triplet<double>(i, (x+1)*resolution_ + y, -dt));
        if ((x-1) >= 0)
            triplets.emplace_back(Triplet<double>(i, (x-1)*resolution_ + y, -dt));
        if ((y+1) < resolution_)
            triplets.emplace_back(Triplet<double>(i, x*resolution_ + (y+1), -dt));
        if ((y-1) >= 0)
            triplets.emplace_back(Triplet<double>(i, x*resolution_ + (y-1), -dt));

        } else {  triplets.emplace_back(Triplet<double>(i, i, dt)); }
    }
    L_.setFromTriplets(triplets.begin(), triplets.end());
}

void MpmHook::initSimulation()
{
    Eigen::initParallel();

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

    addParticleBox(Vector3d(resolution_/2.,resolution_/2.,16.), Vector3i(8,8,1), box_dx_);
    initGrid(resolution_, resolution_); // create domain

    T_ = MatrixXd::Zero(resolution_*resolution_, 1); // Temperature
    f_ = MatrixXd::Zero(resolution_*resolution_, 1); // Source
    buildLaplacian();

    grid_V.resize(0,0);
    grid_F.resize(0,0);
    igl::triangulated_grid(resolution_,resolution_,grid_V,grid_F);
}
bool MpmHook::simulationStep() {
    grid_.v.setZero();
    grid_.mass.setZero();

    f_.setZero();
    //TODO aaa
    T_.setZero();

    if (enable_addbox_ && (clickedVertex != -1) && (button_ == 0)) {
        std::cout << "curPos : " << curPos << std::endl;
        addParticleBox(curPos, Vector3i(8,8,1), box_dx_);
        clickedVertex = -1;
    }

    int ncells = resolution_*resolution_;
    int nparticles = particles_.x.rows();

    bool failure = false;

    // Particles -> Grid
    #pragma omp parallel for
    for (int i = 0; i < nparticles; ++i) {

        if (failure) continue;

        const Vector3d& pos = particles_.x.row(i);
        const Vector3d& vel = particles_.v.row(i);
        const Matrix3d& F = particles_.F[i];
        const double Jp = particles_.Jp(i);
        const double mu = particles_.mu(i);

        double mass = particles_.mass(i);

        // Interpolation
        Vector3i cell_idx = pos.cast<int>();
        Vector3d diff = (pos - cell_idx.cast<double>()).array() - 0.5;
        Matrix3d weights;
        weights.row(0) = 0.5 * (0.5 - diff.array()).pow(2);
        weights.row(1) = 0.75 - diff.array().pow(2); 
        weights.row(2) = 0.5 * (0.5 + diff.array()).pow(2);

        // Lame parameters
        // Initial Lamé parameters
        //const double mu_0 = E / (2 * (1 + nu));
        //const double lambda_0 = E * nu / ((1+nu) * (1 - 2 * nu));
        //double e = std::exp(hardening * (1.0 - Jp));
        //double mu = mu_0 * e;
        //double lambda = lambda_0 * e;

        double volume = Jp;
        // Neohoookean  MPM course eq 48
        Matrix3d F_T_inv = F.transpose().inverse();
        Matrix3d P = mu*(F - F_T_inv) + lambda_*std::log(Jp)*F_T_inv;
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

                    // TODO aaa
                    f_(grid_i) += mass_contrib * particles_.T(i);
                }
            }
        }
    }

    if (failure) return true;

    // Velocity Updates.
    #pragma omp parallel for
    for (int i = 0; i < ncells; ++i) {

        if (grid_.mass(i) > 1e-7) {
            // Converting momentum to velocity and applying gravity force.
            grid_.v.row(i) /= grid_.mass(i);
            grid_.v.row(i) += timestep_ * Vector3d(0, gravity_, 0); 

            // Enforcing boundaries.
            int x = i / resolution_;
            int y = i % resolution_;
            if (x < 2 || x > resolution_ - 3) { grid_.v(i, 0) = 0; }
            if (y < 2 || y > resolution_ - 3) { grid_.v(i, 1) = 0; }

            //std::cout << "f_(i): " << f_(i) << std::endl;
            //std::cout << "T_(i): " << T_(i) << std::endl;
        } else {
            // cells that do no receive mass are considered air cells
            // so the temperature is zeroed out.
            // TODO aaa
            f_(i) = 0.5;

        }
    }

    // -------------------------------------------------------- //
    //std::cout << " f_ : " << f_ << std::endl;
    //f_ = T_; // Set new source!
    f_(0.5*resolution_*(resolution_ + 1)) = 10.0;
    // Solve temperature field update.
    buildLaplacian();
    ConjugateGradient<SparseMatrix<double>, Lower|Upper> solver;
    SparseMatrix<double> I(T_.rows(), T_.rows());
    I.setIdentity();
    solver.compute(L_);
    T_ = solver.solve(f_);
    // ------------------------------------------------------- //

    #pragma omp parallel for
    for (int i = 0; i < ncells; ++i) {
        if (grid_.mass(i) < 1e-7) {
            T_(i) = 0.;
        }
    }

    particles_.v.setZero();
    particles_.T.setZero();

    // Grid -> Particles
    #pragma omp parallel for
    for (int i = 0; i < nparticles; ++i) {
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
                particles_.T(i) += weight * T_(grid_i);
            }
        }
        
        // Melt
        if (particles_.T(i) > 0.8) {
            particles_.mu(i) = 0.;
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

        // Snow plasticity
        double J = std::abs(particles_.F[i].determinant());
        if (enable_snow_) {
            JacobiSVD<Matrix3Xd> svd(particles_.F[i], ComputeFullU | ComputeFullV);
            Vector3d S = svd.singularValues();

            for (int j = 0; j < 3; ++j) {
                S(j) = std::clamp(S(j), 1.0-theta_compression_, 1.0+theta_stretch_);
            }

            double Jp = particles_.Jp(i);
            particles_.F[i] = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();
            double J_new = std::abs(particles_.F[i].determinant());
            J = Jp * J/J_new;
        } 
        particles_.Jp(i) = std::clamp(J, 0.1, 10.0);
    }
    //return true;
    return false;
}
