
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
    mesh_points_ = 10000;
    mesh_scale_ = 1.5;
    mesh_offset_ = 0.3;
    enable_mesh_ = false;

    is_3d_ = false;
    enable_heat_ = false;
    enable_air_ = false;
    use_global_lame_ = false;
    resolution_ = 32;
    dimensions_ = 3;
    point_size_ = 11;
    timestep_ = 0.1;
    gravity_ = -0.4;
    lambda_ = 10.0;
    mu_ = 20.0;
    render_color = 1;
    enable_snow_ = false;
    theta_compression_ = 2.5e-2; // values from snow paper
    theta_stretch_ = 7.5e-3;
    box_dx_ = 0.5;
    point_color_= ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    enable_addbox_ = false;
    enable_heatgun_ = false;
    render_particle_heat_ = false;
    render_grid_heat_ = true;
    alpha_ = 55.0;
    melting_point_ = 0.8;
    transition_heat_ = 5;
    initial_temperature_ = 0.;
}

void MpmHook::addParticleBox(Vector3d pos, Vector3i lengths, double dx) {
    int r0 = particles_.x.rows();

    lengths = (lengths.cast<double>().array() / dx).cast<int>();
    if (!is_3d_) {
        lengths(2) = 1;
    }

    // Create new material for box.
    int id = materials_.size();
    Material mat;
    mat.alpha = alpha_;
    mat.melting_point = melting_point_;
    mat.transition_heat = transition_heat_;
    materials_.emplace_back(mat);

    int new_rows = r0 + lengths.prod();
    particles_.resize(new_rows);

    for (int i = 0; i < lengths(0); ++i) {
        for (int j = 0; j < lengths(1); ++j) {
            for (int k = 0; k < lengths(2); ++k) {
                double x = pos(0) + double(i) * dx;
                double y = pos(1) + double(j) * dx;
                double z = pos(2) + double(k) * dx;
                
                if (!is_3d_) {
                    z = resolution_/2.0;
                }

                int idx;
                if (!is_3d_) {
                    idx = r0 + i*lengths(0) + j;   
                } else {
                    idx = r0 + i*lengths(0)*lengths(1) + j*lengths(1) + k;   
                }

                particles_.material_id(idx) = id;
                particles_.x.row(idx) << x,y,z;         
                particles_.v.row(idx) = Vector3d::Zero();
                particles_.C[idx].setZero();
                particles_.F[idx].setIdentity();
                particles_.Jp(idx) = 1.0;
                particles_.mass(idx) = 1.0;
                particles_.mu(idx) = mu_;
                particles_.T(idx) = initial_temperature_;
                particles_.melting_energy(idx) = 0;
                particles_.color.row(idx) << point_color_.x + double(i)/lengths(0)/5,
                                             point_color_.y + double(j)/lengths(1)/5,
                                             point_color_.z + double(k)/lengths(2)/5;
            }
        }
    }
}

void MpmHook::addParticleMesh() {
    // Reading in mesh.
    MatrixXd V;
    MatrixXi F;
    Util::readMesh(meshFile_, V, F);

    // Scaling mesh to fill bounding box.
    Eigen::Vector3d min = V.colwise().minCoeff();
    V = V.rowwise() - min.transpose();
    V = V / (V.maxCoeff() + 2./resolution_) / mesh_scale_;
    V = V.array() + mesh_offset_;

    // Generating point cloud from mesh.
    MatrixXd P = Util::meshToPoints(V, F, mesh_points_) * resolution_;

    // Create new material for box.
    int id = materials_.size();
    Material mat;
    mat.alpha = alpha_;
    mat.melting_point = melting_point_;
    mat.transition_heat = transition_heat_;
    materials_.emplace_back(mat);

    int r0 = particles_.x.rows();
    int new_rows = r0 + P.rows();
    particles_.resize(new_rows);

    for (int i = 0; i < P.rows(); ++i) {
        Vector3d p = P.row(i);
        
        if (!is_3d_) {
            p(2) = resolution_/2.0;
        }
   
        int idx = r0 + i;
        particles_.material_id(idx) = id;
        particles_.x.row(idx) = p;        
        particles_.v.row(idx) = Vector3d::Zero();
        particles_.C[idx].setZero();
        particles_.F[idx].setIdentity();
        particles_.Jp(idx) = 1.0;
        particles_.mass(idx) = 1.0;
        particles_.mu(idx) = mu_;
        particles_.T(idx) = initial_temperature_;
        particles_.melting_energy(idx) = 0;
        particles_.color.row(idx) << point_color_.x + double(i)/P.rows()/5,
                                     point_color_.y + double(i)/P.rows()/15,
                                     point_color_.z + double(i)/P.rows()/5;
    }
}

void MpmHook::initGrid(int size) {
    grid_.v.resize(size, dimensions_);
    grid_.mass.resize(size, 1);
    grid_.alpha.resize(size, 1);
    T_ = MatrixXd::Zero(size, 1); // Temperature
    f_ = MatrixXd::Zero(size, 1); // Source
}

void MpmHook::drawGUI(igl::opengl::glfw::imgui::ImGuiMenu &menu)
{
    if (ImGui::CollapsingHeader("Mesh", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Enable Mesh", &enable_mesh_);
        ImGui::InputInt("Mesh Points", &mesh_points_);
        ImGui::InputText("Filename", meshFile_);
        ImGui::InputDouble("Mesh Scale", &mesh_scale_);
        ImGui::InputDouble("Mesh Offset", &mesh_offset_);
    }
    if (ImGui::CollapsingHeader("Simulation Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Enable 3D", &is_3d_);
        ImGui::Checkbox("Enable Snow", &enable_snow_);
        ImGui::Checkbox("Enable placement", &enable_addbox_);
        ImGui::Checkbox("Use Global Lame params", &use_global_lame_);
        ImGui::InputInt("Grid Resolution", &resolution_);
        ImGui::InputDouble("Timestep", &timestep_);
        ImGui::InputDouble("Gravity", &gravity_);
        ImGui::InputDouble("Lambda", &lambda_);
        ImGui::InputDouble("Mu", &mu_);
        ImGui::InputDouble("Critical Compression", &theta_compression_);
        ImGui::InputDouble("Critical Stretch", &theta_stretch_);
    }
    if (ImGui::CollapsingHeader("Heat Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Enable Heat", &enable_heat_);
        ImGui::Checkbox("Enable heat gun (right click)", &enable_heatgun_);
        ImGui::Checkbox("Enable Dirichlet Air", &enable_air_);
        ImGui::InputDouble("Thermal Diffusivity", &alpha_);
        ImGui::InputDouble("Melting Point", &melting_point_);
        ImGui::InputDouble("Initial Temperature", &initial_temperature_);
        ImGui::InputInt("Transition Heat", &transition_heat_);
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
        ImGui::ColorEdit3("Point color", (float*)&point_color_);
        ImGui::InputDouble("Spacing", &box_dx_);
    }
}

bool MpmHook::mouseClicked(igl::opengl::glfw::Viewer &viewer, int button) {
    render_mutex.lock();

    MouseEvent me;
    me.button = button;
    int fid;
    Eigen::Vector3f bc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.core().viewport(3) - viewer.current_mouse_y;
    if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core().view,
      viewer.core().proj, viewer.core().viewport, grid_V, grid_F, fid, bc)) {
        me.type = MouseEvent::ME_CLICKED;
        me.vertex = grid_F(fid,0);
        me.pos = grid_V.row(me.vertex) * resolution_;
    } else {
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

    double dt = 1e-0 ;//* alpha_;

    for (int i = 0; i < T_.rows(); ++i) {
        int x = i / resolution_; // (resolution_*resolutoin_ for 3D)
        int y = i % resolution_;

        double r = dt * std::clamp(grid_.alpha(i), 1.0, 1000.0);
        if (!enable_air_) r = dt;
        //std::cout << "r: " << r << std::endl;
        if (!enable_air_ || (grid_.mass(i) > 1e-7)) {

            triplets.emplace_back(Triplet<double>(i, i, 1 + 4*r));

            if ((x+1) < resolution_)
                triplets.emplace_back(Triplet<double>(i, (x+1)*resolution_ + y, -r));
            if ((x-1) >= 0)
                triplets.emplace_back(Triplet<double>(i, (x-1)*resolution_ + y, -r));
            if ((y+1) < resolution_)
                triplets.emplace_back(Triplet<double>(i, x*resolution_ + (y+1), -r));
            if ((y-1) >= 0)
                triplets.emplace_back(Triplet<double>(i, x*resolution_ + (y-1), -r));

        } else {  triplets.emplace_back(Triplet<double>(i, i, dt)); }
    }
    L_.setFromTriplets(triplets.begin(), triplets.end());
}

void MpmHook::initSimulation()
{
    Eigen::initParallel();
    particles_.clear();
    materials_.clear();

    grid_size_ = resolution_*resolution_;
    if (is_3d_) {
        grid_size_ *= resolution_;
    }

    if (enable_mesh_) {
        addParticleMesh();
    } else {
        addParticleBox(Vector3d(resolution_/2.,resolution_/2.,16.), Vector3i(8,8,8), box_dx_);
    }

    initGrid(grid_size_); // create domain
    buildLaplacian();

    // Visualization grid.
    grid_V.resize(0,0);
    grid_F.resize(0,0);
    igl::triangulated_grid(resolution_,resolution_,grid_V,grid_F);
}
bool MpmHook::simulationStep() {
    int nparticles = particles_.x.rows();
    bool failure = false; // Used to exit simulation loop when we hit a NaN.
    grid_.v.setZero();
    grid_.mass.setZero();
    grid_.alpha.setZero();
    f_.setZero();

    if (enable_air_)
        T_.setZero(); //TODO aaa

    if (clickedVertex != -1) {
        if (enable_addbox_ && (button_ == 0)) {
            curPos(2) = resolution_/2.;
            addParticleBox(curPos, Vector3i(8,8,8), box_dx_);
            clickedVertex = -1;
        } else if (enable_heatgun_ && (button_ == 2)) {
            //TODO Only works in 2D
            Vector3i idx = curPos.cast<int>();
            int grid_i = idx(0) * resolution_ + idx(1);
            f_(grid_i) = 100.0;
        }
    }

    // Particles -> Grid
    #pragma omp parallel for
    for (int i = 0; i < nparticles; ++i) {

        if (failure) continue;

        const Vector3d& pos = particles_.x.row(i);
        const Vector3d& vel = particles_.v.row(i);
        const Matrix3d& F = particles_.F[i];
        const double Jp = particles_.Jp(i);
        const double mass = particles_.mass(i);

        double lambda = lambda_;
        double mu = particles_.mu(i);
        if (use_global_lame_) mu = mu_;

        int id = particles_.material_id(i);
        const double alpha = materials_[id].alpha;

        // Interpolation
        Vector3i cell_idx = pos.cast<int>();
        Vector3d diff = (pos - cell_idx.cast<double>()).array() - 0.5;
        Matrix3d weights;
        weights.row(0) = 0.5 * (0.5 - diff.array()).pow(2);
        weights.row(1) = 0.75 - diff.array().pow(2); 
        weights.row(2) = 0.5 * (0.5 + diff.array()).pow(2);

        // Neohoookean  MPM course eq 48
        Matrix3d F_T_inv = F.transpose().inverse();
        Matrix3d P = mu*(F - F_T_inv) + lambda*std::log(Jp)*F_T_inv;
        Matrix3d stress = (1.0 / Jp) * P * F.transpose();
        stress = -Jp * 4 * stress * timestep_; // eq 16 MLS-MPM

        if (isnan(stress(0,0))) {
            failure = true;
            std::cout << "FAILURE: " << cell_idx << std::endl;
            continue;
        }

        int z_max = is_3d_ ? 3 : 1;
        // for all surrounding 9 cells
        for (int x = 0; x < 3; ++x) {
            for (int y = 0; y < 3; ++y) {
                for (int z = 0; z < z_max; ++z) {
                    double weight = weights(x, 0) * weights(y, 1);

                    Vector3i idx = Vector3i(cell_idx(0) + x - 1, cell_idx(1) + y - 1, cell_idx(2) + z - 1);
                    Vector3d dist = (idx.cast<double>() - pos).array() + 0.5;

                    if (!is_3d_) {
                        idx(2) = resolution_/2.;
                        dist(2) = 0;
                    } else {
                        weight *= weights(z, 2);
                    }
                    
                    Vector3d Q = particles_.C[i] * dist;

                    // MPM course, equation 172
                    double weighted_mass = weight * mass;

                    // converting 2D index to 1D
                    int grid_i = get_index(idx);
                    
                    Vector3d vel_momentum = weighted_mass * (vel + Q);
                    Vector3d neohookean_momentum = (weight*stress) * dist;
                    vel_momentum += neohookean_momentum;

                    // scatter mass and momentum contributions to the grid
                    #pragma omp critical
                    {
                        grid_.mass(grid_i) += weighted_mass;
                        grid_.v.row(grid_i) += vel_momentum;
                        grid_.alpha(grid_i) += weighted_mass * alpha;

                        // TODO aaa
                        if (enable_air_)
                            f_(grid_i) += weighted_mass * particles_.T(i);
                    }

                }
            }
        }
    }

    if (failure) return true;

    // Velocity Updates.
    #pragma omp parallel for
    for (int i = 0; i < grid_size_; ++i) {

        if (grid_.mass(i) > 1e-7) {
            // Converting momentum to velocity and applying gravity force.
            grid_.v.row(i) /= grid_.mass(i);
            grid_.v.row(i) += timestep_ * Vector3d(0, gravity_, 0); 

            // Enforcing boundaries.
            Vector3i xyz = get_xyz(i);

            int dims = is_3d_ ? 3 : 2;
            for (int j = 0; j < dims; ++j) {
                if (xyz(j) < 2 || xyz(j) > resolution_ - 3) { 
                    grid_.v(i, j) = 0;
                }
            }
        } else {
            // cells that do no receive mass are considered air cells
            // so the temperature is zeroed out.
            // TODO aaa
            if (enable_air_)
                f_(i) = 0.0;

        }
    }

    // -------------------------------------------------------- //
    if (!enable_air_) {
        f_ = T_; // Set new source!
        f_(0.5*resolution_*(resolution_ + 1)) = 10.0;
    }

    if (enable_heat_) {
        // Solve temperature field update.
        buildLaplacian();
        //ConjugateGradient<SparseMatrix<double>, Lower|Upper> solver;
        SparseLU<SparseMatrix<double>> solver;
        SparseMatrix<double> I(T_.rows(), T_.rows());
        I.setIdentity();
        solver.compute(L_);
        T_ = solver.solve(f_);
        //std::cout << "solver e: " << solver.error() << " iter: " << solver.iterations() << std::endl;
    }

    if (enable_air_) {
        #pragma omp parallel for
        for (int i = 0; i < grid_size_; ++i) {
            if (grid_.mass(i) < 1e-7) {
                T_(i) = 0.;
            }
        }
    }
    // ------------------------------------------------------- //

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

        int z_max = is_3d_ ? 3 : 1;
        // for all surrounding 9 cells
        for (int x = 0; x < 3; ++x) {
            for (int y = 0; y < 3; ++y) {
                for (int z = 0; z < z_max; ++z) {

                    double weight = weights(x, 0) * weights(y, 1);
                    Vector3i idx = Vector3i(cell_idx(0) + x - 1, cell_idx(1) + y - 1, cell_idx(2) + z - 1);
                    Vector3d dist = (idx.cast<double>() - pos).array() + 0.5;

                    if (!is_3d_) {
                        idx(2) = resolution_/2.;
                        dist(2) = 0;
                    } else {
                        weight *= weights(z, 2);
                    }

                    // converting 2D index to 1D
                    int grid_i = get_index(idx);

                    Vector3d weighted_vel = grid_.v.row(grid_i) * weight;

                    // APIC paper equation 10, constructing inner term for B
                    particles_.C[i] += 4 * weighted_vel * dist.transpose();
                    particles_.v.row(i) += weighted_vel;
                    particles_.T(i) += weight * T_(grid_i);
                }
            }
        }
        
        // Melt
        int id = particles_.material_id(i);
        int transition_heat = materials_[id].transition_heat;

        if (particles_.T(i) > materials_[id].melting_point) {
            particles_.melting_energy(i) += 1;

            if (particles_.melting_energy(i) > transition_heat)
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
    return false;
}
