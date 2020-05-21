#ifndef MPMHOOK_H
#define MPMHOOK_H
#include "PhysicsHook.h"
#include <iostream>
#include <Eigen/Core>
#include <string>
#include "igl/triangulated_grid.h"

const Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);

typedef std::tuple<int, int> Edge;

struct MouseEvent
{
    enum METype {
        ME_CLICKED,
        ME_RELEASED,
        ME_DRAGGED
    };

    METype type;
    int vertex;
    int button;
    Eigen::Vector3d pos;
};

struct Material {
    double solid_mu;
    double lambda_mu;
    double alpha;
    double melting_point;
    int transition_heat; // simple counter
    //double specific_heat;
};

struct Particles {
    Eigen::MatrixXd x; // positions
    Eigen::MatrixXd v; // velocities
	Eigen::MatrixXd color;
	Eigen::MatrixXd mass;
    Eigen::MatrixXd Jp;
	Eigen::MatrixXd mu;
    Eigen::MatrixXd T;
    Eigen::MatrixXi melting_energy;
    Eigen::MatrixXd material_id;
	std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> C;
	std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> F;

    void resize(int new_rows) {
        x.conservativeResize(new_rows, 3);
        v.conservativeResize(new_rows, 3);
        color.conservativeResize(new_rows, 3);
        Jp.conservativeResize(new_rows, 1);
        mass.conservativeResize(new_rows, 1);
        mu.conservativeResize(new_rows, 1);
        T.conservativeResize(new_rows, 1);
        material_id.conservativeResize(new_rows, 1);
        melting_energy.conservativeResize(new_rows,1);
        C.resize(new_rows);
        F.resize(new_rows);
    }

    void clear() {
        x.resize(0,0);
        v.resize(0,0);
        color.resize(0,0);
        mass.resize(0,0);
        Jp.resize(0,0);
        mu.resize(0,0);
        T.resize(0,0);
        material_id.resize(0,0);
        melting_energy.resize(0,0);
        C.clear();
        F.clear();
    }
};

struct Grid {
    Eigen::MatrixXd v;    // velocities
	Eigen::MatrixXd mass;  // masses
	Eigen::MatrixXd alpha;

};

class MpmHook : public PhysicsHook
{
public:
    MpmHook();

    virtual void drawGUI(igl::opengl::glfw::imgui::ImGuiMenu &menu);
    virtual void tick();
    virtual void initSimulation();

    virtual void updateGeometry() {
        renderP = particles_.x / resolution_;
        if (render_particle_heat_) {
            igl::colormap(igl::ColorMapType(render_color), particles_.T, false, renderC);
        } else {
            renderC = particles_.color;;
        }
    }

    virtual bool simulationStep();
    virtual void renderGeometry(igl::opengl::glfw::Viewer &viewer) {
        viewer.data().clear();
        viewer.data().point_size = point_size_;
        viewer.data().show_lines = 0;

        Eigen::Vector3d m(0.,0.,0.);
        Eigen::Vector3d M(1.,1.,1.);

        // Corners of the bounding box
        Eigen::MatrixXd V_box(8,3);
        V_box <<
        m(0), m(1), m(2),
        M(0), m(1), m(2),
        M(0), M(1), m(2),
        m(0), M(1), m(2),
        m(0), m(1), M(2),
        M(0), m(1), M(2),
        M(0), M(1), M(2),
        m(0), M(1), M(2);

        // Edges of the bounding box
        Eigen::MatrixXi E_box(12,2);
        E_box << 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6,
                 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 7 ,3;

        // Plot the edges of the bounding box
        for (unsigned i=0;i<E_box.rows(); ++i)
          viewer.data().add_edges(
            V_box.row(E_box(i,0)),
            V_box.row(E_box(i,1)),
            Eigen::RowVector3d(1,0,0)
          );

	    viewer.data().set_points(renderP, renderC);

        if (render_grid_heat_) {
            if (is_3d_) {
                Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(resolution_*resolution_,1);
                igl::colormap(igl::ColorMapType(render_color), zero, false, grid_C);
            } else {
                igl::colormap(igl::ColorMapType(render_color), T_, false, grid_C);
            }
            viewer.data().set_mesh(grid_V,grid_F);
            viewer.data().set_colors(grid_C);
        }
    }

    virtual bool mouseClicked(igl::opengl::glfw::Viewer &viewer, int button);
    virtual bool mouseReleased(igl::opengl::glfw::Viewer &viewer,  int button);

    private:

    int get_index(const Eigen::Vector3i& idx) {
        if (is_3d_) {
            return idx(0)*resolution_*resolution_ + idx(1)*resolution_ + idx(2);
        } else {
            return idx(0)*resolution_ + idx(1);
        }
    }

    Eigen::Vector3i get_xyz(int i) {
        int x,y,z;
        if (is_3d_) {
            x = i / (resolution_*resolution_);
            y = (i / resolution_) % resolution_;
            z = i % resolution_;
        } else {
            x = i / resolution_;
            y = i % resolution_;
            z = resolution_/2.;
        }
        return Eigen::Vector3i(x,y,z);
    }

    void addParticleBox(Eigen::Vector3d pos, Eigen::Vector3i lengths, double dx);
    void addParticleMesh();
    void initGrid(int size);
    void buildLaplacian();

    Eigen::MatrixXd renderP;
    Eigen::MatrixXd renderC;

    Eigen::MatrixXd grid_V;
    Eigen::MatrixXi grid_F;
    Eigen::MatrixXd grid_C;

    Eigen::SparseMatrix<double> L_;
    Eigen::MatrixXd T_;
    Eigen::MatrixXd f_;
    Particles particles_;
    Grid grid_;

    std::vector<Material> materials_;

    int resolution_;
    int dimensions_;
    int point_size_;
    int grid_size_;
    bool is_3d_;
    double timestep_;
    double gravity_;
    double lambda_;
    double mu_;
    double alpha_; // thermal diffusivity
    double melting_point_;
    int transition_heat_; // simple counter
    double initial_temperature_;
    bool use_global_lame_; // use global lame value for particles

    std::string meshFile_;
    int mesh_points_;
    bool enable_mesh_;
    double mesh_scale_;
    double mesh_offset_;

    std::mutex mouseMutex;
    std::vector<MouseEvent> mouseEvents;
    Eigen::Vector3d curPos; // the current position of the mouse cursor in 3D
    int clickedVertex; // the currently selected vertex (-1 if no vertex)
    int button_;

    bool enable_snow_;
    double theta_compression_; // critical compression
    double theta_stretch_;    // critical stretch

    int render_color;
    bool render_particle_heat_ = false;
    bool render_grid_heat_ = false;
    
    ImVec4 point_color_;
    double box_dx_;
    bool enable_addbox_;
    bool enable_heatgun_;
    bool enable_heat_;
    bool enable_air_;

};

#endif // MPMHOOK_H
