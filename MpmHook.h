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
    Eigen::Vector3d pos;
};

struct Particles {
    Eigen::MatrixXd x; // positions
    Eigen::MatrixXd v; // velocities
	Eigen::MatrixXd color;
	Eigen::MatrixXd mass;
    Eigen::MatrixXd Jp;
	std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> C;
	std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> F;

    void clear() {
        x.resize(0,0);
        v.resize(0,0);
        mass.resize(0,0);
        Jp.resize(0,0);
        C.clear();
        F.clear();
    }

};

struct Grid {
    Eigen::MatrixXd v;    // velocities
	Eigen::MatrixXd mass; // masses
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
        renderC = particles_.color;
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

	    viewer.data().set_points(renderP, renderC);//Eigen::RowVector3d(0.564706,0.147059,0.768627));

        // Don't do this everytime
        igl::triangulated_grid(resolution_,resolution_,grid_V,grid_F);
        igl::colormap(igl::ColorMapType(render_color), T_, false, grid_C);
        viewer.data().set_mesh(grid_V,grid_F);
        viewer.data().set_colors(grid_C);
    }

    virtual bool mouseClicked(igl::opengl::glfw::Viewer &viewer, int button);
    virtual bool mouseReleased(igl::opengl::glfw::Viewer &viewer,  int button);

    private:

    void initParticles(int width, int height);
    void addParticleBox(Eigen::Vector3d pos, Eigen::Vector3i lengths, double dx);
    void initGrid(int width, int height);
    void buildLaplacian();
    //void generateCloud();

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

    int resolution_;
    int dimensions_;
    int point_size_;
    bool is_3d_;
    double timestep_;
    double gravity_;
    double lambda_;
    double mu_;

    double particle_mass = 1.0;
    double vol = 1.0;        // Particle Volume
    double hardening = 10.0; // Snow hardening factor
    double E = 1e4;          // Young's Modulus
    double nu = 0.2;         // Poisson ratio
    const bool plastic = true;

    std::string meshFile_;
    std::mutex mouseMutex;
    std::vector<MouseEvent> mouseEvents;
    int clickedVertex; // the currently selected vertex (-1 if no vertex)
    int prevClicked;
    double clickedz;
    Eigen::Vector3d curPos; // the current position of the mouse cursor in 3D
    bool mouseDown;

    int solverIters;
    float solverTol;
    bool enable_heat;

    bool enable_iso;
    int render_color;
    
    ImVec4 point_color_;
    double box_dx_;
    bool enable_addbox_;

};

#endif // MPMHOOK_H
