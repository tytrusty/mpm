#ifndef MPMHOOK_H
#define MPMHOOK_H
#include "PhysicsHook.h"
#include <iostream>
#include <Eigen/Core>

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
	Eigen::MatrixXd mass;
    Eigen::MatrixXd Jp;
	std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> C;
	std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> F;
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
    }

    virtual bool simulationStep();
    virtual void renderGeometry(igl::opengl::glfw::Viewer &viewer) {
        viewer.data().clear();
        viewer.data().point_size = point_size_;

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

	    viewer.data().set_points(renderP, Eigen::RowVector3d(0.564706,0.847059,0.768627));
        //viewer.data().set_mesh(V,F);
    }

    private:

    void initParticles(int width, int height);
    void initGrid(int width, int height);
    //void generateCloud();

    Eigen::MatrixXd renderP;
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    Particles particles_;
    Grid grid_;

    int resolution_ = 64;
    int dimensions_ = 3;
    int point_size_ = 5;
    double timestep_ = 0.05;
    double gravity_ = -0.4;
    double elastic_lambda_ = 10.0;
    double elastic_mu_ = 20.0;

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
};

#endif // MPMHOOK_H
