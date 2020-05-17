#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <Eigen/StdVector>
#include <thread>
#include <iostream>
#include <algorithm>
#include "PhysicsHook.h"
#include "MpmHook.h"

using namespace Eigen;

static PhysicsHook *hook = NULL;

void toggleSimulation()
{
    if (!hook)
        return;

    if (hook->isPaused())
        hook->run();
    else
        hook->pause();
}

void resetSimulation()
{
    if (!hook)
        return;

    hook->reset();
}

bool drawCallback(igl::opengl::glfw::Viewer &viewer)
{
    if (!hook)
        return false;

    hook->render(viewer);
    return false;
}

bool keyCallback(igl::opengl::glfw::Viewer &viewer, unsigned int key, int modifiers)
{
    if (key == ' ') {
        toggleSimulation();
        return true;
    } else if (key == 'r') {
        resetSimulation();
        return true;
    }
    return false;
}

bool mouseDownCallback(igl::opengl::glfw::Viewer &viewer, int button, int modifier) {
    if (!hook)
        return false;
    return hook->mouseClicked(viewer, button);
}

bool mouseUpCallback(igl::opengl::glfw::Viewer &viewer, int button, int modifier) {
    if (!hook)
        return false;
    return hook->mouseReleased(viewer, button);
}

bool mouseMoveCallback(igl::opengl::glfw::Viewer &viewer, int button, int modifier) {
    if (!hook)
        return false;
    return hook->mouseMoved(viewer, button);
}

bool drawGUI(igl::opengl::glfw::imgui::ImGuiMenu &menu) {
    if (ImGui::CollapsingHeader("Simulation Control", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Run/Pause Sim", ImVec2(-1, 0))) {
            toggleSimulation();
        }
        if (ImGui::Button("Reset Sim", ImVec2(-1, 0))) {
            resetSimulation();
        }
    }
    hook->drawGUI(menu);
    return false;
}

int main(int argc, char *argv[]) {
	igl::opengl::glfw::Viewer viewer;

    hook = new MpmHook();
    hook->reset();
    viewer.core().is_animating = true;
    viewer.core().animation_max_fps = 60;

    // Center camera
    Matrix3d bnd;
    bnd << 0.0, 0.0, 0.0,
           0.5, 0.5, 0.0,
           1.0, 1.0, 0.0;
    viewer.core().align_camera_center(bnd);

    // Render points
    viewer.callback_mouse_down = mouseDownCallback;
    viewer.callback_mouse_up = mouseUpCallback;
    viewer.callback_mouse_move = mouseMoveCallback;
    viewer.callback_key_pressed = keyCallback;
    viewer.callback_pre_draw = drawCallback;
    
    // Add UI
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    menu.callback_draw_viewer_menu = [&]() {drawGUI(menu); };

	viewer.launch();
}
