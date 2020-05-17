#ifndef UTIL_H
#define UTIL_H
#include <iostream>
#include <Eigen/Core>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <string>

#include <Eigen/Geometry>
#include <igl/fast_winding_number.h>
#include <igl/random_points_on_mesh.h>
#include <igl/slice_mask.h>

namespace Util {

void readMesh(std::string mesh_file, Eigen::MatrixXd& V, Eigen::MatrixXi& F) {

	std::string meshfname = std::string("../meshes/") + mesh_file;
    std::string ext = mesh_file.substr(mesh_file.find_last_of(".")+1);
    if (ext == "obj") {
        std::cout << "Reading OBJ file" << std::endl;
        if(!igl::readOBJ(meshfname, V, F)) {
            std::cerr << "Couldn't read mesh file" << std::endl;
            exit(-1);
        }
    } else if (ext == "ply") {
        std::cout << "Reading PLY file" << std::endl;
        if(!igl::readPLY(meshfname, V, F)) {
            std::cerr << "Couldn't read mesh file" << std::endl;
            exit(-1);
        }
        std::cout << "Done reading PLY" << std::endl;

    } else if (ext == "off") {
        std::cout << "Reading OFF file" << std::endl;
        if(!igl::readOFF(meshfname, V, F)) {
            std::cerr << "Couldn't read mesh file" << std::endl;
            exit(-1);
        }
        std::cout << "Done reading OFF" << std::endl;
    }
}

Eigen::MatrixXd meshToPoints(Eigen::MatrixXd V, Eigen::MatrixXi F,
        int samples = 10000) {

    const auto time = [](std::function<void(void)> func)->double {
        const double t_before = igl::get_seconds();
        func();
        const double t_after = igl::get_seconds();
        return t_after-t_before;
    };

    // Sample points on mesh and generate normals
    Eigen::MatrixXd P,N;
    {
        Eigen::VectorXi I;
        Eigen::SparseMatrix<double> B;
        igl::random_points_on_mesh(10000,V,F,B,I);
        P = B*V;
        Eigen::MatrixXd FN;
        igl::per_face_normals(V,F,FN);
        N.resize(P.rows(),3);
        for(int p = 0;p<I.rows();p++) {
            N.row(p) = FN.row(I(p));
        }
    }

    // Generate a list of random query points in the bounding box
    Eigen::MatrixXd Q = Eigen::MatrixXd::Random(samples,3);
    const Eigen::RowVector3d Vmin = V.colwise().minCoeff();
    const Eigen::RowVector3d Vmax = V.colwise().maxCoeff();
    const Eigen::RowVector3d Vdiag = Vmax-Vmin;
    for(int q = 0;q<Q.rows();q++) {
      Q.row(q) = (Q.row(q).array()*0.5+0.5)*Vdiag.array() + Vmin.array();
    }


    // Positions of points inside of triangle soup (V,F)
    Eigen::MatrixXd QiV;
    {
      igl::FastWindingNumberBVH fwn_bvh;
      printf("triangle soup precomputation    (% 8ld triangles): %g secs\n",
        F.rows(),
        time([&](){igl::fast_winding_number(V.cast<float>().eval(),F,2,fwn_bvh);}));
      Eigen::VectorXf WiV;
      printf("      triangle soup evaluation  (% 8ld queries):   %g secs\n",
        Q.rows(),
        time([&](){igl::fast_winding_number(fwn_bvh,2,Q.cast<float>().eval(),WiV);}));
      igl::slice_mask(Q,WiV.array()>0.5,1,QiV);
    }
    return QiV;
}


}

#endif // UTIL_H
