#include <iostream>
#include <fstream>
#include <vector>

#include "camera.hpp"
#include "sphere.hpp"
#include "plane.hpp"
#include "disc.hpp"

// Takes a 2d array of pixel values, and converts it to an image of type .ppm.
void twod_vector_to_ppm (std::vector<std::vector<Vector3>> &colors, std::string &file_name) {
  int width {static_cast<int>(colors.size())};
  int height {static_cast<int>(colors[0].size())};
  
  std::ofstream my_file;
  my_file.open(file_name);
  
  my_file << "P3\n" << width << " " << height << "\n255\n";

  // The top left of the image is [0][0] in the array, and the top right is [width - 1][0].
  for (int y {0}; y < height; y++) {
    for (int x {0}; x < width; x++) {
      my_file << (int) (colors[x][y].r() * 255) << " " << (int) (colors[x][y].g() * 255) << " " << (int) (colors[x][y].b() * 255) << "\n";
    }
  }

  my_file.close();
}

// Takes multiple 2d arrays of pixel values, and averages them together to make a single image of type .ppm.
void threed_vector_to_ppm (std::vector<std::vector<std::vector<Vector3>>> &pictures, std::string &file_name) {
  // Just get the width and height of the first picture.
  int width {static_cast<int>(pictures[0].size())};
  int height {static_cast<int>(pictures[0][0].size())};

  std::ofstream my_file;
  my_file.open(file_name);

  my_file << "P3\n" << width << " " << height << "\n255\n";
  
  Vector3 color;
  
  for (int y {0}; y < width; y++) {
    for (int x {0}; x < height; x++) {
      // Reset color.
      color = {0, 0, 0};
      // Take the average of all the pictures for this pixel.
      for (const auto &picture : pictures) {
	color += picture[x][y];
      }
      color /= pictures.size();

      // Output the color.
      my_file << (int) (color.r() * 255) << " " << (int) (color.g() * 255) << " " << (int) (color.b() * 255) << "\n";
    }
  }

  my_file.close();
}

int main () {
  std::cout << "Creating spheres...\n";
  std::vector<Sphere> spheres;

  Vector3 sphere_color {0.122, 0.42, 0.863};
  Material sphere_material {0, sphere_color};
  Vector3 sphere_center {100, 0, 0};
  double sphere_radius {50};
  spheres.push_back(Sphere{sphere_material, sphere_center, sphere_radius});
  sphere_color = {0, 0.686, 0};
  sphere_material.color = sphere_color;
  sphere_center = {75, -35, -20};
  sphere_radius = 20;
  spheres.push_back(Sphere{sphere_material, sphere_center, sphere_radius});

  std::cout << "Creating planes...\n";
  std::vector<Plane> planes;

  Vector3 plane_color {1, 0.039, 0.392};
  Material plane_material {0, plane_color};
  Vector3 plane_origin {0, -40, 0};
  Vector3 plane_normal {0, 1, 0};
  planes.push_back(Plane{plane_material, plane_origin, plane_normal});
  
  // Add a light source far away.
  Vector3 light_shape_color {0, 0, 0};
  double light_shape_reflectivity {0};
  Vector3 light_emit_color {1, 1, 1};
  double light_strength {1};
  Material light_material {light_shape_reflectivity, light_shape_color, light_emit_color, light_strength};
  Vector3 light_center {500, 0, 0};
  Vector3 light_normal {-1, 0, 0};
  planes.push_back(Plane{light_material, light_center, light_normal});
  
  std::cout << "Creating discs...\n";
  std::vector<Disc> discs;

  Vector3 disc_color {0.914, 0.553, 0.082};
  Material disc_material {0, disc_color};;
  Vector3 disc_origin {70, 0, 40};
  Vector3 disc_normal {-1, 0, 1};
  double disc_radius {50};
  discs.push_back(Disc{disc_material, disc_origin, disc_normal, disc_radius});
  
  std::cout << "Combining shapes...\n";
  std::vector<Shape*> shapes;
  
  for (auto &sphere : spheres) {
    shapes.push_back(&sphere);
  }

  for (auto &plane : planes) {
    shapes.push_back(&plane);
  }

  for (auto &disc : discs) {
    shapes.push_back(&disc);
  }
  // 6 5
  std::cout << "Creating camera...\n";
  Vector3 origin {0, 0, 0};
  Vector3 look_direction {1, 0, 0};
  double horizontal_fov {110};
  int horizontal_resolution {800};
  int vertical_resolution {800};
  int num_bounces {5};
  Camera my_camera {origin, look_direction, horizontal_fov, horizontal_resolution, vertical_resolution, num_bounces, shapes};

  std::cout << "Taking pictures...\n";

  int num_samples {5};
  std::vector<std::vector<std::vector<Vector3>>> pictures(num_samples);
  for (int i {0}; i < num_samples; i++) {
    pictures[i] = my_camera.take_picture();
  }
    
  std::string file_name {"num_samples=" + std::to_string(num_samples) + ".ppm"};
  
  std::cout << "Writing to file " + file_name + "...\n";
  threed_vector_to_ppm(pictures, file_name);
  
  return 0;
}
