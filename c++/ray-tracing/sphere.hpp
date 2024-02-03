#include <cmath>

#ifndef __SPHERE_HPP_
#define __SPHERE_HPP_

#include "shape.hpp"

class Sphere : public Shape {
private: 
  Vector3 center;
  double radius;
public:
  Sphere () : center{}, radius{1} {}
  
  Sphere (Vector3 center, double radius) : center{center}, radius{radius} {}

  Sphere (Material material, Vector3 center, double radius) : Shape{material}, center{center}, radius{radius} {}

  Hit_Info hit (const Ray &ray, const bool debug) const;
};

Hit_Info Sphere::hit (const Ray &ray, const bool debug) const {
  Hit_Info hit_info {false};

  // To make the calculations easier, do all calculations relative to the center of the sphere.
  Vector3 offset_ray_origin = ray.origin() - center;
  
  double a {dot(ray.direction(), ray.direction())};
  double b {2 * dot(ray.direction(), offset_ray_origin)};
  double c {dot(offset_ray_origin, offset_ray_origin) - radius * radius};

  double discriminant {b * b - 4 * a * c};
  
  if (debug) {
    std::cout << "\noffset_ray_origin = " << offset_ray_origin << "\n";
    std::cout << "a = " << a << "\nb = " << b << "\nc = " << c << "\n";
    std::cout << "b^2 - 4ac = " << b * b - 4 * a * c << "\n";
    std::cout << "discriminant = " << discriminant << "\n";
  }

  // In the quadratic formula, the discriminant is in a square root, so if the discriminant is negative there are no real solutions.
  if (discriminant < 0) return hit_info;

  // I want to get the minimum distance, because the further distance would be behind the closer distance.
  // Subtracting sqrt(discriminant) will minimize the distance.
  double distance {(-b - std::sqrt(discriminant))/(2 * a)};
  // If the distance is negative, that means that the ray starts inside of another sphere, or that the sphere is entirely behind the ray.
  // In both cases the sphere will not be visisble.
  if (distance < 0) return hit_info;

  hit_info.hit = true;
  hit_info.distance = distance;
  hit_info.hit_point = ray.point_at_distance(distance);
  // The normal vector of a point on the sphere is just the point in space relative to the sphere normalized.
  hit_info.normal = normalize(hit_info.hit_point - center);
  hit_info.material = material;
  
  return hit_info;
}

#endif // __SPHERE_HPP_ not defined
