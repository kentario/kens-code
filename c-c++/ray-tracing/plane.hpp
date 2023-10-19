#include <cmath>

#ifndef __PLANE_HPP_
#define __PLANE_HPP_

#include "shape.hpp"

class Plane : public Shape {
private:
  // Because the plane is infininte, it won't actually have a origin, but this is usefull just for moving it around.
  Vector3 origin;
  Vector3 normal;
public:
  Plane () : origin{}, normal{1, 1, 1} {}

  Plane (Vector3 origin, Vector3 normal) : origin{origin}, normal{normal} {}

  Plane (Material material, Vector3 origin, Vector3 normal) : Shape{material}, origin{origin}, normal{normal} {}

  Hit_Info hit (const Ray &ray, const bool debug) const;
};

Hit_Info Plane::hit (const Ray &ray, const bool debug) const {
  Hit_Info hit_info {false};

  // A point on a plane can be defined as a point some distance from the origin perpendicular to the normal vector.
  // Start with a point some distance from the origin: (point - origin)
  // Then check if it is perpendicular to the normal vector, using dot product: (point - origin) dot normal = 0
  // Some point along the ray can be calculated as (distance * direction + origin).
  // Plug that into the formula for the plane: (distance * direction + ray_origin - plane_origin) dot normal = 0
  // Solve for distance: distance = ((plane_origin - ray_origin) dot normal)/(direction dot normal)
  double denominator {dot(ray.direction(), normal)};

  if (debug) {
    std::cout << "\ndenominator = " << denominator << "\n";
  }

  // If the denominator is 0, then there is no solution because dividing by 0 is not allowed.
  if (denominator == 0) return hit_info;

  double numerator {dot((origin - ray.origin()), normal)};
  double distance {numerator/denominator};

  // If the distance is less than 0, then it is behind the ray.
  if (distance < 0) return hit_info;
  
  if (debug) {
    std::cout << "numerator = " << numerator << "\n";
    std::cout << "distance = " << distance << "\n";
  }
  
  hit_info.hit = true;
  hit_info.distance = distance;
  hit_info.hit_point = ray.point_at_distance(distance);
  hit_info.normal = normal;
  hit_info.material = material;
  
  return hit_info;
}

#endif // __PLANE_HPP_ not defined
