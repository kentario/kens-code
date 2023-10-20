#include <cmath>

#ifndef __DISC_HPP_
#define __DISC_HPP_

#include "shape.hpp"

// The Disc class is almost identical with plane, except that it has a radius.
class Disc : public Shape {
private:
  Vector3 origin;
  Vector3 normal;

  double radius;
public:
  Disc () : origin{}, normal{1, 1, 1} {}

  Disc (Vector3 origin, Vector3 normal, double radius) : origin{origin}, normal{normal}, radius{radius} {}

  Disc (Material material, Vector3 origin, Vector3 normal, double radius) : Shape{material}, origin{origin}, normal{normal}, radius{radius} {}

  Hit_Info hit (const Ray &ray, const bool debug) const;
};

Hit_Info Disc::hit (const Ray &ray, const bool debug) const {
  Hit_Info hit_info {false};
  
  // Use the same calculations as the plane to check if the ray intersects the plane of the disc.
  double denominator {dot(ray.direction(), normal)};
  if (denominator == 0) return hit_info;
  double numerator {dot((origin - ray.origin()), normal)};
  double distance {numerator/denominator};
  if (distance < 0) return hit_info;

  // Check if the hit point is within the radius of the disc.
  hit_info.hit_point = ray.point_at_distance(distance);
  if (hit_info.hit_point.distance(origin) > radius) return hit_info;

  hit_info.hit = true;
  hit_info.distance = distance;
  hit_info.normal = normal;
  hit_info.material = material;
  
  return hit_info;
}

#endif // __DISC_HPP_ not defined
