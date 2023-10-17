#ifndef __OBJECT_HPP_
#define __OBJECT_HPP_

#include "ray.hpp"

struct Hit_Info {
  bool hit;
  double distance;
  Vector3 hit_point;
  Vector3 normal;
  Vector3 color;

  // Returns true if the distance of this hit info is lower than the distance of the other hit info.
  bool is_closer_than (const Hit_Info &hit_info) const {
    return distance < hit_info.distance;
  }
};

inline std::ostream& operator<< (std::ostream &os, const Hit_Info &hit_info) {
  os << "Hit: " << (hit_info.hit ? "true" : "false") << " Distance: " << hit_info.distance << " Hit Point: " << hit_info.hit_point << " Normal: " << hit_info.normal << " Color: " << hit_info.color;
  return os;
}

class Object {
protected:
  Vector3 color;
public:
  Object () {}

  Object (Vector3 color) : color{color} {}
  
  virtual Hit_Info hit (const Ray &ray, const bool debug) const = 0;
};

#endif // __OBJECT_HPP_ not defined
