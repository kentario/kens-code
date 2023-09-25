#include <iostream>
#include <cmath>

class Vector3 {
public:
  double v[3];

  // Constructors with 0 and 3 arguments.
  Vector3 () : v{0, 0, 0} {}
  Vector3 (double a, double b, double c) : v{a, b , c} {}

  inline double x () const {return v[0];}
  inline double y () const {return v[1];}
  inline double z () const {return v[2];}
  inline double r () const {return v[0];}
  inline double g () const {return v[1];}
  inline double b () const {return v[2];}

  // +Vector3
  inline Vector3 operator+ () {return *this;}
  // -Vector3
  inline Vector3 operator- () {return Vector3(-v[0], -v[1], -v[2]);}
  
  inline double operator[] (int i) const {return v[i];}
  inline double& operator[] (int i) {return v[i];}
  
  inline Vector3& operator+= (const Vector3 &other);
  inline Vector3& operator-= (const Vector3 &other);
  inline Vector3& operator*= (const Vector3 &other);
  inline Vector3& operator/= (const Vector3 &other);
  
  inline Vector3& operator*= (const double &value);
  inline Vector3& operator/= (const double &value);

  inline double distance_squared () {return v[0] * v[0], v[1] * v[1], v[2] * v[2];}
  inline double distance () {return sqrt(distance_squared());}
};

// (Vector3 + Vector3) -> Vector3
inline Vector3 operator+ (const Vector3 &a, const Vector3 &b) {
  return Vector3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

// (Vector3 - Vector3) -> Vector3
inline Vector3 operator- (const Vector3 &a, const Vector3 &b) {
  return Vector3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

// (Vector3 * Vector3) -> Vector3
inline Vector3 operator* (const Vector3 &a, const Vector3 &b) {
  return Vector3(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}

// (Vector3 * value) -> Vector3
inline Vector3 operator* (const Vector3 &a, const double &value) {
  return Vector3(a[0] * value, a[1] * value, a[2] * value);
}

// (value * Vector3) -> Vector3
inline Vector3 operator* (const double &value, const Vector3 &a) {
  return Vector3(value * a[0], value * a[1], value * a[2]);
}

// (Vector3/Vector3) -> Vector3
inline Vector3 operator/ (const Vector3 &a, const Vector3 &b) {
  return Vector3(a[0]/b[0], a[1]/b[1], a[2]/b[2]);
}

// (Vector3/value) -> Vector3
inline Vector3 operator/ (const Vector3 &a, const double &value) {
  return Vector3(a[0]/value, a[1]/value, a[2]/value);
}

// (value/Vector3) -> Vector3
inline Vector3 operator/ (const double &value, const Vector3 &a) {
  return Vector3(value/a[0], value/a[1], value/a[2]);
}

inline Vector3& Vector3::operator+= (const Vector3 &other) {
  v[0] += other[0];
  v[1] += other[1];
  v[2] += other[2];
  return *this;
}

inline Vector3& Vector3::operator-= (const Vector3 &other) {
  v[0] -= other[0];
  v[1] -= other[1];
  v[2] -= other[2];
  return *this;
}

inline Vector3& Vector3::operator*= (const Vector3 &other) {
  v[0] *= other[0];
  v[1] *= other[1];
  v[2] *= other[2];
  return *this;
}

inline Vector3& Vector3::operator/= (const Vector3 &other) {
  v[0] /= other[0];
  v[1] /= other[1];
  v[2] /= other[2];
  return *this;
}

inline Vector3& Vector3::operator*= (const double &value) {
  for (int i = 0; i < 3; i++) {
    v[i] *= value;
  }
  return *this;
}

inline Vector3& Vector3::operator/= (const double &value) {
  for (int i = 0; i < 3; i++) {
    v[i] /= value;
  }
  return *this;
}

inline double dot (const Vector3 &a, const Vector3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
