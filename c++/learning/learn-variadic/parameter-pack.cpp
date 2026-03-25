#include <iostream>

/*
 * Template and Function parameter packs
 * https://en.cppreference.com/w/cpp/language/pack
 * https://chatgpt.com/c/67637b93-6a8c-8007-964c-900aa685ee4a
 */

/*
 * VARIADIC CLASS TEMPLATES
 *
 * Can be instantiated with any number of template arguments
 */
template <class... Types>
struct Tuple {};
 
Tuple<> t0;           // Types contains no arguments
Tuple<int> t1;        // Types contains one argument: int
Tuple<int, float> t2; // Types contains two arguments: int and float
//Tuple<0> t3;          // error: 0 is not a type

/*
 * VARIADIC FUNCTION TEMPLATES
 *
 * Can be called with any number of function arguments (template arguments deduced through template argument deduction).
 */
template <class... Types>
void f(Types... args) {}

void example_a (void) { 
  f();       // OK: args contains no arguments
  f(1);      // OK: args contains one argument: int
  f(2, 1.0); // OK: args contains two arguments: int and double
}

// For classes, the parameter pack must be the final parameter.
template <typename U, typename... Ts>    // OK: can deduce U
struct Valid_Struct;
// template <typename... Ts, typename U> // Error: Ts... not at the end
// struct Invalid_Struct;

// For functions, the paramter pack may appear earlier, as long as the following parameters can be deduced from the function arguments, or have a defualt value.
template <typename... Ts, typename U, typename=void>
void valid_func (U, Ts...);    // OK: can deduce U
// void invalid_func (Ts..., U); // Can't be used: Ts... is a non-deduced context in this position



/*
 * PACK EXPANSION
 */

// A pattern followed by an ellipsis, in which the name of at least one pack appears at least once, is expanded into zero or more instantiations of the pattern.
// The name of the pack is replaced by each of the elements from the pack, in order.

// From example_b, p_args is an int*, double*, const char**
template <typename... Ps>
void f (Ps... p_args);
 
template <typename... Ts>
void g (Ts... args) {
  /*
   * “&args...” is a pack expansion
   * “&args” is its pattern
   * operator& is applied to each element of the pack.
   * args... expands to int E1, doulbe E2, const char* E3, and &args... expands to &E1, &E2, &E3.
   */
  f(&args...);
}

// There can also be operations done on the pattern, and the ellipsis should always go at the end.
template <typename... Ts>
void h (Ts... args) {
  // Assuming h is called with a, b, and c: h(a, b, c), the following expands to g((a + a), (b + b), (c + c));
  g((args + args) ...);
}

void example_b (void) {
  h(1, 0.2, std::string{"a"});
}

// If two pack names appear in the same expansion, they are expanded at the same time and thus must have the same length.
template <typename T1, typename T2>
struct Pair {};

template <typename... Args>
void func1 (const Args*... args) {
  (++args, ...);
  (args++, ...);
  (..., ++args);
  (..., args++);
  (++*args, ...);
  (*args++, ...);
  (..., ++*args);
  (..., *args++);
}

template <typename... Args>
void func2 (Args* const... args);

int main (int argc, char *argv[]) {
  return 0;
}
