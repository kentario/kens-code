#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <string.h>

/*
  https://en.wikipedia.org/wiki/Chudnovsky_algorithm
  https://www.craig-wood.com/nick/articles/pi-chudnovsky/
*/

void mpq_to_mpf(mpf_t result, mpq_t input) {
  // Get numerator and denominator as integers.
  mpz_t n_z, d_z;
  mpz_inits(n_z, d_z, NULL);
  mpq_get_num(n_z, input);
  mpq_get_den(d_z, input);

  gmp_printf("%Zd/%Zd\n", n_z, d_z);

  // Convert the integers do doubles.
  // result is acting as the numerator.
  mpf_t d_f;
  mpf_init(d_f);
  mpf_set_z(result, n_z);
  mpf_set_z(d_f, d_z);
  
  // Do the division.
  mpf_div(result, result, d_f);
}

void mpq_mul_ui (mpq_t result, mpq_t x, unsigned long n) {
  mpq_set_ui(result, n, 1);
  mpq_mul(result, result, x);
}

long double sum (int start, int max, long double (*f)(long long)) {
  long double sum = 0;

  for (; start < max; start++) {
    sum += f(start);
  }

  return sum;
}

void inside_a (mpq_t result, unsigned long k) {
  /*
              t1     t2
           (-1)^k * (6k)!
    ----------------------------
    (3k)! * (k!)^3 * 640320^(3k)
     t3        t4        t5
   */

  mpz_t t1, t2, t3, t4, t5, numerator, denominator;
  mpz_inits(t1, t2, t3, t4, t5, numerator, denominator, NULL);
  
  // (-1)^k
  mpz_set_si(t1, -((k % 2) * 2 - 1));
  // (6k)!
  mpz_fac_ui(t2, 6 * k);
  // (3k)!
  mpz_fac_ui(t3, 3 * k);
  // (k!)^3
  mpz_fac_ui(t4, k);
  mpz_pow_ui(t4, t4, 3);
  // 640320^3k
  mpz_set_ui(t5, 640320);
  mpz_pow_ui(t5, t5, 3 * k);

  // t1 * t2
  mpz_mul(numerator, t1, t2);
  // t3 * t4 * t5
  mpz_mul(denominator, t3, t4);
  mpz_mul(denominator, denominator, t5);

  // numerator/denominator
  mpq_set_num(result, numerator);
  mpq_set_den(result, denominator);
}

void inside_b (mpq_t result, unsigned long k, mpq_t this_a) {
  // result = this_a * k
  mpq_set_ui(result, k, 1);
  mpq_mul(result, result, this_a);
}

int main (int argc, char *argv[]) {
  if (argc != 2) {
    printf("Correct usage: %s [num iterations]\n", argv[0]);
    exit(1);
  }

  unsigned long num_iterations = atoi(argv[1]);
  
  mpq_t a, b;
  mpq_inits(a, b, NULL);

  
  for (unsigned long k = 0; k < num_iterations; k++) {
    mpq_t this_a, this_b;
    mpq_inits(this_a, this_b, NULL);
    inside_a(this_a, k);
    inside_b(this_b, k, this_a);
    mpq_add(a, a, this_a);
    mpq_add(b, b, this_b);
  }

  /*
        426880 * sqrt(10005)
    ----------------------------
    13591409 * a + 545140134 * b
           t2            t3
  */

  mpf_set_default_prec(num_iterations * 15);
  
  mpf_t numerator;
  mpf_init(numerator);
  // 426880 * sqrt(10005)
  mpf_sqrt_ui(numerator, 10005);
  mpf_mul_ui(numerator, numerator, 426880);
  
  mpq_t t2, t3, denominator;
  mpq_inits(t2, t3, denominator, NULL);
  // 13591409 * a
  mpq_mul_ui(t2, a, 13591409);
  // 545140134 * b
  mpq_mul_ui(t3, b, 545140134);
  // t2 + t3
  mpq_add(denominator, t2, t3);

  // Convert the floating point numerator to a rational number.
  mpq_t numerator_q;
  mpq_init(numerator_q);
  mpq_set_f(numerator_q, numerator);
  
  mpq_t pi_q;
  mpq_init(pi_q);
  mpq_div(pi_q, numerator_q, denominator);

  mpf_t pi;
  mpf_init(pi);
  mpq_to_mpf(pi, pi_q);
  
  char format_string[20];
  // %.<num decimal places>Ff for formatting strings with an mpf_t
  // Use %% so that the first % gets escaped.
  snprintf(format_string, sizeof(format_string), "PI: %%.%ldFf\n", num_iterations * 14);

  gmp_printf(format_string, pi);
  
  return 0;
}
