#include <stdio.h>
#include <stdarg.h>

void my_printf (char *fmt, ...) {
  va_list args;
  va_start(args, fmt);

  while (*fmt != '\0') {
    if (*fmt == '\\') {
      fmt++;
      switch (*fmt) {
      case 'n':
	printf("\n");
	break;
      case 't':
	printf("\t");
	break;
      case '\\':
	printf("\\");
	break;
      default:
	printf("%c", *fmt);
      }
    } else if (*fmt == '%') {
      fmt++;
      switch (*fmt) {
      case 'd':
	int d = va_arg(args, int);
	printf("%d", d);
	break;
      case 'f':
	double f = va_arg(args, double);
	printf("%f", f);
	break;
      case 'c':
	int c = va_arg(args, int);
	printf("%c", c);
	break;
      default:
	printf("%c", *fmt);
      }
    } else {
      printf("%c", *fmt);
    }
    fmt++;
  }

  va_end(args);
}


#define VALUE_IF_NOT_(...) __VA_ARGS__
#define VALUE_IF_NOT_1(...)

#define VALUE_IF_NOT(cond, ...) VALUE_IF_NOT_##cond(__VA_ARGS__)

// This is the opposite of __VA_OPT__(), which will evaluate to the inside if __VA_ARGS__ has elements, and nothing if __VA_ARGS__ is empty.
// NOT_VA_OPT() will evaluate to "There are no elements" when there are no elements in NOT_VA_OPT
#define NOT_VA_OPT(...) VALUE_IF_NOT(__VA_OPT__(1), "There are no elements")

int main (int argc, char *argv[]) {
  int a = 3;
  char c = 'h';
  double d = 1.234;
  my_printf("a: %d\nb: %d\nc: %c\nd: %f\n", a, 4, c, d);
  
  return 0;
}
