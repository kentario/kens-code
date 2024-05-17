#include <stdio.h>
#include <stdlib.h>

int return_n (int n) {return n;}

int main (int argc, char *argv[]) {
    int x = 3;

    // In all these cases, regardless of what the modification of the x returns, the x is modified before the standalone x is evaluated, even when the standalone x is on the left.
    //                              4 -  3
    printf("a               %d\n", (x - x++));
    x = 3; //                        3  - 4
    printf("a but backwards %d\n", (x++ - x));
    
    x = 3; //                       4 -  4
    printf("b               %d\n", (x - ++x));
    x = 3; //                        4  - 4
    printf("b but backwards %d\n", (++x - x));
    x = 3; //                          4  -  4
    printf("b with function %d\n\n", ((x) - (x = return_n(4))));

    printf("\nIf statements:\n");
    
    x = 3; // First 2 are equivilant.
    if (x = 5 && printf("5 && print          %d\n", x)); // 3
    x = 3;
    if (x = (5 && printf("(5 && print)        %d\n", x))); // 3
    // x == 1 after both of these, because 5 && printf("foo") -> true
    
    x = 3;
    if ((x = 5) && printf("(5) && print        %d\n", x)); // 5
    x = 3;
    if (printf("print && 5          %d\n", x) && (x = 5)); // 3

    printf("return of inner printf: %d\n", printf("\ninner printf: %d\n", 5));

    int y = 5 && 10;
    printf("5 && 10: %d\n\n", y);

    printf("\nArrays vs variables:\n");
    
    int test[1][1] = {{0}};
    x = 0;
    y = 0;
    int *z = &y;
    
    // These 2 are the same but one uses and array and the other a variable, and one uses a pointer to a variable.

    printf("test[0][0]: %d x: %d y: %d *z: %d\n", test[0][0], x, y, *z); // All 0
    
    printf("arr - (arr = 3) %d\n",
	   (test[0][0]) - (test[0][0] = return_n(3)));

    printf("*z - (*z = 3)   %d\n",
	   (*z) - (*z = return_n(3)));

    printf("x - (x = 3)     %d\n",
	   (x) - (x = return_n(3)));
    

    return 0;
}
