#!/bin/bash

echo use let and then an expression
echo 'let a=5+4'
echo if you want to use spaces, use quotes: let \"a = 5 + 4\"
let "a = 5 + 4"
echo $a #9
let a=3*2
echo $a #6
let a++
echo $a #7
let ++a
echo $a #8
echo other variables such as command line arguments can also be used.
let a=$1+2
echo $a
let a+=3
echo $a
let "a = 100 % 3"
echo $a #1
echo
read -p "enter 2 numbers: " first second
let "sum = $first + $second"
echo $first + $second = $sum
echo

echo expr is like let but it prints the result instead of putting it in a variable.
# * is special so it needs to be escaped.
expr 3 \* 5
echo spaces are required, and using quotes, or no spaces will just print the expression.
expr 3+5/2
echo expr can also be used for command substitution
