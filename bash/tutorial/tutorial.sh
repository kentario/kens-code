#!/bin/bash

echo this tutorial comes from 'https://ryanstutorials.net/bash-scripting-tutorial/'
echo

echo name of script \$0: $0
echo input variables are \$1 to \$9
echo number of arguments passed \$0: $0
echo list of all arguments passed $\@: $@
echo the exit status of the most recently run process /$?: $?
echo the process ID of the current program /$$: $$
echo the username of the user running the program /$USER: $USER
echo the hostname of the machine the program is running on /$HOSTNAME: $HOSTNAME
echo number of seconds since this program started \$SECONDS: $SECONDS
echo a random number that is different each time \$RANDOM: $RANDOM
echo the current line number in the program \$LINENO: $LINENO
echo

echo to set a variable, use variable_name=value
echo make sure to leave no spaces around the equals
my_variable=Hello
next_variable=World
my_total="$my_variable $next_variable"
other_total='$my_variable $next_variable'
echo $my_total
echo $other_total
sample_dir=/home/ken
ls $sample_dir
echo

echo command line substitution takes the command line output of a program, and saves it to a variable. When this happens, all the newlines are removed from the output.
echo to do this, place the program in brackets, preceded by a $
echo 'hello_world_output=$(./hello-world.sh)'
hello_world_output=$(./hello-world.sh)
echo $hello_world_output
echo

echo exporting a variable makes it so that that variable is available in all child processes of this process.
var1=foo
var2=bar
echo $0: var1: $var1, var2: $var2
export var1
./script.sh
echo $0: var1: $var1, var2: $var2
echo exporting a variable does not mean that other programs can change the variables value in this process.
echo

echo to get user input, use read.
echo enter something:
read
echo you entered: $REPLY
echo use -p for a prompt, and multiple variables for multiple inputs.
read -p 'enter some more things: ' first second third
echo first argument: $first, second argument: $second, third argument: $third
echo using -s makes the input silent.
read -sp "enter your password (actually don't): " not_password
echo "this is your (not) password: $not_password"
