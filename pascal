#!/usr/bin/perl -w
#	-*- Perl -*-		pascal:	print Pascal's Triangle
#
#	(usage)% pascal LINES
#
#	Inputs:		LINES: the number of lines to print
#			
#	Outputs:	Prints Pascal's Triangle up to LINES
#
#	David and Kentaro Wuertele	Sun Nov 13 17:00:10 2016	Steal This Program!!!

use strict;

my $lines = shift;

die "Must specify a number of lines" if ($lines !~ m/\d+/);

print "Printing $lines lines of Pascal's Triangle.\n";

my @last_line;
foreach my $current_line (1..$lines) {
    my @next_line;
    foreach my $column (0..$#last_line+1) {
	if ($column == 0) {
	    $next_line[$column] = 1;
	} elsif ($column == $#last_line + 1) {
	    $next_line[$column] = 1;
	} else {
	    $next_line[$column] = $last_line[$column-1] + $last_line[$column];
	}
    }
    print " " x (2 * ($lines - $current_line));
    print join (" ", map { sprintf ("%3d", $_) } @next_line), "\n";
    @last_line = @next_line;
}

