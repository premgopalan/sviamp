#!/usr/bin/perl

use strict;
use warnings;

my $eval_bin = "/disk/scratch1/prem/bayesianBKN/c++/svip-0.1/src/eval";
my $svinet_bin = "/disk/scratch1/prem/svip_project/src/svinet";
my @cmds = ("$eval_bin -gen-heldout -file airline.ungraph.txt -n 3237 -k 10",
	    "$eval_bin -gen-heldout -file loc-brightkite_edges.csv -n 56739 -k 10",
	    "$eval_bin -gen-heldout -file ca-AstroPh.csv -n 17903 -k 10",
	    "$eval_bin -gen-heldout -file cond-mat-2005.csv -n 36458 -k 10");

my @cmds2 = ("$svinet_bin -dir loc-brightkite-sets -n 56739 -k %d -link-sampling -svip-mode -eta-type %s -label %s -seed %f",
	     "$svinet_bin -dir ca-AstroPh-sets  -n 17903 -k %d -link-sampling -svip-mode -eta-type %s -label %s -seed %f",
	     "$svinet_bin -dir cond-mat-2005-sets  -n 36458 -k %d -link-sampling -svip-mode -eta-type %s -label %s -seed %f"
);

my @cmds3 = ("$svinet_bin -dir airline-sets -n 3237 -k %d -link-sampling -svip-mode -eta-type %s -label %s -seed %f");

#foreach my $cmd (@cmds) {
#    print $cmd;
#    system($cmd);
#}
#
#exit(0);

for (my $i = 1; $i <= 3; $i++) {
    my $seed = (time() % 1000) * $i;

    foreach my $type ("uniform") {
	#foreach my $cmd (@cmds2) {
	#    foreach my $K (50, 100, 150, 200, 250) {
	#	my $c = sprintf $cmd, $K, $type, $type, $seed;
	#	print "CMD = $c\n";
	#	system("$c 2>&1 > /dev/null &");
	#    }
	#}
	foreach my $cmd (@cmds3) {
	    foreach my $K (10, 15, 20, 25, 30) {
		my $c = sprintf $cmd, $K, $type, $type, $seed;
		print "CMD = $c\n";
		system("$c 2>&1 > /dev/null &");
	    }
	}
    }
}
