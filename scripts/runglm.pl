#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Long;

my $bmark_bin = "/usr/local/bin/benchmark";
my $infer_bin = "/disk/scratch1/prem/pnas2/infer";
my $glm_bin = "/disk/scratch1/prem/glmnew/infer";
my $em_bin = "/disk/scratch1/prem/em/infer";
my $glm_script = "/disk/scratch1/prem/scripts/runglm.pl";
my $F;
my $file = 0;
my $findk = 0;
my $datasets_dir = "/disk/scratch1/prem/glmnew/datasets";
my $seed = 2;

my @datasets = ("ca-AstroPh.csv",
		"ca-GrQc.csv",
		"ca-HepPh.csv",
		"ca-HepTh.csv",
		"cond-mat-2005.csv",
		"loc-brightkite_edges.csv",
		"netscience.csv",
		"usair.csv");
my %fixedK = ();
my %hol = ();

my %nodes = ();
my $stratified_node_sampling = 0;
my $informative_sampling = 0;
my $skip = 0;
my $uniform = 0;
my $LABEL = "";
my $noload = 0;


sub main() {
    my $allreal = 0;
    my $synthetic = 0;
    my $mu = 0.2;
    my $mscale = 1;
    my $all = 0;
    my $checkall = 0;
    my $dataset = "";
    my $em = 0;

    GetOptions ('file' => \$file,
		'synthetic' => \$synthetic,
		'dataset=s' => \$dataset,
		'findk' => \$findk,
		'mu=f' => \$mu,
		'mscale=i' => \$mscale,
		'all' => \$all,
		'checkall' => \$checkall,
		'allreal' => \$allreal,
		'srnode' => \$stratified_node_sampling,
		'massive' => \$informative_sampling,
		'skip' => \$skip,
		'uniform' => \$uniform,
		'seed=i' => \$seed,
		'em' => \$em,
		'label=s' => \$LABEL,
		'noload' => \$noload);
    
    open($F, ">cmds.txt");
    
    if ($synthetic) {
	run_synthetic($mu, $mscale);
    } elsif ($all) {
	run_all();
    } elsif ($checkall) {
	check_all();
    } elsif ($dataset ne "") {
	init();
	if ($em) {
	    run_real_em($dataset);
	} else {
	    run_real($dataset);
	}
    } elsif ($allreal) {
	init();
	run_all_datasets();
    } elsif ($em) {
	init();
	run_all_datasets_em();
    } else {
	die "option unimplemented\n";
    }
    close $F;
}

sub init()
{
    $nodes{"ca-AstroPh.csv"} = 17903;
    $nodes{"ca-GrQc.csv"} = 4158;
    $nodes{"ca-HepPh.csv"} = 11204;
    $nodes{"ca-HepTh.csv"} = 8638;
    $nodes{"cond-mat-2005.csv"} = 36458;
    $nodes{"loc-brightkite_edges.csv"} = 56739;
    $nodes{"netscience.csv"}  = 1461;
    $nodes{"usair.csv"} = 712;
    $nodes{"nslcc.csv"} = 379;

    $fixedK{"ca-AstroPh.csv"} = 100;
    $fixedK{"ca-GrQc.csv"} = 100;
    $fixedK{"ca-HepPh.csv"} = 100;
    $fixedK{"ca-HepTh.csv"} = 100;
    $fixedK{"cond-mat-2005.csv"} = 100;
    $fixedK{"loc-brightkite_edges.csv"} = 100;
    $fixedK{"netscience.csv"}  = 50;
    $fixedK{"usair.csv"} = 20;
    $fixedK{"nslcc.csv"} = 20;

    $hol{"ca-AstroPh.csv"} = 0.01;
    $hol{"ca-GrQc.csv"} = 0.01;
    $hol{"ca-HepPh.csv"} = 0.01;
    $hol{"ca-HepTh.csv"} = 0.01;
    $hol{"cond-mat-2005.csv"} = 0.01;
    $hol{"loc-brightkite_edges.csv"} = 0.01;
    $hol{"netscience.csv"}  = 0.1;
    $hol{"usair.csv"} = 0.1;
    #$hol{"nslcc.csv"} = 0.1;
}


sub run_all_datasets()
{
  LOOP:
    foreach my $d (@datasets) {
	#if ($d eq "cond-mat-2005.csv" || 
	#    $d eq "loc-brightkite_edges.csv") {
	#    next LOOP;
	#}
	my $s = $skip ? "-skip" : "";
	my $cmd = "$glm_script -dataset $d -uniform -seed $seed $s -label $LABEL &";
	system($cmd);
    }
}


sub run_all_datasets_em()
{
    foreach my $d (@datasets) {
	my $cmd = "$glm_script -em -dataset $d -uniform -seed $seed &";
	system($cmd);
    }
}

sub run_real_em($)
{
    my $f = shift @_;
    my $N = $nodes{$f};

    my $file = "$datasets_dir/$f";
    my $dir = "r$seed-$f";
    #my $dir = "out-$f";

    my $K = $fixedK{$f};
    my $cmd = "cd $dir/n$N-k$K-mmsb-batch-ann/n$N-k$K-mmsb-batch; ".
	"$em_bin -file $file -n $N -k $K -em -amm -label EMUniform -seed $seed 2>&1 >> inf.out &";
    run($cmd);
}

sub run_real($)
{
    my $f = shift @_;
    my $N = $nodes{$f};

    my $file = "$datasets_dir/$f";
    my $dir = "r$seed-$f";

    my $cmd = "";
    if (!$skip) {
	$cmd = "mkdir -p $dir; cd $dir; ".
	    "$infer_bin -file $file -n $N -k $N -fastinit ".
	    " 2>&1 >> inf.out";
	print $cmd;
	run($cmd);
    }

    #$cmd = "cd out-$f; wc -l n$N-k$N-mmsb-fastinit-v5/communities.txt";
    #my $K = getK(`$cmd`);

    my $K = $fixedK{$f};
    print "$f $K\n";

    my $lt = 0;
    my $opt = "";
    if (!$uniform) {
	$opt = " -nonuniform";
    }
    if (!$skip) {
	my $h = $hol{$f};
	$cmd = "cd $dir; ".
	    "$infer_bin -file $file -n $N -k $K  -batch -annealing -rfreq 1".
	    " -lt-min-deg $lt -link-thresh 0.9".
	    " -max-iterations 100 $opt -heldout-ratio $h -seed $seed 2>&1 >> inf.out";
	run($cmd);
	
	$cmd = "cd $dir/n$N-k$K-mmsb-batch-ann; ".
	    "$infer_bin -file $file -n $N -k $K -batch -rfreq 1".
	    " -lt-min-deg $lt -link-thresh 0.9".
	    " -max-iterations 100 $opt -load 1 1 -heldout-ratio $h  -seed $seed 2>&1 >> inf.out";
	run($cmd);
    }

    $opt = "-rnode";
    my $rfreq = ($opt eq "-rnode") ? 10 : 1000;
    if ($stratified_node_sampling) {
	$opt = "-massive -onesonly";
    } elsif ($informative_sampling) {
	$rfreq = 1000;
	$opt = "-massive -preprocess";
	
	$cmd = "cd $dir/n$N-k$K-mmsb-batch-ann/n$N-k$K-mmsb-batch; ".
	    "$glm_bin -file $file -n $N -k $K -glm $opt -max-iterations 100000 -load 1 1 ".
	    " -lt-min-deg $lt -link-thresh 0.9 -label $seed".
	    " 2>&1 >> inf.out";
	run($cmd);
	$cmd = "cd $dir/n$N-k$K-mmsb-batch-ann/n$N-k$K-mmsb-batch; ".
	    "mv n$N-k$K-$seed-massiveglm/neighbors.bin .";
	run($cmd);
	$opt = "-massive";
    }

    my $label = $LABEL eq "" ? $seed : $LABEL;
    #my $load = $noload ? "-load 1 1 -amm" : " -load 1 1 ";
    my $load = $noload ? "" : " -load 1 1 ";
    $cmd = "cd $dir/n$N-k$K-mmsb-batch-ann/n$N-k$K-mmsb-batch; ".
	"$glm_bin -file $file -n $N -k $K -glm $opt $load".
	" -lt-min-deg $lt -link-thresh 0.9 -label $label -rfreq $rfreq -seed $seed -max-iterations 100000".
	" 2>&1 >> inf.out &";
    run($cmd);
	
    $cmd = "cd $dir/n$N-k$K-mmsb-batch-ann/n$N-k$K-mmsb-batch; ".
	"$glm_bin -file $file -n $N -k $K -glm $opt  -load 1 1 ".
	" -lt-min-deg $lt -link-thresh 0.9 -label $seed -rfreq $rfreq -seed $seed".
	" -nolambda 2>&1 >> inf.out &";
    # XXXXX
    #run($cmd);

    run_real_em($f);
}

sub run_all()
{
    for (my $mu = 0; $mu <= 0.6; $mu += 0.2) {
	#my $mu = 0.6;
	foreach my $mscale (3,4,5,7) {
	    my $cmd = "$glm_script -synthetic -mu $mu -mscale $mscale &";
	    system($cmd);
	}
    }
}

sub check_all()
{
    for (my $mu = 0; $mu <= 0.4; $mu += 0.2) {
	foreach my $mscale (1,2,3,4,5,6,7,8) {
	    check_synthetic($seed, $mu, $mscale);
	}
    }
}

sub getK($)
{
    my $K = 0;
    my $s = shift @_;
    if ($s =~ /(\d+)\s+.*/) {
	$K = $1 + 0;
    } else {
	die "cannot find K (got $K)\n";
    }
    return $K;
}


sub run_synthetic($$)
{
    my ($mu, $mscale) = @_;
    my $N = 1000;
    my $ovlap = 0.5;
    my $numovlap = 4;
    my $minc = 20;
    my $maxc = 50;
    my $deg_per_comm = 10;
    my $initseed = 21111984 + $seed;

    # my $seed = 1, $mscale = 8, $mu = 0.2;
    # my $seed = 1, $mscale = 8, $mu = 0;
    # my $seed = 1, $mscale = 1, $mu = 0.2;
    # my $seed = 1, $mscale = 1, $mu = 0;
    
    my $dirstr = "r$seed-mu$mu-s$mscale";
    my $dir = sprintf $dirstr, $seed, $mu;
    
    my $mindeg = $numovlap * $deg_per_comm;
    my $maxdeg = $mindeg * $mscale;
    if ($maxdeg <= $mindeg) {
	$maxdeg = $mindeg;
    }

    my $lt = 0;
    if ($mu > .0) {
	$lt = 3;
    } else {
	$lt = 0;
    }

    my $seedcmd = "mkdir -p $dir; cd $dir; echo $seed > time_seed.dat";
    run($seedcmd);
    
    my $bcmdstr = "cd $dir; " .
	"$bmark_bin -N %d -k %d -maxk %d -on %d -om %d -mu %f -minc %d -maxc %d";
    my $bcmd = sprintf $bcmdstr, $N, $mindeg, $maxdeg, int($N * $ovlap), 
    $numovlap, $mu, $minc, $maxc;
    
    my $c = "$bcmd >> bmark.out 2>&1";
    run($c);

    my $file = "network.dat";

    my $cmd = "cd $dir; ".
	"$infer_bin -file $file -n $N -k $N -fastinit ".
	" -nmi community.dat -bmark 2>&1 >> inf.out";
    run($cmd);

    if ($findk) {
	$cmd = "wc -l $dir/n$N-k$N-mmsb-fastinit-v5/communities.txt";
    } else {
	$cmd = "wc -l $dir/ground_truth.txt";
    }
    my $K = getK(`$cmd`);
    
    $cmd = "cd $dir; ".
	"$infer_bin -file $file -n $N -k $K  -batch -annealing -rfreq 1".
	" -lt-min-deg $lt -link-thresh 0.9".
	" -max-iterations 100 -nmi community.dat -bmark 2>&1 >> inf.out";
    run($cmd);

    $cmd = "cd $dir/n$N-k$K-mmsb-batch-ann; ".
	"$infer_bin -file ../$file -n $N -k $K -batch -rfreq 1".
	" -lt-min-deg $lt -link-thresh 0.9".
	" -max-iterations 100 -load 1 1 -nmi ../community.dat -bmark 2>&1 >> inf.out";
    run($cmd);

    $cmd = "cd $dir/n$N-k$K-mmsb-batch-ann/n$N-k$K-mmsb-batch; ".
	"$glm_bin -file ../../$file -n $N -k $K -glm -rpair -max-iterations 3000 -load 1 1 ".
	" -lt-min-deg $lt -link-thresh 0.9".
	" -nmi ../../community.dat -bmark 2>&1 >> inf.out &";
    run($cmd);

    $cmd = "cd $dir/n$N-k$K-mmsb-batch-ann/n$N-k$K-mmsb-batch; ".
	"$glm_bin -file ../../$file -n $N -k $K -glm -rpair -max-iterations 3000 -load 1 1 ".
	" -lt-min-deg $lt -link-thresh 0.9".
	" -nmi ../../community.dat -bmark -nolambda 2>&1 >> inf.out &";
    run($cmd);
}

sub check_synthetic($$$)
{
    my $seed = shift @_;
    my $mu = shift @_;
    my $mscale = shift @_;
    my $N = 1000;
    my $ovlap = 0.5;
    my $numovlap = 4;
    my $minc = 20;
    my $maxc = 50;
    my $deg_per_comm = 10;
    
    my $dirstr = "r${seed}-mu${mu}-s${mscale}";
    my $dir = sprintf $dirstr, $seed, $mu;
    
    my $cmd = "wc -l $dir/ground_truth.txt";
    my $K = `$cmd`;
    if ($K =~ /(\d+)\s+.*/) {
	$K = $1 + 0;
    } else {
	die "cannot find K (got $K)\n";
    }

    my $roc_auc = `grep ROC $dir/n$N-k$K-mmsb-batch-ann/n$N-k$K-mmsb-batch/n$N-k$K-xx-rpairglm/auc-all.txt`;
    if ($roc_auc =~  /ROC\s+(\S+)/) {
	$roc_auc = $1 + 0;
    }
    my $pr_auc = `grep PR $dir/n$N-k$K-mmsb-batch-ann/n$N-k$K-mmsb-batch/n$N-k$K-xx-rpairglm/auc-all.txt`;
    if ($pr_auc =~ /AUC\-PR\s+(\S+)/) {
	$pr_auc = $1 + 0;
    }
    my $roc_auc_nl = `grep ROC $dir/n$N-k$K-mmsb-batch-ann/n$N-k$K-mmsb-batch/n$N-k$K-xx-rpairglm-nolambda/auc-all.txt`;
    if ($roc_auc_nl =~  /ROC\s+(\S+)/) {
	$roc_auc_nl = $1 + 0;
    }
    my $pr_auc_nl = `grep PR $dir/n$N-k$K-mmsb-batch-ann/n$N-k$K-mmsb-batch/n$N-k$K-xx-rpairglm-nolambda/auc-all.txt`;
    if ($pr_auc_nl =~ /AUC\-PR\s+(\S+)/) {
	$pr_auc_nl = $1 + 0;
    }
    my $nmi = `tail -n1 $dir/n$N-k$K-mmsb-batch-ann/n$N-k$K-mmsb-batch/n$N-k$K-xx-rpairglm/mutual.txt`;
    if ($nmi =~ /mutual3:\s+(\S+)/) {
	$nmi = $1 + 0;
    }
    my $nmi_nl = `tail -n1 $dir/n$N-k$K-mmsb-batch-ann/n$N-k$K-mmsb-batch/n$N-k$K-xx-rpairglm-nolambda/mutual.txt`;
    if ($nmi_nl =~ /mutual3:\s+(\S+)/) {
	$nmi_nl = $1 + 0;
    }
    print "$mu $mscale $roc_auc $roc_auc_nl $pr_auc $pr_auc_nl $nmi $nmi_nl\n";
}



sub run($) {
    my $a = shift @_;
    #print "CMD = $a\n";
    print $F "CMD = $a\n";
    if (system($a) != 0) { 
	print $F "$a failed\n";
	return -1;
    }
    return 0;
}

main();
