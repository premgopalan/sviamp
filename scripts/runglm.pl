#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Long;

my $bmark_bin = "/usr/local/bin/benchmark";
my $svinet_bin = "/disk/scratch1/prem/svip_project/src/svinet";
my $glm_bin = "/disk/scratch1/prem/nodepop/src/nodepop";
my $em_bin = "/disk/scratch1/prem/em/infer";
my $glm_script = "/disk/scratch1/prem/nodepop/scripts/runglm.pl";
my $eval_bin = "/disk/scratch1/prem/bayesianBKN/c++/svip-0.1/src/eval";
my $F;
my $file = 0;
my $findk = 0;
my $datasets_dir = "/disk/scratch1/prem/nodepop/datasets";
my $seed = 2;

# my @datasets = ("ca-AstroPh",
# 		"ca-GrQc",
# 		"ca-HepPh",
# 		"ca-HepTh",
# 		"cond-mat-2005",
# 		"loc-brightkite_edges");
# 		"netscience",
# 		"usair");


my @datasets = ("netscience",
		"ca-GrQc",
#		"nslcc",
		"usair",
		"poblogs");
my %fixedK = ();
my %hol = ();

my %nodes = ();
my $stratified_node_sampling = 0;
my $informative_sampling = 0;
my $skip = 0;
my $uniform = 0;
my $LABEL = "";
my $noload = 0;
my $adagrad = 0;
my $globalmu = 0;
my $allopt = 0;


sub main() {
    my $allreal = 0;
    my $synthetic = 0;
    my $mu = 0.2;
    my $mscale = 1;
    my $all = 0;
    my $checkall = 0;
    my $dataset = "";
    my $em = 0;
    my $gensets = 0;
    my $getresults = 0;

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
		'noload' => \$noload,
		'gensets' => \$gensets,
		'adagrad' => \$adagrad,
		'globalmu' => \$globalmu,
		'allopt' => \$allopt,
		'getresults' => \$getresults);
    
    open($F, ">cmds.txt");

    init();
    if ($getresults) {
	getresults();
    } elsif ($gensets) {
	gensets();
    } elsif ($synthetic) {
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
	print "running all datasets\n";
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
    $nodes{"ca-AstroPh"} = 17903;
    $nodes{"ca-GrQc"} = 4158;
    $nodes{"ca-HepPh"} = 11204;
    $nodes{"ca-HepTh"} = 8638;
    $nodes{"cond-mat-2005"} = 36458;
    $nodes{"loc-brightkite_edges"} = 56739;
    $nodes{"netscience"}  = 1461;
    $nodes{"usair"} = 712;
    $nodes{"nslcc"} = 379;
    $nodes{"poblogs"} = 1224;

    $fixedK{"ca-AstroPh"} = 100;
    $fixedK{"ca-GrQc"} = 100;
    $fixedK{"ca-HepPh"} = 100;
    $fixedK{"ca-HepTh"} = 100;
    $fixedK{"cond-mat-2005"} = 100;
    $fixedK{"loc-brightkite_edges"} = 100;
    $fixedK{"netscience"}  = 100;
    $fixedK{"usair"} = 20;
    $fixedK{"nslcc"} = 20;
    $fixedK{"poblogs"} = 2;

    $hol{"ca-AstroPh"} = 0.01;
    $hol{"ca-GrQc"} = 0.01;
    $hol{"ca-HepPh"} = 0.01;
    $hol{"ca-HepTh"} = 0.01;
    $hol{"cond-mat-2005"} = 0.01;
    $hol{"loc-brightkite_edges"} = 0.01;
    $hol{"netscience"}  = 0.1;
    $hol{"usair"} = 0.1;
    $hol{"nslcc"} = 0.1;
    $hol{"poblogs"} = 0.1;
}

sub gensets()
{
    my $cmd_fmt = "$eval_bin -gen-heldout -file %s -n %d -k %d -label sets";
    foreach my $d (@datasets) {
	my $cmd = sprintf $cmd_fmt, "$d.csv", $nodes{$d}, $fixedK{$d};
	print "CMD = $cmd\n";
	system($cmd);
	$cmd = sprintf "rm -rf $d; mv n%d-k%d-sets $d", $nodes{$d}, $fixedK{$d};
	system($cmd);
	print "CMD = $cmd\n";
    }
}

sub gensets2($$$$$)
{
    my ($topdir, $file, $N, $K, $dir) = @_;
    my $cmd = "$eval_bin -gen-heldout -file $file -n $N -k $K -label sets";
    system($cmd);
    $cmd = sprintf "mv n%d-k%d-sets $topdir/$dir", $N, $K;
    system($cmd);
}

sub getresults()
{
    my @dir = split ' ', `ls`;
    my $repc = 1;
    my %maprep = ();
    foreach my $d (@dir) {
	if ($d =~ /r(\d+)-(\S+)/) {
	    my $seed = $1;
	    my $dset = $2;
	    if (!defined $maprep{$seed}) {
		$maprep{$seed} = $repc;
		$repc++;
	    }
	    my $rep = $maprep{$seed};
	    #print "seed = $seed, dataset = $dset\n";
	    my $cmd = sprintf "tail -n1 $d/n%d-k%d-$seed-seed$seed-linksampling/precision-hol.txt",
	    $nodes{$dset}, $fixedK{$dset};
	    my $out = `$cmd`;
	    #print $out;
	    if ($out =~ /\d+\s+\d+\s+(\S+).*/) {
		my $mmsb_hol = $1;
		printf "%s\t%d\t%s\t%.5f\n", $dset, $rep, "mmsb", $mmsb_hol;
	    }
	    $cmd = sprintf "tail -n10 $d/n%d-k%d-$seed-seed$seed-linksampling/n%d-k%d-$seed-rnodeglm-gmu/heldout.txt", $nodes{$dset}, $fixedK{$dset}, $nodes{$dset}, $fixedK{$dset};
	    my @p = split '\n', `$cmd`;
	    my $maxsz = 0;
	    my $mhol = 0;
	    foreach my $q (@p) {
		#print $q . "\n";
		if ($q =~ /\d+\s+\d+\s+(\S+)\s+(\d+).*/) {
		    my $sz = $2 + 0;
		    my $hol = $1 + 0;
		    if ($sz > $maxsz) {
			$mhol = $hol;
			$maxsz = $sz;
		    }
		}
	    }
	    printf "%s\t%d\t%s\t%.5f\n", $dset, $rep, "amp", $mhol;
	}
    }
}

sub run_all_datasets()
{
  LOOP:
    foreach my $d (@datasets) {
	my $s = $skip ? "-skip" : "";
	my $cmd = "$glm_script -dataset $d -seed $seed -allopt &";
	print "CMD = $cmd\n";
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
    my $K = $fixedK{$f};
    print "$f $K\n";

    my $label = $LABEL eq "" ? $seed : $LABEL;
    if (!$skip) {
	$cmd = "mkdir -p $dir; cd $dir; ".
	    "$svinet_bin -dir $file -n $N -k $K -link-sampling -svip-mode -seed $seed".
	    " -label $label -rfreq 10 >> inf.out";
	run($cmd);
    }
    
    my $opt = "-rnode";
    my $rfreq = ($opt eq "-rnode") ? 10 : 1000;

    #my $load = $noload ? "-load 1 1 -amm" : " -load 1 1 ";
    my $load = $noload ? "" : " -load 1 1 ";

    if (!$allopt) {
	my $adagrad_str = !$adagrad ? "" : "-adagrad";
	my $globalmu_str = !$globalmu ? "" : "-globalmu";
	
	$cmd = "cd $dir/n$N-k$K-$label-seed$seed-linksampling; $glm_bin ".
	    "-dir $file -n $N -k $K -glm $opt $load ".
	    " $adagrad_str $globalmu_str ".
	" -label $label -seed $seed -rfreq $rfreq -max-iterations 100000 2>&1 >> inf.out &";
	run($cmd);
    } else {
	$cmd = "cd $dir/n$N-k$K-$label-seed$seed-linksampling; $glm_bin ".
	    "-dir $file -n $N -k $K -glm $opt $load ".
	    " -label $label -seed $seed -rfreq $rfreq -max-iterations 100000 2>&1 >> inf.out &";
	#run($cmd);

	#$cmd = "cd $dir/n$N-k$K-$label-seed$seed-linksampling; $glm_bin ".
	#    "-dir $file -n $N -k $K -glm $opt $load ".
	#    "  -adagrad ".
	#    " -label $label -seed $seed -rfreq $rfreq -max-iterations 100000 2>&1 >> inf.out &";
	#run($cmd);

	#$cmd = "cd $dir/n$N-k$K-$label-seed$seed-linksampling; $glm_bin ".
	#    "-dir $file -n $N -k $K -glm $opt $load ".
	#    "  -globalmu ".
	#    " -label $label -seed $seed -rfreq $rfreq -max-iterations 100000 2>&1 >> inf.out &";
	#run($cmd);

	$cmd = "cd $dir/n$N-k$K-$label-seed$seed-linksampling; $glm_bin ".
	    "-dir $file -n $N -k $K -glm $opt $load ".
	    " -globalmu ".
	    " -label $label -seed $seed -rfreq $rfreq -max-iterations 100000 2>&1 >> inf.out &";
	run($cmd);
    }
}

sub run_all()
{
    for (my $mu = 0; $mu <= 0.4; $mu += 0.2) {
	#my $mu = 0.6;
	foreach my $mscale (2,4,6,8) {
	    my $cmd = "$glm_script -synthetic -mu $mu -mscale $mscale -seed $seed &";
	    run($cmd);
	    sleep(1);
	}
    }
OUT:
}

sub check_all()
{
    print "\"dataset\"\t\"seed\"\t\"mu\"\t\"mscale\"\t\"p10\"\t\"p50\"\t\"p100\"\n";
    for (my $mu = 0; $mu <= 0.4; $mu += 0.2) {
	foreach my $mscale (2,4,6,8) {
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
    my $minc = 200;
    my $maxc = 500;
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
	"$svinet_bin -file $file -n $N -k $N -findk ".
	" -nmi community.dat -bmark 2>&1 >> inf.out";
    run($cmd);

    if ($findk) {
	$cmd = "wc -l $dir/n$N-k$N-mmsb-fastinit-v5/communities.txt";
    } else {
	$cmd = "wc -l $dir/ground_truth.txt";
    }
    my $K = getK(`$cmd`);

    gensets2($dir, "$dir/network.dat", $N, $K, "sets");
    
    $cmd = "cd $dir; ".
	"$svinet_bin -dir sets -n $N -k $K  -link-sampling -svip-mode -rfreq 10".
	" -lt-min-deg $lt -link-thresh 0.9".
	" -max-iterations 100 -nmi community.dat -bmark 2>&1 >> inf.out";
    run($cmd);

    $cmd = "cd $dir/n$N-k$K-mmsb-linksampling; ".
	"$glm_bin -dir ../../$dir/sets -n $N -k $K -glm -rnode -load 1 1 ".
	" -lt-min-deg $lt -link-thresh 0.9".
	" -nmi ../community.dat -bmark -rfreq 10 2>&1 >> inf.out &";
    run($cmd);

    $cmd = "cd $dir/n$N-k$K-mmsb-linksampling; ".
	"$glm_bin -dir ../../$dir/sets -n $N -k $K -glm -rnode -load 1 1 ".
	" -globalmu ".
	" -lt-min-deg $lt -link-thresh 0.9".
	" -nmi ../community.dat -bmark -rfreq 10 -globalmu 2>&1 >> inf.out &";
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

    my $dset1 = "mmsb";
    my $prec1 = `tail -n1 $dir/n$N-k$K-mmsb-linksampling/precision.txt`;
    my $dset2 = "amp";
    my $prec2 = `tail -n1 $dir/n$N-k$K-mmsb-linksampling/n$N-k$K-xx-rnodeglm/precision.txt`;
    print "$dset1\t$seed\t$mu\t$mscale\t$prec1";
    print "$dset2\t$seed\t$mu\t$mscale\t$prec2";
}



sub run($) {
    my $a = shift @_;
    print "CMD = $a\n";
    
    print $F "CMD = $a\n";
    if (system($a) != 0) { 
	print $F "$a failed\n";
	return -1;
    }
    return 0;
}

main();
