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


    if (!$skip) {
	$cmd = "mkdir -p $dir; cd $dir; ".
	    "$svinet_bin -file $file -n $N -k $N -fastinit ".
	    " 2>&1 >> inf.out";
	print $cmd;
	run($cmd);
    }

    #$cmd = "cd out-$f; wc -l n$N-k$N-mmsb-fastinit-v5/communities.txt";
    #my $K = getK(`$cmd`);

