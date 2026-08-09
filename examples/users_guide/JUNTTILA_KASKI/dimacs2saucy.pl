#!/usr/bin/perl

$have_params=0;
while(<>) {
    chomp;
    if(!m/^c/) {
	if(m/^p edge ([0-9]+) ([0-9]+)/) {
	    $have_params=1;
	    $nv=$1;
	    $ne=$2;
	    $ec=0;
	    for($i=0;$i<$nv;$i++) {
		$rp[$i]=$i;
	    }
	    for($i=0;$i<$nv;$i++) {
		$j=$i+int(rand($nv-$i));
		$t=$rp[$i];
		$rp[$i]=$rp[$j];
                $rp[$j]=$t;
	    }
	} else {
	    if(m/^e ([0-9]+) ([0-9]+)/) {
		if(!$have_params) {
		    die "format line missing";
		}
		$i=$rp[$1-1];
		$j=$rp[$2-1];
		if($i > $j) {
		    $t = $i;
		    $i = $j;
		    $j = $t;
		}
		if($i >= $nv || $j >= $nv || $i==$j) {
		    die "invalid edge specification";
		}
		if(defined $edge{"$i $j"}) {
		    die "repeated edge found";
		}
		$edge{"$i $j"}=1;
		$elist[$ec++]=($i)." ".($j);
	    } else {
		if(!m/^ +$/) {
		    die "parse error";
		}
	    }
	}
    }
}
if($ec!=$ne) {
    die "invalid number of edges";
}
print "$nv $ne 1\n";
for($i=0;$i<$ne;$i++) {
    print "$elist[$i]\n"
}
