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
	} else {
	    if(m/^e ([0-9]+) ([0-9]+)/) {
		if(!$have_params) {
		    die "format line missing";
		}
		$i=$1-1;
		$j=$2-1;
		if($i > $j) {
		    $t = $i;
		    $i = $j;
		    $j = $t;
		}
		if($i >= $nv || $j >= $nv || $i==$j) {
		    die "invalid edge specification";
		}
		if(defined $edge{"$i-$j"}) {
		    die "repeated edge found";
		}
		$edge{"$i-$j"}=1;
		$adj{$i}=$adj{$i}." $j";
		$adj{$j}=$adj{$j}." $i";
		$ec++;
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
print "n=$nv g\n";
for($i=0;$i<$nv;$i++) {
    print "$adj{$i};\n"
}
print "x\n";
