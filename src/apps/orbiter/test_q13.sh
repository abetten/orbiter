#!/bin/bash
for f in {0..12}; do
    echo "Testing f=$f"
    ./orbiter.out -v 0 -define F -finite_field -q 13 -end -define P -projective_space -n 2 -field F -v 0 -end -define C -quartic_curve -space P -by_normal_form "3,2,10,4,2,$f" -end -with C -do -quartic_curve_activity -report -end > test_q13_f${f}.log 2>&1
    if [ $? -eq 0 ]; then
        echo "SUCCESS for f=$f"
    else
        echo "FAILED for f=$f"
    fi
done
