ORBITER=src/apps/orbiter/orbiter.out

problem1:
	$(ORBITER) -v 2 \
	  -define Gamma0 -graph -complete 1 -end \
	  -with Gamma0 -do -graph_theoretic_activity -properties -automorphism_group -eigenvalues -end \
	  -define Gamma1 -graph -double_cover Gamma0 -end \
	  -with Gamma1 -do -graph_theoretic_activity -properties -automorphism_group -eigenvalues -end \
	  -define Gamma2 -graph -double_cover Gamma1 -end \
	  -with Gamma2 -do -graph_theoretic_activity -properties -automorphism_group -eigenvalues -end \
	  -define Gamma3 -graph -double_cover Gamma2 -end \
	  -with Gamma3 -do -graph_theoretic_activity -properties -automorphism_group -eigenvalues -end \
	  -define Gamma4 -graph -double_cover Gamma3 -end \
	  -with Gamma4 -do -graph_theoretic_activity -properties -automorphism_group -eigenvalues -end \
	  -define Gamma5 -graph -double_cover Gamma4 -end \
	  -with Gamma5 -do -graph_theoretic_activity -properties -automorphism_group -eigenvalues -end \
	  -define Gamma6 -graph -double_cover Gamma5 -end \
	  -with Gamma6 -do -graph_theoretic_activity -properties -automorphism_group -eigenvalues -end \
	  -define Gamma7 -graph -double_cover Gamma6 -end \
	  -with Gamma7 -do -graph_theoretic_activity -properties -automorphism_group -eigenvalues -end

clean:
	rm -f *_properties.txt
