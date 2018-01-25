// test_hyperoval.C
//
// Anton Betten
// October 10, 2010

#include "orbiter.h"


// global data:

INT t0; // the system time when the program started

int main(int argc, char **argv);

int main(int argc, char **argv)
{
	INT verbose_level = 0;
	//INT f_poly = FALSE;
	INT f_stabilizer = FALSE;
	INT f_regular = FALSE;
	INT f_LunelliSce = FALSE;
	INT f_Subiaco64_1 = FALSE;
	INT f_Subiaco64_2 = FALSE;
	INT f_Adelaide64 = FALSE;
	const BYTE *poly = NULL;
	INT i, j, q;

	t0 = os_ticks();
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-regular") == 0) {
			f_regular = TRUE;
			q = atoi(argv[++i]);
			cout << "-regular " << q << endl;
			}
		else if (strcmp(argv[i], "-LunelliSce") == 0) {
			f_LunelliSce = TRUE;
			poly = "19";
			q = 16;
			cout << "-LunelliSce "<< endl;
			}
		else if (strcmp(argv[i], "-Subiaco64_1") == 0) {
			f_Subiaco64_1 = TRUE;
			q = 64;
			poly = "67";
			cout << "-Subiaco64_1 " << endl;
			}
		else if (strcmp(argv[i], "-Subiaco64_2") == 0) {
			f_Subiaco64_2 = TRUE;
			q = 64;
			poly = "67";
			cout << "-Subiaco64_2" << endl;
			}
		else if (strcmp(argv[i], "-Adelaide64") == 0) {
			f_Adelaide64 = TRUE;
			q = 64;
			poly = "67";
			cout << "-Adelaide64" << endl;
			}
		else if (strcmp(argv[i], "-stabilizer") == 0) {
			f_stabilizer = TRUE;
			cout << "-stabilizer " << endl;
			}
#if 0
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
#endif
		}
	
	cout << "q=" << q << endl;
	cout << "poly=";
	if (poly) {
		cout << poly;
		}
	else {
		cout << endl;
		}
	
	INT n = q + 2;
	INT t, f, a, rk;
	INT *coords;
	INT *pts;
	INT subset[3];
	INT Mtx[9];
	finite_field Fq;

	Fq.init_override_polynomial(q, poly, verbose_level);
	
	coords = NEW_INT(n * 3);
	pts = NEW_INT(n);

	if (f_LunelliSce) {
		LunelliSce(&Fq, pts, verbose_level);
		for (i = 0; i < n; i++) {
			PG_element_unrank_modified(Fq, coords + i * 3, 1, 3, pts[i]);
			}
		}
	else {
		coords[0] = 1;
		coords[1] = 0;
		coords[2] = 0;
		coords[3] = 0;
		coords[4] = 1;
		coords[5] = 0;
		for (t = 0; t < q; t++) {
			if (f_regular) {
				f = Fq.mult(t, t);
				}
			else if (f_Subiaco64_1) {
				f = Subiaco64_1(&Fq, t);
				}
			else if (f_Subiaco64_2) {
				f = Subiaco64_2(&Fq, t);
				}
			else if (f_Adelaide64) {
				f = Adelaide64(&Fq, t);
				}
			coords[6 + t * 3 + 0] = f;
			coords[6 + t * 3 + 1] = t;
			coords[6 + t * 3 + 2] = 1;
			}
		for (i = 0; i < n; i++) {
			PG_element_rank_modified(Fq, coords + i * 3, 1, 3, a);
			pts[i] = a;
			}
		}
	
	first_k_subset(subset, n, 3);
	while (TRUE) {
		cout << "testing subset ";
		INT_vec_print(cout, subset, 3);
		cout << endl;
		
		for (i = 0; i < 3; i++) {
			a = subset[i];
			for (j = 0; j < 3; j++) {
				Mtx[i * 3 + j] = coords[a * 3 + j];
				}
			}
		rk = Fq.Gauss_easy(Mtx, 3, 3);
		if (rk < 3) {
			cout << "not a hyperoval" << endl;
			exit(1);
			}
		if (!next_k_subset(subset, n, 3)) {
			break;
			}
		}
	cout << "passes the hyperoval test" << endl;



	if (f_stabilizer) {
		// computing stabilizer:
		action *A;
		strong_generators *Aut_gens;
		//INT t0 = os_ticks();
	
		A = new action;

		cout << "before init_projective_group" << endl;
		A->init_projective_group(3, &Fq, 
			TRUE /* f_semilinear */, 
			TRUE /* f_basis */, verbose_level);

	
		cout << "computing stabilizer" << endl;
	
		set_stabilizer_compute STAB;
		sims *Stab;
		INT nb_backtrack_nodes;
	
		cout << "computing stabilizer of the hyperoval:" << endl;
		STAB.init(A, pts, n, verbose_level);
		STAB.compute_set_stabilizer(t0, nb_backtrack_nodes, Aut_gens, verbose_level);


		Stab = Aut_gens->create_sims(verbose_level - 1);
	



		longinteger_object go, go2;
		Stab->group_order(go);
		cout << "computing stabilizer of hyperoval done, found a group of order " << go << endl;

		cout << "strong generators:" << endl;
		Aut_gens->print_generators();
	
		delete Aut_gens;
		delete Stab;
		}


	the_end(t0);

}


