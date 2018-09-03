// polar.C
// 
// Anton Betten
// Feb 8, 2010
//
//
// 
//
//

#include "orbiter.h"


// global data:

int t0; // the system time when the program started

int main(int argc, const char **argv);
void init_orthogonal(action *A, int epsilon, int n, finite_field *F, int verbose_level);



int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int i, j;
	int f_epsilon = FALSE;
	int epsilon = 0;
	int f_n = FALSE;
	int n = 0;
	int f_k = FALSE;
	int k = 0;
	int f_q = FALSE;
	int q;
	int f_depth = FALSE;
	int depth = 0;
	int f_group_generators = FALSE;
	int group_generators_data[1000];
	int group_generators_data_size = 0;
	int f_group_order_target = FALSE;
	const char *group_order_target;
	int f_KM = FALSE;
	int KM_t = 0;
	int KM_k = 0;
	int f_group_generators_by_base_image = FALSE;
	int f_print_generators = FALSE;
	int f_cosets = FALSE;
	int cosets_depth, cosets_orbit_idx;
	int f_dual_polar = FALSE;
	int dual_polar_depth, dual_polar_orbit_idx;
	int f_show_stabilizer = FALSE;
	int show_stabilizer_depth, show_stabilizer_orbit_idx;
	int f_action_on_maximals = FALSE;
	
 	t0 = os_ticks();
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-epsilon") == 0) {
			f_epsilon = TRUE;
			epsilon = atoi(argv[++i]);
			cout << "-epsilon " << epsilon << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			k = atoi(argv[++i]);
			cout << "-k " << k << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = atoi(argv[++i]);
			cout << "-depth " << depth << endl;
			}
		else if (strcmp(argv[i], "-cosets") == 0) {
			f_cosets = TRUE;
			cosets_depth = atoi(argv[++i]);
			cosets_orbit_idx = atoi(argv[++i]);
			cout << "-cosets " << cosets_depth << " " << cosets_orbit_idx << endl;
			}
		else if (strcmp(argv[i], "-dual_polar") == 0) {
			f_dual_polar = TRUE;
			dual_polar_depth = atoi(argv[++i]);
			dual_polar_orbit_idx = atoi(argv[++i]);
			cout << "-dual_polar " << dual_polar_depth << " " << dual_polar_orbit_idx << endl;
			}
		else if (strcmp(argv[i], "-show_stabilizer") == 0) {
			f_show_stabilizer = TRUE;
			show_stabilizer_depth = atoi(argv[++i]);
			show_stabilizer_orbit_idx = atoi(argv[++i]);
			cout << "-show_stabilizer " << show_stabilizer_depth << " " << show_stabilizer_orbit_idx << endl;
			}
		else if (strcmp(argv[i], "-action_on_maximals") == 0) {
			f_action_on_maximals = TRUE;
			cout << "-action_on_maximals" << endl;
			}
		else if (strcmp(argv[i], "-G") == 0) {
			f_group_generators = TRUE;
			for (j = 0; ; j++) {
				group_generators_data[j] = atoi(argv[++i]);
				if (group_generators_data[j] == -1)
					break;
				}
			group_generators_data_size = j;
			cout << "-G ";
			int_vec_print(cout, group_generators_data, group_generators_data_size);
			cout << endl;
			}
		else if (strcmp(argv[i], "-GB") == 0) {
			f_group_generators_by_base_image = TRUE;
			for (j = 0; ; j++) {
				group_generators_data[j] = atoi(argv[++i]);
				if (group_generators_data[j] == -1)
					break;
				}
			group_generators_data_size = j;
			cout << "-GB ";
			int_vec_print(cout, group_generators_data, group_generators_data_size);
			cout << endl;
			}
		else if (strcmp(argv[i], "-GBstarting1") == 0) {
			f_group_generators_by_base_image = TRUE;
			for (j = 0; ; j++) {
				group_generators_data[j] = atoi(argv[++i]);
				if (group_generators_data[j] == -1)
					break;
				group_generators_data[j]--;
				}
			group_generators_data_size = j;
			cout << "-GB ";
			int_vec_print(cout, group_generators_data, group_generators_data_size);
			cout << endl;
			}
		else if (strcmp(argv[i], "-GO") == 0) {
			f_group_order_target = TRUE;
			group_order_target = argv[++i];
			cout << "-GO " << group_order_target << endl;
			}
		
		else if (strcmp(argv[i], "-KM") == 0) {
			f_KM = TRUE;
			KM_t = atoi(argv[++i]);
			KM_k = atoi(argv[++i]);
			cout << "-KM " << KM_t << " " << KM_k << endl;
			}
		else if (strcmp(argv[i], "-print_generators") == 0) {
			f_print_generators = TRUE;
			cout << "-print_generators " << endl;
			}
		}
	if (!f_epsilon) {
		cout << "please use -epsilon option" << endl;
		exit(1);
		}
	if (!f_n) {
		cout << "please use -n option" << endl;
		exit(1);
		}
	if (!f_k) {
		cout << "please use -k option" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use -q option" << endl;
		exit(1);
		}
	if (!f_depth) {
		cout << "please use -depth option" << endl;
		exit(1);
		}

	polar P;
	finite_field *F;
	action *A;
	action_on_orthogonal *AO;
	orthogonal *O;

	F = new finite_field;
	A = new action;
	
	F->init(q, 0);
	init_orthogonal(A, epsilon, n, F, verbose_level);
	
	AO = A->G.AO;
	O = AO->O;
	
	cout << "after init_orthogonal" << endl;

	P.init(argc, argv, A, O, epsilon, n, k, F, depth, verbose_level);

#if 0
	{
	int f_vv = (verbose_level >= 2);
	int the_set[] = {0,1,2,7,16,49};
	int set_size = 6;
	set_stabilizer_compute *S;
	sims *Stab;
	vector_ge * Stab_SG;
	longinteger_object go;
	
	S = new set_stabilizer_compute;
	
	//sims Stab;
	int nb_backtrack_nodes;

	Stab = new sims;
	Stab_SG = new vector_ge;
	
	if (f_vv) {
		cout << "initializing set_stabilizer_compute:" << endl;
		}
	S->init(A, Stab, the_set, set_size, verbose_level - 3);
	
	if (f_vv) {
		cout << "computing set stabilizer:" << endl;
		}
	S->compute_set_stabilizer(t0, nb_backtrack_nodes, verbose_level - 2);
	
	if (f_vv) {
		cout << "stabilizer has been computed" << endl;
		cout << "nb_backtrack_nodes=" << nb_backtrack_nodes << endl;
		}

	Stab->group_order(go);
	cout << "the stabilizer has order " << go << endl;
	cout << endl;
	Stab->print_transversal_lengths();
	cout << endl;
	cout << "generators:" << endl;
	Stab->gens.print(cout);
	cout << "######################################" << endl;
	}
#endif









	P.f_print_generators = f_print_generators;
	if (f_group_generators) {
		P.init_group(group_generators_data, group_generators_data_size, 
			f_group_order_target, group_order_target, verbose_level);
		}
	if (f_group_generators_by_base_image) {
		P.init_group_by_base_images(
			group_generators_data, group_generators_data_size, 
			f_group_order_target, group_order_target, verbose_level);
		}
	P.init2(verbose_level);

	
	P.compute_orbits(t0, verbose_level);
	
	cout << "we found " << P.nb_orbits << " orbits at depth " << k << endl;
	
	if (f_KM) {
		P.compute_Kramer_Mesner_matrix(KM_t, KM_k, verbose_level);
		}
	if (f_cosets) {
		P.compute_cosets(cosets_depth, cosets_orbit_idx, verbose_level);
		}
	if (f_dual_polar) {
		longinteger_object *Rank_maximals;
		int nb_maximals;
		
		P.dual_polar_graph(dual_polar_depth, dual_polar_orbit_idx, Rank_maximals, nb_maximals, verbose_level);
		delete [] Rank_maximals;		
		}
	if (f_show_stabilizer) {
		P.show_stabilizer(show_stabilizer_depth, show_stabilizer_orbit_idx, verbose_level);
		}
	if (f_action_on_maximals) {
		longinteger_object *Rank_maximals;
		int *Rank_maximals_int;
		int nb_maximals;
		grassmann *Grass;
		action_on_grassmannian *AG;
		action A2;
		
		P.dual_polar_graph(k, 0, Rank_maximals, nb_maximals, verbose_level);

		Rank_maximals_int = NEW_int(nb_maximals);
		for (i = 0; i < nb_maximals; i++) {
			Rank_maximals_int[i] = Rank_maximals[i].as_int();
			}
		cout << "we have " << nb_maximals << " maximals:" << endl;
		for (i = 0; i < nb_maximals; i++) {
			cout << setw(4) << i << " : " << setw(10) << Rank_maximals_int[i] << endl;
			}

		
		cout << "setting up action on grassmannian:" << endl;
		cout << "before AG->init:" << endl;
		AG = new action_on_grassmannian;
		Grass = new grassmann;
		
		Grass->init(P.n, k, P.F, verbose_level);
		AG->init(*P.A, Grass, verbose_level);
		cout << "before A2.induced_action_on_grassmannian:" << endl;
		A2.induced_action_on_grassmannian(P.A, AG, FALSE /* f_induce_action */, P.A->Sims, verbose_level);

		sims *S;
		longinteger_object go;
		int goi, i, order;
		int *Elt;

		Elt = NEW_int(P.A->elt_size_in_int);
		S = P.A->Sims;
		S->group_order(go);	
		cout << "group of order " << go << endl;
		goi = go.as_int();

#if 0
		for (i = 0; i < goi; i++) {
			S->element_unrank_int(i, Elt);
			order = A->element_order(Elt);
			cout << "element " << i << " of order " << order << ":" << endl;
			A->element_print_quick(Elt, cout);
			A2.element_print_as_permutation(Elt, cout);
			cout << endl;
			}
#endif

		action *A_restr;

		A_restr = new action;
		A_restr->induced_action_by_restriction(A2, 
			FALSE /* f_induce_action */, S, 
			nb_maximals, Rank_maximals_int, verbose_level);
		for (i = 0; i < goi; i++) {
			S->element_unrank_int(i, Elt);
			order = A->element_order(Elt);
			cout << "element " << i << " of order " << order << ":" << endl;
			A->element_print_quick(Elt, cout);
			cout << "in the action on points:" << endl;
			P.A->element_print_as_permutation(Elt, cout);
			cout << endl;
			cout << "in the action on the maximals:" << endl;
			A_restr->element_print_as_permutation(Elt, cout);
			cout << endl;
			}
	
		// do not free AG
		delete A_restr;
		FREE_int(Elt);
		FREE_int(Rank_maximals_int);
		delete [] Rank_maximals;		
		}
	
	delete A;
	
	the_end(t0);
}

void init_orthogonal(action *A, int epsilon, int n, finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	const char *override_poly;
	int p, hh, f_semilinear;
	int f_basis = TRUE;
	int q = F->q;

	if (f_v) {
		cout << "init_orthogonal epsilon=" << epsilon << " n=" << n << " q=" << q << endl;
		}

	is_prime_power(q, p, hh);
	if (hh > 1) {
		f_semilinear = TRUE;
		}
	else {
		f_semilinear = FALSE;
		}

	override_poly = override_polynomial_subfield(q);
	if (f_v && override_poly) {
		cout << "override_poly=" << override_poly << endl;
		}
	if (f_v) {
		cout << "f_semilinear=" << f_semilinear << endl;
		}

	A->init_orthogonal_group(epsilon, 
		n, F, 
		TRUE /* f_on_points */, FALSE /* f_on_lines */, FALSE /* f_on_points_and_lines */, 
		f_semilinear, f_basis, 
		0/*verbose_level*/);

#if 0
	matrix_group *M;
	orthogonal *O;

	M = A->subaction->G.matrix_grp;
	O = M->O;
#endif

	if (f_vv) {
		A->print_base();
		}
	
	
	if (f_v) {
		cout << "init_orthogonal finished, created action:" << endl;
		A->print_info();
		}
}





