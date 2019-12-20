// linear_set_main.cpp
// 
// Anton Betten
// July 8, 2014
//
// based on subspace_orbits.cpp
//
//

#include <orbiter.h>

using namespace std;



using namespace orbiter;
using namespace orbiter::top_level;


// global data:

int t0; // the system time when the program started

int main(int argc, const char **argv);



int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int i;
	int f_n = FALSE;
	int n = 0;
	int f_s = FALSE;
	int s = 0;
	int f_q = FALSE;
	int q;
	int f_depth = FALSE;
	int depth = 0;
	int f_print_generators = FALSE;
	int f_poly_q = FALSE;
	const char *poly_q = NULL;
	int f_poly_Q = FALSE;
	const char *poly_Q = NULL;
	int f_semilinear = FALSE;
	int f_classify = FALSE;
	int f_read = FALSE;
	int f_intersections = FALSE;
	int f_draw_poset = FALSE;
	int f_classify_secondary = FALSE;
	int f_stabilizer = FALSE;
	int stabilizer_level = 0;
	int stabilizer_orbit_at_level = 0;
	int f_identify = FALSE;
	int f_print_level = FALSE;
	int print_level = 0;
	int f_embedded = FALSE;
	int f_sideways = FALSE;
	int f_draw_poset_full = FALSE;
	int draw_poset_full_level = 0;
	int f_plesken = FALSE;
	os_interface Os;

	t0 = Os.os_ticks();
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-s") == 0) {
			f_s = TRUE;
			s = atoi(argv[++i]);
			cout << "-s " << s << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-semilinear") == 0) {
			f_semilinear = TRUE;
			cout << "-semilinear " << endl;
			}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = atoi(argv[++i]);
			cout << "-depth " << depth << endl;
			}
		else if (strcmp(argv[i], "-print_generators") == 0) {
			f_print_generators = TRUE;
			cout << "-print_generators " << endl;
			}
		else if (strcmp(argv[i], "-poly_q") == 0) {
			f_poly_q = TRUE;
			poly_q = argv[++i];
			cout << "-poly_q " << poly_q << endl;
			}
		else if (strcmp(argv[i], "-poly_Q") == 0) {
			f_poly_Q = TRUE;
			poly_Q = argv[++i];
			cout << "-poly_Q " << poly_Q << endl;
			}
		else if (strcmp(argv[i], "-classify") == 0) {
			f_classify = TRUE;
			cout << "-classify " << endl;
			}
		else if (strcmp(argv[i], "-read") == 0) {
			f_read = TRUE;
			cout << "-read " << endl;
			}
		else if (strcmp(argv[i], "-intersections") == 0) {
			f_intersections = TRUE;
			cout << "-intersections " << endl;
			}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
			}
		else if (strcmp(argv[i], "-classify_secondary") == 0) {
			f_classify_secondary = TRUE;
			cout << "-classify_secondary " << endl;
			}
		else if (strcmp(argv[i], "-stabilizer") == 0) {
			f_stabilizer = TRUE;
			stabilizer_level = atoi(argv[++i]);
			stabilizer_orbit_at_level = atoi(argv[++i]);
			cout << "-stabilizer " << stabilizer_level 
				<< " " << stabilizer_orbit_at_level << endl;
			}
		else if (strcmp(argv[i], "-identify") == 0) {
			f_identify = TRUE;
			cout << "-identify " << endl;
			}
		else if (strcmp(argv[i], "-print_level") == 0) {
			f_print_level = TRUE;
			print_level = atoi(argv[++i]);
			cout << "-print_level " << print_level << endl;
			}
		else if (strcmp(argv[i], "-draw_poset_full") == 0) {
			f_draw_poset_full = TRUE;
			draw_poset_full_level = atoi(argv[++i]);
			cout << "-draw_poset_full " << draw_poset_full_level << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded " << endl;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways " << endl;
			}
		else if (strcmp(argv[i], "-plesken") == 0) {
			f_plesken = TRUE;
			cout << "-plesken " << endl;
			}
		
		}
	if (!f_n) {
		cout << "please use -n option" << endl;
		exit(1);
		}
	if (!f_s) {
		cout << "please use -s option" << endl;
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

	int f_v = (verbose_level >= 1);
	


#if 1

	//f_memory_debug = TRUE;
	//f_memory_debug_verbose = TRUE;


	linear_set_classify *LS;

	LS = NEW_OBJECT(linear_set_classify);

	LS->init(argc, argv, s, n, q, poly_q, poly_Q, depth, f_identify, verbose_level);

	if (f_classify) {
		LS->do_classify(verbose_level);
		if (f_print_level) {
			LS->print_orbits_at_level(print_level);
			}
		if (f_draw_poset_full) {
			double x_stretch = 0.4;
			LS->Gen->draw_poset_full(LS->Gen->fname_base, 
				draw_poset_full_level, 0 /* data */,
				f_embedded, f_sideways, x_stretch,
				0 /*verbose_level*/);
			}
		if (f_plesken) {
			latex_interface L;
			int *P;
			int N;
			cout << "computing Plesken matrices:" << endl;
			LS->Gen->Plesken_matrix_up(depth, P, N, 0 /*verbose_level - 2*/);
			cout << "Plesken matrix up:" << endl;
			L.int_matrix_print_tex(cout, P, N, N);

			FREE_int(P);
			LS->Gen->Plesken_matrix_down(depth, P, N, 0/*verbose_level - 2*/);
			cout << "Plesken matrix down:" << endl;
			L.int_matrix_print_tex(cout, P, N, N);

			FREE_int(P);
			}
		}
	else if (f_read) {
		LS->read_data_file(depth, verbose_level);
		}


	strong_generators *strong_gens = NULL;

	if (f_intersections) {
		LS->calculate_intersections(depth, verbose_level);
		}
	else if (f_stabilizer) {
		LS->compute_stabilizer_of_linear_set(argc, argv, 
			stabilizer_level, stabilizer_orbit_at_level, 
			strong_gens, 
			verbose_level);
		longinteger_object go;
		
		strong_gens->group_order(go);
		cout << "Generators for the stabilizer of order " << go << " are:" << endl;
		strong_gens->print_generators();
		}
	

	if (f_classify_secondary) {
		if (strong_gens == NULL) {
			cout << "in order to classify the secondary subspaces, "
					"we need the stabilizer computed first" << endl;
			exit(1);
			}
		LS->classify_secondary(argc, argv, 
			stabilizer_level, stabilizer_orbit_at_level, 
			strong_gens, 
			verbose_level);

		int nb_W, cnt_w;

		nb_W = LS->Gen2->nb_orbits_at_level(LS->secondary_depth);

		for (cnt_w = 0; cnt_w < nb_W; cnt_w++) {
			cout << "Constructing semifield " << cnt_w << " / " << nb_W << ":" << endl;
			LS->construct_semifield(cnt_w, verbose_level);
			}
		}

	if (strong_gens) {
		FREE_OBJECT(strong_gens);
		}
	
	if (f_draw_poset) {
		if (f_v) {
			cout << "arc_generator::compute_starter before gen->draw_poset" << endl;
			}
		LS->Gen->draw_poset(LS->Gen->fname_base, depth, 0 /* data1 */,
				f_embedded, f_sideways, 0 /* gen->verbose_level */);
		}

	
	FREE_OBJECT(LS);

	//registry_dump_sorted();
#endif
	
	the_end(t0);
	//cout << "memory_count_allocate=" << memory_count_allocate << endl;
}


