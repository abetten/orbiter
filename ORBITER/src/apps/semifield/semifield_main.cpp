/*
 * semifield_main.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


// global data:

int t0; // the system time when the program started

int main(int argc, const char **argv);

int main(int argc, const char **argv)
{
	int i;
	int verbose_level = 0;
	int f_poly = FALSE;
	const char *poly = NULL;
	int f_order = FALSE;
	int order = 0;
	int f_dim_over_kernel = FALSE;
	int dim_over_kernel = 0;
	int f_prefix = FALSE;
	const char *prefix = "";
	int f_compute = FALSE;
	int compute_depth = 0;
	int f_print_representatives = FALSE;
	int print_representatives_depth = 0;
	int f_print_full_poset = FALSE;
	int f_draw_poset = FALSE;
	int f_embedded = FALSE;
	int f_sideways = FALSE;

	#if 0
	int f_break_symmetry = FALSE;
	int break_symmetry_depth = 0;
	int f_orbits_light = FALSE;
	int f_read_stabilizers = FALSE;
	int stabilizer_level = 0;
	int f_deep_search = FALSE;
	int deep_search_level = 0;
	int deep_search_first = 0;
	int deep_search_r = 0;
	int deep_search_m = 0;
	int f_compute_automorphism_group = FALSE;
	int compute_automorphism_group_level = 0;
	int compute_automorphism_group_data[1000];
	int compute_automorphism_group_data_size = 0;
	int f_reps3 = FALSE;
	int f_out_path = FALSE;
	const char *out_path = "";
	int f_compute_canonical_form = FALSE;
	int f_compute_canonical_form_all_the_way = FALSE;
	const char *compute_automorphism_group_from_file_fname = NULL;
	int compute_automorphism_group_from_file_first_column = 0;
	int f_test_file = FALSE;
	const char *test_file_fname;
	int f_print_file = FALSE;
	const char *print_file_fname;
	int print_file_first_column;
	int f_knuth_operations = FALSE;
	const char *knuth_operations_fname;
	int knuth_operations_first_column;
	int f_orbits_of_stabilizer = FALSE;
	const char *orbits_of_stabilizer_file_fname;
	int orbits_of_stabilizer_level;
	int orbits_of_stabilizer_po;
	int f_transform = FALSE;
	const char *transform_fname;
	int transform_data_sz = 0;
	int transform_data[1000];
	int f_print_over_extension_field = FALSE;
	int print_over_extension_field_q;
	const char *print_over_extension_field_fname;
	int f_HentzelRua = FALSE;
	int f_John = FALSE;
	int f_randomize = FALSE;
	const char *randomize_fname_in = NULL;
	int randomize_nb_times = 0;
	int randomize_first_column = 0;
	int f_test_invariants = FALSE;
	int test_invariants_level;
	const char *test_invariants_fname;
	int test_invariants_first_column;
	int test_invariants_nb_times;
	int f_test_canonical_form = FALSE;
	int test_canonical_form_level = 0;
	const char *test_canonical_form_fname;
	int test_canonical_form_first_column;
	int test_canonical_form_nb_times;
	int f_John_Sheekey = FALSE;
	int f_test = FALSE;
	int f_deep_search_tree = FALSE;
	int f_make_graphs = FALSE;
	int f_save_strong_generators = FALSE;
	int f_write_class_reps = FALSE;
	int f_write_reps_tex = FALSE;
#endif


	test_typedefs();

	t0 = os_ticks();
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-order") == 0) {
			f_order = TRUE;
			order = atoi(argv[++i]);
			cout << "-order " << order << endl;
			}
		else if (strcmp(argv[i], "-dim_over_kernel") == 0) {
			f_dim_over_kernel = TRUE;
			dim_over_kernel = atoi(argv[++i]);
			cout << "-dim_over_kernel " << dim_over_kernel << endl;
			}
		else if (strcmp(argv[i], "-prefix") == 0) {
			f_prefix = TRUE;
			prefix = argv[++i];
			cout << "-prefix " << prefix << endl;
			}
		else if (strcmp(argv[i], "-compute") == 0) {
			f_compute = TRUE;
			compute_depth = atoi(argv[++i]);
			cout << "-compute " << compute_depth << endl;
			}
		else if (strcmp(argv[i], "-print_representatives") == 0) {
			f_print_representatives = TRUE;
			print_representatives_depth = atoi(argv[++i]);
			cout << "-print_representatives "
					<< print_representatives_depth << endl;
			}
		else if (strcmp(argv[i], "-print_full_poset") == 0) {
			f_print_full_poset = TRUE;
			cout << "-print_full_poset" << endl;
			}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded " << endl;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways " << endl;
			}


		#if 0
		else if (strcmp(argv[i], "-break_symmetry") == 0) {
			f_break_symmetry = TRUE;
			break_symmetry_depth = atoi(argv[++i]);
			cout << "-break_symmetry " << break_symmetry_depth << endl;
			}
		else if (strcmp(argv[i], "-orbits_light") == 0) {
			f_orbits_light = TRUE;
			cout << "-orbits_light " << endl;
			}
		else if (strcmp(argv[i], "-read_stabilizers") == 0) {
			f_read_stabilizers = TRUE;
			stabilizer_level = atoi(argv[++i]);
			cout << "-read_stabilizers " << stabilizer_level << endl;
			}
		else if (strcmp(argv[i], "-deep_search") == 0) {
			f_deep_search = TRUE;
			deep_search_level = atoi(argv[++i]);
			deep_search_first = atoi(argv[++i]);
			deep_search_r = atoi(argv[++i]);
			deep_search_m = atoi(argv[++i]);
			cout << "-deep_search " << deep_search_level << " "
					<< deep_search_first << " "
					<< deep_search_r << " "
					<< deep_search_m << endl;
			}
		else if (strcmp(argv[i], "-deep_search_tree") == 0) {
			f_deep_search_tree = TRUE;
			cout << "-deep_search_tree" << endl;
			}
		else if (strcmp(argv[i], "-compute_automorphism_group") == 0) {
			f_compute_automorphism_group = TRUE;
			compute_automorphism_group_level = atoi(argv[++i]);
			while (TRUE) {
				int a;

				//a = atoi(argv[++i]);
					// dont use atoi,
					// the numbers are too large for atoi
				sscanf(argv[++i], "%d", &a);
				if (a == -1) {
					break;
					}
				compute_automorphism_group_data[
						compute_automorphism_group_data_size++] = a;
				}
			cout << "-compute_automorphism_group "
					<< compute_automorphism_group_level << endl;
			int_vec_print(cout, compute_automorphism_group_data,
					compute_automorphism_group_data_size);
			cout << endl;
			}
		else if (strcmp(argv[i], "-compute_canonical_form") == 0) {
			f_compute_canonical_form = TRUE;
			compute_automorphism_group_level = atoi(argv[++i]);
			f_compute_canonical_form_all_the_way = atoi(argv[++i]);
			compute_automorphism_group_from_file_fname = argv[++i];
			compute_automorphism_group_from_file_first_column =
					atoi(argv[++i]);
			cout << "-compute_canonical_form "
				<< compute_automorphism_group_level << " "
				<< f_compute_canonical_form_all_the_way << " "
				<< compute_automorphism_group_from_file_fname << " "
				<< compute_automorphism_group_from_file_first_column << endl;
			}
		else if (strcmp(argv[i], "-test_canonical_form") == 0) {
			f_test_canonical_form = TRUE;
			test_canonical_form_level = atoi(argv[++i]);
			test_canonical_form_fname = argv[++i];
			test_canonical_form_first_column = atoi(argv[++i]);
			test_canonical_form_nb_times = atoi(argv[++i]);
			cout << "-test_canonical_form "
				<< test_canonical_form_level << " "
				<< test_canonical_form_fname << " "
				<< test_canonical_form_first_column << " "
				<< test_canonical_form_nb_times << " "
				<< endl;
			}
		else if (strcmp(argv[i], "-test_invariants") == 0) {
			f_test_invariants = TRUE;
			test_invariants_level = atoi(argv[++i]);
			test_invariants_fname = argv[++i];
			test_invariants_first_column = atoi(argv[++i]);
			test_invariants_nb_times = atoi(argv[++i]);
			cout << "-test_invariants "
				<< test_invariants_level << " "
				<< test_invariants_fname << " "
				<< test_invariants_first_column << " "
				<< test_invariants_nb_times << " "
				<< endl;
			}
		else if (strcmp(argv[i], "-reps3") == 0) {
			f_reps3 = TRUE;
			cout << "-reps3 " << endl;
			}
		else if (strcmp(argv[i], "-out_path") == 0) {
			f_out_path = TRUE;
			out_path = argv[++i];
			cout << "-out_path " << out_path << endl;
			}
		else if (strcmp(argv[i], "-test_file") == 0) {
			f_test_file = TRUE;
			test_file_fname = argv[++i];
			cout << "-test_file " << test_file_fname << endl;
			}
		else if (strcmp(argv[i], "-print_file") == 0) {
			f_print_file = TRUE;
			print_file_fname = argv[++i];
			print_file_first_column = atoi(argv[++i]);
			cout << "-print_file " << print_file_fname << " "
					<< print_file_first_column << endl;
			}
		else if (strcmp(argv[i], "-knuth_operations") == 0) {
			f_knuth_operations = TRUE;
			knuth_operations_fname = argv[++i];
			knuth_operations_first_column = atoi(argv[++i]);
			cout << "-knuth_operations " << knuth_operations_fname
					<< " " << knuth_operations_first_column << endl;
			}
		else if (strcmp(argv[i], "-orbits_of_stabilizer") == 0) {
			f_orbits_of_stabilizer = TRUE;
			orbits_of_stabilizer_file_fname = argv[++i];
			orbits_of_stabilizer_level = atoi(argv[++i]);
			orbits_of_stabilizer_po = atoi(argv[++i]);
			cout << "-orbits_of_stabilizer "
					<< orbits_of_stabilizer_file_fname << " "
					<< orbits_of_stabilizer_level << " "
					<< orbits_of_stabilizer_po << endl;
			}
		else if (strcmp(argv[i], "-transform") == 0) {
			f_transform = TRUE;
			transform_fname = argv[++i];
			while (TRUE) {
				transform_data[transform_data_sz] = atoi(argv[++i]);
				if (transform_data[transform_data_sz] == -1) {
					break;
					}
				transform_data_sz++;
				}
			}
		else if (strcmp(argv[i], "-print_over_extension_field") == 0) {
			f_print_over_extension_field = TRUE;
			print_over_extension_field_q = atoi(argv[++i]);
			print_over_extension_field_fname = argv[++i];
			}
		else if (strcmp(argv[i], "-HentzelRua") == 0) {
			f_HentzelRua = TRUE;
			cout << "-HentzelRua " << endl;
			}
		else if (strcmp(argv[i], "-John") == 0) {
			f_John = TRUE;
			cout << "-John" << endl;
			}
		else if (strcmp(argv[i], "-randomize") == 0) {
			f_randomize = TRUE;
			randomize_nb_times = atoi(argv[++i]);
			randomize_first_column = atoi(argv[++i]);
			randomize_fname_in = argv[++i];
			}
		else if (strcmp(argv[i], "-John_Sheekey") == 0) {
			f_John_Sheekey = TRUE;
			cout << "-John_Sheekey " << endl;
			}
		else if (strcmp(argv[i], "-test") == 0) {
			f_test = TRUE;
			cout << "-test " << endl;
			}
		else if (strcmp(argv[i], "-make_graphs") == 0) {
			f_make_graphs = TRUE;
			cout << "-make_graphs " << endl;
			}
		else if (strcmp(argv[i], "-save_strong_generators") == 0) {
			f_save_strong_generators = TRUE;
			cout << "-save_strong_generators " << endl;
			}
		else if (strcmp(argv[i], "-write_class_reps") == 0) {
			f_write_class_reps = TRUE;
			cout << "-write_class_reps " << endl;
			}
		else if (strcmp(argv[i], "-write_reps_tex") == 0) {
			f_write_reps_tex = TRUE;
			cout << "-write_reps_tex " << endl;
			}
#endif

		}

	if (!f_order) {
		cout << "please use option -order <order>" << endl;
		exit(1);
		}

	int p, e, e1, n, k, q;
	number_theory_domain NT;

	NT.factor_prime_power(order, p, e);
	cout << "order = " << order << " = " << p << "^" << e << endl;

	if (f_dim_over_kernel) {
		if (e % dim_over_kernel) {
			cout << "dim_over_kernel does not divide e" << endl;
			exit(1);
			}
		e1 = e / dim_over_kernel;
		n = 2 * dim_over_kernel;
		k = dim_over_kernel;
		q = NT.i_power_j(p, e1);
		cout << "order=" << order << " n=" << n << " k=" << k
				<< " q=" << q << endl;
		}
	else {
		n = 2 * e;
		k = e;
		q = p;
		cout << "order=" << order << " n=" << n << " k=" << k
				<< " q=" << q << endl;
		}

#if 0
	if (f_test) {
		cout << "semifield_main: test mode" << endl;
		goto the_end;
		}
#endif

	{
	finite_field *F;
	semifield_classify S;
	file_io Fio;


	F = NEW_OBJECT(finite_field);
	F->init_override_polynomial(q, poly, 0 /* verbose_level */);

	cout << "before S.init" << endl;
	S.init(argc, argv, order, n, k, F,
			4 /* MINIMUM(verbose_level - 1, 2) */);
	cout << "after S.init" << endl;

	cout << "before S.init_poset_classification" << endl;
	S.init_poset_classification(
			argc, argv,
			prefix,
			verbose_level);
	cout << "after S.init_poset_classification" << endl;

	if (f_compute) {
		cout << "before S.compute_orbits" << endl;
		S.compute_orbits(compute_depth, verbose_level);
		cout << "after S.compute_orbits" << endl;
		if (f_print_full_poset) {
			double x_stretch = 0.4;
			cout << "drawing full poset" << endl;
			S.Gen->draw_poset_full(S.Gen->fname_base, k, 0 /* data1 */,
					f_embedded, f_sideways, x_stretch,
					0 /*S.Gen->verbose_level*/);
			cout << "drawing full poset done" << endl;
			}
		if (f_draw_poset) {
			cout << "drawing poset" << endl;
			S.Gen->draw_poset(S.Gen->fname_base, k, 0 /* data1 */,
					f_embedded, f_sideways,
					0 /*S.Gen->verbose_level*/);
			cout << "drawing poset done" << endl;
			}
		}
	else if (f_print_representatives) {
		orbit_rep *R;
		int *v;
		int no, nb;
		char fname[1000];

		R = NEW_OBJECT(orbit_rep);
		v = NEW_int(S.k2);

		sprintf(fname, "%s_lvl_%d", prefix,
				print_representatives_depth);

		nb = Fio.count_number_of_orbits_in_file(fname, verbose_level);

		cout << "there are " << nb << " orbit representatives "
				"in the file " << fname << endl;
		for (no = 0; no < nb; no++) {
			R->init_from_file(S.A /*A_base*/, (char *) prefix,
				print_representatives_depth, no,
				print_representatives_depth - 1/*level_of_candidates_file*/,
				semifield_classify_early_test_func,
				&S,
				verbose_level - 1
				);
			// R has: int *candidates; int nb_candidates;

			for (i = 0; i < print_representatives_depth; i++) {
				cout << R->rep[i] << " ";
				}
			cout << endl;
			for (i = 0; i < print_representatives_depth; i++) {
				cout << R->rep[i] << " = " << endl;
				S.unrank_point(v, R->rep[i], 0 /* verbose_level */);
				int_matrix_print(v, S.k, S.k);
				}
			}
		}
#if 0
	else if (f_break_symmetry) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		//S.break_symmetry(f_orbits_light, verbose_level);
		S.SFS->compute_all_levels(break_symmetry_depth,
			f_write_class_reps, f_write_reps_tex,
			f_make_graphs, f_save_strong_generators,
			verbose_level);
		cout << "Printing all semifields at level "
				<< break_symmetry_depth << endl;
		S.SFS->print_all_semifields_at_level(break_symmetry_depth);
		}
	else if (f_read_stabilizers) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		S.SFS->read_stabilizers(stabilizer_level, verbose_level);
		}
	else if (f_deep_search) {
		cout << "deep_search before S.init_semifield_starter" << endl;
		cout << "deep_search before S.init_semifield_starter "
				"deep_search_level = " << deep_search_level << endl;
		cout << "deep_search before S.init_semifield_starter "
				"f_out_path = " << f_out_path << endl;
		cout << "deep_search before S.init_semifield_starter "
				"out_path = " << out_path << endl;
		S.init_semifield_starter(f_orbits_light, verbose_level);
		cout << "deep_search before S.SFS->deep_search" << endl;
		S.SFS->deep_search(deep_search_level,
			deep_search_first + deep_search_r, deep_search_m,
			f_out_path, out_path, f_deep_search_tree, verbose_level);
		}

#if 0
	else if (f_compute_automorphism_group) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		S.SFS->compute_ag(compute_automorphism_group_level,
			compute_automorphism_group_data,
			compute_automorphism_group_data_size,
			verbose_level);
		}
#endif

	else if (f_compute_canonical_form) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		S.SFS->compute_canonical_form_from_file(
			compute_automorphism_group_level,
			f_compute_canonical_form_all_the_way,
			compute_automorphism_group_from_file_fname,
			compute_automorphism_group_from_file_first_column,
			verbose_level);
		}
	else if (f_test_canonical_form) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		S.SFS->test_canonical_form_from_file(
			test_canonical_form_level,
			test_canonical_form_fname,
			test_canonical_form_first_column,
			test_canonical_form_nb_times,
			verbose_level);
		}
	else if (f_test_invariants) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		S.SFS->test_invariants_from_file(
			test_invariants_level,
			test_invariants_fname,
			test_invariants_first_column,
			test_invariants_nb_times,
			verbose_level);
		}
	else if (f_test_file) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		S.SFS->test_file(test_file_fname, verbose_level);
		}
	else if (f_print_file) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		S.SFS->print_file(print_file_fname,
				print_file_first_column, verbose_level);
		}
	else if (f_knuth_operations) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		S.SFS->Knuth_operations(knuth_operations_fname,
				knuth_operations_first_column, verbose_level);
		}
	else if (f_reps3) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		S.SFS->compute_level_two(FALSE /* f_write_class_reps */,
				FALSE /* f_write_reps_tex */,
				FALSE /* f_make_graphs */,
				FALSE /* f_save_strong_generators */,
				verbose_level);
		S.SFS->classify_representatives_from_info_file_for_level_three(
				verbose_level);
		}
	else if (f_orbits_of_stabilizer) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		S.SFS->orbits_of_stabilizer(orbits_of_stabilizer_file_fname,
				orbits_of_stabilizer_level,
				orbits_of_stabilizer_po,
				verbose_level);
		}
	else if (f_transform) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		S.SFS->map_semifields(transform_fname,
				transform_data, verbose_level);
		}
	else if (f_print_over_extension_field) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		S.SFS->print_over_extension_field(
				print_over_extension_field_fname,
				print_over_extension_field_q,
				verbose_level);
		}
	else if (f_HentzelRua) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		int data[6];
		//int M[6 * 6];
		S.SFS->make_Hentzel_Rua_example(data, verbose_level);
		S.SFS->process_example(data, verbose_level);
		}
	else if (f_John) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		S.SFS->test_John(verbose_level);
		}
	else if (f_randomize) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		S.SFS->randomize(randomize_fname_in,
				randomize_first_column,
				randomize_nb_times,
				verbose_level);
		}
	else if (f_John_Sheekey) {
		S.init_semifield_starter(f_orbits_light, verbose_level);
		S.SFS->John_Sheekey(verbose_level);
		}
#endif


	FREE_OBJECT(F);
	}

//the_end:


	the_end_quietly(t0);
}




