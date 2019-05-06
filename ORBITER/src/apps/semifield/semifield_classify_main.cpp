/*
 * semifield_classify_main.cpp
 *
 *  Created on: Apr 18, 2019
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
	int f_draw_poset = FALSE;
	int f_embedded = FALSE;
	int f_sideways = FALSE;
	int f_report = FALSE;
	int f_memory_debug = FALSE;



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
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report " << endl;
		}
		else if (strcmp(argv[i], "-memory_debug") == 0) {
			f_memory_debug = TRUE;
			cout << "-memory_debug " << endl;
		}
	}




	if (!f_order) {
		cout << "please use option -order <order>" << endl;
		exit(1);
		}

	if (f_memory_debug) {
		cout << "before start_memory_debug" << endl;
		start_memory_debug();
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
	semifield_classify *SC;
	semifield_level_two *L2;
	semifield_lifting *L3;


	F = NEW_OBJECT(finite_field);
	F->init_override_polynomial(q, poly, 0 /* verbose_level */);

	SC = NEW_OBJECT(semifield_classify);
	cout << "before SC->init" << endl;
	SC->init(argc, argv, order, n, k, F,
			4 /* MINIMUM(verbose_level - 1, 2) */);
	cout << "after SC->init" << endl;

	L2 = NEW_OBJECT(semifield_level_two);
	cout << "before L2->init" << endl;
	L2->init(SC, verbose_level);
	cout << "after L2->init" << endl;


	cout << "before L2->compute_level_two" << endl;
	L2->compute_level_two(verbose_level);
	cout << "after L2->compute_level_two" << endl;


	L3 = NEW_OBJECT(semifield_lifting);
	cout << "before L3->compute_level_three" << endl;
	L3->init_level_three(L2,
			SC->f_level_three_prefix, SC->level_three_prefix,
			verbose_level);
	cout << "after L3->compute_level_three" << endl;

	cout << "before L3->compute_level_three" << endl;
	L3->compute_level_three(verbose_level);
	cout << "after L3->compute_level_three" << endl;


	if (f_report) {

		cout << "before report" << endl;
		char fname[1000];
		sprintf(fname, "Semifields_%d.tex", order);

		{
			ofstream fp(fname);
			latex_interface L;

			L.head_easy(fp);


			cout << "before L2->C->report" << endl;
			L2->C->report(fp, verbose_level);
			cout << "before L2->print_representatives" << endl;
			L2->print_representatives(fp, verbose_level);
			cout << "after L2->print_representatives" << endl;

			L.foot(fp);
		}
		cout << "after report" << endl;
	}

	if (f_memory_debug) {
		cout << "before global_mem_object_registry.dump_to_csv_file" << endl;
		global_mem_object_registry.dump_to_csv_file("memory.csv");
		cout << "after global_mem_object_registry.dump_to_csv_file" << endl;
	}

	cout << "before FREE_OBJECT(L2)" << endl;
	FREE_OBJECT(L2);
	cout << "before FREE_OBJECT(SC)" << endl;
	FREE_OBJECT(SC);
	cout << "before FREE_OBJECT(F)" << endl;
	FREE_OBJECT(F);
	cout << "before leaving scope" << endl;
	}
	cout << "after leaving scope" << endl;
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


//the_end:


	the_end_quietly(t0);
}




