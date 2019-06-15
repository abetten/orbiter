// linear_group.cpp
//
// Anton Betten
// October 18, 2018
//
//
//
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;

// global data:

int t0; // the system time when the program started

int main(int argc, const char **argv);
void report(linear_group *LG, int verbose_level);

int main(int argc, const char **argv)
{
	t0 = os_ticks();


	{
	finite_field *F;
	linear_group_description *Descr;
	linear_group *LG;


	int verbose_level = 0;
	int f_linear = FALSE;
	//int q;
	int f_orbits_on_points = FALSE;
	int f_orbits_on_subsets = FALSE;
	int orbits_on_subsets_size = 0;
	int f_draw_poset = FALSE;
	int f_classes = FALSE;
	int f_normalizer = FALSE;
	int f_report = FALSE;
	int f_test_if_geometric = FALSE;
	int test_if_geometric_depth = 0;
	int f_draw_tree = FALSE;
	int f_orbit_of = FALSE;
	int orbit_of_idx = 0;
	int f_orbits_on_set_system_from_file = FALSE;
	const char *orbits_on_set_system_from_file_fname = NULL;
	int orbits_on_set_system_first_column = 0;
	int orbits_on_set_system_number_of_columns = 0;
	int f_orbit_of_set_from_file = FALSE;
	const char *orbit_of_set_from_file_fname = NULL;
	int f_search_subgroup = FALSE;
	int f_print_elements = FALSE;
	int f_print_elements_tex = FALSE;
	int f_multiply = FALSE;
	const char *multiply_a = NULL;
	const char *multiply_b = NULL;
	int f_inverse = FALSE;
	const char *inverse_a = NULL;


	int i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-linear") == 0) {
			f_linear = TRUE;
			Descr = NEW_OBJECT(linear_group_description);
			i += Descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-linear" << endl;
			}
		else if (strcmp(argv[i], "-orbits_on_subsets") == 0) {
			f_orbits_on_subsets = TRUE;
			orbits_on_subsets_size = atoi(argv[++i]);
			cout << "-orbits_on_subsets " << orbits_on_subsets_size << endl;
			}
		else if (strcmp(argv[i], "-orbits_on_points") == 0) {
			f_orbits_on_points = TRUE;
			cout << "-orbits_on_points" << endl;
			}
		else if (strcmp(argv[i], "-test_if_geometric") == 0) {
			f_test_if_geometric = TRUE;
			test_if_geometric_depth = atoi(argv[++i]);
			cout << "-test_if_geometric" << endl;
			}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset" << endl;
			}
		else if (strcmp(argv[i], "-classes") == 0) {
			f_classes = TRUE;
			cout << "-classes" << endl;
			}
		else if (strcmp(argv[i], "-normalizer") == 0) {
			f_normalizer = TRUE;
			cout << "-normalizer" << endl;
			}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report" << endl;
			}
		else if (strcmp(argv[i], "-f_draw_tree") == 0) {
			f_draw_tree = TRUE;
			cout << "-f_draw_tree " << endl;
			}
		else if (strcmp(argv[i], "-orbit_of") == 0) {
			f_orbit_of = TRUE;
			orbit_of_idx = atoi(argv[++i]);
			cout << "-orbit_of " << orbit_of_idx << endl;
			}
		else if (strcmp(argv[i], "-orbit_of_set_from_file") == 0) {
			f_orbit_of_set_from_file = TRUE;
			orbit_of_set_from_file_fname = argv[++i];
			cout << "-orbit_of_set_from_file"
					<< orbit_of_set_from_file_fname << endl;
			}
		else if (strcmp(argv[i], "-orbits_on_set_system_from_file") == 0) {
			f_orbits_on_set_system_from_file = TRUE;
			orbits_on_set_system_from_file_fname = argv[++i];
			orbits_on_set_system_first_column = atoi(argv[++i]);
			orbits_on_set_system_number_of_columns = atoi(argv[++i]);
			cout << "-orbits_on_set_system_from_file"
					<< orbits_on_set_system_from_file_fname
					<< " " << orbits_on_set_system_first_column << " "
					<< orbits_on_set_system_number_of_columns << endl;
			}
		else if (strcmp(argv[i], "-search_subgroup") == 0) {
			f_search_subgroup = TRUE;
			cout << "-search_subgroup " << endl;
			}
		else if (strcmp(argv[i], "-print_elements") == 0) {
			f_print_elements = TRUE;
			cout << "-print_elements " << endl;
			}
		else if (strcmp(argv[i], "-print_elements_tex") == 0) {
			f_print_elements_tex = TRUE;
			cout << "-print_elements_tex " << endl;
			}
		else if (strcmp(argv[i], "-multiply") == 0) {
			f_multiply = TRUE;
			multiply_a = argv[++i];
			multiply_b = argv[++i];
			cout << "-multiply " << multiply_a << " " << multiply_b << endl;
			}
		else if (strcmp(argv[i], "-inverse") == 0) {
			f_inverse = TRUE;
			inverse_a = argv[++i];
			cout << "-inverse " << inverse_a << endl;
			}
	}



	if (!f_linear) {
		cout << "please use option -linear ..." << endl;
		exit(1);
		}


	int f_v = (verbose_level >= 1);
	file_io Fio;


	F = NEW_OBJECT(finite_field);
	F->init(Descr->input_q, 0);

	Descr->F = F;
	//q = Descr->input_q;



	LG = NEW_OBJECT(linear_group);
	if (f_v) {
		cout << "linear_group before LG->init, "
				"creating the group" << endl;
		}

	LG->init(Descr, verbose_level - 1);

	if (f_v) {
		cout << "linear_group after LG->init" << endl;
		}

	action *A;

	A = LG->A2;

	cout << "created group " << LG->prefix << endl;

	schreier *Sch;
	Sch = NEW_OBJECT(schreier);

	cout << "Strong generators are:" << endl;
	LG->Strong_gens->print_generators_tex(cout);
	cout << "Strong generators as permutations are:" << endl;
	LG->Strong_gens->print_generators_as_permutations();


	if (LG->f_has_nice_gens) {
		cout << "we have nice generators, they are:" << endl;
		LG->nice_gens->print(cout);
		cout << "$$" << endl;
		for (i = 0; i < LG->nice_gens->len; i++) {
			//cout << "Generator " << i << " / " << gens->len
			// << " is:" << endl;
			A->element_print_latex(LG->nice_gens->ith(i), cout);
			if (i < LG->nice_gens->len - 1) {
				cout << ", " << endl;
			}
			if (((i + 1) % 3) == 0 && i < LG->nice_gens->len - 1) {
				cout << "$$" << endl;
				cout << "$$" << endl;
				}
			}
		cout << "$$" << endl;
		LG->nice_gens->print_as_permutation(cout);
	}



	cout << "The group acts on the points of PG(" << Descr->n - 1
			<< "," << Descr->input_q << ")" << endl;

	if (A->degree < 1000) {
		for (i = 0; i < A->degree; i++) {
			cout << i << " & ";
			A->print_point(i, cout);
			cout << "\\\\" << endl;
		}
	}
	else {
		cout << "Too many points to print" << endl;
	}

	if (f_classes) {
		A->conjugacy_classes_and_normalizers(verbose_level);
	}

	if (f_multiply) {
		cout << "multiplying" << endl;
		cout << "A=" << multiply_a << endl;
		cout << "B=" << multiply_b << endl;
		int *Elt1;
		int *Elt2;
		int *Elt3;

		Elt1 = NEW_int(A->elt_size_in_int);
		Elt2 = NEW_int(A->elt_size_in_int);
		Elt3 = NEW_int(A->elt_size_in_int);

		A->make_element_from_string(Elt1,
				multiply_a, verbose_level);
		cout << "A=" << endl;
		A->element_print_quick(Elt1, cout);

		A->make_element_from_string(Elt2,
				multiply_b, verbose_level);
		cout << "B=" << endl;
		A->element_print_quick(Elt2, cout);

		A->element_mult(Elt1, Elt2, Elt3, 0);
		cout << "A*B=" << endl;
		A->element_print_quick(Elt3, cout);
		A->element_print_for_make_element(Elt3, cout);
		cout << endl;
		FREE_int(Elt1);
		FREE_int(Elt2);
		FREE_int(Elt3);
	}

	if (f_inverse) {
		cout << "computing the inverse" << endl;
		cout << "A=" << inverse_a << endl;
		int *Elt1;
		int *Elt2;

		Elt1 = NEW_int(A->elt_size_in_int);
		Elt2 = NEW_int(A->elt_size_in_int);

		A->make_element_from_string(Elt1,
				inverse_a, verbose_level);
		cout << "A=" << endl;
		A->element_print_quick(Elt1, cout);

		A->element_invert(Elt1, Elt2, 0);
		cout << "A^-1=" << endl;
		A->element_print_quick(Elt2, cout);
		A->element_print_for_make_element(Elt2, cout);
		cout << endl;
		FREE_int(Elt1);
		FREE_int(Elt2);
	}

	if (f_normalizer) {
		char fname_magma_prefix[1000];
		sims *G;
		sims *H;
		strong_generators *gens_N;
		longinteger_object N_order;


		sprintf(fname_magma_prefix, "%s_normalizer", LG->prefix);

		G = LG->initial_strong_gens->create_sims(verbose_level);
		H = LG->Strong_gens->create_sims(verbose_level);

		cout << "group order G = " << G->group_order_int() << endl;
		cout << "group order H = " << H->group_order_int() << endl;
		cout << "before A->normalizer_using_MAGMA" << endl;
		A->normalizer_using_MAGMA(fname_magma_prefix,
				G, H, gens_N, verbose_level);

		cout << "group order G = " << G->group_order_int() << endl;
		cout << "group order H = " << H->group_order_int() << endl;
		gens_N->group_order(N_order);
		cout << "group order N = " << N_order << endl;
		cout << "Strong generators for the normalizer of H are:" << endl;
		gens_N->print_generators_tex(cout);
		cout << "Strong generators for the normalizer of H as permutations are:" << endl;
		gens_N->print_generators_as_permutations();

		sims *N;
		int N_goi;

		N = gens_N->create_sims(verbose_level);
		N_goi = N->group_order_int();
		cout << "The elements of N are:" << endl;
		N->print_all_group_elements();

		if (N_goi < 30) {
			cout << "creating group table:" << endl;

			char fname[1000];
			int *Table;
			int n;
			N->create_group_table(Table, n, verbose_level);
			cout << "The group table of the normalizer is:" << endl;
			int_matrix_print(Table, n, n, 2);
			sprintf(fname, "normalizer_%d.tex", n);
			{
				ofstream fp(fname);
				latex_interface L;
				L.head_easy(fp);

				fp << "\\begin{sidewaystable}" << endl;
				fp << "$$" << endl;
				int_matrix_print_tex(fp, Table, n, n);
				fp << "$$" << endl;
				fp << "\\end{sidewaystable}" << endl;

				N->print_all_group_elements_tex(fp);

				L.foot(fp);
			}
			FREE_int(Table);
		}
	}

	if (f_report) {
		report(LG, verbose_level);
	}

	if (f_print_elements) {
		sims *H;

		//G = LG->initial_strong_gens->create_sims(verbose_level);
		H = LG->Strong_gens->create_sims(verbose_level);

		//cout << "group order G = " << G->group_order_int() << endl;
		cout << "group order H = " << H->group_order_int() << endl;

		int *Elt;
		longinteger_object go;
		int i, cnt;

		Elt = NEW_int(A->elt_size_in_int);
		H->group_order(go);

		cnt = 0;
		for (i = 0; i < go.as_int(); i++) {
			H->element_unrank_int(i, Elt);

			cout << "Element " << setw(5) << i << " / "
					<< go.as_int() << ":" << endl;
			A->element_print(Elt, cout);
			cout << endl;
			A->element_print_as_permutation(Elt, cout);
			cout << endl;
		}
		FREE_int(Elt);
	}

	if (f_print_elements_tex) {
		sims *H;

		//G = LG->initial_strong_gens->create_sims(verbose_level);
		H = LG->Strong_gens->create_sims(verbose_level);

		//cout << "group order G = " << G->group_order_int() << endl;
		cout << "group order H = " << H->group_order_int() << endl;

		int *Elt;
		longinteger_object go;

		Elt = NEW_int(A->elt_size_in_int);
		H->group_order(go);

		char fname[1000];

		sprintf(fname, "%s_elements.tex", LG->prefix);


				{
					ofstream fp(fname);
					latex_interface L;
					L.head_easy(fp);

					H->print_all_group_elements_tex(fp);

					L.foot(fp);
				}

		FREE_int(Elt);
	}

	if (f_search_subgroup) {
		sims *H;

		//G = LG->initial_strong_gens->create_sims(verbose_level);
		H = LG->Strong_gens->create_sims(verbose_level);

		//cout << "group order G = " << G->group_order_int() << endl;
		cout << "group order H = " << H->group_order_int() << endl;

		int *Elt;
		longinteger_object go;
		int i, cnt;

		Elt = NEW_int(A->elt_size_in_int);
		H->group_order(go);

		cnt = 0;
		for (i = 0; i < go.as_int(); i++) {
			H->element_unrank_int(i, Elt);

#if 0
			cout << "Element " << setw(5) << i << " / "
					<< go.as_int() << ":" << endl;
			A->element_print(Elt, cout);
			cout << endl;
			A->element_print_as_permutation(Elt, cout);
			cout << endl;
#endif
			if (Elt[7] == 0 && Elt[8] == 0 &&
					Elt[11] == 0 && Elt[14] == 0 &&
					Elt[12] == 0 && Elt[19] == 0 &&
					Elt[22] == 0 && Elt[23] == 0) {
				cout << "Element " << setw(5) << i << " / "
						<< go.as_int() << " = " << cnt << ":" << endl;
				A->element_print(Elt, cout);
				cout << endl;
				//A->element_print_as_permutation(Elt, cout);
				//cout << endl;
				cnt++;
			}
		}
		cout << "we found " << cnt << " group elements of the special form" << endl;

		FREE_int(Elt);

	}


	if (f_orbits_on_set_system_from_file) {
		cout << "computing orbits on set system from file "
				<< orbits_on_set_system_from_file_fname << ":" << endl;
		file_io Fio;
		int *M;
		int m, n;
		int *Table;
		int j;

		Fio.int_matrix_read_csv(orbits_on_set_system_from_file_fname, M,
				m, n, verbose_level);
		cout << "read a matrix of size " << m << " x " << n << endl;


		//orbits_on_set_system_first_column = atoi(argv[++i]);
		//orbits_on_set_system_number_of_columns = atoi(argv[++i]);


		Table = NEW_int(m * orbits_on_set_system_number_of_columns);
		for (i = 0; i < m; i++) {
			for (j = 0; j < orbits_on_set_system_number_of_columns; j++) {
				Table[i * orbits_on_set_system_number_of_columns + j] =
						M[i * n + orbits_on_set_system_first_column + j];
			}
		}
		action *A_on_sets;
		int set_size;

		set_size = orbits_on_set_system_number_of_columns;

		cout << "creating action on sets:" << endl;
		A_on_sets = A->create_induced_action_on_sets(m /* nb_sets */,
				set_size, Table,
				verbose_level);

		schreier *Sch;
		int first, a;

		cout << "computing orbits on sets:" << endl;
		A_on_sets->compute_orbits_on_points(Sch,
				LG->Strong_gens->gens, verbose_level);

		cout << "The orbit lengths are:" << endl;
		Sch->print_orbit_lengths(cout);

		cout << "The orbits are:" << endl;
		//Sch->print_and_list_orbits(cout);
		for (i = 0; i < Sch->nb_orbits; i++) {
			cout << " Orbit " << i << " / " << Sch->nb_orbits
					<< " : " << Sch->orbit_first[i] << " : " << Sch->orbit_len[i];
			cout << " : ";

			first = Sch->orbit_first[i];
			a = Sch->orbit[first + 0];
			cout << a << " : ";
			int_vec_print(cout, Table + a * set_size, set_size);
			cout << endl;
			//Sch->print_and_list_orbit_tex(i, ost);
			}
		char fname[1000];

		strcpy(fname, orbits_on_set_system_from_file_fname);
		chop_off_extension(fname);
		strcat(fname, "_orbit_reps.txt");

		{
			ofstream ost(fname);

			for (i = 0; i < Sch->nb_orbits; i++) {

				first = Sch->orbit_first[i];
				a = Sch->orbit[first + 0];
				ost << set_size;
				for (j = 0; j < set_size; j++) {
					ost << " " << Table[a * set_size + j];
				}
				ost << endl;
			}
			ost << -1 << " " << Sch->nb_orbits << endl;
		}

	}

	if (f_orbit_of_set_from_file) {

		cout << "computing orbit of set from file "
				<< orbit_of_set_from_file_fname << ":" << endl;
		file_io Fio;
		int *the_set;
		int set_sz;

		Fio.read_set_from_file(orbit_of_set_from_file_fname,
				the_set, set_sz, verbose_level);
		cout << "read a set of size " << set_sz << endl;

		orbit_of_sets *OS;

		OS = NEW_OBJECT(orbit_of_sets);

		OS->init(A, A, the_set, set_sz,
				LG->Strong_gens->gens, verbose_level);

		//OS->compute(verbose_level);

		cout << "Found an orbit of length " << OS->used_length << endl;

		int *Table;
		int orbit_length, set_size;

		cout << "before OS->get_table_of_orbits" << endl;
		OS->get_table_of_orbits_and_hash_values(Table,
				orbit_length, set_size, verbose_level);
		cout << "after OS->get_table_of_orbits" << endl;

		char str[1000];
		strcpy(str, orbit_of_set_from_file_fname);
		chop_off_extension(str);

		char fname[1000];
		sprintf(fname, "orbit_of_%s_under_%s_with_hash.csv", str, LG->prefix);
		cout << "Writing table to file " << fname << endl;
		Fio.int_matrix_write_csv(fname,
				Table, orbit_length, set_size);
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;

		FREE_int(Table);

		cout << "before OS->get_table_of_orbits" << endl;
		OS->get_table_of_orbits(Table,
				orbit_length, set_size, verbose_level);
		cout << "after OS->get_table_of_orbits" << endl;

		strcpy(str, orbit_of_set_from_file_fname);
		chop_off_extension(str);
		sprintf(fname, "orbit_of_%s_under_%s.txt", str, LG->prefix);
		cout << "Writing table to file " << fname << endl;
		{
			ofstream ost(fname);
			for (i = 0; i < orbit_length; i++) {
				ost << set_size;
				for (int j = 0; j < set_size; j++) {
					ost << " " << Table[i * set_size + j];
				}
				ost << endl;
			}
			ost << -1 << " " << orbit_length << endl;
		}
		//Fio.int_matrix_write_csv(fname,
		//		Table, orbit_length, set_size);
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;


		cout << "before FREE_OBJECT(OS)" << endl;
		FREE_OBJECT(OS);
		cout << "after FREE_OBJECT(OS)" << endl;
	}

	if (f_orbit_of) {

		schreier *Sch;
		Sch = NEW_OBJECT(schreier);

		cout << "computing orbit of point " << orbit_of_idx << ":" << endl;

		//A->all_point_orbits(*Sch, verbose_level);

		Sch->init(A);
		if (!A->f_has_strong_generators) {
			cout << "action::all_point_orbits !f_has_strong_generators" << endl;
			exit(1);
			}
		Sch->init_generators(*LG->Strong_gens->gens /* *strong_generators */);
		Sch->initialize_tables();
		Sch->compute_point_orbit(orbit_of_idx, verbose_level);


		cout << "computing orbit of point done." << endl;

		char fname_tree_mask[1000];

		sprintf(fname_tree_mask, "%s_orbit_of_point_%d.layered_graph",
				LG->prefix, orbit_of_idx);

		Sch->export_tree_as_layered_graph(0 /* orbit_no */,
				fname_tree_mask,
				verbose_level - 1);

		strong_generators *SG_stab;
		longinteger_object full_group_order;

		LG->Strong_gens->group_order(full_group_order);

		cout << "computing the stabilizer of the orbit rep:" << endl;
		SG_stab = Sch->stabilizer_orbit_rep(
				LG->A_linear,
				full_group_order,
				0 /* orbit_idx */, verbose_level);
		cout << "The stabilizer of the orbit rep has been computed:" << endl;
		SG_stab->print_generators();
		SG_stab->print_generators_tex();


		schreier *shallow_tree;

		cout << "computing shallow Schreier tree:" << endl;

#if 0
		enum shallow_schreier_tree_strategy Shallow_schreier_tree_strategy =
				shallow_schreier_tree_standard;
				//shallow_schreier_tree_Seress_deterministic;
				//shallow_schreier_tree_Seress_randomized;
				//shallow_schreier_tree_Sajeeb;
#endif
		int orbit_idx = 0;
		int f_randomized = TRUE;

		Sch->shallow_tree_generators(orbit_idx,
				f_randomized,
				shallow_tree,
				verbose_level);

		cout << "computing shallow Schreier tree done." << endl;

		sprintf(fname_tree_mask, "%s_%%d_shallow.layered_graph", LG->prefix);

		shallow_tree->export_tree_as_layered_graph(0 /* orbit_no */,
				fname_tree_mask,
				verbose_level - 1);



	} // if (f_orbit_of)


	if (f_orbits_on_points) {
		cout << "computing orbits on points:" << endl;
		//A->all_point_orbits(*Sch, verbose_level);
		A->all_point_orbits_from_generators(*Sch,
				LG->Strong_gens,
				verbose_level);



		cout << "computing orbits on points done." << endl;

		Sch->print_and_list_orbits(cout);

		char fname_orbits[1000];

		sprintf(fname_orbits, "%s_orbits.tex", LG->prefix);


		Sch->latex(fname_orbits);
		cout << "Written file " << fname_orbits << " of size "
				<< Fio.file_size(fname_orbits) << endl;


		char fname_tree_mask[1000];

		sprintf(fname_tree_mask, "%s_%%d.layered_graph", LG->prefix);

		Sch->export_tree_as_layered_graph(0 /* orbit_no */,
				fname_tree_mask,
				verbose_level - 1);

		int orbit_idx = 0;
		schreier *shallow_tree;

		cout << "computing shallow Schreier tree:" << endl;

#if 0
		enum shallow_schreier_tree_strategy Shallow_schreier_tree_strategy =
				shallow_schreier_tree_standard;
				//shallow_schreier_tree_Seress_deterministic;
				//shallow_schreier_tree_Seress_randomized;
				//shallow_schreier_tree_Sajeeb;
#endif
		int f_randomized = TRUE;

		Sch->shallow_tree_generators(orbit_idx,
				f_randomized,
				shallow_tree,
				verbose_level);

		cout << "computing shallow Schreier tree done." << endl;

		sprintf(fname_tree_mask, "%s_%%d_shallow.layered_graph", LG->prefix);

		shallow_tree->export_tree_as_layered_graph(0 /* orbit_no */,
				fname_tree_mask,
				verbose_level - 1);
	}

	if (f_orbits_on_subsets) {
		cout << "computing orbits on subsets:" << endl;
		poset_classification *PC;
		poset *Poset;

		Poset = NEW_OBJECT(poset);
		Poset->init_subset_lattice(A, A,
				A->Strong_gens,
				verbose_level);
		PC = Poset->orbits_on_k_sets_compute(
				orbits_on_subsets_size, verbose_level);


		for (int depth = 0; depth <= orbits_on_subsets_size; depth++) {
			cout << "There are " << PC->nb_orbits_at_level(depth)
					<< " orbits on subsets of size " << depth << ":" << endl;

			if (depth < orbits_on_subsets_size) {
				continue;
			}
			PC->list_all_orbits_at_level(depth,
					FALSE /* f_has_print_function */,
					NULL /* void (*print_function)(ostream &ost, int len, int *S, void *data)*/,
					NULL /* void *print_function_data*/,
					TRUE /* f_show_orbit_decomposition */,
					TRUE /* f_show_stab */,
					FALSE /* f_save_stab */,
					FALSE /* f_show_whole_orbit*/);
		}

		if (f_draw_poset) {
			{
			char fname_poset[1000];
			sprintf(fname_poset, "%s_poset_%d", LG->prefix, orbits_on_subsets_size);
			PC->draw_poset(fname_poset,
					orbits_on_subsets_size /*depth*/, 0 /* data1 */,
					TRUE /* f_embedded */,
					FALSE /* f_sideways */,
					0 /* verbose_level */);
			}
		}



		if (f_test_if_geometric) {
			int depth = test_if_geometric_depth;

			//for (depth = 0; depth <= orbits_on_subsets_size; depth++) {

			cout << "Orbits on subsets of size " << depth << ":" << endl;
			PC->list_all_orbits_at_level(depth,
					FALSE /* f_has_print_function */,
					NULL /* void (*print_function)(ostream &ost, int len, int *S, void *data)*/,
					NULL /* void *print_function_data*/,
					TRUE /* f_show_orbit_decomposition */,
					TRUE /* f_show_stab */,
					FALSE /* f_save_stab */,
					TRUE /* f_show_whole_orbit*/);
			int nb_orbits, orbit_idx;

			nb_orbits = PC->nb_orbits_at_level(depth);
			for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {

				int orbit_length;
				int *Orbit;

				cout << "before PC->get_whole_orbit depth " << depth
						<< " orbit " << orbit_idx
						<< " / " << nb_orbits << ":" << endl;
				PC->get_whole_orbit(
						depth, orbit_idx,
						Orbit, orbit_length, verbose_level);
				cout << "depth " << depth << " orbit " << orbit_idx
						<< " / " << nb_orbits << " has length "
						<< orbit_length << ":" << endl;
				int_matrix_print(Orbit, orbit_length, depth);

				action *Aut;
				longinteger_object ago;
				nauty_interface Nauty;

				Aut = Nauty.create_automorphism_group_of_block_system(
					A->degree /* nb_points */,
					orbit_length /* nb_blocks */,
					depth /* block_size */, Orbit,
					verbose_level);
				Aut->group_order(ago);
				cout << "The automorphism group of the set system "
						"has order " << ago << endl;

				FREE_OBJECT(Aut);
				FREE_int(Orbit);
			}
			if (nb_orbits == 2) {
				cout << "the number of orbits at depth " << depth
						<< " is two, we will try create_automorphism_"
						"group_of_collection_of_two_block_systems" << endl;
				int *Orbit1;
				int orbit_length1;
				int *Orbit2;
				int orbit_length2;

				cout << "before PC->get_whole_orbit depth " << depth
						<< " orbit " << orbit_idx
						<< " / " << nb_orbits << ":" << endl;
				PC->get_whole_orbit(
						depth, 0 /* orbit_idx*/,
						Orbit1, orbit_length1, verbose_level);
				cout << "depth " << depth << " orbit " << 0
						<< " / " << nb_orbits << " has length "
						<< orbit_length1 << ":" << endl;
				int_matrix_print(Orbit1, orbit_length1, depth);

				PC->get_whole_orbit(
						depth, 1 /* orbit_idx*/,
						Orbit2, orbit_length2, verbose_level);
				cout << "depth " << depth << " orbit " << 1
						<< " / " << nb_orbits << " has length "
						<< orbit_length2 << ":" << endl;
				int_matrix_print(Orbit2, orbit_length2, depth);

				action *Aut;
				longinteger_object ago;
				nauty_interface Nauty;

				Aut = Nauty.create_automorphism_group_of_collection_of_two_block_systems(
					A->degree /* nb_points */,
					orbit_length1 /* nb_blocks */,
					depth /* block_size */, Orbit1,
					orbit_length2 /* nb_blocks */,
					depth /* block_size */, Orbit2,
					verbose_level);
				Aut->group_order(ago);
				cout << "The automorphism group of the collection of two set systems "
						"has order " << ago << endl;

				FREE_OBJECT(Aut);
				FREE_int(Orbit1);
				FREE_int(Orbit2);

			} // if nb_orbits == 2
		} // if (f_test_if_geometric)


		if (f_draw_poset) {
			{
			char fname_poset[1000];
			sprintf(fname_poset, "%s_%d", LG->prefix, orbits_on_subsets_size);
			PC->draw_poset(fname_poset,
					orbits_on_subsets_size /*depth*/, 0 /* data1 */,
					TRUE /* f_embedded */,
					FALSE /* f_sideways */,
					0 /* verbose_level */);
			}
		}
	}
	}
}

void report(linear_group *LG, int verbose_level)
{
	sims *H;
	action *A;

	A = LG->A2;

	cout << "creating report for group " << LG->prefix << endl;

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = LG->Strong_gens->create_sims(verbose_level);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_int() << endl;

	int *Elt;
	longinteger_object go;

	Elt = NEW_int(A->elt_size_in_int);
	H->group_order(go);

	char fname[1000];
	char title[1000];
	const char *author = "Orbiter";
	const char *extras_for_preamble = "";

	sprintf(fname, "%s_report.tex", LG->prefix);
	sprintf(title, "The group $%s$", LG->label_latex);


	{
		ofstream fp(fname);
		latex_interface L;
		//latex_head_easy(fp);
		L.head(fp,
			FALSE /* f_book */, TRUE /* f_title */,
			title, author,
			FALSE /*f_toc*/, FALSE /* f_landscape*/, FALSE /* f_12pt*/,
			TRUE /*f_enlarged_page*/, TRUE /* f_pagenumbers*/,
			extras_for_preamble);

		//H->print_all_group_elements_tex(fp);

		longinteger_object go;
		sims *G;
		sims *H;

		G = LG->initial_strong_gens->create_sims(verbose_level);
		H = LG->Strong_gens->create_sims(verbose_level);

		G->group_order(go);

		fp << "\\noindent The order of the group $"
				<< LG->label_latex
				<< "$ is " << go << "\\\\" << endl;

		fp << "\\noindent The field ${\\mathbb F}_{"
				<< LG->F->q
				<< "}$ :\\\\" << endl;
		LG->F->cheat_sheet(fp, verbose_level);


		fp << "\\noindent The group acts on a set of size "
				<< A->degree << "\\\\" << endl;

		A->print_points(fp);

		//cout << "Order H = " << H->group_order_int() << "\\\\" << endl;

		if (LG->f_has_nice_gens) {
			cout << "Nice generators:\\\\" << endl;
			LG->nice_gens->print_tex(fp);
		} else {
			cout << "Strong generators:\\\\" << endl;
			LG->Strong_gens->print_generators_tex(fp);
		}

		A->report_basic_orbits(fp);

		sylow_structure *Syl;

		Syl = NEW_OBJECT(sylow_structure);
		Syl->init(G, verbose_level);
		Syl->report(fp);

		A->report_conjugacy_classes_and_normalizers(fp,
				verbose_level);

		L.foot(fp);
	}

	FREE_int(Elt);

}
