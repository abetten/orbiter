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

// global data:

int t0; // the system time when the program started

int main(int argc, const char **argv);

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
	int f_test_if_geometric = FALSE;
	int test_if_geometric_depth = 0;
	int f_draw_tree = FALSE;
	int f_orbit_of = FALSE;
	int orbit_of_idx = 0;
	int f_search_subgroup = FALSE;
	int f_print_elements = FALSE;
	int f_print_elements_tex = FALSE;


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
		else if (strcmp(argv[i], "-f_draw_tree") == 0) {
			f_draw_tree = TRUE;
			cout << "-f_draw_tree " << endl;
			}
		else if (strcmp(argv[i], "-orbit_of") == 0) {
			f_orbit_of = TRUE;
			orbit_of_idx = atoi(argv[++i]);
			cout << "-orbit_of " << orbit_of_idx << endl;
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
	}



	if (!f_linear) {
		cout << "please use option -linear ..." << endl;
		exit(1);
		}


	int f_v = (verbose_level >= 1);


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
				latex_head_easy(fp);

				fp << "\\begin{sidewaystable}" << endl;
				fp << "$$" << endl;
				int_matrix_print_tex(fp, Table, n, n);
				fp << "$$" << endl;
				fp << "\\end{sidewaystable}" << endl;

				N->print_all_group_elements_tex(fp);

				latex_foot(fp);
			}
			FREE_int(Table);
		}
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
					latex_head_easy(fp);

					H->print_all_group_elements_tex(fp);

					latex_foot(fp);
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
				<< file_size(fname_orbits) << endl;


		char fname_tree_mask[1000];

		sprintf(fname_tree_mask, "%s_%%d.layered_graph", LG->prefix);

		Sch->export_tree_as_layered_graph(0 /* orbit_no */,
				fname_tree_mask,
				verbose_level - 1);

		int orbit_idx = 0;
		schreier *shallow_tree;

		cout << "computing shallow Schreier tree:" << endl;

		Sch->shallow_tree_generators(orbit_idx,
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

