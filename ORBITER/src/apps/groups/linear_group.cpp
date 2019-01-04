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
	int f_orbits_on_subsets = FALSE;
	int orbits_on_subsets_size = 0;
	int f_draw_poset = FALSE;


	int i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-linear") == 0) {
			f_linear = TRUE;
			Descr = NEW_OBJECT(linear_group_description);
			i += Descr->read_arguments(argc - (i - 1),
				argv + i, verbose_level);

			cout << "-linear" << endl;
			}
		else if (strcmp(argv[i], "-orbits_on_subsets") == 0) {
			f_orbits_on_subsets = TRUE;
			orbits_on_subsets_size = atoi(argv[++i]);
			cout << "-orbits_on_subsets" << orbits_on_subsets_size << endl;
			}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset" << endl;
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
		LG->nice_gens->print_as_permutation(cout);
	}



	for (i = 0; i < A->degree; i++) {
		cout << i << " & ";
		A->print_point(i, cout);
		cout << "\\\\" << endl;
	}
	cout << "computing orbits on points:" << endl;
	//A->all_point_orbits(*Sch, verbose_level);
	A->all_point_orbits_from_generators(*Sch,
			LG->Strong_gens,
			verbose_level);



	cout << "computing orbits on points done." << endl;

	Sch->print_and_list_orbits(cout);

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

	if (f_orbits_on_subsets) {
		cout << "computing orbits on subsets:" << endl;
		poset_classification *PC;
		poset *Poset;

		Poset = NEW_OBJECT(poset);
		Poset->init_subset_lattice(A, A,
				A->Strong_gens,
				verbose_level);
		PC = orbits_on_k_sets_compute(Poset,
				orbits_on_subsets_size, verbose_level);
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

