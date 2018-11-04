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
	int q;


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
	}



	if (!f_linear) {
		cout << "please use option -linear ..." << endl;
		exit(1);
		}


	int f_v = (verbose_level >= 1);


	F = NEW_OBJECT(finite_field);
	F->init(Descr->input_q, 0);

	Descr->F = F;
	q = Descr->input_q;



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

	cout << "computing orbits on points:" << endl;
	A->all_point_orbits(*Sch, verbose_level);


	cout << "computing orbits on points done." << endl;

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
}

