// poset_description.C
//
// Anton Betten
// November 19, 2018

#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "poset_classification/poset_classification.h"

namespace orbiter {
namespace classification {


poset_description::poset_description()
{
	null();
}

poset_description::~poset_description()
{
	freeself();
}

void poset_description::null()
{
	f_subset_lattice = FALSE;
	f_subspace_lattice = FALSE;
	label[0] = 0;
	f_independence_condition = FALSE;
	independence_condition_value = 0;
}

void poset_description::freeself()
{
	null();
}

void poset_description::read_arguments_from_string(
		const char *str, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	int argc;
	char **argv;
	int i;

	if (f_v) {
		cout << "poset_description::read_arguments_from_string" << endl;
	}
	chop_string(str, argc, argv);

	if (f_vv) {
		cout << "argv:" << endl;
		for (i = 0; i < argc; i++) {
			cout << i << " : " << argv[i] << endl;
		}
	}


	read_arguments(
		argc, (const char **) argv,
		verbose_level);

	for (i = 0; i < argc; i++) {
		FREE_char(argv[i]);
	}
	FREE_pchar(argv);
	if (f_v) {
		cout << "poset_description::read_arguments_from_string "
				"done" << endl;
	}
}


int poset_description::read_arguments(
	int argc, const char **argv,
	int verbose_level)
{
	int i;

	cout << "poset_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

#if 0
		if (argv[i][0] != '-') {
			continue;
			}
#endif

		if (strcmp(argv[i], "-subset_lattice") == 0) {
			f_subset_lattice = TRUE;
			cout << "-subset_lattice " << endl;
			}
		else if (strcmp(argv[i], "-subspace_lattice") == 0) {
			f_subspace_lattice = TRUE;
			dimension = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-subset_lattice " << dimension << " " << q << endl;
			}
		else if (strcmp(argv[i], "-independence_condition") == 0) {
			f_independence_condition = TRUE;
			//independence_condition_max_depth = atoi(argv[++i]);
			independence_condition_value = atoi(argv[++i]);
			cout << "-independence_condition "
					//<< independence_condition_max_depth << " "
					<< independence_condition_value
					<< endl;
			}
		else {
			cout << "poset_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "poset_description::read_arguments done" << endl;
	return i;
}

}}

