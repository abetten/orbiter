// poset_description.cpp
//
// Anton Betten
// November 19, 2018

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


poset_description::poset_description()
{
	f_subset_lattice = false;
	f_subspace_lattice = false;
	dimension = 0;
	q = 0;
	f_independence_condition = false;
	independence_condition_value = 0;
}


poset_description::~poset_description()
{
}

#if 0
void poset_description::read_arguments_from_string(
		const char *str, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	int argc;
	char **argv;
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "poset_description::read_arguments_from_string" << endl;
	}
	ST.chop_string(str, argc, argv);

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
			f_subset_lattice = true;
			cout << "-subset_lattice " << endl;
			}
		else if (strcmp(argv[i], "-subspace_lattice") == 0) {
			f_subspace_lattice = true;
			dimension = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-subset_lattice " << dimension << " " << q << endl;
			}
		else if (strcmp(argv[i], "-independence_condition") == 0) {
			f_independence_condition = true;
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
#endif

}}}


