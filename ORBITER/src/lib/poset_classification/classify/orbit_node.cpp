// orbit_node.C
// 
// Anton Betten
// September 23, 2017
//
//
// 
//
//

#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "poset_classification/poset_classification.h"

namespace orbiter {
namespace classification {

orbit_node::orbit_node()
{
	null();
}

orbit_node::~orbit_node()
{
	freeself();
}

void orbit_node::null()
{
}

void orbit_node::freeself()
{
	null();
}

void orbit_node::init(classification_step *C, int orbit_index,
	strong_generators *gens, int *Rep, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_node::init "
				"orbit_index=" << orbit_index << " rep=";
		int_vec_print(cout, Rep, C->representation_sz);
		cout << endl;
		}
	orbit_node::C = C;
	orbit_node::orbit_index = orbit_index;
	orbit_node::gens = gens;
	int_vec_copy(Rep,
			C->Rep + orbit_index * C->representation_sz,
			C->representation_sz);
}

void orbit_node::write_file(ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "orbit_node::write_file" << endl;
		}
	gens->write_to_file_binary(fp, 0 /* verbose_level */);

	if (f_v) {
		cout << "orbit_node::write_file finished" << endl;
		}
}

void orbit_node::read_file(ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "orbit_node::read_file" << endl;
		}
	gens = NEW_OBJECT(strong_generators);
	gens->read_from_file_binary(C->A, fp, 0 /* verbose_level */);

	if (f_v) {
		cout << "orbit_node::read_file finished" << endl;
		}
}


}}

