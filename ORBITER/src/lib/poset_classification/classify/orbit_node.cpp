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
#include "groups_and_group_actions/groups_and_group_actions.h"
#include "poset_classification/poset_classification.h"

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

void orbit_node::init(classification *C, INT orbit_index, 
	strong_generators *gens, INT *Rep, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_node::init "
				"orbit_index=" << orbit_index << " rep=";
		INT_vec_print(cout, Rep, C->representation_sz);
		cout << endl;
		}
	orbit_node::C = C;
	orbit_node::orbit_index = orbit_index;
	orbit_node::gens = gens;
	INT_vec_copy(Rep, C->Rep + orbit_index * C->representation_sz,
			C->representation_sz);
}

void orbit_node::write_file(ofstream &fp, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "orbit_node::write_file" << endl;
		}
	gens->write_to_file_binary(fp, 0 /* verbose_level */);

	if (f_v) {
		cout << "orbit_node::write_file finished" << endl;
		}
}

void orbit_node::read_file(ifstream &fp, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "orbit_node::read_file" << endl;
		}
	gens = new strong_generators;
	gens->read_from_file_binary(C->A, fp, 0 /* verbose_level */);

	if (f_v) {
		cout << "orbit_node::read_file finished" << endl;
		}
}


