// orbit_node.cpp
// 
// Anton Betten
// September 23, 2017
//
//
// 
//
//

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace invariant_relations {

orbit_node::orbit_node()
{
	C = NULL;
	orbit_index = 0;
	gens = NULL;
	extra_data = NULL;
}

orbit_node::~orbit_node()
{
}

void orbit_node::init(
		classification_step *C,
		int orbit_index,
		groups::strong_generators *gens,
		long int *Rep, void *extra_data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_node::init "
				"orbit_index=" << orbit_index << " rep=";
		Lint_vec_print(cout, Rep, C->representation_sz);
		cout << endl;
	}
	orbit_node::C = C;
	orbit_node::orbit_index = orbit_index;
	orbit_node::gens = gens;
	Lint_vec_copy(
			Rep,
			C->Rep + orbit_index * C->representation_sz,
			C->representation_sz);
	orbit_node::extra_data = extra_data;
}


void orbit_node::write_file(
		std::ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "orbit_node::write_file" << endl;
	}
	gens->write_to_file_binary(fp, verbose_level);

	if (f_v) {
		cout << "orbit_node::write_file finished" << endl;
	}
}

void orbit_node::read_file(
		std::ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "orbit_node::read_file" << endl;
	}
	gens = NEW_OBJECT(groups::strong_generators);
	gens->read_from_file_binary(C->A, fp, verbose_level);

	if (f_v) {
		cout << "orbit_node::read_file finished" << endl;
	}
}



}}}


