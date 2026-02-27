// poset_classification.cpp
//
// Anton Betten
// December 29, 2003

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {

poset_of_orbits *poset_classification::get_Poo()
{
	return Poo;
}

orbit_tracer *poset_classification::get_Orbit_tracer()
{
	return Orbit_tracer;
}

std::string &poset_classification::get_problem_label_with_path()
{
	return problem_label_with_path;
}

std::string &poset_classification::get_problem_label()
{
	return problem_label;
}

int poset_classification::first_node_at_level(
		int i)
{
	return Poo->first_node_at_level(i);
}

poset_orbit_node *poset_classification::get_node(
		int node_idx)
{
	return Poo->get_node(node_idx);
}

data_structures_groups::vector_ge *poset_classification::get_transporter()
{
	return Orbit_tracer->get_transporter();
}

int *poset_classification::get_transporter_i(
		int i)
{
	return Orbit_tracer->get_transporter()->ith(i);
}

int poset_classification::get_sz()
{
	return sz;
}

int poset_classification::get_max_set_size()
{
	return max_set_size;
}

long int *poset_classification::get_S()
{
	return set_S;
}

long int *poset_classification::get_set_i(
		int i)
{
	return Orbit_tracer->get_set_i(i);
}

long int *poset_classification::get_set0()
{
	return Poo->get_set0();
}

long int *poset_classification::get_set1()
{
	return Poo->get_set1();
}

long int *poset_classification::get_set3()
{
	return Poo->get_set3();
}

int poset_classification::allowed_to_show_group_elements()
{
	return Control->f_allowed_to_show_group_elements;
}

int poset_classification::do_group_extension_in_upstep()
{
	return Control->f_do_group_extension_in_upstep;
}

layer3_group_actions::combinatorics_with_groups::poset_with_group_action *poset_classification::get_poset()
{
	return Poset;
}

poset_classification_control *poset_classification::get_control()
{
	return Control;
}

actions::action *poset_classification::get_A()
{
	return Poset->A;
}

actions::action *poset_classification::get_A2()
{
	return Poset->A2;
}

algebra::linear_algebra::vector_space *poset_classification::get_VS()
{
	return Poset->VS;
}

data_structures_groups::schreier_vector_handler *poset_classification::get_schreier_vector_handler()
{
	return Schreier_vector_handler;
}

int &poset_classification::get_depth()
{
	return depth;
}

int poset_classification::has_base_case()
{
	return f_base_case;
}

int poset_classification::has_invariant_subset_for_root_node()
{
	return Control->f_has_invariant_subset_for_root_node;
}

int poset_classification::size_of_invariant_subset_for_root_node()
{
	return Control->invariant_subset_for_root_node_size;
}

int *poset_classification::get_invariant_subset_for_root_node()
{
	return Control->invariant_subset_for_root_node;
}



classification_base_case *poset_classification::get_Base_case()
{
	return Base_case;
}

int poset_classification::node_has_schreier_vector(
		int node_idx)
{
	if (Poo->get_node(node_idx)->has_Schreier_vector()) {
		return true;
	}
	else {
		return false;
	}
}

int poset_classification::max_number_of_orbits_to_print()
{
	return Control->downstep_orbits_print_max_orbits;
}

int poset_classification::max_number_of_points_to_print_in_orbit()
{
	return Control->downstep_orbits_print_max_points_per_orbit;
}

void poset_classification::print_extension_type(
		std::ostream &ost, int t)
{
	if (t == EXTENSION_TYPE_UNPROCESSED) {
		ost << "   unprocessed";
	}
	else if (t == EXTENSION_TYPE_EXTENSION) {
		ost << "     extension";
	}
	else if (t == EXTENSION_TYPE_FUSION) {
		ost << "        fusion";
	}
	else if (t == EXTENSION_TYPE_PROCESSING) {
		ost << "    processing";
	}
	else if (t == EXTENSION_TYPE_NOT_CANONICAL) {
		ost << " not canonical";
	}
	else {
		ost << "type=" << t;
	}
}



}}}





