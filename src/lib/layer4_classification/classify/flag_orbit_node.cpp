// flag_orbit_node.cpp
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

flag_orbit_node::flag_orbit_node()
{
	Flag_orbits = NULL;
	flag_orbit_index = -1;
	downstep_primary_orbit = -1;
	downstep_secondary_orbit = -1;
	f_long_orbit = FALSE;
	upstep_primary_orbit = -1;
	upstep_secondary_orbit = -1;
	downstep_orbit_len = 0;
	f_fusion_node = FALSE;
	fusion_with = -1;
	fusion_elt = NULL;
	gens = NULL;

	nb_received = 0;
	Receptacle = NULL;

	//null();
}

flag_orbit_node::~flag_orbit_node()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbit_node::~flag_orbit_node" << endl;
	}
	if (fusion_elt) {
		FREE_int(fusion_elt);
	}
	if (f_v) {
		cout << "flag_orbit_node::~flag_orbit_node before FREE_OBJECT(gens)" << endl;
	}
	if (gens) {
		FREE_OBJECT(gens);
	}
	if (Receptacle) {
		int i;

		for (i = 0; i < nb_received; i++) {
			(*Flag_orbits->func_to_free_received_trace)(Receptacle[i],
					Flag_orbits->free_received_trace_data, verbose_level);
		}
		FREE_pvoid(Receptacle);
	}
	if (f_v) {
		cout << "flag_orbit_node::~flag_orbit_node done" << endl;
	}
}

void flag_orbit_node::init(
	flag_orbits *Flag_orbits,
	int flag_orbit_index,
	int downstep_primary_orbit, int downstep_secondary_orbit, 
	int downstep_orbit_len,
	int f_long_orbit,
	long int *pt_representation,
	groups::strong_generators *Strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbit_node::init" << endl;
	}
	flag_orbit_node::Flag_orbits = Flag_orbits;
	flag_orbit_node::flag_orbit_index = flag_orbit_index;
	flag_orbit_node::downstep_primary_orbit = downstep_primary_orbit;
	flag_orbit_node::downstep_secondary_orbit = downstep_secondary_orbit;
	flag_orbit_node::downstep_orbit_len = downstep_orbit_len;
	flag_orbit_node::f_long_orbit = FALSE;
	Lint_vec_copy(pt_representation,
			Flag_orbits->Pt +
			flag_orbit_index * Flag_orbits->pt_representation_sz,
			Flag_orbits->pt_representation_sz);
	gens = Strong_gens;

	nb_received = 0;
	Receptacle = NEW_pvoid(Flag_orbits->upper_bound_for_number_of_traces);

	if (f_v) {
		cout << "flag_orbit_node::init done" << endl;
	}
}

void flag_orbit_node::receive_trace_result(
		void *trace_result, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbit_node::receive_trace_result" << endl;
	}
	if (nb_received == Flag_orbits->upper_bound_for_number_of_traces) {
		cout << "flag_orbit_node::receive_trace_result Receptacle is full" << endl;
		exit(1);
	}
	Receptacle[nb_received++] = trace_result;

	if (Flag_orbits->free_received_trace_data == NULL) {
		cout << "flag_orbit_node::receive_trace_result Warning: Flag_orbits->free_received_trace_data == NULL" << endl;
	}
	if (f_v) {
		cout << "flag_orbit_node::receive_trace_result done" << endl;
	}
}



void flag_orbit_node::write_file(std::ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "flag_orbit_node::write_file" << endl;
	}
	fp.write((char *) &downstep_primary_orbit, sizeof(int));
	fp.write((char *) &downstep_secondary_orbit, sizeof(int));
	fp.write((char *) &downstep_orbit_len, sizeof(int));
	fp.write((char *) &upstep_primary_orbit, sizeof(int));
	fp.write((char *) &upstep_secondary_orbit, sizeof(int));
	fp.write((char *) &f_fusion_node, sizeof(int));
	if (f_fusion_node) {
		if (f_v) {
			cout << "flag_orbit_node::write_file f_fusion_node" << endl;
		}
		fp.write((char *) &fusion_with, sizeof(int));
		Flag_orbits->A->Group_element->element_write_to_file_binary(fusion_elt, fp, 0);
	}
	if (f_v) {
		cout << "flag_orbit_node::write_file "
				"before gens->write_to_file_binary" << endl;
	}
	gens->write_to_file_binary(fp, 0 /* verbose_level */);

	if (f_v) {
		cout << "flag_orbit_node::write_file finished" << endl;
	}
}

void flag_orbit_node::read_file(std::ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "flag_orbit_node::read_file" << endl;
	}
	
	fp.read((char *) &downstep_primary_orbit, sizeof(int));
	fp.read((char *) &downstep_secondary_orbit, sizeof(int));
	fp.read((char *) &downstep_orbit_len, sizeof(int));
	fp.read((char *) &upstep_primary_orbit, sizeof(int));
	fp.read((char *) &upstep_secondary_orbit, sizeof(int));
	fp.read((char *) &f_fusion_node, sizeof(int));
	if (f_fusion_node) {
		if (f_v) {
			cout << "flag_orbit_node::read_file f_fusion_node" << endl;
		}
		fp.read((char *) &fusion_with, sizeof(int));
		fusion_elt = NEW_int(Flag_orbits->A->elt_size_in_int);
		Flag_orbits->A->Group_element->element_read_from_file_binary(fusion_elt, fp, 0);
	}

	if (f_v) {
		cout << "flag_orbit_node::read_file "
				"before gens->read_from_file_binary" << endl;
	}
	gens = NEW_OBJECT(groups::strong_generators);
	gens->read_from_file_binary(Flag_orbits->A, fp, verbose_level);

	if (f_v) {
		cout << "flag_orbit_node::read_file finished" << endl;
	}
}

void flag_orbit_node::print_latex(flag_orbits *Flag_orbits,
		ostream &ost,
		int f_print_stabilizer_gens)
{
	ring_theory::longinteger_object go;

	ost << "Flag orbit " << flag_orbit_index << " / " << Flag_orbits->nb_flag_orbits
			<< " down=(" << downstep_primary_orbit
			<< "," << downstep_secondary_orbit
			<< ")"
			<< " up=(" << upstep_primary_orbit
			<< "," << upstep_secondary_orbit
			<< ")";
	if (f_fusion_node) {
		ost << " fuse to " << fusion_with;
	}
	ost << " is ";

	Lint_vec_print(ost, Flag_orbits->Pt +
			flag_orbit_index * Flag_orbits->pt_representation_sz,
			Flag_orbits->pt_representation_sz);

	ost << " with a stabilizer of order ";
	gens->group_order(go);
	ost << go << "\\\\" << endl;
	if (f_print_stabilizer_gens) {
		gens->print_generators_tex(ost);
	}
	if (f_fusion_node) {
		ost << "Fusion element:\\\\" << endl;
		ost << "$$" << endl;
		Flag_orbits->A->Group_element->element_print_latex(fusion_elt, ost);
		ost << "$$" << endl;
	}

	ost << "nb received = " << nb_received << "\\\\" << endl;

	if (nb_received) {
		if (Flag_orbits->func_latex_report_trace) {
			(*Flag_orbits->func_latex_report_trace)(ost, Receptacle[0],
					Flag_orbits->free_received_trace_data, 0 /*verbose_level*/);
		}
	}
}

}}}



