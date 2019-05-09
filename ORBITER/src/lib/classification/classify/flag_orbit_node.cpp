// flag_orbit_node.C
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
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace classification {

flag_orbit_node::flag_orbit_node()
{
	null();
}

flag_orbit_node::~flag_orbit_node()
{
	freeself();
}

void flag_orbit_node::null()
{
	Flag_orbits = NULL;
	flag_orbit_index = -1;
	downstep_primary_orbit = -1;
	downstep_secondary_orbit = -1;
	upstep_primary_orbit = -1;
	upstep_secondary_orbit = -1;
	downstep_orbit_len = 0;
	f_fusion_node = FALSE;
	fusion_with = -1;
	fusion_elt = NULL;
	gens = NULL;
}

void flag_orbit_node::freeself()
{
	if (fusion_elt) {
		FREE_int(fusion_elt);
		}
	if (gens) {
		FREE_OBJECT(gens);
		}
	null();
}

void flag_orbit_node::init(
	flag_orbits *Flag_orbits, int flag_orbit_index,
	int downstep_primary_orbit, int downstep_secondary_orbit, 
	int downstep_orbit_len, int f_long_orbit, 
	int *pt_representation, strong_generators *Strong_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbit_node::init" << endl;
		}
	if (Flag_orbits->f_lint) {
		cout << "flag_orbit_node::init Flag_orbits->f_lint" << endl;
		exit(1);
	}
	flag_orbit_node::Flag_orbits = Flag_orbits;
	flag_orbit_node::flag_orbit_index = flag_orbit_index;
	flag_orbit_node::downstep_primary_orbit = downstep_primary_orbit;
	flag_orbit_node::downstep_secondary_orbit = downstep_secondary_orbit;
	flag_orbit_node::downstep_orbit_len = downstep_orbit_len;
	flag_orbit_node::f_long_orbit = FALSE;
	int_vec_copy(pt_representation,
			Flag_orbits->Pt +
			flag_orbit_index * Flag_orbits->pt_representation_sz,
			Flag_orbits->pt_representation_sz);
	gens = Strong_gens;
	if (f_v) {
		cout << "flag_orbit_node::init done" << endl;
		}
}

void flag_orbit_node::init_lint(
	flag_orbits *Flag_orbits, int flag_orbit_index,
	int downstep_primary_orbit, int downstep_secondary_orbit,
	int downstep_orbit_len, int f_long_orbit,
	long int *pt_representation, strong_generators *Strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "flag_orbit_node::init" << endl;
		}
	if (!Flag_orbits->f_lint) {
		cout << "flag_orbit_node::init_lint !Flag_orbits->f_lint" << endl;
		exit(1);
	}
	flag_orbit_node::Flag_orbits = Flag_orbits;
	flag_orbit_node::flag_orbit_index = flag_orbit_index;
	flag_orbit_node::downstep_primary_orbit = downstep_primary_orbit;
	flag_orbit_node::downstep_secondary_orbit = downstep_secondary_orbit;
	flag_orbit_node::downstep_orbit_len = downstep_orbit_len;
	flag_orbit_node::f_long_orbit = FALSE;
	lint_vec_copy(pt_representation,
			Flag_orbits->Pt_lint +
			flag_orbit_index * Flag_orbits->pt_representation_sz,
			Flag_orbits->pt_representation_sz);
	gens = Strong_gens;
	if (f_v) {
		cout << "flag_orbit_node::init done" << endl;
		}
}

void flag_orbit_node::write_file(ofstream &fp, int verbose_level)
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
		fp.write((char *) &fusion_with, sizeof(int));
		Flag_orbits->A->element_write_to_file_binary(fusion_elt, fp, 0);
		}
	gens->write_to_file_binary(fp, 0 /* verbose_level */);

	if (f_v) {
		cout << "flag_orbit_node::write_file finished" << endl;
		}
}

void flag_orbit_node::read_file(ifstream &fp, int verbose_level)
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
		fp.read((char *) &fusion_with, sizeof(int));
		fusion_elt = NEW_int(Flag_orbits->A->elt_size_in_int);
		Flag_orbits->A->element_read_from_file_binary(fusion_elt, fp, 0);
		}

	if (FALSE) {
		cout << "flag_orbit_node::read_file "
				"before gens->read_from_file_binary" << endl;
		}
	gens = NEW_OBJECT(strong_generators);
	gens->read_from_file_binary(Flag_orbits->A, fp, verbose_level);

	if (f_v) {
		cout << "flag_orbit_node::read_file finished" << endl;
		}
}

void flag_orbit_node::print_latex(flag_orbits *Flag_orbits,
		ostream &ost,
		int f_print_stabilizer_gens)
{
	longinteger_object go;

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
	if (Flag_orbits->f_lint) {
		lint_vec_print(ost, Flag_orbits->Pt_lint +
				flag_orbit_index * Flag_orbits->pt_representation_sz,
				Flag_orbits->pt_representation_sz);
	}
	else {
		int_vec_print(ost, Flag_orbits->Pt +
				flag_orbit_index * Flag_orbits->pt_representation_sz,
				Flag_orbits->pt_representation_sz);
	}
	ost << " with a stabilizer of order ";
	gens->group_order(go);
	ost << go << "\\\\" << endl;
	if (f_print_stabilizer_gens) {
		gens->print_generators_tex(ost);
	}
	if (f_fusion_node) {
		ost << "Fusion element:\\\\" << endl;
		ost << "$$" << endl;
		Flag_orbits->A->element_print_latex(fusion_elt, ost);
		ost << "$$" << endl;
	}
}

}}



