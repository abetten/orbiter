/*
 * semifield_downstep_node.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



semifield_downstep_node::semifield_downstep_node()
{
	SC = NULL;
	SL = NULL;
	F = NULL;
	k = 0;
	k2 = 0;
	level = 0;
	orbit_number = 0;
	Candidates = NULL;
	nb_candidates = 0;
	subspace_basis = NULL;
	//subspace_base_cols = NULL;
	on_cosets = NULL;
	A_on_cosets = NULL;
	Sch = NULL;
	first_flag_orbit = 0;
	//null();
}

semifield_downstep_node::~semifield_downstep_node()
{
	if (subspace_basis) {
		FREE_int(subspace_basis);
	}
	if (A_on_cosets) {
		FREE_OBJECT(A_on_cosets);
	}
	//freeself();
}

void semifield_downstep_node::init(
		semifield_lifting *SL, int level, int orbit_number,
		long int *Candidates, int nb_candidates, int first_flag_orbit,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	sorting Sorting;

	if (f_v) {
		cout << "semifield_downstep_node::init level=" << level
				<< " orbit_number=" << orbit_number
				<< " nb_candidates=" << nb_candidates << endl;
		}

	semifield_downstep_node::SL = SL;
	semifield_downstep_node::SC = SL->SC;
	semifield_downstep_node::F = SC->Mtx->GFq;
	semifield_downstep_node::k = SC->k;
	semifield_downstep_node::k2 = SC->k2;

	semifield_downstep_node::level = level;
	semifield_downstep_node::orbit_number = orbit_number;
	semifield_downstep_node::Candidates = Candidates;
	semifield_downstep_node::nb_candidates = nb_candidates;
	semifield_downstep_node::first_flag_orbit = first_flag_orbit;


	subspace_basis = NEW_int(level * k2);
	//subspace_base_cols = NEW_int(level);

	int pivots[2]; // not used

	if (level == 2) {
		SL->L2->get_basis_and_pivots(orbit_number,
				subspace_basis, pivots, verbose_level);
	}
	else {
		SL->get_basis(orbit_number,
				subspace_basis, //subspace_base_cols,
				verbose_level);
	}

	if (f_v) {
		cout << "subspace_basis:" << endl;
		int_matrix_print(subspace_basis, level, k2);
		//cout << "base_cols:" << endl;
		//int_vec_print(cout, subspace_base_cols, level);
		//cout << endl;
		}

#if 0
	if (f_v) {
		cout << "semifield_downstep_node::init "
				"sorting the " << nb_candidates << " candidates" << endl;
		}
	Sorting.lint_vec_heapsort(Candidates, nb_candidates);
	if (f_v) {
		cout << "downstep_node::init "
				"sorting done" << endl;
		}
#endif

	on_cosets = NEW_OBJECT(action_on_cosets);


	if (f_v) {
		cout << "semifield_downstep_node::init "
				"initializing on_cosets:" << endl;
		}
	on_cosets->init_lint(
		nb_candidates, Candidates,
		SC->AS,
		F,
		level /* dimension_of_subspace */,
		k * k /* n */,
		subspace_basis,
		SC->desired_pivots,
		coset_action_unrank_point,
		coset_action_rank_point,
		this /*rank_unrank_data*/,
		verbose_level);
	if (f_v) {
		cout << "semifield_downstep_node::init "
				"initializing on_cosets done" << endl;
		}


	A_on_cosets = NEW_OBJECT(action);

	if (f_vv) {
		cout << "semifield_downstep_node::init "
				"initializing A_on_cosets:" << endl;
		}
	A_on_cosets->induced_action_on_cosets(
		on_cosets,
		FALSE /* f_induce_action */, NULL /* old_G */,
		verbose_level);
	if (f_v) {
		cout << "semifield_downstep_node::init "
				"initializing A_on_cosets done, "
				"the degree of the action is " << A_on_cosets->degree << endl;
		}


	if (f_v) {
		cout << "semifield_downstep_node::init "
				"before orbits_on_points_schreier" << endl;
		}


	strong_generators *sg;
	longinteger_object go;


	sg = SL->get_stabilizer_generators(
		level, orbit_number,
		verbose_level);
	sg->group_order(go);
	if (f_v) {
		cout << "semifield_downstep_node::init "
				"initializing A_on_cosets done, "
				"the group order is " << go << endl;
	}
	if (f_vv) {
		cout << "semifield_downstep_node::init the generators for "
				"the stabilizer are:" << endl;
		sg->print_generators_tex(cout);
	}

#if 0
	if (SFS->f_orbits_light) {


		orbits_light(SFS, level, orbit_number,
			Candidates, nb_candidates,
			sg,
			verbose_level);


		}
	else {
#endif
		file_io File_io;
		string fname;

		SL->make_file_name_schreier(fname, level, orbit_number);

		if (File_io.file_size(fname) > 0) {
			Sch = NEW_OBJECT(schreier);
			Sch->A = A_on_cosets;
			cout << "semifield_downstep_node::init "
					"Reading schreier data structure from "
					"file " << fname << endl;
			Sch->read_file_binary(fname, verbose_level);
			}
		else {
			if (f_v) {
				cout << "semifield_downstep_node::init "
						"before sg->orbits_on_points_schreier" << endl;
				}
			Sch = sg->orbits_on_points_schreier(
					A_on_cosets, verbose_level);
			cout << "Writing schreier data structure to "
					"file " << fname << endl;
			Sch->write_file_binary(fname, verbose_level);

			Sch->delete_images(); // make space

			//delete Sch; // make space in memory
			}
		//}

	if (f_vv) {
		cout << "semifield_downstep_node::init "
				"after orbits_on_points_schreier" << endl;
		}




	if (f_v) {
		cout << "semifield_downstep_node::init done" << endl;
	}
}

int semifield_downstep_node::find_point(long int a)
{
	int idx;
	sorting Sorting;

	if (!Sorting.lint_vec_search(Candidates, nb_candidates,
			a, idx, 0 /* verbose_level */)) {
		cout << "semifield_downstep_node::find_point point " << a
				<< " cannot be found in the Candidates array" << endl;
		//cout << "The " << nb_candidates << " Candidates:" << endl;
		//int_vec_print(cout, Candidates, nb_candidates);
		//cout << endl;
		//SFS->matrix_print_numeric(a);
		exit(1);
		}
	return idx;
}

// #############################################################################
// global functions:
// #############################################################################


void coset_action_unrank_point(int *v, long int a, void *data)
{
	semifield_downstep_node *DN = (semifield_downstep_node *) data;
	semifield_classify *SC = DN->SC;

	SC->matrix_unrank(a, v);
}

long int coset_action_rank_point(int *v, void *data)
{
	semifield_downstep_node *DN = (semifield_downstep_node *) data;
	semifield_classify *SC = DN->SC;

	return SC->matrix_rank(v);
}



}}

