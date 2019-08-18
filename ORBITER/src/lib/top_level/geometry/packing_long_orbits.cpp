/*
 * packing_long_orbits.cpp
 *
 *  Created on: Aug 13, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


packing_long_orbits::packing_long_orbits()
{
	P = NULL;

	fixpoints_idx = 0;
	fixpoints_clique_case_number = 0;
	fixpoint_clique_size = 0;
	fixpoint_clique = NULL;
	long_orbit_idx = 0;

	Filtered_orbits = NULL;
	fname_graph[0] = 0;

	CG = NULL;
}

packing_long_orbits::~packing_long_orbits()
{
	if (Filtered_orbits) {
		FREE_OBJECT(Filtered_orbits);
	}
	if (CG) {
		FREE_OBJECT(CG);
	}
}

void packing_long_orbits::init(packing_was *P,
		int fixpoints_idx,
		int fixpoints_clique_case_number,
		int fixpoint_clique_size,
		int *fixpoint_clique,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_long_orbits::init" << endl;
	}
	packing_long_orbits::P = P;
	packing_long_orbits::fixpoints_idx = fixpoints_idx;
	packing_long_orbits::fixpoints_clique_case_number = fixpoints_clique_case_number;
	packing_long_orbits::fixpoint_clique_size = fixpoint_clique_size;
	packing_long_orbits::fixpoint_clique = fixpoint_clique;


	long_orbit_idx = P->find_orbits_of_length(P->long_orbit_length);
	if (f_v) {
		cout << "packing_long_orbits::init long_orbit_idx=" << long_orbit_idx << endl;
	}

	if (f_v) {
		cout << "packing_long_orbits::init done" << endl;
	}
}

void packing_long_orbits::filter_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t, i, b;

	if (f_v) {
		cout << "packing_long_orbits::filter_orbits" << endl;
		}


	set_of_sets *Input;

	Input = P->reduced_spread_orbits_under_H->Orbits_classified;


	Filtered_orbits = NEW_OBJECT(set_of_sets);

	Filtered_orbits->init_basic(
			Input->underlying_set_size,
			Input->nb_sets,
			Input->Set_size, 0 /* verbose_level */);

	int_vec_zero(Filtered_orbits->Set_size,
			Input->nb_sets);

	for (t = 0; t < Input->nb_sets; t++) {
		if (t == fixpoints_idx) {
			continue;
			}

		int *Orb;
		int orbit_length;
		int len1;

		orbit_length = P->reduced_spread_orbits_under_H->Orbits_classified_length[t];
		Orb = NEW_int(orbit_length);
		Filtered_orbits->Set_size[t] = 0;

		for (i = 0; i < Input->Set_size[t]; i++) {
			b = Input->element(t, i);

			P->reduced_spread_orbits_under_H->Sch->get_orbit(b,
					Orb, len1, 0 /* verbose_level*/);
			if (len1 != orbit_length) {
				cout << "packing_long_orbits::filter_orbits len1 != orbit_length" << endl;
				exit(1);
			}
			if (P->test_if_pair_of_orbits_are_adjacent(
					fixpoint_clique, fixpoint_clique_size,
					Orb, orbit_length, verbose_level)) {

				// add b to the list in Reduced_Orbits_by_length:

				Filtered_orbits->add_element(t, b);
				}
			}

		FREE_int(Orb);
		}

	if (f_v) {
		cout << "packing_long_orbits::filter_orbits "
				"we found the following number of live orbits:" << endl;
		cout << "t : nb" << endl;
		for (t = 0; t < Input->nb_sets; t++) {
			cout << t << " : " << Filtered_orbits->Set_size[t]
				<< endl;
			}
		}
	if (f_v) {
		cout << "packing_long_orbits::filter_orbits "
				"done" << endl;
		}
}

void packing_long_orbits::create_graph_on_remaining_long_orbits(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//file_io Fio;

	if (f_v) {
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits" << endl;
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"long_orbit_idx = " << long_orbit_idx << endl;
		//cout << "long_orbits_fixpoint_case::create_graph_on_remaining_long_orbits "
		//		"clique_size = " << Paat->clique_size << endl;
		//cout << "long_orbits_fixpoint_case::create_graph_on_remaining_long_orbits "
		//		"clique_no = " << clique_no << endl;
		}

	create_fname_graph_on_remaining_long_orbits();

	if (f_v) {
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"fname=" << fname_graph << endl;
		}

	//selected_fixpoints, clique_size,

	cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
			"creating the graph on long orbits with "
			<< Filtered_orbits->Set_size[long_orbit_idx]
			<< " vertices" << endl;


	int user_data_sz;
	int *user_data;

	user_data_sz = fixpoint_clique_size;
	user_data = NEW_int(user_data_sz);
	int_vec_apply(fixpoint_clique,
			P->reduced_spread_orbits_under_H->Orbits_classified->Sets[fixpoints_idx],
			user_data, fixpoint_clique_size);

	create_graph_and_save_to_file(
				CG,
				fname_graph,
				P->long_orbit_length /* orbit_length */,
				TRUE /* f_has_user_data */, user_data, user_data_sz,
				verbose_level);



	FREE_int(user_data);
	cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
			"the graph on long orbits has been created with "
			<< CG->nb_points
			<< " vertices" << endl;


#if 0
	CG->save(fname_graph, verbose_level);

	cout << "Written file " << fname_graph
			<< " of size " << Fio.file_size(fname_graph)
			<< endl;
#endif

	if (f_v) {
		cout << "packing_long_orbits::create_graph_on_remaining_long_orbits "
				"done" << endl;
		}
}

void packing_long_orbits::create_fname_graph_on_remaining_long_orbits()
{
	if (P->f_output_path) {
		sprintf(fname_graph, "%s%s_fpc%d_graph", P->output_path, P->H_LG->prefix, fixpoints_clique_case_number);
	}
	else {
		sprintf(fname_graph, "%s_fpc%d_graph", P->H_LG->prefix, fixpoints_clique_case_number);
	}

}

void packing_long_orbits::create_graph_and_save_to_file(
	colored_graph *&CG,
	const char *fname,
	int orbit_length,
	int f_has_user_data, int *user_data, int user_data_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_long_orbits::create_graph_and_save_to_file" << endl;
		}


	int type_idx;

	P->reduced_spread_orbits_under_H->create_graph_on_orbits_of_a_certain_length_override_orbits_classified(
		CG,
		fname,
		orbit_length,
		type_idx,
		f_has_user_data, user_data, user_data_size,
		packing_long_orbit_test_function,
		this /* void *test_function_data */,
		Filtered_orbits,
		verbose_level);

	CG->save(fname, verbose_level);

	//FREE_OBJECT(CG);

	if (f_v) {
		cout << "packing_long_orbits::create_graph_and_save_to_file done" << endl;
		}
}

void packing_long_orbits::create_graph_on_long_orbits(
		colored_graph *&CG,
		int *user_data, int user_data_sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "packing_long_orbits::create_graph_on_long_orbits" << endl;
	}


	create_graph_and_save_to_file(
			CG,
			fname_graph,
			P->long_orbit_length /* orbit_length */,
			TRUE /* f_has_user_data */, user_data, user_data_sz,
			verbose_level);

	if (f_v) {
		cout << "packing_long_orbits::create_graph_on_long_orbits done" << endl;
	}
}


void packing_long_orbits::report_filtered_orbits(ostream &ost)
{
	int i;

	//Sch->print_orbit_lengths_tex(ost);
	ost << "Type : orbit length : number of orbits of this length\\\\" << endl;
	for (i = 0; i < Filtered_orbits->nb_sets; i++) {
		ost << i << " : " << P->reduced_spread_orbits_under_H->Orbits_classified_length[i] << " : "
				<< Filtered_orbits->Set_size[i] << "\\\\" << endl;
		}
}

// #############################################################################
// global functions:
// #############################################################################


int packing_long_orbit_test_function(int *orbit1, int len1,
		int *orbit2, int len2, void *data)
{
	packing_long_orbits *L = (packing_long_orbits *) data;

	return L->P->test_if_pair_of_orbits_are_adjacent(
			orbit1, len1, orbit2, len2, 0 /*verbose_level*/);
}


}}

