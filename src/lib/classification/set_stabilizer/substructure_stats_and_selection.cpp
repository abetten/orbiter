/*
 * substructure_stats_and_selection.cpp
 *
 *  Created on: Jun 9, 2021
 *      Author: betten
 */



#include "foundations/foundations.h"
#include "discreta/discreta.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {



substructure_stats_and_selection::substructure_stats_and_selection()
{
	//std::string fname_case_out;
	SubC = NULL;
	Pts = NULL;
	nb_pts = 0;
	nCk = 0;
	isotype = NULL;
	orbit_frequencies = NULL;
	nb_orbits = 0;
	T = NULL;

	SoS = NULL;
	types = NULL;
	nb_types = 0;
	selected_type = 0;
	selected_orbit = 0;
	selected_frequency = 0;

	interesting_subsets = NULL;
	nb_interesting_subsets = 0;

	gens = NULL;
	//transporter_to_canonical_form = NULL;
	//Gens_stabilizer_original_set = NULL;
}


substructure_stats_and_selection::~substructure_stats_and_selection()
{
	if (gens) {
		FREE_OBJECT(gens);
	}
	if (SoS) {
		FREE_OBJECT(SoS);
	}
	if (types) {
		FREE_int(types);
	}
	if (isotype) {
		FREE_int(isotype);
	}
	if (orbit_frequencies) {
		FREE_int(orbit_frequencies);
	}
	if (T) {
		FREE_OBJECT(T);
	}
	if (interesting_subsets) {
		FREE_lint(interesting_subsets);
	}

}


void substructure_stats_and_selection::init(
		std::string &fname_case_out,
		substructure_classifier *SubC,
		long int *Pts,
		int nb_pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "substructure_stats_and_selection::init, fname_case_out=" << fname_case_out << " nb_pts=" << nb_pts << endl;
	}

	substructure_stats_and_selection::fname_case_out.assign(fname_case_out);

	substructure_stats_and_selection::SubC = SubC;
	substructure_stats_and_selection::Pts = Pts;
	substructure_stats_and_selection::nb_pts = nb_pts;

	if (f_v) {
		cout << "substructure_stats_and_selection::init before PC->trace_all_k_subsets_and_compute_frequencies" << endl;
	}

	SubC->PC->trace_all_k_subsets_and_compute_frequencies(
			Pts, nb_pts, SubC->substructure_size, nCk, isotype, orbit_frequencies, nb_orbits,
			0 /*verbose_level*/);

	if (f_v) {
		cout << "substructure_stats_and_selection::init after PC->trace_all_k_subsets_and_compute_frequencies" << endl;
	}




	T = NEW_OBJECT(tally);

	T->init(orbit_frequencies, nb_orbits, FALSE, 0);


	if (f_v) {
		cout << "substructure_stats_and_selection::init Pts=";
		Orbiter->Lint_vec->print(cout, Pts, nb_pts);
		cout << endl;
		cout << "substructure_stats_and_selection::init orbit isotype=";
		Orbiter->Int_vec->print(cout, isotype, nCk);
		cout << endl;
		cout << "substructure_stats_and_selection::init orbit frequencies=";
		Orbiter->Int_vec->print(cout, orbit_frequencies, nb_orbits);
		cout << endl;
		cout << "substructure_stats_and_selection::init orbit frequency types=";
		T->print_naked(FALSE /* f_backwards */);
		cout << endl;
	}


	ring_theory::longinteger_domain D;
	int i, f, l, idx;
	int j;



	SoS = T->get_set_partition_and_types(types, nb_types, verbose_level);

	ring_theory::longinteger_object go_min;


	selected_type = -1;

	for (i = 0; i < nb_types; i++) {
		f = T->type_first[i];
		l = T->type_len[i];
		cout << types[i];
		cout << " : ";
		Orbiter->Lint_vec->print(cout, SoS->Sets[i], SoS->Set_size[i]);
		cout << " : ";


		for (j = 0; j < SoS->Set_size[i]; j++) {

			idx = SoS->Sets[i][j];

			ring_theory::longinteger_object go;

			SubC->PC->get_stabilizer_order(SubC->substructure_size, idx, go);

			if (types[i]) {

				// types[i] must be greater than zero
				// so the type really appears.

				if (selected_type == -1) {
					selected_type = j;
					selected_orbit = idx;
					selected_frequency = types[i];
					go.assign_to(go_min);
				}
				else {
					if (D.compare_unsigned(go, go_min) < 0) {
						selected_type = j;
						selected_orbit = idx;
						selected_frequency = types[i];
						go.assign_to(go_min);
					}
				}
			}

			cout << go;
			if (j < SoS->Set_size[i] - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}

	if (f_v) {
		cout << "selected_type = " << selected_type
			<< " selected_orbit = " << selected_orbit
			<< " selected_frequency = " << selected_frequency
			<< " go_min = " << go_min << endl;
	}




	if (f_v) {
		cout << "substructure_stats_and_selection::init" << endl;
		cout << "selected_orbit = " << selected_orbit << endl;
	}

	if (f_v) {
		cout << "substructure_stats_and_selection::init "
				"we decide to go for subsets of size " << SubC->substructure_size
				<< ", selected_frequency = " << selected_frequency << endl;
	}

	j = 0;
	interesting_subsets = NEW_lint(selected_frequency);
	for (i = 0; i < nCk; i++) {
		if (isotype[i] == selected_orbit) {
			interesting_subsets[j++] = i;
			//cout << "subset of rank " << i << " is isomorphic to orbit " << orb_idx << " j=" << j << endl;
			}
		}
	if (j != selected_frequency) {
		cout << "substructure_stats_and_selection::init j != selected_frequency" << endl;
		exit(1);
		}
	nb_interesting_subsets = selected_frequency;
#if 0
	if (f_vv) {
		print_interesting_subsets(nb_pts, intermediate_subset_size, nb_interesting_subsets, interesting_subsets);
		}
#endif


	SubC->PC->get_stabilizer_generators(
		gens,
		SubC->substructure_size, selected_orbit, verbose_level);



	if (f_v) {
		cout << "substructure_stats_and_selection::init" << endl;
		cout << "stabilizer generators are:" << endl;
		gens->print_generators_tex(cout);
	}

	if (f_v) {
		cout << "substructure_stats_and_selection::init done" << endl;
	}


}


}}


