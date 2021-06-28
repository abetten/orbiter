/*
 * stabilizer_orbits_and_types.cpp
 *
 *  Created on: Jun 11, 2021
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace classification {



stabilizer_orbits_and_types::stabilizer_orbits_and_types()
{
	CS = NULL;

	selected_set_stab_gens = NULL;
	selected_set_stab = NULL;


	reduced_set_size = 0; // = set_size - SubSt->SubC->substructure_size




	reduced_set1 = NULL;
	reduced_set2 = NULL;
	reduced_set1_new_labels = NULL;
	reduced_set2_new_labels = NULL;
	canonical_set1 = NULL;
	canonical_set2 = NULL;
	elt1 = NULL;
	Elt1 = NULL;
	Elt1_inv = NULL;
	new_automorphism = NULL;
	Elt4 = NULL;
	elt2 = NULL;
	Elt2 = NULL;
	transporter0 = NULL;


	//longinteger_object go_G;

	Schreier = NULL;
	nb_orbits = 0;
	orbit_count1 = NULL; // [nb_orbits]
	orbit_count2 = NULL; // [nb_orbits]


	nb_interesting_subsets_reduced = 0;
	interesting_subsets_reduced = NULL;

	Orbit_patterns = NULL; // [nb_interesting_subsets * nb_orbits]


	orbit_to_interesting_orbit = NULL; // [nb_orbits]
	nb_interesting_orbits = 0;
	interesting_orbits = NULL;
	nb_interesting_points = 0;
	interesting_points = NULL;
	interesting_orbit_first = NULL;
	interesting_orbit_len = NULL;
	local_idx1 = local_idx2 = 0;



}

stabilizer_orbits_and_types::~stabilizer_orbits_and_types()
{
	if (Elt1) {
		FREE_int(Elt1);
	}

	if (selected_set_stab_gens) {
		FREE_OBJECT(selected_set_stab_gens);
	}
	if (selected_set_stab) {
		FREE_OBJECT(selected_set_stab);
	}

	if (interesting_subsets_reduced) {
		FREE_lint(interesting_subsets_reduced);
	}
	if (Orbit_patterns) {
		FREE_int(Orbit_patterns);
	}

	if (Schreier) {
		FREE_OBJECT(Schreier);
	}

	if (orbit_count1) {
		FREE_int(orbit_count1);
		FREE_int(orbit_count2);
	}
	if (reduced_set1) {
		FREE_lint(reduced_set1);
		FREE_lint(reduced_set2);
		FREE_lint(reduced_set1_new_labels);
		FREE_lint(reduced_set2_new_labels);
		FREE_lint(canonical_set1);
		FREE_lint(canonical_set2);
		FREE_int(elt1);
		FREE_int(elt2);
		FREE_int(Elt1);
		FREE_int(Elt2);
		FREE_int(Elt1_inv);
		FREE_int(new_automorphism);
		FREE_int(Elt4);
		FREE_int(transporter0);
	}
	if (orbit_to_interesting_orbit) {
		FREE_int(orbit_to_interesting_orbit);
	}

	if (interesting_points) {
		FREE_lint(interesting_points);
		FREE_int(interesting_orbits);
		FREE_int(interesting_orbit_first);
		FREE_int(interesting_orbit_len);
	}
}

void stabilizer_orbits_and_types::init(compute_stabilizer *CS, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "stabilizer_orbits_and_types::init" << endl;
	}

	stabilizer_orbits_and_types::CS = CS;


	if (f_v) {
		cout << "stabilizer_orbits_and_types::init before CS->SubSt->SubC->PC->get_stabilizer_generators" << endl;
	}
	CS->SubSt->SubC->PC->get_stabilizer_generators(
			selected_set_stab_gens,
			CS->SubSt->SubC->substructure_size, CS->SubSt->selected_orbit, 0 /*verbose_level*/);
	if (f_v) {
		cout << "stabilizer_orbits_and_types::init after PC->get_stabilizer_generators" << endl;
		selected_set_stab_gens->print_generators_tex();
	}
	selected_set_stab_gens->group_order(go_G);
	if (f_v) {
		cout << "stabilizer_orbits_and_types::init go_G=" << go_G << endl;
	}

	if (f_v) {
		cout << "stabilizer_orbits_and_types::init before selected_set_stab_gens->create_sims" << endl;
	}
	selected_set_stab = selected_set_stab_gens->create_sims(verbose_level);
	if (f_v) {
		cout << "stabilizer_orbits_and_types::init after selected_set_stab_gens->create_sims" << endl;
	}


	reduced_set1 = NEW_lint(CS->SubSt->nb_pts);
	reduced_set2 = NEW_lint(CS->SubSt->nb_pts);
	reduced_set1_new_labels = NEW_lint(CS->SubSt->nb_pts);
	reduced_set2_new_labels = NEW_lint(CS->SubSt->nb_pts);
	canonical_set1 = NEW_lint(CS->SubSt->nb_pts);
	canonical_set2 = NEW_lint(CS->SubSt->nb_pts);
	elt1 = NEW_int(CS->SubSt->SubC->A->elt_size_in_int);
	elt2 = NEW_int(CS->SubSt->SubC->A->elt_size_in_int);
	Elt1 = NEW_int(CS->SubSt->SubC->A->elt_size_in_int);
	Elt2 = NEW_int(CS->SubSt->SubC->A->elt_size_in_int);
	Elt1_inv = NEW_int(CS->SubSt->SubC->A->elt_size_in_int);
	new_automorphism = NEW_int(CS->SubSt->SubC->A->elt_size_in_int);
	Elt4 = NEW_int(CS->SubSt->SubC->A->elt_size_in_int);
	transporter0 = NEW_int(CS->SubSt->SubC->A->elt_size_in_int);


	reduced_set_size = CS->SubSt->nb_pts - CS->SubSt->SubC->substructure_size;

	if (f_v) {
		cout << "stabilizer_orbits_and_types::init before compute_stabilizer_orbits_and_find_minimal_pattern" << endl;
	}
	compute_stabilizer_orbits_and_find_minimal_pattern(verbose_level);
	if (f_v) {
		cout << "stabilizer_orbits_and_types::init after compute_stabilizer_orbits_and_find_minimal_pattern" << endl;
		cout << "stabilizer_orbits_and_types::init nb_interesting_subsets_reduced = " << nb_interesting_subsets_reduced << endl;
	}

	if (f_v) {
		cout << "stabilizer_orbits_and_types::init before find_interesting_orbits" << endl;
	}
	find_interesting_orbits(verbose_level);
	if (f_v) {
		cout << "stabilizer_orbits_and_types::init after find_interesting_orbits" << endl;
	}


	if (f_v) {
		cout << "stabilizer_orbits_and_types::init done" << endl;
	}
}


void stabilizer_orbits_and_types::compute_stabilizer_orbits_and_find_minimal_pattern(int verbose_level)
// uses selected_set_stab_gens to compute orbits on points in action A2
{
	int f_v = (verbose_level >= 1);
	sorting Sorting;

	if (f_v) {
		cout << "stabilizer_orbits_and_types::compute_stabilizer_orbits_and_find_minimal_pattern" << endl;
	}
	if (f_v) {
		cout << "stabilizer_orbits_and_types::compute_stabilizer_orbits_and_find_minimal_pattern computing orbits on points" << endl;
	}
	Schreier = selected_set_stab_gens->orbits_on_points_schreier(CS->SubSt->SubC->A2, 0 /*verbose_level*/);
	if (f_v) {
		cout << "stabilizer_orbits_and_types::compute_stabilizer_orbits_and_find_minimal_pattern computing orbits on points done, "
				"we found " << Schreier->nb_orbits << " orbits" << endl;
	}

	nb_orbits = Schreier->nb_orbits;
	orbit_count1 = NEW_int(nb_orbits);
	orbit_count2 = NEW_int(nb_orbits);
	Orbiter->Int_vec.zero(orbit_count1, nb_orbits);

	int cnt;

	interesting_subsets_reduced = NEW_lint(CS->SubSt->nb_interesting_subsets);
	Orbit_patterns = NEW_int(CS->SubSt->nb_interesting_subsets * nb_orbits);

	if (f_v) {
		cout << "stabilizer_orbits_and_types::compute_orbits_and_find_minimal_pattern computing Orbit_patterns" << endl;
	}

	for (cnt = 0; cnt < CS->SubSt->nb_interesting_subsets; cnt++) {

		if ((cnt % 10000) == 0) {
			cout << "stabilizer_orbits_and_types::compute_stabilizer_orbits_and_find_minimal_pattern "
					"computing Orbit_patterns cnt = " << cnt << " / " << CS->SubSt->nb_interesting_subsets << endl;
		}
		find_orbit_pattern(cnt, elt1 /* transp */, verbose_level - 4);


		Orbiter->Int_vec.copy(orbit_count1, Orbit_patterns + cnt * nb_orbits, nb_orbits);

	}

	if (f_v) {
		cout << "stabilizer_orbits_and_types::compute_stabilizer_orbits_and_find_minimal_pattern computing Orbit_patterns done" << endl;
	}


	if (f_v) {
		cout << "orbit patterns (top row is orbit length): " << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Schreier->orbit_len, 1, nb_orbits, nb_orbits, 2);
		if (CS->SubSt->nb_interesting_subsets < 100) {
			Orbiter->Int_vec.print_integer_matrix_width(cout,
				Orbit_patterns, CS->SubSt->nb_interesting_subsets, nb_orbits, nb_orbits, 2);
		}
		else {
			cout << "too large to print" << endl;
		}
		//Orbiter->Int_vec.print(cout, Orbit_patterns, nb_orbits);
		//cout << endl;
	}


	for (cnt = 0; cnt < CS->SubSt->nb_interesting_subsets; cnt++) {
		if (cnt == 0) {
			Orbiter->Int_vec.copy(Orbit_patterns + cnt * nb_orbits, orbit_count2, nb_orbits);
			nb_interesting_subsets_reduced = 0;
			interesting_subsets_reduced[nb_interesting_subsets_reduced++] = CS->SubSt->interesting_subsets[cnt];
		}
		else {
			int cmp;

			cmp = Sorting.integer_vec_compare(Orbit_patterns + cnt * nb_orbits, orbit_count2, nb_orbits);

			if (cmp > 0) {
				Orbiter->Int_vec.copy(Orbit_patterns + cnt * nb_orbits, orbit_count2, nb_orbits);
				nb_interesting_subsets_reduced = 0;
				interesting_subsets_reduced[nb_interesting_subsets_reduced++] = CS->SubSt->interesting_subsets[cnt];
			}
			else if (cmp == 0) {
				interesting_subsets_reduced[nb_interesting_subsets_reduced++] = CS->SubSt->interesting_subsets[cnt];
			}
		}

	}

#if 1
	if (f_v) {
		cout << "minimal orbit pattern : " << endl;
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				Schreier->orbit_len, 1, nb_orbits, nb_orbits, 2);
		Orbiter->Int_vec.print_integer_matrix_width(cout,
				orbit_count2, 1, nb_orbits, nb_orbits, 2);
		cout << "nb_interesting_subsets_reduced = " << nb_interesting_subsets_reduced << endl;
		cout << "interesting_subsets_reduced:" << endl;
		Orbiter->Lint_vec.print(cout, interesting_subsets_reduced, nb_interesting_subsets_reduced);
		cout << endl;
	}
#endif

	if (f_v) {
		cout << "stabilizer_orbits_and_types::compute_stabilizer_orbits_and_find_minimal_pattern done" << endl;
	}
}

void stabilizer_orbits_and_types::find_orbit_pattern(int cnt, int *transp, int verbose_level)
// computes transporter to transp
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "stabilizer_orbits_and_types::find_orbit_pattern cnt=" << cnt
				<< " interesting_subsets[cnt]=" << CS->SubSt->interesting_subsets[cnt] << endl;
	}
	sorting Sorting;

	if (f_v) {
		cout << "stabilizer_orbits_and_types::find_orbit_pattern before PC->map_to_canonical_k_subset" << endl;
	}
	CS->SubSt->SubC->PC->map_to_canonical_k_subset(CS->SubSt->Pts, CS->SubSt->nb_pts,
			CS->SubSt->SubC->substructure_size /* subset_size */, CS->SubSt->interesting_subsets[cnt],
			reduced_set1, transp /*transporter */, local_idx1, verbose_level - 4);
		// reduced_set1 has size set_size - level (=reduced_set_size)
	if (f_v) {
		cout << "stabilizer_orbits_and_types::find_orbit_pattern after PC->map_to_canonical_k_subset" << endl;
	}


	Sorting.lint_vec_heapsort(reduced_set1, reduced_set_size);
	if (FALSE) {
		cout << "stabilizer_orbits_and_types::find_orbit_pattern STABILIZER "
				<< setw(4) << cnt << " : " << setw(4) << CS->SubSt->interesting_subsets[cnt] << " : ";
		Orbiter->Lint_vec.print(cout, reduced_set1, reduced_set_size);
		cout << endl;
	}

	Schreier->compute_orbit_statistic_lint(reduced_set1, reduced_set_size,
			orbit_count1, verbose_level - 1);

	if (f_v) {
		cout << "stabilizer_orbits_and_types::find_orbit_pattern" << endl;
	}
}

void stabilizer_orbits_and_types::find_interesting_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "stabilizer_orbits_and_types::find_interesting_orbits" << endl;
	}
	// An orbit is interesting if it contains points from reduced_set1[],
	// i.e., orbit i is interesting if orbit_count1[i] is not equal to zero
	// Let interesting_orbits[nb_interesting_orbits] be the list of interesting orbits

	int i;

	nb_interesting_orbits = 0;
	nb_interesting_points = 0;
	orbit_to_interesting_orbit = NEW_int(nb_orbits);
	interesting_orbits = NEW_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		if (orbit_count2[i]) {
			orbit_to_interesting_orbit[i] = nb_interesting_orbits;
			interesting_orbits[nb_interesting_orbits++] = i;
			nb_interesting_points += Schreier->orbit_len[i];
		}
		else {
			orbit_to_interesting_orbit[i] = -1;
		}
	}
	if (f_v) {
		cout << "nb_interesting_orbits = " << nb_interesting_orbits << endl;
		cout << "nb_interesting_points = " << nb_interesting_points << endl;
		cout << "interesting_orbits:" << endl;
		Orbiter->Int_vec.print(cout, interesting_orbits, nb_interesting_orbits);
		cout << endl;
		cout << "orbit_to_interesting_orbit:" << endl;
		Orbiter->Int_vec.print(cout, orbit_to_interesting_orbit, nb_orbits);
		cout << endl;
	}

	interesting_points = NEW_lint(nb_interesting_points);

	interesting_orbit_first = NEW_int(nb_interesting_orbits);
	interesting_orbit_len = NEW_int(nb_interesting_orbits);

	int idx, j, f, l, k, ii;
	sorting Sorting;

	j = 0;
	for (k = 0; k < nb_interesting_orbits; k++) {
		idx = interesting_orbits[k];
		f = Schreier->orbit_first[idx];
		l = Schreier->orbit_len[idx];
		interesting_orbit_first[k] = j;
		interesting_orbit_len[k] = l;
		for (ii = 0; ii < l; ii++) {
			interesting_points[j++] = Schreier->orbit[f + ii];
		}
		Sorting.lint_vec_heapsort(interesting_points + interesting_orbit_first[k], l);
	}

	if (f_v) {
		cout << "interesting_points:" << endl;
		for (k = 0; k < nb_interesting_orbits; k++) {
			f = interesting_orbit_first[k];
			l = interesting_orbit_len[k];
			Orbiter->Lint_vec.print(cout, interesting_points + f, l);
			if (k < nb_interesting_orbits - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}

	if (f_v) {
		cout << "stabilizer_orbits_and_types::find_interesting_orbits done" << endl;
	}

}

void stabilizer_orbits_and_types::compute_local_labels(long int *set_in, long int *set_out, int sz, int verbose_level)
{
	// Let reduced_set1_new_labels[] be the set reduced_set1[] in the restricted action,
	// and let the set be ordered increasingly:

	int i, idx, idx1, f, l, pos_local;
	long int a;
	sorting Sorting;

	for (i = 0; i < sz; i++) {
		a = set_in[i];
		idx = Schreier->orbit_number(a);
		idx1 = orbit_to_interesting_orbit[idx];
		f = interesting_orbit_first[idx1];
		l = interesting_orbit_len[idx1];
		if (!Sorting.lint_vec_search(interesting_points + f, l, a, pos_local, 0 /* verbose_level */)) {
			cout << "stabilizer_orbits_and_types::compute_local_labels did not find point " << a << endl;
			exit(1);
		}
		set_out[i] = f + pos_local;
	}

	Sorting.lint_vec_heapsort(set_out, sz);

}

void stabilizer_orbits_and_types::map_subset_and_compute_local_labels(int cnt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "stabilizer_orbits_and_types::map_subset_and_compute_local_labels" << endl;
	}

	if (f_vv) {
		cout << "stabilizer_orbits_and_types::map_subset_and_compute_local_labels "
				"before map_reduced_set_and_do_orbit_counting" << endl;
	}
	map_reduced_set_and_do_orbit_counting(cnt,
			interesting_subsets_reduced[cnt],
			elt1,
			verbose_level - 2);
	if (f_vv) {
		cout << "stabilizer_orbits_and_types::map_subset_and_compute_local_labels "
				"after map_reduced_set_and_do_orbit_counting" << endl;
	}

	if (f_v) {
		Orbiter->Lint_vec.print(cout, reduced_set1, reduced_set_size);
		cout << endl;
	}


	if (!check_orbit_count()) {
		cout << "stabilizer_orbits_and_types::map_subset_and_compute_local_labels !Stab_orbits->check_orbit_count()" << endl;
		cout << "reduced_set1: ";
		Orbiter->Lint_vec.print(cout, reduced_set1, reduced_set_size);
		cout << endl;
		cout << "orbit_count1: ";
		Orbiter->Int_vec.print(cout, orbit_count1, nb_orbits);
		cout << endl;
		cout << "orbit_count2: ";
		Orbiter->Int_vec.print(cout, orbit_count2, nb_orbits);
		cout << endl;
		exit(1);
	}


	if (f_vv) {
		cout << "stabilizer_orbits_and_types::map_subset_and_compute_local_labels before compute_local_labels" << endl;
	}
	compute_local_labels(reduced_set1,
			reduced_set1_new_labels,
			reduced_set_size,
			verbose_level - 2);
	if (f_vv) {
		cout << "stabilizer_orbits_and_types::map_subset_and_compute_local_labels after compute_local_labels" << endl;
	}
	if (f_v) {
		cout << "local labels:" << endl;
		Orbiter->Lint_vec.print(cout, reduced_set1_new_labels, reduced_set_size);
		cout << endl;
	}

	if (f_v) {
		cout << "stabilizer_orbits_and_types::map_subset_and_compute_local_labels done" << endl;
	}
}

#if 0
void stabilizer_orbits_and_types::map_the_first_set_and_do_orbit_counting(int cnt, int verbose_level)
{
	sorting Sorting;

	CS->SubSt->SubC->PC->map_to_canonical_k_subset(
			CS->SubSt->Pts, CS->SubSt->nb_pts,
			CS->SubSt->SubC->substructure_size /* subset_size */,
			CS->SubSt->interesting_subsets[cnt],
			reduced_set1,
			elt1 /*transporter */,
			local_idx1,
			verbose_level - 4);

		// map the chosen subset interesting_subsets[cnt]
		// to the canonical orbit rep and move it to the beginning.
		// The remaining points are mapped as well and are arranged after the canonical subset.
		// the remaining points are stored in reduced_set1.
		// local_idx1 is the (local) orbit index of the chosen set in the orbits at level
		// reduced_set1 has size set_size - level (=reduced_set_size)


	Sorting.lint_vec_heapsort(reduced_set1, reduced_set_size);
	if (FALSE) {
		cout << setw(4) << cnt << " : " << setw(4) << CS->SubSt->interesting_subsets[cnt] << " : ";
		Orbiter->Lint_vec.print(cout, reduced_set1, reduced_set_size);
		cout << endl;
	}
	if (FALSE) {
		cout << "elt1:" << endl;
		CS->SubSt->SubC->A->element_print(elt1, cout);
		cout << endl;
	}


	// compute orbit_count1[] for reduced_set1[].
	// orbit_count1[i] is the number of points from reduced_set1[] contained in orbit i

	Schreier->compute_orbit_statistic_lint(
			reduced_set1,
			reduced_set_size,
			orbit_count1, verbose_level - 1);
}
#endif

void stabilizer_orbits_and_types::map_reduced_set_and_do_orbit_counting(int cnt,
		long int subset_idx, int *transporter, int verbose_level)
{
	sorting Sorting;

	CS->SubSt->SubC->PC->map_to_canonical_k_subset(
			CS->SubSt->Pts, CS->SubSt->nb_pts,
			CS->SubSt->SubC->substructure_size /* subset_size */,
			subset_idx,
			reduced_set1,
			transporter,
			local_idx1,
			verbose_level - 4);
		// reduced_set2 has size set_size - level (=reduced_set_size)


	Sorting.lint_vec_heapsort(reduced_set1, reduced_set_size);
	if (FALSE) {
		cout << "stabilizer_orbits_and_types::map_the_second_set STABILIZER "
				<< setw(4) << cnt << " : " << setw(4) << interesting_subsets_reduced[cnt] << " : ";
		Orbiter->Lint_vec.print(cout,
				reduced_set1,
				reduced_set_size);
		cout << endl;
	}

	Schreier->compute_orbit_statistic_lint(reduced_set1,
			reduced_set_size,
			orbit_count1,
			verbose_level - 1);
}


#if 0
int stabilizer_orbits_and_types::compute_second_reduced_set()
{
	int i, j, a;
	sorting Sorting;

	for (i = 0; i < reduced_set_size; i++) {
		a = reduced_set2[i];
		for (j = 0; j < nb_interesting_points; j++) {
			if (interesting_points[j] == a) {
				reduced_set2_new_labels[i] = j;
				break;
			}
		}
		if (j == nb_interesting_points) {
			break;
		}
	}
	if (i < reduced_set_size) {
		return FALSE;
	}

	Sorting.lint_vec_heapsort(reduced_set2_new_labels, reduced_set_size);
#if 0
	if (f_vv) {
		cout << "reduced_set2_new_labels:" << endl;
		INT_vec_print(cout, reduced_set2_new_labels, reduced_set_size);
		cout << endl;
	}
#endif
#if 0
	if (f_vv) {
		cout << "sorted: ";
		INT_vec_print(cout, reduced_set2_new_labels, reduced_set_size);
		cout << endl;
		cout << "orbit invariant: ";
		for (i = 0; i < nb_orbits; i++) {
			if (orbit_count2[i] == 0)
				continue;
			cout << i << "^" << orbit_count2[i] << " ";
		}
		cout << endl;
	}
#endif

	return TRUE;
}
#endif

int stabilizer_orbits_and_types::check_orbit_count()
{
	int i;

	for (i = 0; i < nb_orbits; i++) {
		if (orbit_count2[i] != orbit_count1[i]) {
			break;
		}
	}
	if (i < nb_orbits) {
		return FALSE;
	}
	else {
		return TRUE;
	}
}

void stabilizer_orbits_and_types::print_orbit_count(int f_both)
{
	int i;

	cout << "orbit count:" << endl;
	for (i = 0; i < nb_orbits; i++) {
		cout << i << " : " << orbit_count1[i];
		if (f_both) {
			cout << " - " << orbit_count2[i];
		}
		cout << endl;
	}
}



}}
