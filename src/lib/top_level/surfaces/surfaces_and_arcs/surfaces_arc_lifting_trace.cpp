/*
 * surfaces_arc_lifting_trace.cpp
 *
 *  Created on: Jul 30, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

surfaces_arc_lifting_trace::surfaces_arc_lifting_trace()
{
	Up = NULL;

	f = f2 = po = so = 0;

	Elt_alpha2 = NULL;
	Elt_beta1 = NULL;
	Elt_beta2 = NULL;

	Elt_T1 = NULL;
	Elt_T2 = NULL;
	Elt_T3 = NULL;
	Elt_T4 = NULL;

	Elt_Alpha1 = NULL;
	Elt_Alpha2 = NULL;
	Elt_Beta1 = NULL;
	Elt_Beta2 = NULL;
	Elt_Beta3 = NULL;



	upstep_idx = 0;

	seventytwo_case_idx = 0;

}

surfaces_arc_lifting_trace::~surfaces_arc_lifting_trace()
{
	if (Elt_alpha2) {
		FREE_int(Elt_alpha2);
	}
	if (Elt_beta1) {
		FREE_int(Elt_beta1);
	}
	if (Elt_beta2) {
		FREE_int(Elt_beta2);
	}
	if (Elt_T1) {
		FREE_int(Elt_T1);
	}
	if (Elt_T2) {
		FREE_int(Elt_T2);
	}
	if (Elt_T3) {
		FREE_int(Elt_T3);
	}
	if (Elt_T4) {
		FREE_int(Elt_T4);
	}
	if (Elt_Alpha1) {
		FREE_int(Elt_Alpha1);
	}
	if (Elt_Alpha2) {
		FREE_int(Elt_Alpha2);
	}
	if (Elt_Beta1) {
		FREE_int(Elt_Beta1);
	}
	if (Elt_Beta2) {
		FREE_int(Elt_Beta2);
	}
	if (Elt_Beta3) {
		FREE_int(Elt_Beta3);
	}

}


void surfaces_arc_lifting_trace::init(surfaces_arc_lifting_upstep *Up,
		int seventytwo_case_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surfaces_arc_lifting_trace::init" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		cout << "flag orbit " << Up->f << " / " << Up->Lift->Flag_orbits->nb_flag_orbits
		<< " tritangent_plane_idx = " << Up->tritangent_plane_idx
		<< " seventytwo_case_idx = " << seventytwo_case_idx << endl;
	}
	surfaces_arc_lifting_trace::Up = Up;
	surfaces_arc_lifting_trace::seventytwo_case_idx = seventytwo_case_idx;
	f = Up->f;

	The_case = Up->Seventytwo[seventytwo_case_idx];


	Elt_alpha2 = NEW_int(Up->Lift->Surf_A->A->elt_size_in_int);
	Elt_beta1 = NEW_int(Up->Lift->Surf_A->A->elt_size_in_int);
	Elt_beta2 = NEW_int(Up->Lift->Surf_A->A->elt_size_in_int);

	Elt_T1 = NEW_int(Up->Lift->Surf_A->A->elt_size_in_int);
	Elt_T2 = NEW_int(Up->Lift->Surf_A->A->elt_size_in_int);
	Elt_T3 = NEW_int(Up->Lift->Surf_A->A->elt_size_in_int);
	Elt_T4 = NEW_int(Up->Lift->Surf_A->A->elt_size_in_int);

	Elt_Alpha1 = NEW_int(Up->Lift->Surf_A->A->elt_size_in_int);
	Elt_Alpha2 = NEW_int(Up->Lift->Surf_A->A->elt_size_in_int);
	Elt_Beta1 = NEW_int(Up->Lift->Surf_A->A->elt_size_in_int);
	Elt_Beta2 = NEW_int(Up->Lift->Surf_A->A->elt_size_in_int);
	Elt_Beta3 = NEW_int(Up->Lift->Surf_A->A->elt_size_in_int);


	po = Up->Lift->Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
	so = Up->Lift->Flag_orbits->Flag_orbit_node[f].downstep_secondary_orbit;
	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::process_flag_orbit "
				"po=" << po << " so=" << so << endl;
	}

	upstep_idx = Up->tritangent_plane_idx * 72 + seventytwo_case_idx; //Up->line_idx * 24 + Up->cnt;

	if (f_v) {
		cout << "surfaces_arc_lifting_trace::init done" << endl;
	}
}


void surfaces_arc_lifting_trace::process_flag_orbit(surfaces_arc_lifting_upstep *Up, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);


	if (f_v) {
		cout << "surfaces_arc_lifting_trace::process_flag_orbit" << endl;
	}


	if (f_vv) {
		cout << "f=" << f << " / " << Up->Lift->Flag_orbits->nb_flag_orbits
				<< ", tritangent_plane_idx=" << Up->tritangent_plane_idx << " / 45 "
				<< ", seventytwo_case_idx=" << seventytwo_case_idx << " / 72 "
				<< ", upstep " << upstep_idx << " / " << 3240
				<< " before move_arc" << endl;
	}

	move_arc(verbose_level - 2);

	if (f_vv) {
		cout << "f=" << f << " / " << Up->Lift->Flag_orbits->nb_flag_orbits
				<< ", tritangent_plane_idx=" << Up->tritangent_plane_idx << " / 45 "
				<< ", seventytwo_case_idx=" << seventytwo_case_idx << " / 72 "
				<< ", upstep " << upstep_idx << " / " << 3240
				<< " after move_arc" << endl;
	}


	if (f_vv) {
		cout << "f=" << f << " / " << Up->Lift->Flag_orbits->nb_flag_orbits
				<< ", upstep " << upstep_idx << " / " << 3240;
		cout << " f2=" << f2 << " before lift_group_elements_and_move_two_lines";
		cout << endl;
	}
	lift_group_elements_and_move_two_lines(verbose_level - 2);

	if (f_vv) {
		cout << "f=" << f << " / " << Up->Lift->Flag_orbits->nb_flag_orbits
				<< ", upstep " << upstep_idx << " / " << 3240;
		cout << " f2=" << f2 << " after lift_group_elements_and_move_two_lines";
		cout << endl;
	}
	if (f_vvv) {
		cout << "f=" << f << " / " << Up->Lift->Flag_orbits->nb_flag_orbits
				<< ", upstep "
				"tritangent_plane_idx=" << Up->tritangent_plane_idx << " / 45 ";
		cout << " line_idx=" << The_case.line_idx
				<< " l1=" << The_case.l1 << " l2=" << The_case.l2;
		cout << " f2=" << f2 << " the lifted group elements are:";
		cout << endl;

		cout << "Alpha1=" << endl;
		Up->Lift->A4->element_print_quick(Elt_Alpha1, cout);
		cout << "Alpha2=" << endl;
		Up->Lift->A4->element_print_quick(Elt_Alpha2, cout);
		cout << "Beta1=" << endl;
		Up->Lift->A4->element_print_quick(Elt_Beta1, cout);
		cout << "Beta2=" << endl;
		Up->Lift->A4->element_print_quick(Elt_Beta2, cout);
		cout << "Beta3=" << endl;
		Up->Lift->A4->element_print_quick(Elt_Beta3, cout);
	}

	Up->Lift->A4->element_mult(Elt_Alpha1, Elt_Alpha2, Elt_T1, 0);
	Up->Lift->A4->element_mult(Elt_T1, Elt_Beta1, Elt_T2, 0);
	Up->Lift->A4->element_mult(Elt_T2, Elt_Beta2, Elt_T3, 0);
	Up->Lift->A4->element_mult(Elt_T3, Elt_Beta3, Elt_T4, 0);


	if (f_vvv) {
		cout << "f=" << f << " / " << Up->Lift->Flag_orbits->nb_flag_orbits
				<< ", upstep "
				"tritangent_plane_idx=" << Up->tritangent_plane_idx << " / 45 ";
		cout << " line_idx=" << The_case.line_idx
				<< " l1=" << The_case.l1 << " l2=" << The_case.l2;
		cout << " f2=" << f2;
		cout << endl;
		cout << "T4 = Alpha1 * Alpha2 * Beta1 * Beta2 * Beta3 = " << endl;
		Up->Lift->A4->element_print_quick(Elt_T4, cout);
		cout << endl;
	}



	if (f_v) {
		cout << "surfaces_arc_lifting_trace::process_flag_orbit done" << endl;
	}

}


void surfaces_arc_lifting_trace::move_arc(int verbose_level)
// computes alpha1 (4x4), alpha2 (3x3), beta1 (3x3) and beta2 (3x3) and f2
// The following data is computed but not stored:
// P6_local, orbit_not_on_conic_idx, pair_orbit_idx, the_partition4
//
// This function defines a 4x4 projectivity Elt_alpha1
// which maps the chosen plane tritangent_plane_idx
// to the standard plane W=0.
// P6a is the image of P6 under alpha1, preserving the order of elements.
// After that, P6_local will be computed to contain the local coordinates of the arc.
// After that, a 3x3 collineation alpha2 will be computed to map
// P6_local to the canonical orbit representative from the classification
// of non-conical six-arcs computed earlier.
// After that, 3x3 collineations beta1 and beta2 will be computed.
// beta1 takes the pair P1,P2 to the canonical orbit representative
// under the stabilizer of the arc.
// beta2 takes the set-partition imposed by ({P2,P3},{P4,P5})
// to the canonical orbit representative under that stabilizer of the arc
// and the pair of points {P1,P2}
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_arc" << endl;
		cout << "verbose_level = " << verbose_level;
		cout << " f=" << f << " / " << Up->Lift->Flag_orbits->nb_flag_orbits
				<< ", "
				"tritangent_plane_idx=" << Up->tritangent_plane_idx << " / 45, "
				"line_idx=" << The_case.line_idx << " / 3, "
				"l1=" << The_case.l1 << " l2=" << The_case.l2;
		//cout << " transversals4=";
		//lint_vec_print(cout, transversals4, 4);
		cout << " P6=";
		Orbiter->Lint_vec->print(cout, The_case.P6, 6);
		cout << endl;
	}


	// compute Elt_alpha1 which is 4x4:
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_arc before move_plane_and_arc" << endl;
	}
	move_plane_and_arc(The_case.P6a, verbose_level - 2);
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_arc after move_plane_and_arc" << endl;
	}



	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_arc before compute_local_coordinates_of_arc" << endl;
	}
	Up->Lift->Surf->compute_local_coordinates_of_arc(The_case.P6a, The_case.P6_local, verbose_level - 2);
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_arc after compute_local_coordinates_of_arc" << endl;
	}





	// compute Elt_alpha2 which is 3x3:

	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_arc before make_arc_canonical" << endl;
	}
	make_arc_canonical(The_case.P6_local, The_case.P6_local_canonical,
			The_case.orbit_not_on_conic_idx, verbose_level);
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_arc after make_arc_canonical" << endl;
	}





	// compute beta1 which is 3x3:
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_arc before compute_beta1" << endl;
	}
	compute_beta1(&The_case, verbose_level);
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_arc after compute_beta1" << endl;
	}


	// compute beta2 which is 3x3:
	// also, compute f2



	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_arc before compute_beta2" << endl;
	}
	compute_beta2(The_case.orbit_not_on_conic_idx,
			The_case.pair_orbit_idx,
			The_case.partition_orbit_idx,
			The_case.the_partition4, verbose_level);

	The_case.f2 = f2;

	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_arc after compute_beta2" << endl;
	}


	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_arc done" << endl;
	}
}

void surfaces_arc_lifting_trace::move_plane_and_arc(long int *P6a, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_plane_and_arc" << endl;
		cout << "verbose_level = " << verbose_level;
		cout << " f=" << f << " / " << Up->Lift->Flag_orbits->nb_flag_orbits
				<< ", "
				"tritangent_plane_idx=" << Up->tritangent_plane_idx << " / 45, "
				"line_idx=" << The_case.line_idx << " / 3, "
				"l1=" << The_case.l1 << " l2=" << The_case.l2;
		//cout << " transversals4=";
		//lint_vec_print(cout, transversals4, 4);
		cout << " P6=";
		Orbiter->Lint_vec->print(cout, The_case.P6, 6);
		cout << endl;
	}

	int i;

	The_case.tritangent_plane_rk = Up->D->SO->SOP->Tritangent_plane_rk[Up->tritangent_plane_idx];

	if (f_v) {
		cout << "surfaces_arc_lifting_upstep::move_plane_and_arc" << endl;
		cout << "tritangent_plane_rk = " << The_case.tritangent_plane_rk << endl;
	}

	Up->Lift->Surf_A->Surf->Gr3->unrank_embedded_subspace_lint_here(The_case.Basis_pi,
			The_case.tritangent_plane_rk, 0 /*verbose_level - 5*/);

	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_plane_and_arc" << endl;
		cout << "Basis=" << endl;
		Orbiter->Int_vec->matrix_print(The_case.Basis_pi, 4, 4);
	}

	Up->Lift->Surf_A->Surf->F->Linear_algebra->invert_matrix(The_case.Basis_pi, The_case.Basis_pi_inv, 4, 0 /* verbose_level */);
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_plane_and_arc" << endl;
		cout << "Basis_inv=" << endl;
		Orbiter->Int_vec->matrix_print(The_case.Basis_pi_inv, 4, 4);
	}

	The_case.Basis_pi_inv[16] = 0; // in case the group is semilinear

	Up->Lift->Surf_A->A->make_element(Elt_Alpha1, The_case.Basis_pi_inv, 0 /*verbose_level*/);
	for (i = 0; i < 6; i++) {
		P6a[i] = Up->Lift->Surf_A->A->image_of(Elt_Alpha1, The_case.P6[i]);
	}
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_plane_and_arc" << endl;
		cout << "P6a=" << endl;
		Orbiter->Lint_vec->print(cout, P6a, 6);
		cout << endl;
	}


	if (f_v) {
		cout << "surfaces_arc_lifting_trace::move_plane_and_arc done" << endl;
	}
}


void surfaces_arc_lifting_trace::make_arc_canonical(
		long int *P6_local, long int *P6_local_canonical,
		int &orbit_not_on_conic_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting_trace::make_arc_canonical" << endl;
	}
	int i;

	Up->Lift->Six_arcs->recognize(P6_local, Elt_alpha2,
			orbit_not_on_conic_idx, verbose_level - 2);

	if (f_v) {
		cout << "surfaces_arc_lifting_trace::make_arc_canonical" << endl;
		cout << "P6_local=" << endl;
		Orbiter->Lint_vec->print(cout, P6_local, 6);
		cout << " orbit_not_on_conic_idx=" << orbit_not_on_conic_idx << endl;
	}
	for (i = 0; i < 6; i++) {
		P6_local_canonical[i] = Up->Lift->A3->image_of(Elt_alpha2, P6_local[i]);
	}
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::make_arc_canonical" << endl;
		cout << "P6_local_canonical=" << endl;
		Orbiter->Lint_vec->print(cout, P6_local_canonical, 6);
		cout << " orbit_not_on_conic_idx=" << orbit_not_on_conic_idx << endl;
		cout << "The flag orbit f satisfies "
				<< Up->Lift->flag_orbit_fst[orbit_not_on_conic_idx]
				<< " <= f < "
				<< Up->Lift->flag_orbit_fst[orbit_not_on_conic_idx] +
				Up->Lift->flag_orbit_len[orbit_not_on_conic_idx] << endl;
	}

	if (f_v) {
		cout << "surfaces_arc_lifting_trace::make_arc_canonical done" << endl;
	}
}

void surfaces_arc_lifting_trace::compute_beta1(seventytwo_cases *The_case, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting_trace::compute_beta1" << endl;
	}
	int i;

	data_structures::sorting Sorting;
	long int P6_orbit_rep[6];
	int idx;

	Orbiter->Lint_vec->copy(The_case->P6_local_canonical, P6_orbit_rep, 6);
	Sorting.lint_vec_heapsort(P6_orbit_rep, 6);
	for (i = 0; i < 6; i++) {
		Sorting.lint_vec_search_linear(P6_orbit_rep, 6, The_case->P6_local_canonical[i], idx);
		The_case->P6_perm[i] = idx;
	}
	The_case->pair[0] = The_case->P6_perm[0];
	The_case->pair[1] = The_case->P6_perm[1];
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::compute_beta1" << endl;
		cout << "P6_orbit_rep=" << endl;
		Orbiter->Lint_vec->print(cout, P6_orbit_rep, 6);
		cout << endl;
		cout << "P6_perm=" << endl;
		Orbiter->Lint_vec->print(cout, The_case->P6_perm, 6);
		cout << endl;
	}


	// compute beta1 which is 3x3:


	if (f_v) {
		cout << "surfaces_arc_lifting_trace::compute_beta1 before "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
	}
	Up->Lift->Table_orbits_on_pairs[The_case->orbit_not_on_conic_idx].recognize(The_case->pair, Elt_beta1,
			The_case->pair_orbit_idx, verbose_level - 4);
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::compute_beta1 after "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
		cout << "pair_orbit_idx=" << The_case->pair_orbit_idx << endl;
	}
	for (i = 0; i < 6; i++) {
		The_case->P6_perm_mapped[i] =
				Up->Lift->Table_orbits_on_pairs[The_case->orbit_not_on_conic_idx].A_on_arc->image_of(
				Elt_beta1, The_case->P6_perm[i]);
	}


	if (f_v) {
		cout << "surfaces_arc_lifting_trace::compute_beta1 before "
			"The_case->compute_partition" << endl;
	}
	The_case->compute_partition(verbose_level);
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::compute_beta1 after "
			"The_case->compute_partition" << endl;
	}


	if (f_v) {
		cout << "surfaces_arc_lifting_trace::compute_beta1 after "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
		cout << "the_partition4=";
		Orbiter->Int_vec->print(cout, The_case->the_partition4, 4);
		cout << endl;
	}

	if (f_v) {
		cout << "surfaces_arc_lifting_trace::compute_beta1 done" << endl;
	}
}

void surfaces_arc_lifting_trace::compute_beta2(
		int orbit_not_on_conic_idx,
		int pair_orbit_idx,
		int &partition_orbit_idx,
		int *the_partition4, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surfaces_arc_lifting_trace::compute_beta2" << endl;
	}

	if (f_v) {
		cout << "surfaces_arc_lifting_trace::compute_beta2 before "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
	}
	Up->Lift->Table_orbits_on_pairs[orbit_not_on_conic_idx].
		Table_orbits_on_partition[pair_orbit_idx].recognize(
			the_partition4, Elt_beta2,
			partition_orbit_idx, verbose_level - 4);
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::compute_beta2 after "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
		cout << "pair_orbit_idx=" << pair_orbit_idx << endl;
		cout << "partition_orbit_idx=" << partition_orbit_idx << endl;
	}

	f2 = Up->Lift->flag_orbit_fst[orbit_not_on_conic_idx] +
			Up->Lift->Table_orbits_on_pairs[orbit_not_on_conic_idx].
			partition_orbit_first[pair_orbit_idx] + partition_orbit_idx;


	if (f_v) {
		cout << "surfaces_arc_lifting_trace::compute_beta2 after "
			"Table_orbits_on_pairs[orbit_not_on_conic_idx].recognize" << endl;
		cout << "pair_orbit_idx=" << pair_orbit_idx << endl;
		cout << "partition_orbit_idx=" << partition_orbit_idx << endl;
		cout << "f2=" << f2 << endl;
	}

	if (Up->Lift->flag_orbit_on_arcs_not_on_a_conic_idx[f2] != orbit_not_on_conic_idx) {
		cout << "flag_orbit_on_arcs_not_on_a_conic_idx[f2] != orbit_not_on_conic_idx" << endl;
		exit(1);
	}
	if (Up->Lift->flag_orbit_on_pairs_idx[f2] != pair_orbit_idx) {
		cout << "flag_orbit_on_pairs_idx[f2] != pair_orbit_idx" << endl;
		exit(1);

	}
	if (Up->Lift->flag_orbit_on_partition_idx[f2] != partition_orbit_idx) {
		cout << "flag_orbit_on_partition_idx[f2] != partition_orbit_idx" << endl;
		exit(1);

	}
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::compute_beta2 done" << endl;
	}

}

void surfaces_arc_lifting_trace::lift_group_elements_and_move_two_lines(int verbose_level)
// uses Elt_Alpha2, Elt_Beta1, Elt_Beta2, Elt_Beta3, Elt_T1, Elt_T2, Elt_T3
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);


	if (f_v) {
		cout << "surfaces_arc_lifting_trace::lift_group_elements_and_move_two_lines" << endl;
		cout << "verbose_level = " << verbose_level;
		cout << " f=" << f << " / " << Up->Lift->Flag_orbits->nb_flag_orbits
				<< ", "
				"tritangent_plane_idx=" << Up->tritangent_plane_idx << " / 45, "
				"line_idx=" << The_case.line_idx << " / 3, "
				"l1=" << The_case.l1 << " l2=" << The_case.l2;
		cout << " f2 = " << f2 << endl;
	}


	if (f_vv) {
		cout << "surfaces_arc_lifting_trace::lift_group_elements_and_move_two_lines "
				"before embedding" << endl;
		cout << "Elt_alpha2=" << endl;
		Up->Lift->A3->element_print_quick(Elt_alpha2, cout);
		cout << "Elt_beta1=" << endl;
		Up->Lift->A3->element_print_quick(Elt_beta1, cout);
		cout << "Elt_beta2=" << endl;
		Up->Lift->A3->element_print_quick(Elt_beta2, cout);
	}



	if (f_v) {
		cout << "surfaces_arc_lifting_trace::lift_group_elements_and_move_two_lines "
				"before embed Elt_alpha2" << endl;
	}
	embed(Elt_alpha2, Elt_Alpha2, verbose_level - 2);
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::lift_group_elements_and_move_two_lines "
				"before embed Elt_alpha2" << endl;
	}
	embed(Elt_beta1, Elt_Beta1, verbose_level - 2);
	embed(Elt_beta2, Elt_Beta2, verbose_level - 2);

	if (f_vv) {
		cout << "surfaces_arc_lifting_trace::lift_group_elements_and_move_two_lines "
				"after embedding" << endl;
		cout << "Elt_Alpha2=" << endl;
		Up->Lift->A4->element_print_quick(Elt_Alpha2, cout);
		cout << "Elt_Beta1=" << endl;
		Up->Lift->A4->element_print_quick(Elt_Beta1, cout);
		cout << "Elt_Beta2=" << endl;
		Up->Lift->A4->element_print_quick(Elt_Beta2, cout);
	}


	Up->Lift->A4->element_mult(Elt_Alpha1, Elt_Alpha2, Elt_T1, 0);
	Up->Lift->A4->element_mult(Elt_T1, Elt_Beta1, Elt_T2, 0);
	Up->Lift->A4->element_mult(Elt_T2, Elt_Beta2, Elt_T3, 0);


	// map the two lines:

	//long int L1, L2;
	int beta3[17];



	The_case.L1 = Up->Lift->Surf_A->A2->element_image_of(Up->Lines[The_case.l1], Elt_T3, 0 /* verbose_level */);
	The_case.L2 = Up->Lift->Surf_A->A2->element_image_of(Up->Lines[The_case.l2], Elt_T3, 0 /* verbose_level */);
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::lift_group_elements_and_move_two_lines "
				"L1=" << The_case.L1 << " L2=" << The_case.L2 << endl;
	}

	// compute beta 3:

	//int orbit_not_on_conic_idx;
	//int pair_orbit_idx;
	//int partition_orbit_idx;
	long int line1_to, line2_to;


	//orbit_not_on_conic_idx = flag_orbit_on_arcs_not_on_a_conic_idx[f2];
	//pair_orbit_idx = flag_orbit_on_pairs_idx[f2];
	//partition_orbit_idx = flag_orbit_on_partition_idx[f2];

#if 0
	line1_to = Table_orbits_on_pairs[orbit_not_on_conic_idx].
			Table_orbits_on_partition[pair_orbit_idx].;
#endif

	//int pt_representation_sz;

	//pt_representation_sz = 6 + 1 + 2 + 1 + 1 + 2 + 20 + 27;

		// Flag[0..5]   : 6 for the arc P1,...,P6
		// Flag[6]      : 1 for orb, the selected orbit on pairs
		// Flag[7..8]   : 2 for the selected pair, i.e., {0,1} for P1,P2.
		// Flag[9]      : 1 for orbit, the selected orbit on set_partitions
		// Flag[10]     : 1 for the partition of the remaining points; values=0,1,2
		// Flag[11..12] : 2 for the chosen lines line1 and line2 through P1 and P2
		// Flag[13..32] : 20 for the equation of the surface
		// Flag[33..59] : 27 for the lines of the surface

	//Flag2_representation = NEW_lint(pt_representation_sz);

	Orbiter->Lint_vec->copy(
			Up->Lift->Flag_orbits->Pt + f2 * Up->pt_representation_sz,
			Up->Flag2_representation, Up->pt_representation_sz);


	line1_to = Up->Flag2_representation[11];
	line2_to = Up->Flag2_representation[12];

	if (f_vv) {
		cout << "surfaces_arc_lifting_trace::lift_group_elements_and_move_two_lines "
				"line1_to=" << line1_to << " line2_to=" << line2_to << endl;
		int A[8];
		int B[8];
		Up->Lift->Surf_A->Surf->P->unrank_line(A, line1_to);
		cout << "line1_to=" << line1_to << "=" << endl;
		Orbiter->Int_vec->matrix_print(A, 2, 4);
		Up->Lift->Surf_A->Surf->P->unrank_line(B, line2_to);
		cout << "line2_to=" << line2_to << "=" << endl;
		Orbiter->Int_vec->matrix_print(B, 2, 4);
	}

	if (f_vv) {
		cout << "surfaces_arc_lifting_trace::lift_group_elements_and_move_two_lines "
				"L1=" << The_case.L1 << " L2=" << The_case.L2 << endl;
		int A[8];
		int B[8];
		Up->Lift->Surf_A->Surf->P->unrank_line(A, The_case.L1);
		cout << "L1=" << The_case.L1 << "=" << endl;
		Orbiter->Int_vec->matrix_print(A, 2, 4);
		Up->Lift->Surf_A->Surf->P->unrank_line(B, The_case.L2);
		cout << "L2=" << The_case.L2 << "=" << endl;
		Orbiter->Int_vec->matrix_print(B, 2, 4);
	}

	// test if L1 and line1_to are skew then switch L1 and L2:

	//long int tritangent_plane_rk;
	long int p1, p2;

	//tritangent_plane_rk = SO->Tritangent_plane_rk[tritangent_plane_idx];

	p1 = Up->Lift->Surf_A->Surf->P->point_of_intersection_of_a_line_and_a_plane_in_three_space(
			The_case.L1 /* line */,
			0 /* plane */, 0 /* verbose_level */);

	p2 = Up->Lift->Surf_A->Surf->P->point_of_intersection_of_a_line_and_a_plane_in_three_space(
			line1_to /* line */,
			0 /* plane */, 0 /* verbose_level */);

	if (f_vv) {
		cout << "surfaces_arc_lifting_trace::lift_group_elements_and_move_two_lines "
				"p1=" << p1 << " p2=" << p2 << endl;
	}

	if (p1 != p2) {

		if (f_vv) {
			cout << "L1 and line1_to do not intersect the plane in "
					"the same point, so we switch L1 and L2" << endl;
		}
		int t;

		t = The_case.L1;
		The_case.L1 = The_case.L2;
		The_case.L2 = t;
	}
	else {
		if (f_vv) {
			cout << "no need to switch" << endl;
		}
	}

	if (f_v) {
		cout << "surfaces_arc_lifting_trace::lift_group_elements_and_move_two_lines "
				"before hyperplane_lifting_with_two_lines_moved" << endl;
	}
	Up->Lift->Surf_A->Surf->P->hyperplane_lifting_with_two_lines_moved(
			The_case.L1 /* line1_from */, line1_to,
			The_case.L2 /* line2_from */, line2_to,
			beta3,
			verbose_level - 4);
	beta3[16] = 0;
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::lift_group_elements_and_move_two_lines "
				"after hyperplane_lifting_with_two_lines_moved" << endl;
	}

	Up->Lift->A4->make_element(Elt_Beta3, beta3, 0);

	if (f_vv) {
		cout << "surfaces_arc_lifting_trace::lift_group_elements_and_move_two_lines" << endl;
		cout << "Elt_beta3=" << endl;
		Orbiter->Int_vec->matrix_print(Elt_Beta3, 4, 4);
		cout << "Elt_beta3=" << endl;
		Up->Lift->A4->element_print_quick(Elt_Beta3, cout);
		cout << endl;
	}



	//FREE_lint(Flag2_representation);
	if (f_v) {
		cout << "surfaces_arc_lifting_trace::lift_group_elements_and_move_two_lines done" << endl;
	}
}

void surfaces_arc_lifting_trace::embed(int *Elt_A3, int *Elt_A4, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int M3[9];
	int M4[17];
	int i, j, a;


	if (f_v) {
		cout << "surfaces_arc_lifting_trace::embed" << endl;
	}
	Orbiter->Int_vec->copy(Elt_A3, M3, 9);
	Up->Lift->F->PG_element_normalize(M3, 1, 9);
	Orbiter->Int_vec->zero(M4, 17);
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			a = M3[i * 3 + j];
			M4[i * 4 + j] = a;
		}
	}
	M4[3 * 4 + 3] = 1;
	if (FALSE) {
		cout << "surfaces_arc_lifting_trace::embed M4=" << endl;
		Orbiter->Int_vec->print(cout, M4, 17);
		cout << endl;
	}
	if (Up->Lift->f_semilinear) {
		M4[16] = Elt_A3[9];
	}
	if (FALSE) {
		cout << "surfaces_arc_lifting_trace::embed before make_element" << endl;
	}
	Up->Lift->A4->make_element(Elt_A4, M4, 0);


	if (f_v) {
		cout << "surfaces_arc_lifting_trace::embed done" << endl;
	}
}

void surfaces_arc_lifting_trace::report_product(ostream &ost, int *Elt, int verbose_level)
{
	ost << "$$" << endl;
	Up->Lift->A4->element_print_latex(Elt_Alpha1, ost);
	Up->Lift->A4->element_print_latex(Elt_Alpha2, ost);
	Up->Lift->A4->element_print_latex(Elt_Beta1, ost);
	Up->Lift->A4->element_print_latex(Elt_Beta2, ost);
	Up->Lift->A4->element_print_latex(Elt_Beta3, ost);
	ost << "=";
	Up->Lift->A4->element_print_latex(Elt, ost);
	ost << "$$" << endl;

	int f_print_as_exponentials_save = Up->Lift->F->f_print_as_exponentials;

	if (f_print_as_exponentials_save == FALSE) {
		return;
	}

	Up->Lift->F->f_print_as_exponentials = FALSE;

	ost << "$$" << endl;
	Up->Lift->A4->element_print_latex(Elt_Alpha1, ost);
	Up->Lift->A4->element_print_latex(Elt_Alpha2, ost);
	Up->Lift->A4->element_print_latex(Elt_Beta1, ost);
	Up->Lift->A4->element_print_latex(Elt_Beta2, ost);
	Up->Lift->A4->element_print_latex(Elt_Beta3, ost);
	ost << "=";
	Up->Lift->A4->element_print_latex(Elt, ost);
	ost << "$$" << endl;

	Up->Lift->F->f_print_as_exponentials = f_print_as_exponentials_save;


}


}}
