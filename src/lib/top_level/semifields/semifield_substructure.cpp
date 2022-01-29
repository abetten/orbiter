/*
 * semifield_substructure.cpp
 *
 *  Created on: May 15, 2019
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace semifields {



semifield_substructure::semifield_substructure()
{
	SCWS = NULL;
	SC = NULL;
	L3 = NULL;
	Gr3 = NULL;
	Gr2 = NULL;
	Non_unique_cases_with_non_trivial_group = NULL;
	nb_non_unique_cases_with_non_trivial_group = 0;

	Need_orbits_fst = NULL;
	Need_orbits_len = NULL;

	TR = NULL;
	N = 0;
	N2 = 0;
	f = 0;
	Data = NULL;
	nb_solutions = 0;
	data_size = 0;
	start_column = 0;
	FstLen = NULL;
	Len = NULL;
	nb_orbits_at_level_3 = 0;
	nb_orb_total = 0;
	All_Orbits = NULL;
	Nb_orb = NULL;
	Orbit_idx = NULL;
	Position = NULL;
	Fo_first = NULL;
	nb_flag_orbits = 0;
	Flag_orbits = NULL;
	data1 = NULL;
	data2 = NULL;
	Basis1 = NULL;
	Basis2 = NULL;
	//Basis3 = NULL;
	B = NULL;
	v1 = NULL;
	v2 = NULL;
	v3 = NULL;
	transporter1 = NULL;
	transporter2 = NULL;
	transporter3 = NULL;
	Elt1 = NULL;
	coset_reps = NULL;

}

semifield_substructure::~semifield_substructure()
{

}

void semifield_substructure::init()
// allocates the arrays and matrices
{
	Basis1 = NEW_int(SC->k * SC->k2);
	Basis2 = NEW_int(SC->k * SC->k2);
	//Basis3 = NEW_int(SC->k * SC->k2);
	B = NEW_int(SC->k2);
	Gr3 = NEW_OBJECT(geometry::grassmann);
	Gr2 = NEW_OBJECT(geometry::grassmann);
	transporter1 = NEW_int(SC->A->elt_size_in_int);
	transporter2 = NEW_int(SC->A->elt_size_in_int);
	transporter3 = NEW_int(SC->A->elt_size_in_int);


	Gr3->init(SC->k, 3, SC->Mtx->GFq, 0 /* verbose_level */);
	N = Gr3->nb_of_subspaces(0 /* verbose_level */);
	Gr2->init(SC->k, 2, SC->Mtx->GFq, 0 /* verbose_level */);
	N2 = Gr2->nb_of_subspaces(0 /* verbose_level */);
	data1 = NEW_lint(SC->k);
	data2 = NEW_lint(SC->k);
	v1 = NEW_int(SC->k);
	v2 = NEW_int(SC->k);
	v3 = NEW_int(SC->k);
	Elt1 = NEW_int(SC->A->elt_size_in_int);
}

void semifield_substructure::compute_cases(
		int nb_non_unique_cases,
		int *Non_unique_cases, long int *Non_unique_cases_go,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, sum;

	if (f_v) {
		cout << "computing non-unique cases with non-trivial group:" << endl;
		}

	Non_unique_cases_with_non_trivial_group = NEW_int(nb_non_unique_cases);
	nb_non_unique_cases_with_non_trivial_group = 0;
	for (i = 0; i < nb_non_unique_cases; i++) {
		a = Non_unique_cases[i];
		if (Non_unique_cases_go[i] > 1) {
			Non_unique_cases_with_non_trivial_group
				[nb_non_unique_cases_with_non_trivial_group++] = a;
			}
		}
	if (f_v) {
		cout << "There are " << nb_non_unique_cases_with_non_trivial_group
			<< " cases with more than one solution and with a "
				"non-trivial group" << endl;
		cout << "They are:" << endl;
		Orbiter->Int_vec->matrix_print(Non_unique_cases_with_non_trivial_group,
			nb_non_unique_cases_with_non_trivial_group / 10 + 1, 10);
		}


	Need_orbits_fst = NEW_int(nb_non_unique_cases_with_non_trivial_group);
	Need_orbits_len = NEW_int(nb_non_unique_cases_with_non_trivial_group);
	for (i = 0; i < nb_non_unique_cases_with_non_trivial_group; i++) {
		a = Non_unique_cases_with_non_trivial_group[i];
		Need_orbits_fst[i] = FstLen[2 * a + 0];
		Need_orbits_len[i] = FstLen[2 * a + 1];
		}


	sum = 0;
	for (i = 0; i < nb_non_unique_cases_with_non_trivial_group; i++) {
		sum += Need_orbits_len[i];
		}
	{
	tally C;

	C.init(Need_orbits_len,
			nb_non_unique_cases_with_non_trivial_group, FALSE,
			0);
	if (f_v) {
		cout << "classification of Len amongst the need orbits cases:" << endl;
		C.print_naked(TRUE);
		cout << endl;
		}
	}
	if (f_v) {
		cout << "For a total of " << sum << " semifields" << endl;
		}
}


void semifield_substructure::compute_orbits(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int a, o, fst, len;

	if (f_v) {
		cout << "semifield_substructure::compute_orbits" << endl;
		}
	if (f_v) {
		cout << "computing all orbits:" << endl;
		}



	All_Orbits =
			new orbit_of_subspaces **[nb_non_unique_cases_with_non_trivial_group];
	Nb_orb = NEW_int(nb_non_unique_cases_with_non_trivial_group);
	Position = NEW_pint(nb_non_unique_cases_with_non_trivial_group);
	Orbit_idx = NEW_pint(nb_non_unique_cases_with_non_trivial_group);

	nb_orb_total = 0;
	for (o = 0; o < nb_non_unique_cases_with_non_trivial_group; o++) {



		a = Non_unique_cases_with_non_trivial_group[o];

		fst = Need_orbits_fst[o];
		len = Need_orbits_len[o];

		All_Orbits[o] = new orbit_of_subspaces *[len];

		if (f_v) {
			cout << "case " << o << " / "
				<< nb_non_unique_cases_with_non_trivial_group
				<< " with " << len
				<< " semifields" << endl;
			}

		int *f_reached;
		int *position;
		int *orbit_idx;

		f_reached = NEW_int(len);
		position = NEW_int(len);
		orbit_idx = NEW_int(len);

		Orbiter->Int_vec->zero(f_reached, len);
		Orbiter->Int_vec->mone(position, len);

		int cnt, f;

		cnt = 0;

		for (f = 0; f < len; f++) {
			if (f_reached[f]) {
				continue;
				}
			orbit_of_subspaces *Orb;
			long int *input_data;


			input_data = Data + (fst + f) * data_size + start_column;

			if (f_vvv) {
				cout << "case " << o << " / "
					<< nb_non_unique_cases_with_non_trivial_group
					<< " is original case " << a << " at "
					<< fst << " with " << len
					<< " semifields. Computing orbit of "
					"semifield " << f << " / " << len << endl;
				cout << "Orbit rep "
					<< f << ":" << endl;
				Orbiter->Lint_vec->print(cout, input_data, SC->k);
				cout << endl;
				}
			if (FALSE) {
				cout << "The stabilizer is:" << endl;
				L3->Stabilizer_gens[a].print_generators(cout);
			}

			SC->compute_orbit_of_subspaces(input_data,
				&L3->Stabilizer_gens[a],
				Orb,
				verbose_level - 4);
			if (f_vv) {
				cout << "case " << o << " / "
					<< nb_non_unique_cases_with_non_trivial_group
					<< " is original case " << a << " at "
					<< fst << " with " << len
					<< " semifields. Orbit of semifield " << f << " / "
					<< len << " has length "
					<< Orb->used_length << endl;
				}

			int idx, g, c;

			c = 0;
			for (g = 0; g < len; g++) {
				if (f_reached[g]) {
					continue;
					}
				if (FALSE) {
					cout << "testing subspace " << g << " / " << len << ":" << endl;
				}
				if (Orb->find_subspace_lint(
						Data + (fst + g) * data_size + start_column,
						idx, 0 /*verbose_level*/)) {
					f_reached[g] = TRUE;
					position[g] = idx;
					orbit_idx[g] = cnt;
					c++;
					}
				}
			if (c != Orb->used_length) {
				cout << "c != Orb->used_length" << endl;
				cout << "Orb->used_length=" << Orb->used_length << endl;
				cout << "c=" << c << endl;
				exit(1);
				}
			All_Orbits[o][cnt] = Orb;

			cnt++;

			}

		if (f_vv) {
			cout << "case " << o << " / "
				<< nb_non_unique_cases_with_non_trivial_group
				<< " with " << len << " semifields done, there are "
				<< cnt << " orbits in this case." << endl;
			}

		Nb_orb[o] = cnt;
		nb_orb_total += cnt;

		Position[o] = position;
		Orbit_idx[o] = orbit_idx;

		FREE_int(f_reached);



		}

	if (f_v) {
		cout << "Number of orbits:" << endl;
		for (o = 0; o < nb_non_unique_cases_with_non_trivial_group; o++) {
			cout << o << " : " << Nb_orb[o] << endl;
			}
		cout << "Total number of orbits = " << nb_orb_total << endl;
		}
	if (f_v) {
		cout << "semifield_substructure::compute_orbits done" << endl;
		}
}


void semifield_substructure::compute_flag_orbits(int verbose_level)
// initializes Fo_first and Flag_orbits
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int o, idx, h, fst, g;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "semifield_substructure::compute_flag_orbits" << endl;
		}
	if (f_v) {
		cout << "Counting number of flag orbits:" << endl;
		}
	nb_flag_orbits = 0;
	for (o = 0; o < nb_orbits_at_level_3; o++) {

		if (FALSE) {
			cout << "orbit " << o << " number of semifields = "
				<< Len[o] << " group order = "
				<< L3->Stabilizer_gens[o].group_order_as_lint() << endl;
			}
		if (Len[o] == 0) {
			}
		else if (Len[o] == 1) {
			nb_flag_orbits += 1;
			}
		else if (L3->Stabilizer_gens[o].group_order_as_lint() == 1) {
			nb_flag_orbits += Len[o];
			}
		else {
			if (!Sorting.int_vec_search(
				Non_unique_cases_with_non_trivial_group,
				nb_non_unique_cases_with_non_trivial_group, o, idx)) {
				cout << "cannot find orbit " << o
						<< " in the list " << endl;
				exit(1);
				}
			if (f_vv) {
				cout << "Found orbit " << o
						<< " at position " << idx << endl;
				}
			nb_flag_orbits += Nb_orb[idx];
			}

		} // next o
	if (f_v) {
		cout << "nb_flag_orbits = " << nb_flag_orbits << endl;
		}


	if (f_v) {
		cout << "Computing Flag_orbits:" << endl;
		}


	Fo_first = NEW_int(nb_orbits_at_level_3);

	Flag_orbits = NEW_OBJECT(invariant_relations::flag_orbits);
	Flag_orbits->init(
		SC->A, SC->AS,
		nb_orbits_at_level_3 /* nb_primary_orbits_lower */,
		SC->k /* pt_representation_sz */,
		nb_flag_orbits /* nb_flag_orbits */,
		1 /* upper_bound_for_number_of_traces */, // ToDo
		NULL /* void (*func_to_free_received_trace)(void *trace_result, void *data, int verbose_level) */,
		NULL /* void (*func_latex_report_trace)(std::ostream &ost, void *trace_result, void *data, int verbose_level)*/,
		NULL /* void *free_received_trace_data */,
		verbose_level);


	h = 0;
	for (o = 0; o < nb_orbits_at_level_3; o++) {

		long int *data;

		Fo_first[o] = h;
		fst = FstLen[2 * o + 0];

		if (Len[o] == 0) {
			// nothing to do here
		}
		else if (Len[o] == 1) {
			data = Data +
					(fst + 0) * data_size + start_column;
			Flag_orbits->Flag_orbit_node[h].init(
					Flag_orbits,
				h /* flag_orbit_index */,
				o /* downstep_primary_orbit */,
				0 /* downstep_secondary_orbit */,
				1 /* downstep_orbit_len */,
				FALSE /* f_long_orbit */,
				data /* int *pt_representation */,
				&L3->Stabilizer_gens[o],
				0 /*verbose_level - 2 */);
			h++;
		}
		else if (L3->Stabilizer_gens[o].group_order_as_lint() == 1) {
			for (g = 0; g < Len[o]; g++) {
				data = Data +
						(fst + g) * data_size + start_column;
				Flag_orbits->Flag_orbit_node[h].init(
						Flag_orbits,
					h /* flag_orbit_index */,
					o /* downstep_primary_orbit */,
					g /* downstep_secondary_orbit */,
					1 /* downstep_orbit_len */,
					FALSE /* f_long_orbit */,
					data /* int *pt_representation */,
					&L3->Stabilizer_gens[o],
					0 /*verbose_level - 2*/);
				h++;
			}
		}
		else {
			if (!Sorting.int_vec_search(
					Non_unique_cases_with_non_trivial_group,
					nb_non_unique_cases_with_non_trivial_group, o, idx)) {
				cout << "cannot find orbit " << o << " in the list " << endl;
				exit(1);
			}
			if (FALSE) {
				cout << "Found orbit " << o
					<< " at position " << idx << endl;
			}
			for (g = 0; g < Nb_orb[idx]; g++) {
				orbit_of_subspaces *Orb;
				groups::strong_generators *gens;
				ring_theory::longinteger_object go;

				Orb = All_Orbits[idx][g];
				data = Orb->Subspaces_lint[Orb->position_of_original_subspace];
				L3->Stabilizer_gens[o].group_order(go);
				gens = Orb->stabilizer_orbit_rep(
					go /* full_group_order */, 0 /*verbose_level - 1*/);
				Flag_orbits->Flag_orbit_node[h].init(
						Flag_orbits,
					h /* flag_orbit_index */,
					o /* downstep_primary_orbit */,
					g /* downstep_secondary_orbit */,
					Orb->used_length /* downstep_orbit_len */,
					FALSE /* f_long_orbit */,
					data /* int *pt_representation */,
					gens,
					0 /*verbose_level - 2*/);
				h++;
			}
		}


	}
	if (h != nb_flag_orbits) {
		cout << "h != nb_flag_orbits" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "Finished initializing flag orbits. "
				"Number of flag orbits = " << nb_flag_orbits << endl;
	}
	if (f_v) {
		cout << "semifield_substructure::compute_flag_orbits done" << endl;
		}
}


void semifield_substructure::do_classify(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	os_interface Os;

	if (f_v) {
		cout << "semifield_substructure::do_classify" << endl;
		}
	if (f_v) {
		cout << "Computing classification:" << endl;
	}

	//int depth0 = 3;





	int po, so;
	ring_theory::longinteger_object go;
	int f_skip = FALSE;
	int i;



	int *f_processed; // [nb_flag_orbits]
	int nb_processed;


	f_processed = NEW_int(nb_flag_orbits);
	Orbiter->Int_vec->zero(f_processed, nb_flag_orbits);
	nb_processed = 0;




	SC->A->group_order(go);

	SCWS->Semifields->init(SC->A, SC->AS,
			nb_flag_orbits, SC->k, go, verbose_level);


	Flag_orbits->nb_primary_orbits_upper = 0;

	for (f = 0; f < nb_flag_orbits; f++) {


		double progress;

		if (f_processed[f]) {
			continue;
		}

		progress = ((double) nb_processed * 100. ) /
				(double) nb_flag_orbits;

		if (f_v) {
			Os.time_check(cout, SCWS->t0);
			cout << " : Defining new orbit "
				<< Flag_orbits->nb_primary_orbits_upper
				<< " from flag orbit " << f << " / "
				<< nb_flag_orbits << " progress="
				<< progress << "% "
				"nb semifields = " << Flag_orbits->nb_primary_orbits_upper << endl;

		}
		Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit =
				Flag_orbits->nb_primary_orbits_upper;



		po = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
		so = Flag_orbits->Flag_orbit_node[f].downstep_secondary_orbit;
		if (f_v) {
			cout << "po=" << po << " so=" << so << endl;
		}
		Orbiter->Lint_vec->copy(
				Flag_orbits->Pt + f * Flag_orbits->pt_representation_sz,
				data1, SC->k);
		if (f_v) {
			cout << "data1=";
			Orbiter->Lint_vec->print(cout, data1, SC->k);
			cout << endl;
		}

		groups::strong_generators *Aut_gens;
		ring_theory::longinteger_object go;

		Aut_gens = Flag_orbits->Flag_orbit_node[f].gens->create_copy();
		coset_reps = NEW_OBJECT(data_structures_groups::vector_ge);
		coset_reps->init(SC->A, verbose_level - 2);


		for (i = 0; i < SC->k; i++) {
			SC->matrix_unrank(data1[i], Basis1 + i * SC->k2);
		}
		if (f_v) {
			cout << "Basis1=" << endl;
			Orbiter->Int_vec->matrix_print(Basis1, SC->k, SC->k2);
			SC->basis_print(Basis1, SC->k);
		}
		f_skip = FALSE;
		for (i = 0; i < SC->k; i++) {
			v3[i] = Basis1[2 * SC->k2 + i * SC->k + 0];
		}
		if (!SC->Mtx->GFq->Linear_algebra->is_unit_vector(v3, SC->k, SC->k - 1)) {
			cout << "flag orbit " << f << " / "
					<< nb_flag_orbits
					<< " 1st col of third matrix is = ";
			Orbiter->Int_vec->print(cout, v3, SC->k);
			cout << " which is not the (k-1)-th unit vector, "
					"so we skip" << endl;
			f_skip = TRUE;
		}

		if (f_skip) {
			if (f_v) {
				cout << "flag orbit " << f << " / "
						<< nb_flag_orbits
					<< ", first vector is not the unit vector, "
					"so we skip" << endl;
			}
			f_processed[f] = TRUE;
			nb_processed++;
			continue;
		}
		if (f_v) {
			cout << "flag orbit " << f << " / " << nb_flag_orbits
				<< ", looping over the " << N << " subspaces" << endl;
		}

		TR = NEW_OBJECTS(trace_record, N);


		if (f_v) {
			Os.time_check(cout, SCWS->t0);
			cout << " : flag orbit " << f << " / " << nb_flag_orbits
				<< ", looping over the " << N << " subspaces, "
				"before loop_over_all_subspaces" << endl;
		}


		loop_over_all_subspaces(f_processed, nb_processed,
				verbose_level - 3);


		if (f_v) {
			cout << "flag orbit " << f << " / " << nb_flag_orbits
				<< ", looping over the " << N << " subspaces, "
				"after loop_over_all_subspaces" << endl;
		}


		save_trace_record(
				TR,
				SCWS->f_trace_record_prefix, SCWS->trace_record_prefix,
				Flag_orbits->nb_primary_orbits_upper,
			f, po, so, N);


		FREE_OBJECTS(TR);

		int cl;
		ring_theory::longinteger_object go1, Cl, ago, ago1;
		ring_theory::longinteger_domain D;

		Aut_gens->group_order(go);
		cl = coset_reps->len;
		Cl.create(cl, __FILE__, __LINE__);
		D.mult(go, Cl, ago);
		if (f_v) {
			cout << "Semifield "
					<< Flag_orbits->nb_primary_orbits_upper
					<< ", Ago = starter * number of cosets = " << ago
					<< " = " << go << " * " << Cl
					<< " created from flag orbit " << f << " / "
				<< nb_flag_orbits << " progress="
				<< progress << "%" << endl;
		}

		groups::strong_generators *Stab;

		Stab = NEW_OBJECT(groups::strong_generators);
		if (f_v) {
			cout << "flag orbit " << f << " / " << nb_flag_orbits
				<< ", semifield isotopy class "
				<< Flag_orbits->nb_primary_orbits_upper <<
				"computing stabilizer" << endl;
		}
		Stab->init_group_extension(Aut_gens,
				coset_reps, cl /* index */,
				verbose_level);
		Stab->group_order(ago1);
		if (f_v) {
			cout << "flag orbit " << f << " / " << nb_flag_orbits
				<< ", semifield isotopy class "
				<< Flag_orbits->nb_primary_orbits_upper <<
				" computing stabilizer done, order = " <<  ago1 << endl;
		}


		SCWS->Semifields->Orbit[Flag_orbits->nb_primary_orbits_upper].init(
				SCWS->Semifields,
			Flag_orbits->nb_primary_orbits_upper,
			Stab, data1, NULL /* extra_data */, verbose_level);

		FREE_OBJECT(Aut_gens);

		f_processed[f] = TRUE;
		nb_processed++;
		Flag_orbits->nb_primary_orbits_upper++;



	} // next f

	FREE_int(f_processed);


	SCWS->Semifields->nb_orbits = Flag_orbits->nb_primary_orbits_upper;


	if (f_v) {
		cout << "Computing classification done, we found "
				<< SCWS->Semifields->nb_orbits
				<< " semifields" << endl;
		Os.time_check(cout, SCWS->t0);
		cout << endl;
	}
	if (f_v) {
		cout << "semifield_substructure::do_classify done" << endl;
		}
}

void semifield_substructure::loop_over_all_subspaces(
		int *f_processed, int &nb_processed, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	field_theory::finite_field *F;
	int rk, i, f2;
	int k, k2;
	int f_skip;
	int trace_po;
	int ret;
	data_structures::sorting Sorting;


#if 0
	if (f == 1) {
		verbose_level += 10;
		f_v = f_vv = f_vvv = TRUE;
		cout << "CASE 1, STARTS HERE" << endl;
	}
#endif

	if (f_v) {
		cout << "loop_over_all_subspaces" << endl;
	}


	k = SC->k;
	k2 = SC->k2;
	F = SC->Mtx->GFq;

	for (rk = 0; rk < N; rk++) {

		trace_record *T;

		T = TR + rk;

		T->coset = rk;


		if (f_vv) {
			cout << "flag orbit " << f << " / "
				<< nb_flag_orbits << ", subspace "
				<< rk << " / " << N << ":" << endl;
		}

		for (i = 0; i < k; i++) {
			SC->matrix_unrank(data1[i], Basis1 + i * k2);
		}
		if (f_vvv) {
			SC->basis_print(Basis1, k);
		}


		// unrank the subspace:
		Gr3->unrank_lint_here_and_extend_basis(B, rk,
				0 /* verbose_level */);

		// multiply the matrices to get the matrices
		// adapted to the subspace:
		// the first three matrices are the generators
		// for the subspace.
		F->Linear_algebra->mult_matrix_matrix(B, Basis1, Basis2, k, k, k2,
				0 /* verbose_level */);


		if (f_vvv) {
			cout << "base change matrix B=" << endl;
			Orbiter->Int_vec->matrix_print_bitwise(B, k, k);

			cout << "Basis2 = B * Basis1 (before trace)=" << endl;
			Orbiter->Int_vec->matrix_print_bitwise(Basis2, k, k2);
			SC->basis_print(Basis2, k);
		}


		if (f_vv) {
			cout << "before trace_to_level_three" << endl;
		}
		ret = L3->trace_to_level_three(
			Basis2,
			k /* basis_sz */,
			transporter1,
			trace_po,
			verbose_level - 3);

		f_skip = FALSE;

		if (ret == FALSE) {
			cout << "flag orbit " << f << " / "
				<< nb_flag_orbits << ", subspace "
				<< rk << " / " << N
				<< " : trace_to_level_three return FALSE" << endl;

			f_skip = TRUE;
		}

		T->trace_po = trace_po;

		if (f_vvv) {
			cout << "After trace, trace_po = "
					<< trace_po << endl;
			cout << "Basis2 (after trace)=" << endl;
			Orbiter->Int_vec->matrix_print_bitwise(Basis2, k, k2);
			SC->basis_print(Basis2, k);
		}


		for (i = 0; i < k; i++) {
			v1[i] = Basis2[0 * k2 + i * k + 0];
		}
		for (i = 0; i < k; i++) {
			v2[i] = Basis2[1 * k2 + i * k + 0];
		}
		for (i = 0; i < k; i++) {
			v3[i] = Basis2[2 * k2 + i * k + 0];
		}
		if (f_skip == FALSE) {
			if (!F->Linear_algebra->is_unit_vector(v1, k, 0) ||
					!F->Linear_algebra->is_unit_vector(v2, k, 1) ||
					!F->Linear_algebra->is_unit_vector(v3, k, k - 1)) {
				f_skip = TRUE;
			}
		}
		T->f_skip = f_skip;

		if (f_skip) {
			if (f_vv) {
				cout << "skipping this case " << trace_po
					<< " because pivot is not "
						"in the last row." << endl;
			}
		}
		else {

			F->Linear_algebra->Gauss_int_with_given_pivots(
				Basis2 + 3 * k2,
				FALSE /* f_special */,
				TRUE /* f_complete */,
				SC->desired_pivots + 3,
				k - 3 /* nb_pivots */,
				k - 3, k2,
				0 /*verbose_level - 2*/);
			if (f_vvv) {
				cout << "Basis2 after RREF(2)=" << endl;
				Orbiter->Int_vec->matrix_print_bitwise(Basis2, k, k2);
				SC->basis_print(Basis2, k);
			}

			for (i = 0; i < k; i++) {
				data2[i] = SC->matrix_rank(Basis2 + i * k2);
			}
			if (f_vvv) {
				cout << "data2=";
				Orbiter->Lint_vec->print(cout, data2, k);
				cout << endl;
			}

			int solution_idx;

			if (f_vvv) {
				cout << "before find_semifield_in_table" << endl;
			}
			if (!find_semifield_in_table(
				trace_po,
				data2 /* given_data */,
				solution_idx,
				verbose_level)) {

				cout << "flag orbit " << f << " / "
					<< nb_flag_orbits << ", subspace "
					<< rk << " / " << N << ":" << endl;

				cout << "find_semifield_in_table returns FALSE" << endl;

				cout << "trace_po=" << trace_po << endl;
				cout << "data2=";
				Orbiter->Lint_vec->print(cout, data2, k);
				cout << endl;

				cout << "Basis2 after RREF(2)=" << endl;
				Orbiter->Int_vec->matrix_print_bitwise(Basis2, k, k2);
				SC->basis_print(Basis2, k);

				exit(1);
			}


			T->solution_idx = solution_idx;
			T->nb_sol = Len[trace_po];

			if (f_vvv) {
				cout << "solution_idx=" << solution_idx << endl;
			}

			long int go;
			go = L3->Stabilizer_gens[trace_po].group_order_as_lint();

			T->go = go;
			T->pos = -1;
			T->so = -1;
			T->orbit_len = -1;
			T->f2 = -1;

			if (f_vv) {
				cout << "flag orbit " << f << " / "
					<< nb_flag_orbits << ", subspace "
					<< rk << " / " << N << " trace_po="
					<< trace_po << " go=" << go
					<< " solution_idx=" << solution_idx << endl;
			}


			if (go == 1) {
				if (f_vv) {
					cout << "This starter case has a trivial "
							"group order" << endl;
				}

				f2 = Fo_first[trace_po] + solution_idx;

				T->so = solution_idx;
				T->orbit_len = 1;
				T->f2 = f2;

				if (f2 == f) {
					if (f_vv) {
						cout << "We found an automorphism" << endl;
					}
					coset_reps->append(transporter1, verbose_level - 2);
				}
				else {
					if (!f_processed[f2]) {
						if (f_vv) {
							cout << "We are identifying with "
									"flag orbit " << f2 << ", which is "
									"po=" << trace_po << " so="
									<< solution_idx << endl;
						}
						Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit =
								Flag_orbits->nb_primary_orbits_upper;
						Flag_orbits->Flag_orbit_node[f2].f_fusion_node
							= TRUE;
						Flag_orbits->Flag_orbit_node[f2].fusion_with
							= f;
						Flag_orbits->Flag_orbit_node[f2].fusion_elt
							= NEW_int(SC->A->elt_size_in_int);
						SC->A->element_invert(
							transporter1,
							Flag_orbits->Flag_orbit_node[f2].fusion_elt,
							0);
						f_processed[f2] = TRUE;
						nb_processed++;
						if (f_vv) {
							cout << "Flag orbit f2 = " << f2
								<< " has been fused with flag orbit "
								<< f << endl;
						}
					}
					else {
						if (f_vv) {
							cout << "Flag orbit f2 = " << f2
									<< " has already been processed, "
											"nothing to do here" << endl;
						}
					}
				}
			}
			else {
				if (Len[trace_po] == 1) {
					if (f_vv) {
						cout << "This starter case has only "
								"one solution" << endl;
					}
					f2 = Fo_first[trace_po] + 0;

					T->pos = -1;
					T->so = 0;
					T->orbit_len = 1;
					T->f2 = f2;


					if (f2 == f) {
						if (f_vv) {
							cout << "We found an automorphism" << endl;
						}
						coset_reps->append(transporter1, verbose_level - 2);
					}
					else {
						if (!f_processed[f2]) {
							if (f_vv) {
								cout << "We are identifying with po="
										<< trace_po << " so=" << 0
										<< ", which is flag orbit "
										<< f2 << endl;
							}
							Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit =
									Flag_orbits->nb_primary_orbits_upper;
							Flag_orbits->Flag_orbit_node[f2].f_fusion_node
								= TRUE;
							Flag_orbits->Flag_orbit_node[f2].fusion_with
								= f;
							Flag_orbits->Flag_orbit_node[f2].fusion_elt
								= NEW_int(SC->A->elt_size_in_int);
							SC->A->element_invert(transporter1,
								Flag_orbits->Flag_orbit_node[f2].fusion_elt,
								0);
							f_processed[f2] = TRUE;
							nb_processed++;
							if (f_vv) {
								cout << "Flag orbit f2 = " << f2
									<< " has been fused with "
										"flag orbit " << f << endl;
							}
						}
						else {
							if (f_vv) {
								cout << "Flag orbit f2 = " << f2
									<< " has already been processed, "
										"nothing to do here" << endl;
							}
						}
					}
				}
				else {
					// now we have a starter_case with more
					// than one solution and
					// with a non-trivial group.
					// Those cases are collected in
					// Non_unique_cases_with_non_trivial_group
					// [nb_non_unique_cases_with_non_trivial_group];

					int non_unique_case_idx, orbit_idx, position;
					orbit_of_subspaces *Orb;

					if (!Sorting.int_vec_search(
						Non_unique_cases_with_non_trivial_group,
						nb_non_unique_cases_with_non_trivial_group,
						trace_po, non_unique_case_idx)) {
						cout << "cannot find in Non_unique_cases_with_"
								"non_trivial_group array" << endl;
						exit(1);
					}
					orbit_idx = Orbit_idx[non_unique_case_idx][solution_idx];
					position = Position[non_unique_case_idx][solution_idx];
					Orb = All_Orbits[non_unique_case_idx][orbit_idx];
					f2 = Fo_first[trace_po] + orbit_idx;



					T->pos = position;
					T->so = orbit_idx;
					T->orbit_len = Orb->used_length;
					T->f2 = f2;



					if (f_vv) {
						cout << "orbit_idx = " << orbit_idx
								<< " position = " << position
								<< " f2 = " << f2 << endl;
					}
					Orb->get_transporter(position, transporter2,
							0 /*verbose_level */);
					SC->A->element_invert(transporter2, Elt1, 0);
					SC->A->element_mult(transporter1, Elt1,
							transporter3, 0);

					if (f2 == f) {
						if (f_vv) {
							cout << "We found an automorphism" << endl;
							}
						coset_reps->append(transporter3, verbose_level - 2);
					}
					else {
						if (!f_processed[f2]) {
							if (f_vv) {
								cout << "We are identifying with po="
									<< trace_po << " so=" << solution_idx
									<< ", which is flag orbit "
									<< f2 << endl;
							}
							Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit =
									Flag_orbits->nb_primary_orbits_upper;
							Flag_orbits->Flag_orbit_node[f2].f_fusion_node
								= TRUE;
							Flag_orbits->Flag_orbit_node[f2].fusion_with
								= f;
							Flag_orbits->Flag_orbit_node[f2].fusion_elt =
								NEW_int(SC->A->elt_size_in_int);
							SC->A->element_invert(transporter3,
								Flag_orbits->Flag_orbit_node[f2].fusion_elt,
								0);
							f_processed[f2] = TRUE;
							nb_processed++;
							if (f_vv) {
								cout << "Flag orbit f2 = " << f2
									<< " has been fused with flag "
									"orbit " << f << endl;
							}
						}
						else {
							if (f_vv) {
								cout << "Flag orbit f2 = " << f2
									<< " has already been processed, "
											"nothing to do here" << endl;
							}
						}
					}

				}
			}
		} // if !f_skip
	} // next rk
	if (f_v) {
		cout << "loop_over_all_subspaces done" << endl;
	}

}

void semifield_substructure::all_two_dimensional_subspaces(
		int *Trace_po, int verbose_level)
// input is in data1[]
// Trace_po[N2]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	field_theory::finite_field *F;
	int rk, i;
	int k, k2;
	int trace_po;
	data_structures::sorting Sorting;



	if (f_v) {
		cout << "semifield_substructure::all_two_dimensional_subspaces" << endl;
	}


	k = SC->k;
	k2 = SC->k2;
	F = SC->Mtx->GFq;

	for (rk = 0; rk < N2; rk++) {


		if (f_vv) {
			cout << "semifield_substructure::all_two_dimensional_subspaces "
				"subspace "
				<< rk << " / " << N2 << ":" << endl;
		}

		for (i = 0; i < k; i++) {
			SC->matrix_unrank(data1[i], Basis1 + i * k2);
		}
		if (f_vvv) {
			SC->basis_print(Basis1, k);
		}


		// unrank the subspace:
		Gr2->unrank_lint_here_and_extend_basis(B, rk,
				0 /* verbose_level */);

		// multiply the matrices to get the matrices
		// adapted to the subspace:
		// the first three matrices are the generators
		// for the subspace.
		F->Linear_algebra->mult_matrix_matrix(B, Basis1, Basis2, k, k, k2,
				0 /* verbose_level */);


		if (f_vvv) {
			cout << "base change matrix B=" << endl;
			Orbiter->Int_vec->matrix_print_bitwise(B, k, k);

			cout << "Basis2 = B * Basis1 (before trace)=" << endl;
			Orbiter->Int_vec->matrix_print_bitwise(Basis2, k, k2);
			SC->basis_print(Basis2, k);
		}


		if (f_vv) {
			cout << "before trace_to_level_two" << endl;
		}

		L3->trace_to_level_two(
				Basis2,
				k /* basis_sz */,
				//Basis3,
				transporter1,
				trace_po,
				verbose_level - 3);

		Trace_po[rk] = trace_po;

		if (f_vvv) {
			cout << "After trace, trace_po = "
					<< trace_po << endl;
			//cout << "Basis3 (after trace)=" << endl;
			//int_matrix_print_bitwise(Basis3, k, k2);
			//SC->basis_print(Basis3, k);
		}
	}
	if (f_v) {
		cout << "semifield_substructure::all_two_dimensional_subspaces "
				"done" << endl;
	}
}


int semifield_substructure::identify(long int *data,
		int &rk, int &trace_po, int &fo, int &po,
		int *transporter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	field_theory::finite_field *F;
	int i, f2;
	int k, k2;
	int f_skip;
	//int trace_po;
	int ret;
	data_structures::sorting Sorting;
	int solution_idx;



	if (f_v) {
		cout << "semifield_substructure::identify" << endl;
	}


	k = SC->k;
	k2 = SC->k2;
	F = SC->Mtx->GFq;

	for (rk = 0; rk < N; rk++) {

		for (i = 0; i < k; i++) {
			SC->matrix_unrank(data[i], Basis1 + i * k2);
		}
		if (f_vvv) {
			SC->basis_print(Basis1, k);
		}


		// unrank the subspace:
		Gr3->unrank_lint_here_and_extend_basis(B, rk,
				0 /* verbose_level */);

		// multiply the matrices to get the matrices
		// adapted to the subspace:
		// the first three matrices are the generators
		// for the subspace.
		F->Linear_algebra->mult_matrix_matrix(B, Basis1, Basis2, k, k, k2,
				0 /* verbose_level */);


		if (f_vvv) {
			cout << "semifield_substructure::identify "
					"base change matrix B=" << endl;
			Orbiter->Int_vec->matrix_print_bitwise(B, k, k);

			cout << "semifield_substructure::identify "
					"Basis2 = B * Basis1 (before trace)=" << endl;
			Orbiter->Int_vec->matrix_print_bitwise(Basis2, k, k2);
			SC->basis_print(Basis2, k);
		}


		if (f_vv) {
			cout << "semifield_substructure::identify "
					"before trace_to_level_three" << endl;
		}
		ret = L3->trace_to_level_three(
			Basis2,
			k /* basis_sz */,
			transporter1,
			trace_po,
			verbose_level - 3);

		f_skip = FALSE;

		if (ret == FALSE) {
			cout << "semifield_substructure::identify "
					"trace_to_level_three return FALSE" << endl;

			f_skip = TRUE;
		}


		if (f_vvv) {
			cout << "semifield_substructure::identify "
					"After trace, trace_po = "
					<< trace_po << endl;
			cout << "semifield_substructure::identify "
					"Basis2 (after trace)=" << endl;
			Orbiter->Int_vec->matrix_print_bitwise(Basis2, k, k2);
			SC->basis_print(Basis2, k);
		}


		for (i = 0; i < k; i++) {
			v1[i] = Basis2[0 * k2 + i * k + 0];
		}
		for (i = 0; i < k; i++) {
			v2[i] = Basis2[1 * k2 + i * k + 0];
		}
		for (i = 0; i < k; i++) {
			v3[i] = Basis2[2 * k2 + i * k + 0];
		}
		if (f_skip == FALSE) {
			if (!F->Linear_algebra->is_unit_vector(v1, k, 0) ||
					!F->Linear_algebra->is_unit_vector(v2, k, 1) ||
					!F->Linear_algebra->is_unit_vector(v3, k, k - 1)) {
				f_skip = TRUE;
			}
		}

		if (f_skip) {
			if (f_vv) {
				cout << "semifield_substructure::identify "
						"skipping this case " << trace_po
					<< " because pivot is not "
						"in the last row." << endl;
			}
		}
		else {

			F->Linear_algebra->Gauss_int_with_given_pivots(
				Basis2 + 3 * k2,
				FALSE /* f_special */,
				TRUE /* f_complete */,
				SC->desired_pivots + 3,
				k - 3 /* nb_pivots */,
				k - 3, k2,
				0 /*verbose_level - 2*/);
			if (f_vvv) {
				cout << "semifield_substructure::identify "
						"Basis2 after RREF(2)=" << endl;
				Orbiter->Int_vec->matrix_print_bitwise(Basis2, k, k2);
				SC->basis_print(Basis2, k);
			}

			for (i = 0; i < k; i++) {
				data2[i] = SC->matrix_rank(Basis2 + i * k2);
			}
			if (f_vvv) {
				cout << "semifield_substructure::identify data2=";
				Orbiter->Lint_vec->print(cout, data2, k);
				cout << endl;
			}

			if (f_vvv) {
				cout << "semifield_substructure::identify "
						"before find_semifield_in_table" << endl;
			}
			if (!find_semifield_in_table(
				trace_po,
				data2 /* given_data */,
				solution_idx,
				verbose_level)) {

				cout << "semifield_substructure::identify "
						"find_semifield_in_table returns FALSE" << endl;

				cout << "semifield_substructure::identify "
						"trace_po=" << trace_po << endl;
				cout << "semifield_substructure::identify data2=";
				Orbiter->Lint_vec->print(cout, data2, k);
				cout << endl;

				cout << "semifield_substructure::identify "
						"Basis2 after RREF(2)=" << endl;
				Orbiter->Int_vec->matrix_print_bitwise(Basis2, k, k2);
				SC->basis_print(Basis2, k);

				return FALSE;
			}

			long int go;
			go = L3->Stabilizer_gens[trace_po].group_order_as_lint();

			//T->solution_idx = solution_idx;
			//T->nb_sol = Len[trace_po];
			if (go == 1) {
				if (f_vv) {
					cout << "This starter case has a trivial "
							"group order" << endl;
				}

				f2 = Fo_first[trace_po] + solution_idx;

				if (Flag_orbits->Flag_orbit_node[f2].f_fusion_node) {
					fo = Flag_orbits->Flag_orbit_node[f2].fusion_with;
					SC->A->element_mult(transporter1,
						Flag_orbits->Flag_orbit_node[f2].fusion_elt,
						transporter,
						0);
				}
				else {
					fo = f2;
					SC->A->element_move(transporter1,
						transporter,
						0);
				}
			}
			else if (Len[trace_po] == 1) {
				f2 = Fo_first[trace_po] + 0;
				if (Flag_orbits->Flag_orbit_node[f2].f_fusion_node) {
					fo = Flag_orbits->Flag_orbit_node[f2].fusion_with;
					SC->A->element_mult(transporter1,
						Flag_orbits->Flag_orbit_node[f2].fusion_elt,
						transporter,
						0);
				}
				else {
					fo = f2;
					SC->A->element_move(transporter1,
						transporter,
						0);
				}
			}
			else {
				// now we have a starter_case with more
				// than one solution and
				// with a non-trivial group.
				// Those cases are collected in
				// Non_unique_cases_with_non_trivial_group
				// [nb_non_unique_cases_with_non_trivial_group];

				int non_unique_case_idx, orbit_idx, position;
				orbit_of_subspaces *Orb;

				if (!Sorting.int_vec_search(
					Non_unique_cases_with_non_trivial_group,
					nb_non_unique_cases_with_non_trivial_group,
					trace_po, non_unique_case_idx)) {
					cout << "semifield_substructure::identify "
							"cannot find in Non_unique_cases_with_"
							"non_trivial_group array" << endl;
					exit(1);
				}
				orbit_idx = Orbit_idx[non_unique_case_idx][solution_idx];
				position = Position[non_unique_case_idx][solution_idx];
				Orb = All_Orbits[non_unique_case_idx][orbit_idx];
				f2 = Fo_first[trace_po] + orbit_idx;

				if (f_vv) {
					cout << "orbit_idx = " << orbit_idx
							<< " position = " << position
							<< " f2 = " << f2 << endl;
				}
				Orb->get_transporter(position, transporter2,
						0 /*verbose_level */);
				SC->A->element_invert(transporter2, Elt1, 0);
				SC->A->element_mult(transporter1, Elt1,
						transporter3, 0);
				SC->A->element_move(transporter3,
						transporter1, 0);
				if (Flag_orbits->Flag_orbit_node[f2].f_fusion_node) {
					fo = Flag_orbits->Flag_orbit_node[f2].fusion_with;
					SC->A->element_mult(transporter1,
						Flag_orbits->Flag_orbit_node[f2].fusion_elt,
						transporter,
						0);
				}
				else {
					fo = f2;
					SC->A->element_move(transporter1,
						transporter,
						0);
				}

			} // go != 1

			po = Flag_orbits->Flag_orbit_node[fo].upstep_primary_orbit;


			if (f_vvv) {
				cout << "semifield_substructure::identify done "
						"solution_idx=" << solution_idx
						<< "trace_po=" << trace_po
						<< "f2=" << f2
						<< "fo=" << fo
						<< "po=" << po
						<< endl;
			}
			return TRUE;
		} // end else


	} // next rk

	if (f_v) {
		cout << "semifield_substructure::identify done" << endl;
	}
	return FALSE;
}

int semifield_substructure::find_semifield_in_table(
	int po,
	long int *given_data,
	int &idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int fst, len, g;
	data_structures::sorting Sorting;


	if (f_v) {
		cout << "find_semifield_in_table" << endl;
		}

	idx = -1;

	if (f_v) {
		cout << "searching for: ";
		Orbiter->Lint_vec->print(cout, given_data, SC->k);
		cout << endl;
		}
	fst = FstLen[2 * po + 0];
	len = FstLen[2 * po + 1];
	if (f_v) {
		cout << "find_semifield_in_table po = " << po
				<< " len = " << len << endl;
		}
	// make sure that the first three integers
	// agree with what is stored
	// in the table for orbit po:
	if (len == 0) {
		cout << "find_semifield_in_table len == 0" << endl;
		return FALSE;
		}

	if (Sorting.lint_vec_compare(Data + fst * data_size + start_column,
			given_data, 3)) {
		cout << "find_semifield_in_table the first three entries "
				"of given_data do not match with what "
				"is in Data" << endl;
		exit(1);
	}

	if (len == 1) {
		if (Sorting.lint_vec_compare(Data + fst * data_size + start_column,
				given_data, SC->k)) {
			cout << "find_semifield_in_table len is 1 and the first six "
				"entries of given_data do not match with what "
				"is in Data" << endl;
			exit(1);
			}
		idx = 0;
	}
	else {
		for (g = 0; g < len; g++) {
			if (Sorting.lint_vec_compare(Data + (fst + g) * data_size + start_column,
					given_data, SC->k) == 0) {
				idx = g;
				break;
				}
		}
		if (g == len) {
			cout << "find_semifield_in_table cannot find the "
					"semifield in the table" << endl;
			for (g = 0; g < len; g++) {
				cout << g << " : " << fst + g << " : ";
				Orbiter->Lint_vec->print(cout,
						Data + (fst + g) * data_size + start_column,
						SC->k);
				cout << " : ";
				cout << Sorting.lint_vec_compare(
						Data + (fst + g) * data_size + start_column,
						given_data, SC->k);
				cout << endl;
			}
			return FALSE;
		}
	}

	if (f_v) {
		cout << "find_semifield_in_table done, idx = " << idx << endl;
	}
	return TRUE;
}



}}}


