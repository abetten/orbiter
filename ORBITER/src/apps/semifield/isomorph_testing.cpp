/*
 * isomorph_testing.cpp
 *
 *  Created on: May 8, 2019
 *      Author: betten
 *
 *  originally created on March 7, 2018
 *
 */


#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;




typedef class trace_record trace_record;

// global data:

int t0; // the system time when the program started

int find_semifield_in_table(semifield_lifting *L3,
	long int *Data, int nb_semifields, int data_size,
	int start_column, int *FstLen, int nb_orbits, int po,
	long int *given_data, int verbose_level);
int is_unit_vector(int *v, int len, int k);
void save_trace_record(trace_record *T,
		int iso, int f, int po, int so, int N);


#define MAX_FILES 1000

class trace_record {
public:
	int coset;
	int trace_po;
	int f_skip;
	int solution_idx;
	int nb_sol;
	int go;
	int pos;
	int so;
	int orbit_len;
	int f2;
};


int main(int argc, const char **argv)
{
	int i;
	int verbose_level = 0;
	int f_poly = FALSE;
	const char *poly = NULL;
	int f_order = FALSE;
	int order = 0;
	int f_dim_over_kernel = FALSE;
	int dim_over_kernel = 0;
	int f_prefix = FALSE;
	const char *prefix = "";
	int f_orbits_light = FALSE;

	t0 = os_ticks();
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-order") == 0) {
			f_order = TRUE;
			order = atoi(argv[++i]);
			cout << "-order " << order << endl;
			}
		else if (strcmp(argv[i], "-dim_over_kernel") == 0) {
			f_dim_over_kernel = TRUE;
			dim_over_kernel = atoi(argv[++i]);
			cout << "-dim_over_kernel " << dim_over_kernel << endl;
			}
		else if (strcmp(argv[i], "-prefix") == 0) {
			f_prefix = TRUE;
			prefix = argv[++i];
			cout << "-prefix " << prefix << endl;
			}
		else if (strcmp(argv[i], "-orbits_light") == 0) {
			f_orbits_light = TRUE;
			cout << "-orbits_light " << endl;
			}
		}


	if (!f_order) {
		cout << "please use option -order <order>" << endl;
		exit(1);
		}

	int p, e, e1, n, k, q, k2;
	number_theory_domain NT;

	NT.factor_prime_power(order, p, e);
	cout << "order = " << order << " = " << p << "^" << e << endl;

	if (f_dim_over_kernel) {
		if (e % dim_over_kernel) {
			cout << "dim_over_kernel does not divide e" << endl;
			exit(1);
			}
		e1 = e / dim_over_kernel;
		n = 2 * dim_over_kernel;
		k = dim_over_kernel;
		q = NT.i_power_j(p, e1);
		cout << "order=" << order << " n=" << n
			<< " k=" << k << " q=" << q << endl;
		}
	else {
		n = 2 * e;
		k = e;
		q = p;
		cout << "order=" << order << " n=" << n
			<< " k=" << k << " q=" << q << endl;
		}
	k2 = k * k;


	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	{




	const char *fname_FstLen = "semi64_liftings_FstLen.csv";
	const char *fname_liftings = "semi64_liftings_Data.csv";
	int *FstLen;
	int *Len;
	int nb_orbits;
	long int *Data;
	int nb_solutions, data_size;
	int start_column;
	int *Existing_cases;
	int *Existing_cases_fst;
	int *Existing_cases_len;
	int nb_existing_cases;
	int a;
	int *Non_unique_cases;
	int *Non_unique_cases_fst;
	int *Non_unique_cases_len;
	int *Non_unique_cases_go;
	int nb_non_unique_cases;
	int *Non_unique_cases_with_non_trivial_group;
	int nb_non_unique_cases_with_non_trivial_group;
	int *Need_orbits_fst;
	int *Need_orbits_len;
	int sum;
	int o, fst, len;
	long int *input_data;
	orbit_of_subspaces ***All_Orbits;
	int **Position;
	int **Orbit_idx;
	int *Nb_orb;
	int nb_orb_total;
	int nb_middle_orbits;
	int idx;
	int *Po_first; // [nb_orbits]
	flag_orbits *Flag_orbits;
	int h, g;
	long int *data;
	sorting Sorting;
	file_io Fio;




	{
	finite_field *F;
	semifield_classify *SC;
	semifield_level_two *L2;
	semifield_lifting *L3;


	F = NEW_OBJECT(finite_field);
	F->init_override_polynomial(q, poly, 0 /* verbose_level */);

	SC = NEW_OBJECT(semifield_classify);
	cout << "before SC->init" << endl;
	SC->init(argc, argv, order, n, k, F,
			4 /* MINIMUM(verbose_level - 1, 2) */);
	cout << "after SC->init" << endl;

	L2 = NEW_OBJECT(semifield_level_two);
	cout << "before L2->init" << endl;
	L2->init(SC, verbose_level);
	cout << "after L2->init" << endl;


#if 1
	cout << "before L2->compute_level_two" << endl;
	L2->compute_level_two(verbose_level);
	cout << "after L2->compute_level_two" << endl;
#else
	L2->read_level_info_file(verbose_level);
#endif

	L3 = NEW_OBJECT(semifield_lifting);
	cout << "before L3->init_level_three" << endl;
	L3->init_level_three(L2,
			SC->f_level_three_prefix, SC->level_three_prefix,
			verbose_level);
	cout << "after L3->init_level_three" << endl;

	cout << "before L3->recover_level_three_from_file" << endl;
	//L3->compute_level_three(verbose_level);
	L3->recover_level_three_from_file(TRUE /* f_read_flag_orbits */, verbose_level);
	cout << "after L3->recover_level_three_from_file" << endl;






	if (f_v) {
		cout << "before reading files " << fname_FstLen
			<< " and " << fname_liftings << endl;
		}



	start_column = 4;
	{
	classify C;
	int mtx_n;


	// We need 64 bit integers for the Data array
	// because semifields of order 64 need 6 x 6 matrices,
	// which need to be encoded using 36 bits.

	if (sizeof(long int) < 8) {
		cout << "sizeof(long int) < 8" << endl;
		exit(1);
		}
	Fio.int_matrix_read_csv(fname_FstLen,
		FstLen, nb_orbits, mtx_n, verbose_level);
	Len = NEW_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		Len[i] = FstLen[i * 2 + 1];
		}
	Fio.lint_matrix_read_csv(fname_liftings, Data,
		nb_solutions, data_size, verbose_level);


	if (f_v) {
		cout << "Read " << nb_solutions
			<< " solutions arising from "
			<< nb_orbits << " orbits" << endl;
		}



#if 0
	cout << "Reading stabilizers at level 3:" << endl;
	S->SFS->read_stabilizers(3 /* level */, verbose_level);
	cout << "Reading stabilizers at level 3 done" << endl;
#endif



	C.init(Len, nb_orbits, FALSE, 0);
	if (f_v) {
		cout << "classification of Len:" << endl;
		C.print_naked(TRUE);
		cout << endl;
		}
	}

	if (f_v) {
		cout << "computing existing cases:" << endl;
		}



	Existing_cases = NEW_int(nb_orbits);
	nb_existing_cases = 0;

	for (i = 0; i < nb_orbits; i++) {
		if (Len[i]) {
			Existing_cases[nb_existing_cases++] = i;
			}
		}
	Existing_cases_fst = NEW_int(nb_existing_cases);
	Existing_cases_len = NEW_int(nb_existing_cases);
	for (i = 0; i < nb_existing_cases; i++) {
		a = Existing_cases[i];
		Existing_cases_fst[i] = FstLen[2 * a + 0];
		Existing_cases_len[i] = FstLen[2 * a + 1];
		}
	if (f_v) {
		cout << "There are " << nb_existing_cases
			<< " cases which exist" << endl;
		}

	if (f_v) {
		cout << "computing non-unique cases:" << endl;
		}

	Non_unique_cases = NEW_int(nb_existing_cases);
	nb_non_unique_cases = 0;
	for (i = 0; i < nb_existing_cases; i++) {
		a = Existing_cases[i];
		if (Existing_cases_len[i] > 1) {
			Non_unique_cases[nb_non_unique_cases++] = a;
			}
		}

	if (f_v) {
		cout << "There are " << nb_non_unique_cases
			<< " cases which have more than one solution" << endl;
		}
	Non_unique_cases_fst = NEW_int(nb_non_unique_cases);
	Non_unique_cases_len = NEW_int(nb_non_unique_cases);
	Non_unique_cases_go = NEW_int(nb_non_unique_cases);
	for (i = 0; i < nb_non_unique_cases; i++) {
		a = Non_unique_cases[i];
		Non_unique_cases_fst[i] = FstLen[2 * a + 0];
		Non_unique_cases_len[i] = FstLen[2 * a + 1];
		Non_unique_cases_go[i] =
			L3->Stabilizer_gens[a].group_order_as_int();
		}

	{
	classify C;

	C.init(Non_unique_cases_len, nb_non_unique_cases, FALSE, 0);
	if (f_v) {
		cout << "classification of Len amongst the non-unique cases:" << endl;
		C.print_naked(TRUE);
		cout << endl;
		}
	}
	{
	classify C;

	C.init(Non_unique_cases_go, nb_non_unique_cases, FALSE, 0);
	if (f_v) {
		cout << "classification of group orders amongst "
				"the non-unique cases:" << endl;
		C.print_naked(TRUE);
		cout << endl;
		}
	}

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
		int_matrix_print(Non_unique_cases_with_non_trivial_group,
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
	classify C;

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

		if (f_vv) {
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

		int_vec_zero(f_reached, len);
		int_vec_mone(position, len);

		int cnt, f;

		cnt = 0;

		for (f = 0; f < len; f++) {
			if (f_reached[f]) {
				continue;
				}
			orbit_of_subspaces *Orb;


			input_data = Data + (fst + f) * data_size + start_column;

			if (FALSE) {
				cout << "case " << o << " / "
					<< nb_non_unique_cases_with_non_trivial_group
					<< " is original case " << a << " at "
					<< fst << " with " << len
					<< " semifields. Orbit rep "
					<< f << ":" << endl;
				//int_vec_print(cout, input_data, 6);
				//cout << endl;
				}

			SC->compute_orbit_of_subspaces(input_data,
				&L3->Stabilizer_gens[a],
				Orb,
				0 /*verbose_level*/);
			if (FALSE) {
				cout << "Found an orbit of length "
					<< Orb->used_length << endl;
				}

			int idx, g, c;

			c = 0;
			for (g = 0; g < len; g++) {
				if (f_reached[g]) {
					continue;
					}
				if (Orb->find_subspace_lint(
						Data + (fst + g) * data_size + start_column,
						idx, 0 /* verbose_level */)) {
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
			//delete Orb;

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
		//FREE_int(position);
		//FREE_int(orbit_idx);



		}

	if (f_v) {
		cout << "Number of orbits:" << endl;
		for (o = 0; o < nb_non_unique_cases_with_non_trivial_group; o++) {
			cout << o << " : " << Nb_orb[o] << endl;
			}
		cout << "Total number of orbits = " << nb_orb_total << endl;
		}


	if (f_v) {
		cout << "Counting number of middle orbits:" << endl;
		}
	nb_middle_orbits = 0;
	for (o = 0; o < nb_orbits; o++) {

		if (FALSE) {
			cout << "orbit " << o << " number of semifields = "
				<< Len[o] << " group order = "
				<< L3->Stabilizer_gens[o].group_order_as_int() << endl;
			}
		if (Len[o] == 0) {
			}
		else if (Len[o] == 1) {
			nb_middle_orbits += 1;
			}
		else if (L3->Stabilizer_gens[o].group_order_as_int() == 1) {
			nb_middle_orbits += Len[o];
			}
		else {
			if (!Sorting.int_vec_search(Non_unique_cases_with_non_trivial_group,
				nb_non_unique_cases_with_non_trivial_group, o, idx)) {
				cout << "cannot find orbit " << o
						<< " in the list " << endl;
				exit(1);
				}
			if (f_vv) {
				cout << "Found orbit " << o
						<< " at position " << idx << endl;
				}
			nb_middle_orbits += Nb_orb[idx];
			}

		} // next o
	if (f_v) {
		cout << "nb_middle_orbits = " << nb_middle_orbits << endl;
		}


	if (f_v) {
		cout << "Computing Flag_orbits:" << endl;
		}


	Po_first = NEW_int(nb_orbits);

	Flag_orbits = NEW_OBJECT(flag_orbits);
	Flag_orbits->init_lint(
		SC->A, SC->AS,
		nb_orbits /* nb_primary_orbits_lower */,
		6 /* pt_representation_sz */,
		nb_middle_orbits /* nb_flag_orbits */,
		verbose_level);


	h = 0;
	for (o = 0; o < nb_orbits; o++) {

		Po_first[o] = h;
		fst = FstLen[2 * o + 0];

		if (Len[o] == 0) {
			// nothing to do here
			}
		else if (Len[o] == 1) {
			data = Data +
					(fst + 0) * data_size + start_column;
			Flag_orbits->Flag_orbit_node[h].init_lint(
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
		else if (L3->Stabilizer_gens[o].group_order_as_int() == 1) {
			for (g = 0; g < Len[o]; g++) {
				data = Data +
						(fst + g) * data_size + start_column;
				Flag_orbits->Flag_orbit_node[h].init_lint(
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
				strong_generators *gens;
				longinteger_object go;

				Orb = All_Orbits[idx][g];
				data = Orb->Subspaces_lint[Orb->position_of_original_subspace];
				L3->Stabilizer_gens[o].group_order(go);
				gens = Orb->stabilizer_orbit_rep(
					go /* full_group_order */, 0 /*verbose_level - 1*/);
				Flag_orbits->Flag_orbit_node[h].init_lint(
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
	if (h != nb_middle_orbits) {
		cout << "h != nb_middle_orbits" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "Finished initializing flag orbits. "
				"Number of flag orbits = " << nb_middle_orbits << endl;
		}



	if (f_v) {
		cout << "Computing classification:" << endl;
		}

	int *Basis1;
	int *Basis2;
	int *B;
	grassmann *Gr;
	int *transporter1;
	int *transporter2;
	int *transporter3;
	int rk, N;
	int trace_po;
	//int trace_po0;
	//int cnt_aut;
	int depth0 = 3;
	int desired_pivot_rows[6] = {0, 1, 5, 4, 3, 2};
	int desired_pivots[6];


	for (i = 0; i < 6; i++) {
		desired_pivots[i] = desired_pivot_rows[i] * 6;
		}
	if (f_v) {
		cout << "desired_pivot_rows:";
		int_vec_print(cout, desired_pivot_rows, 6);
		cout << endl;
		cout << "desired_pivots:";
		int_vec_print(cout, desired_pivots, 6);
		cout << endl;
		}


	Basis1 = NEW_int(k * k2);
	Basis2 = NEW_int(k * k2);
	B = NEW_int(k * k);
	Gr = NEW_OBJECT(grassmann);
	transporter1 = NEW_int(SC->A->elt_size_in_int);
	transporter2 = NEW_int(SC->A->elt_size_in_int);
	transporter3 = NEW_int(SC->A->elt_size_in_int);


	Gr->init(k, depth0, F, 0 /* verbose_level */);
	N = Gr->nb_of_subspaces(0 /* verbose_level */);



	classification_step *Semifields;
	int *f_processed; // [nb_middle_orbits]
	int nb_processed, po, so, f, f2;
	long int data1[6];
	long int data2[6];
	int *Elt1;
	int *Elt2;
	int *Elt3;
	longinteger_object go;
	int v1[6];
	int v2[6];
	int v3[6];
	int f_skip = FALSE;



	f_processed = NEW_int(nb_middle_orbits);
	int_vec_zero(f_processed, nb_middle_orbits);
	nb_processed = 0;

	Elt1 = NEW_int(SC->A->elt_size_in_int);
	Elt2 = NEW_int(SC->A->elt_size_in_int);
	Elt3 = NEW_int(SC->A->elt_size_in_int);

	Semifields = NEW_OBJECT(classification_step);

	SC->A->group_order(go);

	Semifields->init_lint(SC->A, SC->AS,
			nb_middle_orbits, 6, go, verbose_level);


	Flag_orbits->nb_primary_orbits_upper = 0;

	for (f = 0; f < nb_middle_orbits; f++) {


		double progress;

		if (f_processed[f]) {
			continue;
			}

		progress = ((double) nb_processed * 100. ) /
				(double) nb_middle_orbits;

		if (f_v) {
			cout << "Defining new orbit "
				<< Flag_orbits->nb_primary_orbits_upper
				<< " from flag orbit " << f << " / "
				<< nb_middle_orbits << " progress="
				<< progress << "%" << endl;
			}
		Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit =
			Flag_orbits->nb_primary_orbits_upper;



		po = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
		so = Flag_orbits->Flag_orbit_node[f].downstep_secondary_orbit;
		if (f_v) {
			cout << "po=" << po << " so=" << so << endl;
			}
		lint_vec_copy(Flag_orbits->Pt_lint + f * 6, data1, 6);

		strong_generators *Aut_gens;
		vector_ge *coset_reps;
		longinteger_object go;

		Aut_gens = Flag_orbits->Flag_orbit_node[f].gens->create_copy();
		coset_reps = NEW_OBJECT(vector_ge);
		coset_reps->init(SC->A);


		for (i = 0; i < k; i++) {
			SC->matrix_unrank(data1[i],
					Basis1 + i * k2);
			}
		if (f_vvv) {
			cout << "Basis1=" << endl;
			int_matrix_print(Basis1, k, k2);
			for (i = 0; i < k; i++) {
				cout << "Matrix " << i << ":" << endl;
				int_matrix_print(Basis1 + i * k2, k, k);
				}
			}
		f_skip = FALSE;
		for (i = 0; i < k; i++) {
			v3[i] = Basis1[2 * k2 + i * k + 0];
			}
		if (!is_unit_vector(v3, 6, 5)) {
			cout << "flag orbit " << f << " / "
					<< nb_middle_orbits
					<< " 1st col of third matrix is = ";
			int_vec_print(cout, v3, 6);
			cout << " which is not the 5th unit vector, "
					"so we skip" << endl;
			f_skip = TRUE;
			}

		if (f_skip) {
			if (f_v) {
				cout << "flag orbit " << f << " / "
						<< nb_middle_orbits
					<< ", first vector is not the unit vector, "
					"so we skip" << endl;
				}
			f_processed[f] = TRUE;
			nb_processed++;
			continue;
			}
		if (f_v) {
			cout << "flag orbit " << f << " / " << nb_middle_orbits
				<< ", looping over the " << N << " subspaces" << endl;
			}

		trace_record *TR;

		TR = NEW_OBJECTS(trace_record, N);

		for (rk = 0; rk < N; rk++) {

			trace_record *T;

			T = TR + rk;

			T->coset = rk;

			if (f_v) {
				cout << "flag orbit " << f << " / "
					<< nb_middle_orbits << ", subspace "
					<< rk << " / " << N << ":" << endl;
				}

			// we do it again:
			for (i = 0; i < k; i++) {
				SC->matrix_unrank(data1[i], Basis1 + i * k2);
				}


			// unrank the subspace:
			Gr->unrank_int_here_and_extend_basis(B, rk,
					0 /* verbose_level */);

			// multiply the matrices to get the matrices
			// adapted to the subspace:
			// the first three matrices are the generators
			// for the subspace.
			F->mult_matrix_matrix(B, Basis1, Basis2, k, k, k2,
					0 /* verbose_level */);


			if (f_v) {
				cout << "base change matrix B=" << endl;
				int_matrix_print_bitwise(B, k, k);
				cout << "Basis2 (before trace)=" << endl;
				int_matrix_print_bitwise(Basis2, k, k2);
				for (i = 0; i < k; i++) {
					cout << "Matrix " << i << ":" << endl;
					int_matrix_print(Basis2 + i * k2, k, k);
					}
				}


			trace_po = L3->trace_to_level_three(
				Basis2,
				k /* basis_sz */,
				transporter1,
				verbose_level - 4);

			T->trace_po = trace_po;

			if (f_v) {
				cout << "After trace, trace_po = "
						<< trace_po << endl;
				cout << "Basis2 (after trace)=" << endl;
				int_matrix_print_bitwise(Basis2, k, k2);
				for (i = 0; i < k; i++) {
					cout << "Matrix " << i << ":" << endl;
					int_matrix_print(Basis2 + i * k2, k, k);
					}
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
			if (!is_unit_vector(v1, 6, 0)) {
				f_skip = TRUE;
				}
			if (!is_unit_vector(v2, 6, 1)) {
				f_skip = TRUE;
				}
			if (!is_unit_vector(v3, 6, 5)) {
				f_skip = TRUE;
				}
			T->f_skip = f_skip;

			if (f_skip) {
				if (f_v) {
					cout << "skipping this case " << trace_po
						<< " because pivot is not "
							"in the last row." << endl;
					}
				}
			else {
				F->Gauss_int_with_given_pivots(
					Basis2,
					FALSE /* f_special */,
					TRUE /* f_complete */,
					desired_pivots,
					k /* nb_pivots */,
					k, k2,
					0 /*verbose_level - 2*/);
				if (f_v) {
					cout << "Basis2 after RREF=" << endl;
					int_matrix_print_bitwise(Basis2, k, k2);
					for (i = 0; i < k; i++) {
						cout << "Matrix " << i << ":" << endl;
						int_matrix_print(Basis2 + i * k2, k, k);
						}
					}

				for (i = 0; i < k; i++) {
					data2[i] = SC->matrix_rank(Basis2 + i * k2);
					}
				if (f_v) {
					cout << "data2=";
					lint_vec_print(cout, data2, 6);
					cout << endl;
					}

				int solution_idx;

				solution_idx = find_semifield_in_table(
					L3,
					Data,
					nb_solutions /* nb_semifields */,
					data_size,
					start_column,
					FstLen,
					nb_orbits,
					trace_po,
					data2 /* given_data */,
					verbose_level);


				T->solution_idx = solution_idx;
				T->nb_sol = Len[trace_po];

				if (f_v) {
					cout << "solution_idx=" << solution_idx << endl;
					}

				int go;
				go = L3->Stabilizer_gens
						[trace_po].group_order_as_int();

				T->go = go;
				T->pos = -1;
				T->so = -1;
				T->orbit_len = -1;
				T->f2 = -1;

				if (f_v) {
					cout << "flag orbit " << f << " / "
						<< nb_middle_orbits << ", subspace "
						<< rk << " / " << N << " trace_po="
						<< trace_po << " go=" << go
						<< " solution_idx=" << solution_idx << endl;
					}


				if (go == 1) {
					if (f_v) {
						cout << "This starter case has a trivial "
								"group order" << endl;
						}

					f2 = Po_first[trace_po] + solution_idx;

					T->so = solution_idx;
					T->orbit_len = 1;
					T->f2 = f2;

					if (f2 == f) {
						if (f_v) {
							cout << "We found an automorphism" << endl;
							}
						coset_reps->append(transporter1);
						}
					else {
						if (!f_processed[f2]) {
							if (f_v) {
								cout << "We are identifying with "
										"po=" << trace_po << " so="
										<< solution_idx << ", which is "
											"flag orbit " << f2 << endl;
								}
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
							if (f_v) {
								cout << "Flag orbit f2 = " << f2
									<< " has been fused with flag orbit "
									<< f << endl;
								}
							}
						else {
							if (f_v) {
								cout << "Flag orbit f2 = " << f2
										<< " has already been processed, "
												"nothing to do here" << endl;
								}
							}
						}
					}
				else {
					if (Len[trace_po] == 1) {
						if (f_v) {
							cout << "This starter case has only "
									"one solution" << endl;
							}
						f2 = Po_first[trace_po] + 0;

						T->pos = -1;
						T->so = 0;
						T->orbit_len = 1;
						T->f2 = f2;


						if (f2 == f) {
							if (f_v) {
								cout << "We found an automorphism" << endl;
								}
							coset_reps->append(transporter1);
							}
						else {
							if (!f_processed[f2]) {
								if (f_v) {
									cout << "We are identifying with po="
											<< trace_po << " so=" << 0
											<< ", which is flag orbit "
											<< f2 << endl;
									}
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
								if (f_v) {
									cout << "Flag orbit f2 = " << f2
										<< " has been fused with "
											"flag orbit " << f << endl;
									}
								}
							else {
								if (f_v) {
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
						f2 = Po_first[trace_po] + orbit_idx;



						T->pos = position;
						T->so = orbit_idx;
						T->orbit_len = Orb->used_length;
						T->f2 = f2;



						if (f_v) {
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
							if (f_v) {
								cout << "We found an automorphism" << endl;
								}
							coset_reps->append(transporter3);
							}
						else {
							if (!f_processed[f2]) {
								if (f_v) {
									cout << "We are identifying with po="
										<< trace_po << " so=" << solution_idx
										<< ", which is flag orbit "
										<< f2 << endl;
									}
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
								if (f_v) {
									cout << "Flag orbit f2 = " << f2
										<< " has been fused with flag "
										"orbit " << f << endl;
									}
								}
							else {
								if (f_v) {
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

		save_trace_record(TR,
			Flag_orbits->nb_primary_orbits_upper, f, po, so, N);


		FREE_OBJECTS(TR);

		int cl;
		longinteger_object go1, Cl, ago;
		longinteger_domain D;

		Aut_gens->group_order(go);
		cl = coset_reps->len;
		Cl.create(cl);
		D.mult(go, Cl, ago);
		if (f_v) {
			cout << "Semifield "
					<< Flag_orbits->nb_primary_orbits_upper
					<< ", Ago = starter * number of cosets = " << ago
					<< " = " << go << " * " << Cl
					<< " created from flag orbit " << f << " / "
				<< nb_middle_orbits << " progress="
				<< progress << "%" << endl;
			}

#if 0
		Aut_gens->add_generators(coset_reps,
				cl /* group_index */, 0 /*verbose_level*/);
		Aut_gens->group_order(go1);
		if (f_v) {
			cout << "We have created a group of order " << go1 << endl;
			}
		if (D.compare(ago, go1) != 0) {
			cout << "The group orders differ, something is wrong" << endl;
			exit(1);
			}
#endif


		Semifields->Orbit[Flag_orbits->nb_primary_orbits_upper].init_lint(
			Semifields,
			Flag_orbits->nb_primary_orbits_upper,
			Aut_gens, data1, verbose_level);

		FREE_OBJECT(Aut_gens);

		f_processed[f] = TRUE;
		nb_processed++;
		Flag_orbits->nb_primary_orbits_upper++;



		} // next f




	if (f_v) {
		cout << "Computing classification done, we found "
				<< Flag_orbits->nb_primary_orbits_upper
				<< " semifields" << endl;
		}

	FREE_OBJECT(Gr);
	FREE_int(transporter1);
	FREE_int(transporter2);
	FREE_int(transporter3);
	FREE_int(Basis1);
	FREE_int(Basis2);
	FREE_int(B);
	FREE_OBJECT(Flag_orbits);

	cout << "before freeing L3" << endl;
	FREE_OBJECT(L3);
	cout << "before freeing L2" << endl;
	FREE_OBJECT(L2);
	cout << "before freeing SC" << endl;
	FREE_OBJECT(SC);
	cout << "before freeing F" << endl;
	FREE_OBJECT(F);
	cout << "before leaving scope" << endl;
	}
	cout << "after leaving scope" << endl;


	}


	the_end(t0);
}

int find_semifield_in_table(semifield_lifting *L3,
	long int *Data, int nb_semifields, int data_size,
	int start_column, int *FstLen, int nb_orbits, int po,
	long int *given_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int fst, len, g;
	int idx = -1;

	if (f_v) {
		cout << "find_semifield_in_table" << endl;
		}
	if (f_v) {
		cout << "searching for: ";
		lint_vec_print(cout, given_data, 6);
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
		exit(1);
		}

	if (lint_vec_compare(Data + fst * data_size + start_column,
			given_data, 3)) {
		cout << "find_semifield_in_table the first three entries "
				"of given_data do not match with what "
				"is in Data" << endl;
		exit(1);
		}

	if (len == 1) {
		if (lint_vec_compare(Data + fst * data_size + start_column,
				given_data, 6)) {
			cout << "find_semifield_in_table len is 1 and the first six "
				"entries of given_data do not match with what "
				"is in Data" << endl;
			exit(1);
			}
		idx = 0;
		}
	else {
		for (g = 0; g < len; g++) {
			if (lint_vec_compare(Data + (fst + g) * data_size + start_column,
					given_data, 6) == 0) {
				idx = g;
				break;
				}
			}
		if (g == len) {
			cout << "find_semifield_in_table cannot find the "
					"semifield in the table" << endl;
			exit(1);
			}
		}

	if (f_v) {
		cout << "find_semifield_in_table done, idx = " << idx << endl;
		}
	return idx;
}

int is_unit_vector(int *v, int len, int k)
{
	int i;

	for (i = 0; i < len; i++) {
		if (i == k) {
			if (v[i] != 1) {
				return FALSE;
				}
			}
		else {
			if (v[i] != 0) {
				return FALSE;
				}
			}
		}
	return TRUE;
}


void save_trace_record(trace_record *T,
		int iso, int f, int po, int so, int N)
{
	int *M;
	int w = 10;
	char fname[1000];
	const char *column_label[] = {
		"coset",
		"trace_po",
		"f_skip",
		"solution_idx",
		"nb_sol",
		"go",
		"pos",
		"so",
		"orbit_len",
		"f2"
		};
	int i;
	file_io Fio;

	M = NEW_int(N * w);
	for (i = 0; i < N; i++) {
		M[i * w + 0] = T[i].coset;
		M[i * w + 1] = T[i].trace_po;
		M[i * w + 2] = T[i].f_skip;
		M[i * w + 3] = T[i].solution_idx;
		M[i * w + 4] = T[i].nb_sol;
		M[i * w + 5] = T[i].go;
		M[i * w + 6] = T[i].pos;
		M[i * w + 7] = T[i].so;
		M[i * w + 8] = T[i].orbit_len;
		M[i * w + 9] = T[i].f2;
		}

	sprintf(fname, "trace_record_%03d_f%05d_po%d_so%d.csv", iso, f, po, so);
	Fio.int_matrix_write_csv_with_labels(fname, M, N, w, column_label);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
}

