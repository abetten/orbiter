/*
 * tdo_refine_3designs.cpp
 *
 *  Created on: Sep 15, 2026
 *      Author: betten
 */





#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace tactical_decompositions {


tdo_refine_3designs::tdo_refine_3designs()
{
	Record_birth();

	Tdo_scheme_synthetic = NULL;
	Row_split = NULL;
	Col_split = NULL;

}

tdo_refine_3designs::~tdo_refine_3designs()
{
	Record_death();

	if (Row_split) {
		FREE_OBJECT(Row_split);
	}
	if (Col_split) {
		FREE_OBJECT(Col_split);
	}
}

void tdo_refine_3designs::init(
		tdo_scheme_synthetic *Tdo_scheme_synthetic,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "tdo_refine_3designs::init" << endl;
	}

	tdo_refine_3designs::Tdo_scheme_synthetic = Tdo_scheme_synthetic;

	if (f_v) {
		cout << "tdo_refine_3designs::init done" << endl;
	}
}

// #############################################################################
// TDO parameter refinement for 3-designs - row refinement
// #############################################################################


int tdo_refine_3designs::td3_refine_rows(
		tdo_refinement_output *&Output,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "tdo_refine_3designs::td3_refine_rows" << endl;
	}
	Tdo_scheme_synthetic->check_init();

	int R, /*l1,*/ l2, r;
	int nb_eqns, nb_vars = 0;
	int point_types_allocated;
	other::data_structures::partitionstack P;
	int lambda2;
	int nb_points;
	int nb_sol;
	tdo_data T;


	int *point_types;
	int nb_point_types, point_type_len;
	int *distributions;
	int nb_distributions;



	nb_points = Tdo_scheme_synthetic->m;
	lambda2 = Tdo_scheme_synthetic->Descr->lambda3 * (nb_points - 2) / (Tdo_scheme_synthetic->Descr->block_size - 2);
	if (f_v) {
		cout << "nb_points = " << nb_points
				<< " lambda2 = " << lambda2 << endl;
	}
	if ((Tdo_scheme_synthetic->Descr->block_size - 2) * lambda2 != Tdo_scheme_synthetic->Descr->lambda3 * (nb_points - 2)) {
		cout << "parameters are wrong" << endl;
		exit(1);
	}


	//other::data_structures::partitionstack *Col_split;

	Col_split = Tdo_scheme_synthetic->get_column_split_partition(verbose_level);

	R = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];
	//l1 = nb_col_classes[ROW_SCHEME];
	l2 = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];


	T.allocate(R);

	T.types_first[0] = 0;

	point_types_allocated = 100;
	nb_point_types = 0;
	point_types = NEW_int(point_types_allocated * l2);
	point_type_len = l2;

	T.nb_only_one_type = 0;
	T.nb_multiple_types = 0;

	for (r = 0; r < R; r++) {

		if (f_vvv) {
			cout << "r=" << r << endl;
		}
		if (!td3_rows_setup_first_system(
				Tdo_scheme_synthetic->Descr->lambda3,
				Tdo_scheme_synthetic->Descr->block_size,
				lambda2,
			T, r, Col_split,
			nb_vars, nb_eqns,
			point_types, nb_point_types,
			verbose_level - 1)) {
			FREE_int(point_types);
			return false;
		}

		nb_sol = T.solve_first_system(
			point_types, nb_point_types, point_types_allocated,
			verbose_level - 1);

		if (f_vv) {
			cout << "r = " << r << ", found " << nb_sol
				<< " refined point types" << endl;
		}
		if (nb_sol == 0) {
			FREE_int(point_types);
			return false;
		}

		T.types_len[r] = nb_sol;
		T.types_first[r + 1] = T.types_first[r] + nb_sol;

		if (nb_sol == 1) {
			if (f_vv) {
				cout << "only one solution in block r=" << r << endl;
			}
			T.only_one_type[T.nb_only_one_type++] = r;
		}
		else {
			T.multiple_types[T.nb_multiple_types++] = r;
		}

		//T.D1->freeself();
		FREE_OBJECT(T.D1);
		T.D1 = NEW_OBJECT(solvers::diophant);
		//diophant_close(T.D1);
		//T.D1 = NULL;

	} // next r


	// now we compute the distributions:
	//
	int Nb_vars, Nb_eqns;

	if (!td3_rows_setup_second_system(
			Tdo_scheme_synthetic->Descr->lambda3,
			Tdo_scheme_synthetic->Descr->block_size,
			lambda2,
		T,
		nb_vars, Nb_vars, Nb_eqns,
		point_types, nb_point_types,
		verbose_level)) {
		FREE_int(point_types);
		return false;
	}

	if (Nb_vars == 0) {
		int h, r, u;

		distributions = NEW_int(1 * nb_point_types);
		nb_distributions = 0;
		for (h = 0; h < T.nb_only_one_type; h++) {
			r = T.only_one_type[h];
			u = T.types_first[r];
			//cout << "only one type, r=" << r << " u=" << u
			//<< " row_classes_len[ROW][r]="
			//<< row_classes_len[ROW][r] << endl;
			distributions[nb_distributions * nb_point_types + u] =
					Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][r];
		}
		nb_distributions++;

		Output = NEW_OBJECT(tdo_refinement_output);

		Output->types = point_types;
		Output->nb_types = nb_point_types;
		Output->type_len = point_type_len;
		Output->distributions = distributions;
		Output->nb_distributions = nb_distributions;

#if 0
		int *point_types;
		int nb_point_types, point_type_len;
		int *distributions;
		int nb_distributions;
#endif


		return true;
	}

	int f_scale = false;
	int scaling = 0;

	T.solve_second_system(
			false /* f_use_mckay */,
			Tdo_scheme_synthetic->Descr->f_once,
			Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME],
			f_scale, scaling,
		point_types, nb_point_types, distributions, nb_distributions,
		verbose_level - 1);


	if (f_v) {
		cout << "tdo_refine_3designs::td3_refine_rows "
				"found " << nb_distributions
				<< " distributions." << endl;
	}

	Output = NEW_OBJECT(tdo_refinement_output);

	Output->types = point_types;
	Output->nb_types = nb_point_types;
	Output->type_len = point_type_len;
	Output->distributions = distributions;
	Output->nb_distributions = nb_distributions;


	//FREE_OBJECT(Col_split);

	if (f_v) {
		cout << "tdo_refine_3designs::td3_refine_rows done" << endl;
	}
	return true;
}

int tdo_refine_3designs::td3_rows_setup_first_system(
	int lambda3, int block_size, int lambda2,
	tdo_data &T, int r,
	other::data_structures::partitionstack *Col_split,
	int &nb_vars, int &nb_eqns,
	int *&point_types, int &nb_point_types,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "tdo_refine_3designs::td3_rows_setup_first_system r=" << r << endl;
	}

	int i, j, R, l1, l2, r2, r3, S, I, J, f, l, s;
	int eqn_offset, eqn_cnt;
	other_combinatorics::combinatorics_domain Combi;


	// create all partitions which are refined
	// point types of points in block r

	R = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];
	l1 = Tdo_scheme_synthetic->nb_col_classes[ROW_SCHEME];
	l2 = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];

	nb_vars = l2;
	nb_eqns = R + R + (R - 1) + (((R - 1) * (R - 2)) >> 1) + l1;


	T.D1->open(nb_eqns, nb_vars, verbose_level - 1);
	S = 0;


	Int_vec_zero(T.D1->A, nb_eqns * nb_vars);

	for (I = 0; I < nb_eqns; I++) {
		T.D1->RHS[I] = 9999;
	}

	// pair joinings
	for (r2 = 0; r2 < R; r2++) {
		if (r2 == r) {
			// connections within the same row-partition
			for (J = 0; J < nb_vars; J++) {
				T.D1->A[r2 * nb_vars + J] =
					Combi.minus_one_if_positive(Tdo_scheme_synthetic->the_col_scheme[r2 * l2 + J]);
			}
			T.D1->RHS[r2] = (Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][r2] - 1) * lambda2;
		}
		else {
			// connections to the point from different row-partitions
			for (J = 0; J < nb_vars; J++) {
				T.D1->A[r2 * nb_vars + J] = Tdo_scheme_synthetic->the_col_scheme[r2 * l2 + J];
			}
			T.D1->RHS[r2] = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][r2] * lambda2;
		}
	}
	if (f_vv) {
		cout << "r=" << r << " after pair joining, the system is" << endl;
		T.D1->print();
	}

	// triple joinings
	eqn_offset = R;
	for (r2 = 0; r2 < R; r2++) {
		if (r2 == r) {
			// connections to pairs within the same row-partition
			for (J = 0; J < nb_vars; J++) {
				T.D1->A[(eqn_offset + r2) * nb_vars + J] =
					Combi.binomial2(Combi.minus_one_if_positive(
							Tdo_scheme_synthetic->the_col_scheme[r2 * l2 + J]));
			}
			T.D1->RHS[eqn_offset + r2] =
				Combi.binomial2((Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][r2] - 1)) * lambda3;
		}
		else {
			// connections to pairs with one
			// in the same and one in the other part
			for (J = 0; J < nb_vars; J++) {
				T.D1->A[(eqn_offset + r2) * nb_vars + J] =
					Combi.minus_one_if_positive(Tdo_scheme_synthetic->the_col_scheme[r * l2 + J])
						* Tdo_scheme_synthetic->the_col_scheme[r2 * l2 + J];
			}
			T.D1->RHS[eqn_offset + r2] =
				(Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][r] - 1) *
				Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][r2] * lambda3;
		}
	}
	if (f_vv) {
		cout << "tdo_refine_3designs::td3_rows_setup_first_system "
				"r=" << r << " after triple joining, the system is" << endl;
		T.D1->print();
	}

	eqn_offset += R;
	eqn_cnt = 0;
	for (r2 = 0; r2 < R; r2++) {
		if (r2 == r) {
			continue;
		}
		// connections to pairs from one different row-partition
		for (J = 0; J < nb_vars; J++) {
			T.D1->A[(eqn_offset + eqn_cnt) * nb_vars + J] =
				Combi.binomial2(Tdo_scheme_synthetic->the_col_scheme[r2 * l2 + J]);
		}
		T.D1->RHS[eqn_offset + eqn_cnt] = Combi.binomial2(
				Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][r2]) * lambda3;
		eqn_cnt++;
	}
	if (f_vv) {
		cout << "tdo_refine_3designs::td3_rows_setup_first_system "
				"r=" << r << " after connections to pairs "
				"from one different row-partition, the system is" << endl;
		T.D1->print();
	}

	eqn_offset += (R - 1);
	eqn_cnt = 0;
	for (r2 = 0; r2 < R; r2++) {
		if (r2 == r) {
			continue;
		}
		for (r3 = r2 + 1; r3 < R; r3++) {
			if (r3 == r) {
				continue;
			}
			// connections to pairs from two different row-partitions
			for (J = 0; J < nb_vars; J++) {
				T.D1->A[(eqn_offset + eqn_cnt) * nb_vars + J] =
						Tdo_scheme_synthetic->the_col_scheme[r2 * l2 + J] * Tdo_scheme_synthetic->the_col_scheme[r3 * l2 + J];
			}
			T.D1->RHS[eqn_offset + eqn_cnt] =
					Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][r2] * Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][r3] * lambda3;
			eqn_cnt++;
		}
	}
	eqn_offset += eqn_cnt;
	if (f_vv) {
		cout << "tdo_refine_3designs::td3_rows_setup_first_system "
				"r=" << r << " after connections to pairs from two "
				"different row-partitions, the system is" << endl;
		T.D1->print();
	}

	S = 0;
	for (i = 0; i < l1; i++) {
		s = Tdo_scheme_synthetic->the_row_scheme[r * l1 + i];
		if (f_vvv) {
			cout << "r=" << r << " i=" << i << " s=" << s << endl;
		}
		T.D1->RHS[eqn_offset + i] = s;
		S += s;
		f = Col_split->startCell[i];
		l = Col_split->cellSize[i];
		if (f_vvv) {
			cout << "f=" << f << " l=" << l << endl;
		}

		for (j = 0; j < l; j++) {
			T.D1->A[(eqn_offset + i) * nb_vars + f + j] = 1;
			T.D1->x_min[f + j] = 0;
			T.D1->x_max[f + j] = s;
		}
	}
	if (f_vv) {
		cout << "tdo_refine_3designs::td3_rows_setup_first_system "
				"r=" << r << " after adding extra equations, "
				"the system is" << endl;
		T.D1->print();
	}

	T.D1->f_has_sum = true;
	T.D1->sum = S;
	//T.D1->f_x_max = true;


	if (f_vv) {
		cout << "tdo_refine_3designs::td3_rows_setup_first_system "
				"r=" << r << " the system is" << endl;
		T.D1->print();
	}

	if (f_v) {
		cout << "tdo_refine_3designs::td3_rows_setup_first_system done" << endl;
	}

	return true;
}

int tdo_refine_3designs::td3_rows_setup_second_system(
	int lambda3, int block_size, int lambda2,
	tdo_data &T,
	int nb_vars, int &Nb_vars, int &Nb_eqns,
	int *&point_types, int &nb_point_types,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "tdo_refine_3designs::td3_rows_setup_second_system" << endl;
	}

	int l2, i, I, r, nb_eqns_counting;
	int S;

	l2 = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];

	nb_eqns_counting = T.nb_multiple_types * (l2 + 1);
	Nb_eqns = nb_eqns_counting;
	Nb_vars = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		T.types_first2[i] = Nb_vars;
		Nb_vars += T.types_len[r];
	}


	T.D2->open(Nb_eqns, Nb_vars, verbose_level - 1);
	if (f_v) {
		cout << "td3_rows_setup_second_system: "
			"opening second system with "
			<< Nb_eqns << " equations and "
			<< Nb_vars << " variables" << endl;
	}

	Int_vec_zero(T.D2->A, Nb_eqns * Nb_vars);
	for (I = 0; I < Nb_eqns; I++) {
		T.D2->RHS[I] = 9999;
	}


	if (!td3_rows_counting_flags(
		lambda3, block_size, lambda2, S,
		T,
		nb_vars, Nb_vars,
		point_types, nb_point_types, 0,
		verbose_level)) {
		return false;
	}


	T.D2->f_has_sum = true;
	T.D2->sum = S;


	if (f_vv) {
		cout << "The second system is" << endl;

		T.D2->print();
	}

	if (f_v) {
		cout << "tdo_refine_3designs::td3_rows_setup_second_system "
				"done" << endl;
	}
	return true;

}

int tdo_refine_3designs::td3_rows_counting_flags(
	int lambda3, int block_size, int lambda2, int &S,
	tdo_data &T,
	int nb_vars, int Nb_vars,
	int *&point_types, int &nb_point_types, int eqn_offset,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int I, i, r, f, l, j, c, J, a, b, rr, p, u, l2, h, s;

	l2 = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];

	if (f_v) {
		cout << "tdo_refine_3designs::td3_rows_counting_flags "
				"eqn_offset=" << eqn_offset
			<< " nb_multiple_types=" << T.nb_multiple_types << endl;
	}
	// counting flags, a block diagonal system with
	// nb_multiple_types * (l2 + 1) equations
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (I = 0; I < l2; I++) {
			for (j = 0; j < l; j++) {
				c = f + j;
				J = T.types_first2[i] + j;
				a = point_types[c * nb_vars + I];
				T.D2->A[(eqn_offset + i * (l2 + 1) + I) * Nb_vars + J] = a;
			}
			a = Tdo_scheme_synthetic->the_col_scheme[r * Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME] + I];
			b = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][I];
			T.D2->RHS[eqn_offset + i * (l2 + 1) + I] = a * b;
			for (h = 0; h < T.nb_only_one_type; h++) {
				rr = T.only_one_type[h];
				p = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][rr];
				u = T.types_first[rr];
				a = point_types[u * nb_vars + I];
				T.D2->RHS[eqn_offset + i * (l2 + 1) + I] -= a * p;
				if (T.D2->RHS[eqn_offset + i * (l2 + 1) + I] < 0) {
					if (f_v) {
						cout << "td3_rows_counting_flags: RHS[nb_eqns_joining + i * (l2 + 1) + I] "
							"is negative, no solution for the distribution" << endl;
					}
					return false;
				}
			} // next h
		} // next I
	} // next i


	S = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		// one extra equation for the sum
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			// counting: extra row of ones
			T.D2->A[(eqn_offset + i * (l2 + 1) + l2) * Nb_vars + J] = 1;
		}

		s = Tdo_scheme_synthetic->row_classes_len[COL_SCHEME][r];
		T.D2->RHS[eqn_offset + i * (l2 + 1) + l2] = s;
		S += s;
	}
	if (f_vvv) {
		cout << "tdo_refine_3designs::td3_rows_counting_flags, the system is" << endl;
		T.D2->print();
	}

	if (f_v) {
		cout << "tdo_refine_3designs::td3_rows_counting_flags done" << endl;
	}

	return true;
}

// #############################################################################
// TDO parameter refinement for 3-designs - column refinement
// #############################################################################



int tdo_refine_3designs::td3_refine_columns(
		tdo_refinement_output *&Output,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "tdo_refine_3designs::td3_refine_columns" << endl;
	}
	Tdo_scheme_synthetic->check_init();


	int R, /*l1,*/ l2, r, nb_eqns, nb_vars = 0;
	int line_types_allocated;
	other::data_structures::partitionstack P;
	int lambda2;
	int nb_points;
	int nb_sol;
	tdo_data T;


	int *line_types;
	int nb_line_types, line_type_len;
	int *distributions;
	int nb_distributions;

	nb_points = Tdo_scheme_synthetic->m;
	lambda2 = Tdo_scheme_synthetic->Descr->lambda3 * (nb_points - 2) / (Tdo_scheme_synthetic->Descr->block_size - 2);
	if (f_v) {
		cout << "nb_points = " << nb_points
			<< " lambda2 = " << lambda2 << endl;
	}
	if ((Tdo_scheme_synthetic->Descr->block_size - 2) * lambda2 != Tdo_scheme_synthetic->Descr->lambda3 * (nb_points - 2)) {
		cout << "tdo_refine_3designs::td3_refine_columns parameters are wrong" << endl;
		exit(1);
	}


	//other::data_structures::partitionstack *Row_split;



	Row_split = Tdo_scheme_synthetic->get_row_split_partition(verbose_level);

	R = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];
	//l1 = nb_row_classes[COL_SCHEME];
	l2 = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];

	T.allocate(R);

	T.types_first[0] = 0;

	line_types_allocated = 100;
	nb_line_types = 0;
	line_types = NEW_int(line_types_allocated * l2);
	line_type_len = l2;

	T.nb_only_one_type = 0;
	T.nb_multiple_types = 0;

	for (r = 0; r < R; r++) {

		if (f_vvv) {
			cout << "r=" << r << endl;
		}
		if (!td3_columns_setup_first_system(
				Tdo_scheme_synthetic->Descr->lambda3,
				Tdo_scheme_synthetic->Descr->block_size,
				lambda2,
			T, r, P,
			nb_vars, nb_eqns,
			line_types, nb_line_types,
			verbose_level - 1)) {

			FREE_int(line_types);
			return false;
		}

		nb_sol = T.solve_first_system(
			line_types, nb_line_types, line_types_allocated,
			verbose_level - 1);

		if (f_vv) {
			cout << "r = " << r << ", found " << nb_sol
				<< " refine line types" << endl;
		}
		if (nb_sol == 0) {
			FREE_int(line_types);
			return false;
		}

		T.types_len[r] = nb_sol;
		T.types_first[r + 1] = T.types_first[r] + nb_sol;

		if (nb_sol == 1) {
			if (f_vv) {
				cout << "only one solution in block r=" << r << endl;
			}
			T.only_one_type[T.nb_only_one_type++] = r;
		}
		else {
			T.multiple_types[T.nb_multiple_types++] = r;
		}

		//T.D1->freeself();
		FREE_OBJECT(T.D1);
		T.D1 = NEW_OBJECT(solvers::diophant);
		//diophant_close(T.D1);
		//T.D1 = NULL;

	} // next r


	// now we compute the distributions:
	//
	int Nb_vars, Nb_eqns;

	if (!td3_columns_setup_second_system(
			Tdo_scheme_synthetic->Descr->lambda3,
			Tdo_scheme_synthetic->Descr->block_size,
			lambda2,
			Tdo_scheme_synthetic->Descr->f_scale,
			Tdo_scheme_synthetic->Descr->scaling,
			T,
			nb_vars, Nb_vars, Nb_eqns,
			line_types, nb_line_types,
			verbose_level)) {

		FREE_int(line_types);
		return false;
	}

	T.solve_second_system(
		false /* f_use_mckay */,
		Tdo_scheme_synthetic->Descr->f_once,
		Tdo_scheme_synthetic->col_classes_len[COL_SCHEME],
		Tdo_scheme_synthetic->Descr->f_scale,
		Tdo_scheme_synthetic->Descr->scaling,
		line_types, nb_line_types,
		distributions, nb_distributions,
		verbose_level - 1);



	Output = NEW_OBJECT(tdo_refinement_output);

	Output->types = line_types;
	Output->nb_types = nb_line_types;
	Output->type_len = line_type_len;
	Output->distributions = distributions;
	Output->nb_distributions = nb_distributions;

#if 0
	int *line_types;
	int nb_line_types, line_type_len;
	int *distributions;
	int nb_distributions;
#endif


	//FREE_OBJECT(Row_split);

	if (f_v) {
		cout << "tdo_refine_3designs::td3_refine_columns: found "
			<< nb_distributions << " distributions." << endl;
	}
	return true;
}

int tdo_refine_3designs::td3_columns_setup_first_system(
	int lambda3, int block_size, int lambda2,
	tdo_data &T, int r,
	other::data_structures::partitionstack &P,
	int &nb_vars,int &nb_eqns,
	int *&line_types, int &nb_line_types,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_first_system r=" << r << endl;
	}

	int j, R, l1, l2, S, I, J, f, l, a, a2;
	int s, d, d2, d3, o, h, rr, p, u, a3, e;
	other_combinatorics::combinatorics_domain Combi;


	// create all partitions which are refined line types

	R = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];
	l1 = Tdo_scheme_synthetic->nb_row_classes[COL_SCHEME];
	l2 = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];

	nb_vars = l2; // P.n
	nb_eqns = l1; // = P.ht


	T.D1->open(nb_eqns, nb_vars, verbose_level - 1);
	S = 0;

	Int_vec_zero(T.D1->A, nb_eqns * nb_vars);
#if 0
	for (I = 0; I < nb_eqns; I++) {
		for (J = 0; J < nb_vars; J++) {
			T.D1->A[I * nb_vars + J] = 0;
		}
	}
#endif

	for (I = 0; I < nb_eqns; I++) {
		f = P.startCell[I];
		l = P.cellSize[I];
		for (j = 0; j < l; j++) {
			J = f + j;
			T.D1->A[I * nb_vars + J] = 1;
			a = Tdo_scheme_synthetic->the_row_scheme[J * R + r];
			if (a == 0) {
				T.D1->x_max[J] = 0;
			}
			else {
				T.D1->x_max[J] = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][J];
			}
			T.D1->x_min[J] = 0;
		}
		s = Tdo_scheme_synthetic->the_col_scheme[I * R + r];
		T.D1->RHS[I] = s;
		S += s;
	}

	// try to reduce the upper bounds:

	for (j = 0; j < nb_vars; j++) {
		//cout << "j=" << j << endl;
		if (T.D1->x_max[j] == 0) {
			continue;
		}
		d = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][j];
		d2 = Combi.binomial2(d) * lambda2;
		o = d2;
		for (h = 0; h < T.nb_only_one_type; h++) {
			rr = T.only_one_type[h];
			p = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][rr];

			u = T.types_first[rr];
			//cout << "u=" << u << endl;

			a = line_types[u * nb_vars + j];
			a2 = Combi.binomial2(a);
			o -= a2 * p;
			if (o < 0) {
				if (f_vvv) {
					cout << "tdo_refine_3designs::td3_columns_setup_first_system "
							"only one type, but no solution because "
						"of joining in row-class " << j << endl;
					//cout << "u=" << u << " j=" << j << endl;
				}
				return false;
			}
		}
		e = Combi.largest_binomial2_below(o);
		T.D1->x_min[j] = 0;
		T.D1->x_max[j] = MINIMUM(T.D1->x_max[j], e);
	}
	for (j = 0; j < nb_vars; j++) {
		//cout << "j=" << j << endl;
		if (T.D1->x_max[j] == 0) {
			continue;
		}
		d = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][j];
		d3 = Combi.binomial3(d) * lambda3;
		o = d3;
		for (h = 0; h < T.nb_only_one_type; h++) {
			rr = T.only_one_type[h];
			p = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][rr];
				u = T.types_first[rr];
			//cout << "u=" << u << endl;

			a = line_types[u * nb_vars + j];
			a3 = Combi.binomial3(a);
			o -= a3 * p;
			if (o < 0) {
				if (f_vvv) {
					cout << "tdo_refine_3designs::td3_columns_setup_first_system "
							"only one type, but no solution because "
						"of joining in row-class " << j << endl;
					//cout << "u=" << u << " j=" << j << endl;
				}
				return false;
			}
		}
		e = Combi.largest_binomial3_below(o);
		T.D1->x_min[j] = 0;
		T.D1->x_max[j] = MINIMUM(T.D1->x_max[j], e);
	}

	T.D1->f_has_sum = true;
	T.D1->sum = S;
	//T.D1->f_x_max = true;

	if (f_vv) {
		cout << "r=" << r << " the system is" << endl;
		T.D1->print();
	}
	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_first_system done" << endl;
	}
	return true;
}


int tdo_refine_3designs::td3_columns_setup_second_system(
	int lambda3, int block_size, int lambda2,
	int f_scale, int scaling,
	tdo_data &T,
	int nb_vars, int &Nb_vars, int &Nb_eqns,
	int *&line_types, int &nb_line_types,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_second_system" << endl;
	}

	int l2, i, r, I, a;
	int S;
	int nb_eqns_joining, nb_eqns_joining_pairs;
	int nb_eqns_joining_triples, nb_eqns_counting;
	other_combinatorics::combinatorics_domain Combi;

	l2 = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];

	nb_eqns_joining_triples = l2 + l2 * (l2 - 1) + Combi.binomial3(l2);
		// l2 times: triples within a given class
		// l2 * (l2 - 1) times (ordered pairs from an l2 set):
		//     triples with 2 in a given class, 1 in another given class
		// binomial3(l2) triples from different classes
	nb_eqns_joining_pairs = l2 + Combi.binomial2(l2);
		// l2 times: pairs within a given class
		// binomial2(l2) pairs from different classes
	nb_eqns_joining = nb_eqns_joining_triples + nb_eqns_joining_pairs;
	nb_eqns_counting = T.nb_multiple_types * (l2 + 1);
	Nb_eqns = nb_eqns_joining + nb_eqns_counting;
	Nb_vars = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		T.types_first2[i] = Nb_vars;
		Nb_vars += T.types_len[r];
	}

	T.D2->open(Nb_eqns, Nb_vars, verbose_level - 1);
	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_second_system: "
			"opening second system with "
			<< Nb_eqns << " equations and "
			<< Nb_vars << " variables" << endl;
	}

	Int_vec_zero(T.D2->A, Nb_eqns * Nb_vars);
#if 0
	for (I = 0; I < Nb_eqns; I++) {
		for (J = 0; J < Nb_vars; J++) {
			T.D2->A[I * Nb_vars + J] = 0;
		}
	}
#endif
	for (I = 0; I < Nb_eqns; I++) {
		T.D2->RHS[I] = 9999;
	}


	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_second_system "
				"before td3_columns_triples_same_class" << endl;
	}
	if (!td3_columns_triples_same_class(
		lambda3, block_size,
		T,
		nb_vars, Nb_vars,
		line_types, nb_line_types, 0,
		verbose_level - 2)) {

		return false;
	}
	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_second_system "
				"after td3_columns_triples_same_class" << endl;
	}

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_second_system "
				"before td3_columns_pairs_same_class" << endl;
	}
	if (!td3_columns_pairs_same_class(
		lambda3, block_size, lambda2,
		T,
		nb_vars, Nb_vars,
		line_types, nb_line_types, nb_eqns_joining_triples,
		verbose_level - 2)) {

		return false;
	}
	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_second_system "
				"after td3_columns_pairs_same_class" << endl;
	}

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_second_system "
				"before td3_columns_counting_flags" << endl;
	}
	if (!td3_columns_counting_flags(
			lambda3, block_size, lambda2, S,
			T,
			nb_vars, Nb_vars,
			line_types, nb_line_types, nb_eqns_joining,
			verbose_level - 2)) {

		return false;
	}
	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_second_system "
				"after td3_columns_counting_flags" << endl;
	}

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_second_system "
				"before td3_columns_lambda2_joining_pairs_from_different_classes" << endl;
	}
	if (!td3_columns_lambda2_joining_pairs_from_different_classes(
		lambda3, block_size, lambda2,
		T,
		nb_vars, Nb_vars,
		line_types, nb_line_types, nb_eqns_joining_triples + l2,
		verbose_level - 2)) {

		return false;
	}
	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_second_system "
				"after td3_columns_lambda2_joining_pairs_from_different_classes" << endl;
	}

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_second_system "
				"before td3_columns_lambda3_joining_triples_2_1" << endl;
	}
	if (!td3_columns_lambda3_joining_triples_2_1(
		lambda3, block_size, lambda2,
		T,
		nb_vars, Nb_vars,
		line_types, nb_line_types, l2,
		verbose_level - 2)) {

		return false;
	}
	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_second_system "
				"after td3_columns_lambda3_joining_triples_2_1" << endl;
	}

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_second_system "
				"before td3_columns_lambda3_joining_triples_1_1_1" << endl;
	}
	if (!td3_columns_lambda3_joining_triples_1_1_1(
		lambda3, block_size, lambda2,
		T,
		nb_vars, Nb_vars,
		line_types, nb_line_types, l2 + l2 * (l2 - 1),
		verbose_level - 2)) {

		return false;
	}
	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_second_system "
				"before td3_columns_lambda3_joining_triples_1_1_1" << endl;
	}

	if (f_scale) {
		if (S % scaling) {
			cout << "cannot scale by " << scaling
				<< " b/c S=" << S << endl;
			exit(1);
		}
		S /= scaling;
		for (I = 0; I < Nb_eqns; I++) {
			a = T.D2->RHS[I];
			if (a % scaling) {
				if (a % scaling) {
					cout << "cannot scale by " << scaling
						<< " b/c RHS[" << I << "]=" << a << endl;
				}
				exit(1);
			}
			a /= scaling;
			T.D2->RHS[I] = a;
		}
#if 0
		for (I = 0; I < Nb_eqns; I++)
			for (J = 0; J < Nb_vars; J++)
				T.D2->A[I * Nb_vars + J] *= scaling;
#endif
	}



	T.D2->f_has_sum = true;
	T.D2->sum = S;


	if (f_vv) {
		cout << "The second system is" << endl;

		T.D2->print();
	}

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_setup_second_system done" << endl;
	}
	return true;

}


int tdo_refine_3designs::td3_columns_triples_same_class(
	int lambda3, int block_size,
	tdo_data &T,
	int nb_vars, int Nb_vars,
	int *&line_types, int &nb_line_types, int eqn_offset,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);


	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_triples_same_class: "
				"eqn_offset=" << eqn_offset << endl;
	}


	int I, i, r, f, l, j, c, J, a, a3, rr, p, u, l2, h;
	other_combinatorics::combinatorics_domain Combi;

	l2 = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];

	// triples from the same class:
	for (I = 0; I < l2; I++) {
		for (i = 0; i < T.nb_multiple_types; i++) {
			r = T.multiple_types[i];
			f = T.types_first[r];
			l = T.types_len[r];
			for (j = 0; j < l; j++) {
				c = f + j;
				J = T.types_first2[i] + j;
				a = line_types[c * nb_vars + I];
				a3 = Combi.binomial3(a);
				// joining triples from the same class:
				T.D2->A[(eqn_offset + I) * Nb_vars + J] = a3;
			}
		}
		a = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][I];
		a3 = Combi.binomial3(a);
		T.D2->RHS[eqn_offset + I] = a3 * lambda3;
		for (h = 0; h < T.nb_only_one_type; h++) {
			rr = T.only_one_type[h];
			p = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][rr];
			u = T.types_first[rr];
			a = line_types[u * nb_vars + I];
			a3 = Combi.binomial3(a);
			T.D2->RHS[eqn_offset + I] -= a3 * p;
			if (T.D2->RHS[eqn_offset + I] < 0) {
				if (f_v) {
					cout << "td3_refine_columns: RHS[I] is negative, "
						"no solution for the distribution" << endl;
				}
				return false;
			}
		}
	}
	if (f_vvv) {
		cout << "triples from the same class, the system is" << endl;
		T.D2->print();
	}
	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_triples_same_class done" << endl;
	}
	return true;
}

int tdo_refine_3designs::td3_columns_pairs_same_class(
	int lambda3, int block_size, int lambda2,
	tdo_data &T,
	int nb_vars, int Nb_vars,
	int *&line_types, int &nb_line_types, int eqn_offset,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_pairs_same_class: "
			"eqn_offset=" << eqn_offset << endl;
	}

	int I, i, r, f, l, j, c, J, a, a2, rr, p, u, l2, h;
	other_combinatorics::combinatorics_domain Combi;

	l2 = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];

	// pairs from the same class:
	for (I = 0; I < l2; I++) {
		for (i = 0; i < T.nb_multiple_types; i++) {
			r = T.multiple_types[i];
			f = T.types_first[r];
			l = T.types_len[r];
			for (j = 0; j < l; j++) {
				c = f + j;
				J = T.types_first2[i] + j;
				a = line_types[c * nb_vars + I];
				a2 = Combi.binomial2(a);
				// joining pairs from the same class:
				T.D2->A[(eqn_offset + I) * Nb_vars + J] = a2;
			}
		}
		a = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][I];
		a2 = Combi.binomial2(a);
		T.D2->RHS[eqn_offset + I] = a2 * lambda2;
		for (h = 0; h < T.nb_only_one_type; h++) {
			rr = T.only_one_type[h];
			p = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][rr];
			u = T.types_first[rr];
			a = line_types[u * nb_vars + I];
			a2 = Combi.binomial2(a);
			T.D2->RHS[eqn_offset + I] -= a2 * p;
			if (T.D2->RHS[eqn_offset + I] < 0) {
				if (f_v) {
					cout << "td3_refine_columns: RHS[eqn_offset + I] "
						"is negative, no solution for the "
						"distribution" << endl;
				}
				return false;
			}
		}
	}
	if (f_vvv) {
		cout << "pairs from the same class, the system is" << endl;
		T.D2->print();
	}

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_pairs_same_class done" << endl;
	}
	return true;
}

int tdo_refine_3designs::td3_columns_counting_flags(
	int lambda3, int block_size, int lambda2, int &S,
	tdo_data &T,
	int nb_vars, int Nb_vars,
	int *&line_types, int &nb_line_types, int eqn_offset,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int I, i, r, f, l, j, c, J, a, b, rr, p, u, l2, h, s;

	l2 = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_counting_flags: "
				"eqn_offset=" << eqn_offset << endl;
	}
	// counting flags, a block diagonal system with
	// nb_multiple_types * (l2 + 1) equations

	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];

		for (I = 0; I < l2; I++) {

			for (j = 0; j < l; j++) {
				c = f + j;
				J = T.types_first2[i] + j;
				a = line_types[c * nb_vars + I];
				T.D2->A[(eqn_offset + i * (l2 + 1) + I) * Nb_vars + J] = a;
			}
			a = Tdo_scheme_synthetic->the_row_scheme[I * Tdo_scheme_synthetic->nb_col_classes[ROW_SCHEME] + r];
			b = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][I];
			T.D2->RHS[eqn_offset + i * (l2 + 1) + I] = a * b;

			for (h = 0; h < T.nb_only_one_type; h++) {
				rr = T.only_one_type[h];
				p = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][rr];
				u = T.types_first[rr];
				a = line_types[u * nb_vars + I];
				T.D2->RHS[eqn_offset + i * (l2 + 1) + I] -= a * p;

				if (T.D2->RHS[eqn_offset + i * (l2 + 1) + I] < 0) {
					if (f_v) {
						cout << "td3_columns_counting_flags: "
							"RHS[nb_eqns_joining + i * (l2 + 1) + I] "
							"is negative, no solution for the "
							"distribution" << endl;
					}
					return false;
				}
			} // next h
		} // next I
	} // next i


	S = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];

		// one extra equation for the sum
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			// counting: extra row of ones
			T.D2->A[(eqn_offset + i * (l2 + 1) + l2) * Nb_vars + J] = 1;
		}

		s = Tdo_scheme_synthetic->col_classes_len[ROW_SCHEME][r];
		T.D2->RHS[eqn_offset + i * (l2 + 1) + l2] = s;
		S += s;
	}
	if (f_vvv) {
		cout << "tdo_refine_3designs::td3_columns_counting_flags, the system is" << endl;
		T.D2->print();
	}

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_counting_flags done" << endl;
	}

	return true;
}

int tdo_refine_3designs::td3_columns_lambda2_joining_pairs_from_different_classes(
	int lambda3, int block_size, int lambda2,
	tdo_data &T,
	int nb_vars, int Nb_vars,
	int *&line_types, int &nb_line_types, int eqn_offset,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_lambda2_joining_pairs_from_different_classes "
				"eqn_offset=" << eqn_offset << endl;
	}

	int I1, I2, i, r, f, l, j, c, J, a, b, ab, k, rr, p, u, l2, h;
	other_combinatorics::combinatorics_domain Combi;

	l2 = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];

	// lambda2: joining pairs from different classes
	for (I1 = 0; I1 < l2; I1++) {

		for (I2 = I1 + 1; I2 < l2; I2++) {
			k = Combi.ij2k(I1, I2, l2);

			for (i = 0; i < T.nb_multiple_types; i++) {
				r = T.multiple_types[i];
				f = T.types_first[r];
				l = T.types_len[r];

				for (j = 0; j < l; j++) {
					c = f + j;
					J = T.types_first2[i] + j;
					a = line_types[c * nb_vars + I1];
					b = line_types[c * nb_vars + I2];
					ab = a * b;
					// joining pairs from different classes:
					T.D2->A[(eqn_offset + k) * Nb_vars + J] = ab;
				}
			}
			a = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][I1];
			b = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][I2];
			T.D2->RHS[eqn_offset + k] = a * b * lambda2;

			for (h = 0; h < T.nb_only_one_type; h++) {
				rr = T.only_one_type[h];
				p = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][rr];
				u = T.types_first[rr];
				a = line_types[u * nb_vars + I1];
				b = line_types[u * nb_vars + I2];
				T.D2->RHS[eqn_offset + k] -= a * b * p;

				if (T.D2->RHS[eqn_offset + k] < 0) {
					if (f_v) {
						cout << "td3_columns_lambda2_joining_pairs_"
							"from_different_classes: RHS[eqn_offset + k] "
							"is negative, no solution for the "
							"distribution" << endl;
					}
					return false;
				}
			} // next h
		}
	}
	if (f_vvv) {
		cout << "td3_columns_lambda2_joining_pairs_from_different_classes, "
				"the system is" << endl;
		T.D2->print();
	}

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_lambda2_joining_pairs_from_different_classes done" << endl;
	}
	return true;
}

int tdo_refine_3designs::td3_columns_lambda3_joining_triples_2_1(
	int lambda3, int block_size, int lambda2,
	tdo_data &T,
	int nb_vars, int Nb_vars,
	int *&line_types, int &nb_line_types, int eqn_offset,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_lambda3_joining_triples_2_1: "
			"eqn_offset=" << eqn_offset << endl;
	}

	int I1, I2, i, r, f, l, j, c, J, a, a2, ab, b, k, rr, p, u, l2, h;
	int length_first, length_first2, length_second;
	other_combinatorics::combinatorics_domain Combi;

	l2 = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];

	// lambda3: joining triples with two in the first
	// class and one in the second class
	for (I1 = 0; I1 < l2; I1++) {
		length_first = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][I1];
		length_first2 = Combi.binomial2(length_first);

		for (I2 = 0; I2 < l2; I2++) {
			if (I2 == I1) {
				continue;
			}

			length_second = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][I2];
			k = Combi.ordered_pair_rank(I1, I2, l2);

			for (i = 0; i < T.nb_multiple_types; i++) {
				r = T.multiple_types[i];
				f = T.types_first[r];
				l = T.types_len[r];

				for (j = 0; j < l; j++) {
					c = f + j;
					J = T.types_first2[i] + j;
					a = line_types[c * nb_vars + I1];
					b = line_types[c * nb_vars + I2];
					ab = Combi.binomial2(a) * b;
					T.D2->A[(l2 + k) * Nb_vars + J] = ab;
				}
			}
			T.D2->RHS[l2 + k] = length_first2 * length_second * lambda3;

			for (h = 0; h < T.nb_only_one_type; h++) {
				rr = T.only_one_type[h];
				p = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][rr];
				u = T.types_first[rr];
				a = line_types[u * nb_vars + I1];
				a2 = Combi.binomial2(a);
				b = line_types[u * nb_vars + I2];
				T.D2->RHS[l2 + k] -= a2 * b * p;

				if (T.D2->RHS[l2 + k] < 0) {
					if (f_v) {
						cout << "td3_columns_lambda3_joining_triples_2_1: "
							"RHS[l2 + k] is negative, no solution for "
							"the distribution" << endl;
					}
					return false;
				}
			} // next h
		}
	}
	if (f_vvv) {
		cout << "td3_columns_lambda3_joining_triples_2_1, "
				"the system is" << endl;
		T.D2->print();
	}

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_lambda3_joining_triples_2_1 done" << endl;
	}
	return true;
}

int tdo_refine_3designs::td3_columns_lambda3_joining_triples_1_1_1(
	int lambda3, int block_size, int lambda2,
	tdo_data &T,
	int nb_vars, int Nb_vars,
	int *&line_types, int &nb_line_types, int eqn_offset,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_lambda3_joining_triples_1_1_1 "
			"eqn_offset=" << eqn_offset << endl;
	}

	int I1, I2, I3, i, r, f, l, j, c, J, a, b, k, rr, p, u, l2, h, g;
	int length_first, length_second, length_third;
	other_combinatorics::combinatorics_domain Combi;

	l2 = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];

	// lambda3: joining triples with all in different classes
	for (I1 = 0; I1 < l2; I1++) {
		length_first = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][I1];

		for (I2 = I1 + 1; I2 < l2; I2++) {
			length_second = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][I2];

			for (I3 = I2 + 1; I3 < l2; I3++) {
				length_third = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][I3];

				k = Combi.ijk_rank(I1, I2, I3, l2);
				for (i = 0; i < T.nb_multiple_types; i++) {
					r = T.multiple_types[i];
					f = T.types_first[r];
					l = T.types_len[r];
					for (j = 0; j < l; j++) {
						c = f + j;
						J = T.types_first2[i] + j;
						a = line_types[c * nb_vars + I1];
						b = line_types[c * nb_vars + I2];
						g = line_types[c * nb_vars + I3];
						T.D2->A[(l2 + l2 * (l2 - 1) + k) *
								Nb_vars + J] = a * b * g;
					}
				}
				T.D2->RHS[l2 + l2 * (l2 - 1) + k] = length_first *
						length_second * length_third * lambda3;
				for (h = 0; h < T.nb_only_one_type; h++) {
					rr = T.only_one_type[h];
					p = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][rr];
					u = T.types_first[rr];
					a = line_types[u * nb_vars + I1];
					b = line_types[u * nb_vars + I2];
					g = line_types[u * nb_vars + I3];
					T.D2->RHS[l2 + l2 * (l2 - 1) + k] -= a * b * g * p;
					if (T.D2->RHS[l2 + l2 * (l2 - 1) + k] < 0) {
						if (f_v) {
							cout << "td3_columns_lambda3_joining_triples_"
								"1_1_1: RHS[l2 + l2 * (l2 - 1) + k] is "
								"negative, no solution for the "
								"distribution" << endl;
						}
						return false;
					}
				} // next h
			}
		}
	}
	if (f_vvv) {
		cout << "td3_columns_lambda3_joining_triples_1_1_1, "
				"the system is" << endl;
		T.D2->print();
	}
	if (f_v) {
		cout << "tdo_refine_3designs::td3_columns_lambda3_joining_triples_1_1_1 done" << endl;
	}

	return true;
}


}}}}

