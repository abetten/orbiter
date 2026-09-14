/*
 * tdo_refine_cols.cpp
 *
 *  Created on: Sep 14, 2026
 *      Author: betten
 */





#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace tactical_decompositions {


tdo_refine_cols::tdo_refine_cols()
{
	Record_birth();

	Tdo_scheme_synthetic = NULL;
	Row_split = NULL;

}

tdo_refine_cols::~tdo_refine_cols()
{
	Record_death();

	if (Row_split) {
		FREE_OBJECT(Row_split);
	}
}

void tdo_refine_cols::init(
		tdo_scheme_synthetic *Tdo_scheme_synthetic,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "tdo_refine_cols::init" << endl;
	}

	tdo_refine_cols::Tdo_scheme_synthetic = Tdo_scheme_synthetic;

	if (f_v) {
		cout << "tdo_refine_cols::init done" << endl;
	}
}

int tdo_refine_cols::refine_columns(
		tdo_refinement_output *&Output,
		int &cnt_second_system,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "tdo_refine_cols::refine_columns" << endl;
	}
	Tdo_scheme_synthetic->check_init();

	int f_easy;
	int l1, l2, R;
	int ret = false;


	if (f_vv) {
		cout << "f_omit1=" << Tdo_scheme_synthetic->Descr->f_omit1 << " Descr->omit1=" << Tdo_scheme_synthetic->Descr->omit1 << endl;
		cout << "f_omit2=" << Tdo_scheme_synthetic->Descr->f_omit2 << " omit2=" << Tdo_scheme_synthetic->Descr->omit2 << endl;
		cout << "f_use_packing_numbers=" << Tdo_scheme_synthetic->Descr->f_use_packing_numbers << endl;
		cout << "f_D1_upper_bound_x0=" << Tdo_scheme_synthetic->Descr->f_D1_upper_bound_x0 << endl;
		cout << "f_use_mckay_solver=" << Tdo_scheme_synthetic->Descr->f_use_mckay_solver << endl;
	}
	R = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];
	l1 = Tdo_scheme_synthetic->nb_row_classes[COL_SCHEME];
	l2 = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];
	if (f_vv) {
		cout << "l1=" << l1 << endl;
		cout << "l2=" << l2 << endl;
		cout << "R=" << R << endl;
	}

	//other::data_structures::partitionstack *Row_split;


	Row_split = Tdo_scheme_synthetic->get_row_split_partition(0 /*verbose_level*/);

	if (f_vv) {
		cout << "tdo_refine_cols::refine_columns "
				"row split partition: " << endl;
		Row_split->print(cout);
		cout << endl;
	}
	if (Row_split->ht != l1) {
		cout << "P.ht != l1" << endl;
	}

	if ((R == 1) && (l1 == 1) && (Tdo_scheme_synthetic->the_col_scheme[0] == -1)) {
		f_easy = true;
		if (false) {
			cout << "easy mode" << endl;
		}
	}
	else {
		f_easy = false;
		if (false) {
			cout << "full mode" << endl;
		}
	}


	if (f_easy) {
		cout << "tdo_refine_cols::refine_columns "
				"refine_cols_easy nyi" << endl;
		exit(1);

	}
	else {
		ret = refine_cols_hard(
				Output,
			cnt_second_system,
			verbose_level - 1);
	}

	//FREE_OBJECT(Row_split);


	if (f_v) {
		cout << "tdo_refine_cols::refine_columns finished" << endl;
	}
	return ret;
}

int tdo_refine_cols::refine_cols_hard(
		tdo_refinement_output *&Output,
		int &cnt_second_system,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);


	if (f_v) {
		cout << "tdo_refine_cols::refine_cols_hard" << endl;
	}

	Tdo_scheme_synthetic->check_init();

	//int nb_eqns, nb_vars;
	int R, /*l1,*/ l2, L1, L2, r;
	int nb_sol, nb_sol1, f_survive;


	int *line_types;
	int nb_line_types;
	int line_type_len;
	int *distributions;
	int nb_distributions;
	int line_types_allocated;



	{
		tdo_data T;
		int i, j, u;

		if (f_vv) {
			cout << "f_omit1=" << Tdo_scheme_synthetic->Descr->f_omit1 << " Descr->omit1=" << Tdo_scheme_synthetic->Descr->omit1 << endl;
			cout << "f_omit=" << Tdo_scheme_synthetic->Descr->f_omit2 << " omit2=" << Tdo_scheme_synthetic->Descr->omit2 << endl;
			cout << "f_use_packing_numbers=" << Tdo_scheme_synthetic->Descr->f_use_packing_numbers << endl;
			cout << "f_D1_upper_bound_x0=" << Tdo_scheme_synthetic->Descr->f_D1_upper_bound_x0 << endl;
			cout << "D1_upper_bound_x0=" << Tdo_scheme_synthetic->Descr->D1_upper_bound_x0 << endl;
			cout << "f_use_mckay_solver=" << Tdo_scheme_synthetic->Descr->f_use_mckay_solver << endl;
		}
		R = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];
		//l1 = nb_row_classes[COL_SCHEME];
		l2 = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];

		if (f_v) {
			cout << "tdo_refine_cols::refine_cols_hard "
					"the_row_scheme is:" << endl;
			for (i = 0; i < l2; i++) {
				for (j = 0; j < R; j++) {
					cout << setw(4) << Tdo_scheme_synthetic->the_row_scheme[i * R + j];
				}
				cout << endl;
			}
		}

		column_refinement_L1_L2(
				Tdo_scheme_synthetic->Descr->f_omit1,
				Tdo_scheme_synthetic->Descr->omit1,
				L1, L2,
				verbose_level);

		T.allocate(R);

		T.types_first[0] = 0;



		line_types_allocated = 100;
		nb_line_types = 0;
		line_types = NEW_int(line_types_allocated * l2);
		line_type_len = l2;

		T.nb_only_one_type = 0;
		T.nb_multiple_types = 0;


		for (r = 0; r < R; r++) {

			if (f_v) {
				cout << "tdo_refine_cols::refine_cols_hard r=" << r << " / " << R << endl;
			}

			if (f_v) {
				cout << "tdo_refine_cols::refine_cols_hard "
						"before tdo_columns_setup_first_system" << endl;
			}
			if (!tdo_columns_setup_first_system(
					T, r,
					line_types, nb_line_types,
					verbose_level)) {
				if (f_v) {
					cout << "tdo_refine_cols::refine_cols_hard "
							"tdo_columns_setup_first_system returns false" << endl;
				}
				FREE_int(line_types);
				return false;
			}
			if (f_v) {
				cout << "tdo_refine_cols::refine_cols_hard "
						"after tdo_columns_setup_first_system" << endl;
			}

			if (Tdo_scheme_synthetic->Descr->f_D1_upper_bound_x0) {
				T.D1->x_min[0] = 0;
				T.D1->x_max[0] = Tdo_scheme_synthetic->Descr->D1_upper_bound_x0;
				cout << "setting upper bound for D1->x[0] to "
						<< T.D1->x_max[0] << endl;
			}


#if 0
			// ATTENTION, this is from a specific problem
			// on arcs in a plane (MARUTA)

			// now we are interested in (42,6)_8 arcs
			// a line intersects the arc in at most 6 points:
			//T.D1->x_max[0] = 6;

			// now we are interested in (33,5)_8 arcs
			//T.D1->x_max[0] = 5;
			//cout << "ATTENTION: MARUTA, limiting x_max[0] to 5" << endl;

			// now we are interested in (49,7)_8 arcs
			//T.D1->x_max[0] = 7;
			//cout << "ATTENTION: MARUTA, limiting x_max[0] to 7" << endl;

			// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#endif


			if (f_vv) {
				string label;

				label = "first_" + std::to_string(r);
				T.D1->write_xml(cout, label);
			}

			if (f_v) {
				cout << "tdo_refine_cols::refine_cols_hard "
						"before T.solve_first_system" << endl;
			}
			nb_sol = T.solve_first_system(
				line_types, nb_line_types, line_types_allocated,
				verbose_level - 1);
			if (f_v) {
				cout << "tdo_refine_cols::refine_cols_hard "
						"after T.solve_first_system" << endl;
			}

			if (f_v) {
				cout << "tdo_refine_cols::refine_cols_hard "
						"r = " << r << ", found " << nb_sol
						<< " refined line types" << endl;
			}
			if (f_vv) {
				Int_vec_print_integer_matrix_width(
						cout,
					line_types + T.types_first[r] * L2, nb_sol, L2, L2, 2);
			}


			// Some very specialized test:

			// if in some row block i, and column block r,
			// the number of flags in the row equals the number of columns in the column partition,
			// then the refined lines in this part must have at least one entry in the row block i.
			// So, zero is not allowed.


			nb_sol1 = 0;
			for (u = 0; u < nb_sol; u++) {
				f_survive = true;
				for (i = 0; i < L2; i++) {
					int len1, len2, flags;
					len1 = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][i];
					if (len1 > 1) {
						continue;
					}
					len2 = Tdo_scheme_synthetic->col_classes_len[ROW_SCHEME][r];
					flags = Tdo_scheme_synthetic->the_row_scheme[i * R + r];
					if (flags == len2) {
						if (line_types[(T.types_first[r] + u) * L2 + i] == 0) {
							f_survive = false;
							if (f_vv) {
								cout << "line type " << u << " eliminated, "
										"line_types[] = 0" << endl;
								cout << "row block " << i << endl;
								cout << "col block=" << r << endl;
								cout << "length of col block " << len2 << endl;
								cout << "flags " << flags << endl;

							}
							break;
						}
					}
				}
				if (f_survive) {
					for (i = 0; i < L2; i++) {
						line_types[(T.types_first[r] + nb_sol1) * L2 + i] =
							line_types[(T.types_first[r] + u) * L2 + i];
					}
					nb_sol1++;
				}
			}
			if (nb_sol1 < nb_sol) {
				if (f_v) {
					cout << "tdo_scheme_synthetic::refine_cols_hard "
							"eliminated " << nb_sol - nb_sol1
							<< " types" << endl;
				}
				nb_sol = nb_sol1;
				nb_line_types = T.types_first[r] + nb_sol1;
				if (f_v) {
					cout << "tdo_scheme_synthetic::refine_cols_hard "
							"r = " << r << ", found " << nb_sol
							<< " refined line types" << endl;
				}
			}

			if (f_vv) {
				Int_vec_print_integer_matrix_width(
						cout,
					line_types + T.types_first[r] * L2, nb_sol, L2, L2, 2);
			}
			if (nb_sol == 0) {
				FREE_int(line_types);
				return false;
			}

			T.types_len[r] = nb_sol;
			T.types_first[r + 1] = T.types_first[r] + nb_sol;

			if (nb_sol == 1) {
				if (f_v) {
					cout << "tdo_refine_cols::refine_cols_hard "
							"only one solution in block "
						"r=" << r << endl;
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



		if (f_v) {
			cout << "tdo_refine_cols::refine_cols_hard "
					"R=" << R << endl;
			cout << "r : T.types_first[r] : T.types_len[r]" << endl;
			for (r = 0; r < R; r++) {
				cout << r << " : " << T.types_first[r] << " : "
						<< T.types_len[r] << endl;
			}
		}
		if (f_vv) {
			Int_vec_print_integer_matrix_width(
					cout, line_types,
					nb_line_types, line_type_len, line_type_len, 3);
		}

		// now we compute the distributions:
		//
		int f_scale = false;
		int scaling = 0;

		if (!tdo_columns_setup_second_system(
				T,
				line_types, nb_line_types,
				verbose_level)) {

			FREE_int(line_types);
			if (f_v) {
				cout << "tdo_refine_cols::refine_cols_hard tdo_columns_setup_second_system return false" << endl;
			}
			Output = NEW_OBJECT(tdo_refinement_output);
			return false;
		}


#if 0
		// ATTENTION, this is for the classification
		// of (42,6)_8 arcs where a_1 = 0 (MARUTA)

		//T.D2->x_max[5] = 0; // a_1 is known to be zero

		// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#endif



		if (f_vv) {
			string label;

			label = "second";
			T.D2->write_xml(cout, label);
		}




		int idx, /*f,*/ l;
		idx = 0;
		for (r = 0; r < R; r++) {
			l = T.types_len[r];
			if (l > 1) {
				if (T.multiple_types[idx] != r) {
					cout << "T.multiple_types[idx] != r" << endl;
					exit(1);
				}
				//f = T.types_first2[idx];
				idx++;
			}
			else {
				//f = -1;
			}
		}

		if (f_v) {
			cout << "tdo_refine_cols::refine_cols_hard "
					"solving second system "
					<< cnt_second_system << " which is " << T.D2->m
					<< " x " << T.D2->n << endl;
			cout << "variable blocks:" << endl;
			cout << "i : r : col_classes_len[COL][r] : types_first2[i] : "
				"types_len[r]" << endl;
			int f, l;
			for (i = 0; i < T.nb_multiple_types; i++) {
				r = T.multiple_types[i];
				f = T.types_first2[i];
				l = T.types_len[r];
				cout << i << " : " << r << " : " << setw(3)
					<< Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][r] << " : " << setw(3)
					<< f << " : " << setw(3) << l << endl;
			}
		}

		if (Tdo_scheme_synthetic->Descr->f_omit2) {
			if (f_v) {
				cout << "tdo_refine_cols::refine_cols_hard "
						"before T.solve_second_system_omit" << endl;
			}
			T.solve_second_system_omit(
					Tdo_scheme_synthetic->col_classes_len[COL_SCHEME],
				line_types, nb_line_types,
				distributions, nb_distributions,
				Tdo_scheme_synthetic->Descr->omit2,
				verbose_level);
			if (f_v) {
				cout << "tdo_refine_cols::refine_cols_hard "
						"after T.solve_second_system_omit" << endl;
			}
		}
		else {
			if (f_v) {
				cout << "tdo_refine_cols::refine_cols_hard "
						"before T.solve_second_system_with_help" << endl;
			}
			T.solve_second_system_with_help(
					Tdo_scheme_synthetic->Descr->f_use_mckay_solver, Tdo_scheme_synthetic->Descr->f_once,
					Tdo_scheme_synthetic->col_classes_len[COL_SCHEME], f_scale, scaling,
				line_types, nb_line_types,
				distributions, nb_distributions,
				cnt_second_system, Tdo_scheme_synthetic->Descr->Sol,
				verbose_level);
			if (f_v) {
				cout << "tdo_refine_cols::refine_cols_hard "
						"after T.solve_second_system_with_help" << endl;
			}
		}

		if (f_v) {
			cout << "tdo_refine_cols::refine_cols_hard "
					"second system " << cnt_second_system
				<< " found " << nb_distributions << " distributions." << endl;
		}
		if (f_v) {
			cout << "tdo_refine_cols::refine_cols_hard "
					"The distributions are:" << endl;
			Int_matrix_print(distributions, nb_distributions, nb_line_types);
		}

#if 0
		// ATTENTION: this is from a specific problem of CHEON !!!!

		cout << "ATTENTION, we are running specific code "
				"for a problem of Cheon" << endl;
		int cnt, h, x0;

		cnt = 0;
		for (h = 0; h < nb_distributions; h++) {
			x0 = distributions[h * nb_line_types + 0];
			if (x0 == 12) {
				for (j = 0; j < nb_line_types; j++) {
					distributions[cnt * nb_line_types + j] =
							distributions[h * nb_line_types + j];
					}
				cnt++;
				}
			if (x0 > 12) {
				cout << "x0 > 12, something is wrong" << endl;
				exit(1);
				}
			}
		cout << "CHEON: we found " << cnt << " refinements with x0=12" << endl;
		nb_distributions = cnt;

		// ATTENTION
#endif

		cnt_second_system++;
		if (f_v) {
			cout << "tdo_refine_cols::refine_cols_hard before freeing T" << endl;
		}
	}
	if (f_v) {
		cout << "tdo_refine_cols::refine_cols_hard after closing T." << endl;
	}


#if 0
	// output is now in
	int *line_types;
	int nb_line_types;
	int line_type_len;
	int *distributions;
	int nb_distributions;
	int line_types_allocated;
#endif

	Output = NEW_OBJECT(tdo_refinement_output);

	Output->types_allocated = line_types_allocated;
	Output->nb_types = nb_line_types;
	Output->type_len = line_type_len;
	Output->types = line_types;
	Output->distributions = distributions;
	Output->nb_distributions = nb_distributions;



	if (f_v) {
		cout << "tdo_refine_cols::refine_cols_hard done" << endl;
	}
	return true;
}

void tdo_refine_cols::column_refinement_L1_L2(
		//other::data_structures::partitionstack *Row_split,
		int f_omit, int omit,
		int &L1, int &L2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int l1, l2, omit2, i;
	l1 = Tdo_scheme_synthetic->nb_row_classes[COL_SCHEME];
	l2 = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME]; // the finer scheme

	omit2 = 0;
	if (f_omit) {
		for (i = l1 - omit; i < l1; i++) {
			omit2 += Row_split->cellSize[i];
		}
	}
	L1 = l1 - omit;
	L2 = l2 - omit2;
	if (f_v) {
		cout << "tdo_refine_cols::column_refinement_L1_L2 "
				"l1 = " << l1
			<< " l2=" << l2 << " L1=" << L1 << " L2=" << L2 << endl;
	}
}

int tdo_refine_cols::tdo_columns_setup_first_system(
		tdo_data &T, int r,
		//other::data_structures::partitionstack *Row_split,
		int *&line_types, int &nb_line_types,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);


	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_first_system "
				"r=" << r << endl;
	}
	Tdo_scheme_synthetic->check_init();


	int i, j, f, l, I, J, rr, R, S, a, a2, s, /*l1, l2,*/ L1, L2;
	int h, u, d, d2, o, e, p, eqn_number, nb_vars, nb_eqns;
	other_combinatorics::combinatorics_domain Combi;
	int local_omit;

	// create all partitions which are refined line types

	if (!Tdo_scheme_synthetic->Descr->f_omit1) {
		local_omit = 0;
	}
	else {
		local_omit = Tdo_scheme_synthetic->Descr->omit1;
	}

	if (f_v) {
		if (Tdo_scheme_synthetic->Descr->f_omit1) {
			cout << "omit1=" << local_omit << endl;
		}
	}

	R = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];
	//l1 = nb_row_classes[COL_SCHEME];
	//l2 = nb_row_classes[ROW_SCHEME]; // the finer scheme

	column_refinement_L1_L2(
			Tdo_scheme_synthetic->Descr->f_omit1, local_omit,
			L1, L2,
			verbose_level);

	nb_vars = L2;
	nb_eqns = L1 + 1 + (R - 1);


	T.D1->open(nb_eqns, nb_vars, verbose_level - 1);
	S = 0;

	Int_vec_zero(T.D1->A, nb_eqns * nb_vars);

	// the m equalities that come from the fact that the new type
	// is a partition of the old type.

	// we are in the r-th column class (r is given)

	for (I = 0; I < L1; I++) {
		f = Row_split->startCell[I];
		l = Row_split->cellSize[I];
		for (j = 0; j < l; j++) {
			J = f + j;
			T.D1->Aij(I, J) = 1;
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
		T.D1->type[I] = t_EQ;
		S += s;
	}

	eqn_number = L1;

	for (i = 0; i < L2; i++) {
		a = Combi.minus_one_if_positive(
				Tdo_scheme_synthetic->the_row_scheme[i * R + r]);
		T.D1->Aij(eqn_number, i) = a;
	}
	T.D1->RHS[eqn_number] = Tdo_scheme_synthetic->col_classes_len[ROW_SCHEME][r] - 1;
			// the -1 was missing!!!
	T.D1->type[eqn_number] = t_LE;
	eqn_number++;

	for (j = 0; j < R; j++) {
		if (j == r) {
			continue;
		}
		for (i = 0; i < L2; i++) {
			a = Tdo_scheme_synthetic->the_row_scheme[i * R + j];
			T.D1->Aij(eqn_number, i) = a;
		}
		T.D1->RHS[eqn_number] = Tdo_scheme_synthetic->col_classes_len[ROW_SCHEME][j];
		T.D1->type[eqn_number] = t_LE;
		eqn_number++;
	}
	T.D1->m = eqn_number;


	// try to reduce the upper bounds:

	for (h = 0; h < T.nb_only_one_type; h++) {
		rr = T.only_one_type[h];
		u = T.types_first[rr];
		//cout << "u=" << u << endl;
		for (j = 0; j < nb_vars; j++) {
			//cout << "j=" << j << endl;
			if (T.D1->x_max[j] == 0) {
				continue;
			}
			d = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][j];
			p = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][rr];
			d2 = Combi.binomial2(d);

			a = line_types[u * nb_vars + j]; // wait, ToDo! they have not been initialized yet ??? maybe yes


			a2 = Combi.binomial2(a);
			o = d2 - a2 * p;
			if (o < 0) {
				if (f_vv) {
					cout << "only one type, but no solution because of "
						"joining in row-class " << j << endl;
					//cout << "u=" << u << " j=" << j << endl;
				}
				return false;
			}
			e = Combi.largest_binomial2_below(o);
			T.D1->x_min[j] = 0;
			T.D1->x_max[j] = MINIMUM(T.D1->x_max[j], e);
		}
	}


	T.D1->f_has_sum = true;
	T.D1->sum = S;
	//T.D1->f_x_max = true;

	T.D1->eliminate_zero_rows_quick(verbose_level);

	if (f_vv) {
		T.D1->print();
	}
	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_first_system done" << endl;
	}
	return true;
}

int tdo_refine_cols::tdo_columns_setup_second_system(
		tdo_data &T,
		//other::data_structures::partitionstack *Row_split,
		int *&line_types, int &nb_line_types,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_second_system" << endl;
		cout << "f_use_packing_numbers="
				<< Tdo_scheme_synthetic->Descr->f_use_packing_numbers << endl;
	}
	Tdo_scheme_synthetic->check_init();

	int i, len, r, L1, L2, Nb_eqns, Nb_vars;
	other_combinatorics::combinatorics_domain Combi;

	int nb_eqns_joining, nb_eqns_counting;
	int nb_eqns_upper_bound, nb_eqns_used;
	int l2;

	l2 = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME]; // the finer scheme

	column_refinement_L1_L2(
			Tdo_scheme_synthetic->Descr->f_omit1,
			Tdo_scheme_synthetic->Descr->omit1,
			L1, L2,
			verbose_level);

	nb_eqns_joining = L2 + Combi.binomial2(L2);
	nb_eqns_counting = T.nb_multiple_types * (L2 + 1);
	nb_eqns_upper_bound = 0;
	if (Tdo_scheme_synthetic->Descr->f_use_packing_numbers) {
		for (i = 0; i < l2; i++) {
			len = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][i];
			if (len > 2) {
				nb_eqns_upper_bound += len - 2;
			}
		}
	}

	Nb_eqns = nb_eqns_joining + nb_eqns_counting + nb_eqns_upper_bound;
	Nb_vars = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		T.types_first2[i] = Nb_vars;
		Nb_vars += T.types_len[r];
	}

	T.D2->open(Nb_eqns, Nb_vars, verbose_level - 1);
	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_second_system "
				"opening second system with "
			<< Nb_eqns << " equations and " << Nb_vars
			<< " variables" << endl;
	}
	if (f_vv) {
		cout << "l2=" << l2 << endl;
		cout << "L2=" << L2 << endl;
		cout << "nb_eqns_joining=" << nb_eqns_joining << endl;
		cout << "nb_eqns_counting=" << nb_eqns_counting << endl;
		cout << "nb_eqns_upper_bound=" << nb_eqns_upper_bound << endl;
		cout << "T.nb_multiple_types=" << T.nb_multiple_types << endl;
		cout << "i : r = T.multiple_types[i] : T.types_first2[i] "
				": T.types_len[r]" << endl;
		for (i = 0; i < T.nb_multiple_types; i++) {
			r = T.multiple_types[i];
			cout << i << " : " << r << " : " << T.types_first2[i]
				<< " : " << T.types_len[r] << endl;
		}
	}


	Int_vec_zero(T.D2->A, Nb_eqns * Nb_vars);

	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_second_system "
				"before tdo_columns_setup_second_system_eqns_joining" << endl;
	}
	if (!tdo_columns_setup_second_system_eqns_joining(
			T,
			line_types, nb_line_types,
			0 /*eqn_start*/,
			verbose_level)) {
		if (f_v) {
			T.D2->print();
		}
		return false;
	}
	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_second_system "
				"after tdo_columns_setup_second_system_eqns_joining" << endl;
	}


	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_second_system "
				"before tdo_columns_setup_second_system_eqns_counting" << endl;
	}
	tdo_columns_setup_second_system_eqns_counting(
			T,
			line_types, nb_line_types,
			nb_eqns_joining /* eqn_start */,
			verbose_level);
	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_second_system "
				"after tdo_columns_setup_second_system_eqns_counting" << endl;
	}

	if (Tdo_scheme_synthetic->Descr->f_use_packing_numbers) {
		if (f_v) {
			cout << "tdo_refine_cols::tdo_columns_setup_second_system "
					"before tdo_columns_setup_second_system_eqns_upper_bound" << endl;
		}
		if (!tdo_columns_setup_second_system_eqns_upper_bound(
				T,
				line_types, nb_line_types,
				nb_eqns_joining + nb_eqns_counting /* eqn_start */,
				nb_eqns_used,
				verbose_level)) {
			if (f_v) {
				T.D2->print();
			}
			return false;
		}
		if (f_v) {
			cout << "tdo_refine_cols::tdo_columns_setup_second_system "
					"after tdo_columns_setup_second_system_eqns_upper_bound" << endl;
		}
	}






	T.D2->eliminate_zero_rows_quick(verbose_level);


	if (f_vv) {
		cout << "tdo_refine_cols::tdo_columns_setup_second_system, "
				"The second system is" << endl;
		T.D2->print();
	}
	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_second_system done" << endl;
	}
	return true;
}

int tdo_refine_cols::tdo_columns_setup_second_system_eqns_joining(
		tdo_data &T,
		//other::data_structures::partitionstack *Row_split,
		int *line_types, int nb_line_types,
		int eqn_start,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_second_system_eqns_joining" << endl;
	}
	Tdo_scheme_synthetic->check_init();

	int l2, L1, L2, i, r, f, l, j, c;
	int J, I, I1, I2, a, b, ab, a2, k, h, rr, p, u;
	other_combinatorics::combinatorics_domain Combi;

	l2 = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];
	column_refinement_L1_L2(
			Tdo_scheme_synthetic->Descr->f_omit1,
			Tdo_scheme_synthetic->Descr->omit1,
			L1, L2,
			verbose_level);

	for (I = 0; I < L2; I++) {

		string label;

		label = "J_{" + std::to_string(I + 1) + "}";
		T.D2->init_eqn_label(eqn_start + I, label);
	}
	for (I1 = 0; I1 < L2; I1++) {
		for (I2 = I1 + 1; I2 < L2; I2++) {
			k = Combi.ij2k(I1, I2, L2);

			string label;
			label = "J_{" + std::to_string(I1 + 1) + "," + std::to_string(I2 + 1) + "}";
			T.D2->init_eqn_label(eqn_start + L2 + k, label);
		}
	}
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			for (I = 0; I < L2; I++) {
				a = line_types[c * L2 + I];
				a2 = Combi.binomial2(a);
				T.D2->Aij(eqn_start + I, J) = a2;
			}
			for (I1 = 0; I1 < L2; I1++) {
				for (I2 = I1 + 1; I2 < L2; I2++) {
					k = Combi.ij2k(I1, I2, L2);
					a = line_types[c * L2 + I1];
					b = line_types[c * L2 + I2];
					ab = a * b;
					T.D2->Aij(eqn_start + L2 + k, J) = ab;
				}
			}
		}
	}

	// prepare RHS:

	for (I = 0; I < L2; I++) {
		a = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][I];
		a2 = Combi.binomial2(a);
		T.D2->RHS[eqn_start + I] = a2;
	}
	for (I1 = 0; I1 < L2; I1++) {
		a = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][I1];
		for (I2 = I1 + 1; I2 < L2; I2++) {
			b = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][I2];
			k = Combi.ij2k(I1, I2, L2);
			T.D2->RHS[eqn_start + l2 + k] = a * b;
		}
	}

	// now subtract the contribution from one-type blocks:
	for (h = 0; h < T.nb_only_one_type; h++) {
		rr = T.only_one_type[h];
		p = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][rr];
		u = T.types_first[rr];
		for (I = 0; I < L2; I++) {
			a = line_types[u * L2 + I];
			a2 = Combi.binomial2(a);
			T.D2->RHS[eqn_start + I] -= a2 * p;
			if (T.D2->RHS[eqn_start + I] < 0) {
				if (f_v) {
					cout << "tdo_columns_setup_second_system_eqns_"
							"joining: RHS is negative, no solution for "
							"the distribution" << endl;
				}
				return false;
			}
		}
		for (I1 = 0; I1 < L2; I1++) {
			a = line_types[u * L2 + I1];
			for (I2 = I1 + 1; I2 < L2; I2++) {
				b = line_types[u * L2 + I2];
				k = Combi.ij2k(I1, I2, L2);
				ab = a * b * p;
				T.D2->RHS[eqn_start + L2 + k] -= ab;
				if (T.D2->RHS[eqn_start + L2 + k] < 0) {
					if (f_v) {
						cout << "tdo_columns_setup_second_system_eqns_"
								"joining: RHS is negative, no solution for "
								"the distribution" << endl;
					}
					return false;
				}
			}
		}
	}
	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_second_system_eqns_joining done" << endl;
	}
	return true;
}

void tdo_refine_cols::tdo_columns_setup_second_system_eqns_counting(
		tdo_data &T,
		//other::data_structures::partitionstack *Row_split,
		int *line_types, int nb_line_types,
		int eqn_start,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_second_system_eqns_counting" << endl;
	}
	Tdo_scheme_synthetic->check_init();

	int L1, L2, i, r, f, l, j, c, J, I, a, b, S, s;
	//l2 = nb_row_classes[ROW];
	column_refinement_L1_L2(
			Tdo_scheme_synthetic->Descr->f_omit1,
			Tdo_scheme_synthetic->Descr->omit1,
			L1, L2,
			verbose_level);

	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			for (I = 0; I < L2; I++) {
				string label;

				label = "F_{" + std::to_string(r + 1) + "," + std::to_string(I + 1) + "}";
				T.D2->init_eqn_label(eqn_start + i * (L2 + 1) + I, label);
			}
		}

		string label;

		label = "F_{" + std::to_string(r + 1) + "}";
		T.D2->init_eqn_label(eqn_start + i * (L2 + 1) + L2, label);
	}

	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			for (I = 0; I < L2; I++) {
				a = line_types[c * L2 + I];
				T.D2->Aij(eqn_start + i * (L2 + 1) + I, J) = a;
			}
			T.D2->Aij(eqn_start + i * (L2 + 1) + L2, J) = 1;
		}
	}

	// set upper bound x_max:

	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		s = Tdo_scheme_synthetic->col_classes_len[ROW_SCHEME][r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			T.D2->x_min[J] = 0;
			T.D2->x_max[J] = s;
			if (f_v) {
				cout << "tdo_refine_cols::tdo_columns_setup_second_system_eqns_counting x_max[" << J << "] = " << s << endl;
			}
		}
	}

	// prepare RHS:

	S = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		for (I = 0; I < L2; I++) {
			a = Tdo_scheme_synthetic->the_row_scheme[I * Tdo_scheme_synthetic->nb_col_classes[ROW_SCHEME] + r];
			b = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][I];
			T.D2->RHS[eqn_start + i * (L2 + 1) + I] = a * b;
		}
		s = Tdo_scheme_synthetic->col_classes_len[ROW_SCHEME][r];
		T.D2->RHS[eqn_start + i * (L2 + 1) + L2] = s;
		S += s;
	}

	T.D2->f_has_sum = true;
	T.D2->sum = S;
	//T.D2->f_x_max = true;
	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_second_system_eqns_counting done" << endl;
	}
}

int tdo_refine_cols::tdo_columns_setup_second_system_eqns_upper_bound(
		tdo_data &T,
		//other::data_structures::partitionstack *Row_split,
		int *line_types, int nb_line_types,
		int eqn_start, int &nb_eqns_used,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_second_system_eqns_upper_bound" << endl;
	}
	Tdo_scheme_synthetic->check_init();

	int nb_eqns_packing;
	int L1, L2, i, r, f, l, j, c, J, I;
	int k, h, rr, p, u, a, len, f_used;
	geometry::other_geometry::geometry_global Gg;

	nb_eqns_packing = 0;
	//l2 = nb_row_classes[ROW];
	column_refinement_L1_L2(
			Tdo_scheme_synthetic->Descr->f_omit1,
			Tdo_scheme_synthetic->Descr->omit1,
			L1, L2,
			verbose_level);
	for (I = 0; I < L2; I++) {
		len = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][I];
		if (len <= 2) {
			continue;
		}
		for (k = 3; k <= len; k++) {
			f_used = false;
			for (i = 0; i < T.nb_multiple_types; i++) {
				r = T.multiple_types[i];
				f = T.types_first[r];
				l = T.types_len[r];
				for (j = 0; j < l; j++) {
					c = f + j;
					J = T.types_first2[i] + j;
					a = line_types[c * L2 + I];
					if (a < k) {
						continue;
					}
					f_used = true;
					T.D2->Aij(eqn_start + nb_eqns_packing, J) = 1;
				}
			} // next i
			if (f_used) {
				int bound;

				bound = Gg.TDO_upper_bound(len, k);
				T.D2->RHS[eqn_start + nb_eqns_packing] = bound;
				T.D2->type[eqn_start + nb_eqns_packing] = t_LE;
				for (h = 0; h < T.nb_only_one_type; h++) {
					rr = T.only_one_type[h];
					p = Tdo_scheme_synthetic->col_classes_len[ROW_SCHEME][rr];
					u = T.types_first[rr];
					a = line_types[u * L2 + I];
					if (a < k) {
						continue;
					}
					T.D2->RHS[eqn_start + nb_eqns_packing] -= p;
					if (T.D2->RHS[eqn_start + nb_eqns_packing] < 0) {
						if (f_v) {
							cout << "tdo_refine_cols::tdo_columns_setup_second_system_eqns_upper_bound "
									"RHS < 0" << endl;
						}
						return false;
					}
				}
				string label;

				label = "P_{" + std::to_string(I + 1) + "," + std::to_string(k) + "} \\,\\mbox{using}\\, "
						"P(" + std::to_string(len) + "," + std::to_string(k) + ")=" + std::to_string(bound);
				T.D2->init_eqn_label(eqn_start + nb_eqns_packing, label);
				nb_eqns_packing++;
			}
		} // next k
	}
	nb_eqns_used = nb_eqns_packing;
	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_second_system_eqns_upper_bound "
				"nb_eqns_used = " << nb_eqns_used << endl;
	}
	if (f_v) {
		cout << "tdo_refine_cols::tdo_columns_setup_second_system_eqns_upper_bound done" << endl;
	}
	return true;
}





}}}}


