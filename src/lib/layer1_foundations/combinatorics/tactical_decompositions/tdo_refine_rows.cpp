/*
 * tdo_refine_rows.cpp
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


tdo_refine_rows::tdo_refine_rows()
{
	Record_birth();

	Tdo_scheme_synthetic = NULL;
	Col_split = NULL;

}

tdo_refine_rows::~tdo_refine_rows()
{
	Record_death();

	if (Col_split) {
		FREE_OBJECT(Col_split);
	}
}

void tdo_refine_rows::init(
		tdo_scheme_synthetic *Tdo_scheme_synthetic,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "tdo_refine_rows::init" << endl;
	}

	tdo_refine_rows::Tdo_scheme_synthetic = Tdo_scheme_synthetic;

	if (f_v) {
		cout << "tdo_refine_rows::init done" << endl;
	}
}



int tdo_refine_rows::refine_rows(
		tdo_refinement_output *&Output,
		int &cnt_second_system,
		int verbose_level)
// called from tdo_refinement::do_row_refinement
// Even if the function returns false, Output must be deallocated.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "tdo_refine_rows::refine_rows" << endl;
	}
	Tdo_scheme_synthetic->check_init();

	int l1, l2, R;

	//other::data_structures::partitionstack *Col_split;




	if (f_vv) {
		cout << "f_omit1=" << Tdo_scheme_synthetic->Descr->f_omit1 << " omit1=" << Tdo_scheme_synthetic->Descr->omit1 << endl;
		cout << "f_omit2=" << Tdo_scheme_synthetic->Descr->f_omit2 << " omit2=" << Tdo_scheme_synthetic->Descr->omit2 << endl;
		cout << "f_use_packing_numbers=" << Tdo_scheme_synthetic->Descr->f_use_packing_numbers << endl;
		cout << "f_dual_is_linear_space=" << Tdo_scheme_synthetic->Descr->f_dual_is_linear_space << endl;
		cout << "f_use_mckay=" << Tdo_scheme_synthetic->Descr->f_use_mckay_solver << endl;
	}

	if (Tdo_scheme_synthetic->row_level >= 2) {

		R = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];
		l1 = Tdo_scheme_synthetic->nb_col_classes[ROW_SCHEME];
		l2 = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];
		if (f_vv) {
			cout << "l1=" << l1 << " at level " << Tdo_scheme_synthetic->level[ROW_SCHEME] << endl;
			cout << "l2=" << l2 << " at level " << Tdo_scheme_synthetic->level[COL_SCHEME] << endl;
			cout << "R=" << R << endl;
		}



		// prepare the column split partition:


		Col_split = Tdo_scheme_synthetic->get_column_split_partition(0 /*verbose_level*/);


		if (f_vv) {
			cout << "column split partition: " << endl;
			Col_split->print(cout);
			cout << endl;
		}
		if (Col_split->ht != l1) {
			cout << "Col_split->ht != l1" << endl;
			exit(1);
		}
		if ((R == 1) && (l1 == 1) && (Tdo_scheme_synthetic->the_row_scheme[0] == -1)) {
			if (f_v) {
				cout << "tdo_refine_rows::refine_rows "
						"before refine_rows_easy" << endl;
			}
			if (!refine_rows_easy(
					Output,
					cnt_second_system,
					verbose_level - 1)) {

				if (f_v) {
					cout << "tdo_refine_rows::refine_rows "
							"refine_rows_easy returns false" << endl;
				}

				return false;
			}
			if (f_v) {
				cout << "tdo_refine_rows::refine_rows "
						"after refine_rows_easy" << endl;
			}
		}
		else {
			if (f_v) {
				cout << "tdo_refine_rows::refine_rows "
						"before refine_rows_hard" << endl;
			}
			if (!refine_rows_hard(
					Output,
					cnt_second_system,
					verbose_level - 1)) {

				if (f_v) {
					cout << "tdo_refine_rows::refine_rows "
							"refine_rows_hard returns false" << endl;
				}

				return false;
			}
			if (f_v) {
				cout << "tdo_refine_rows::refine_rows "
						"after refine_rows_hard" << endl;
			}
		}
	}
	else {
		if (f_v) {
			cout << "tdo_refine_rows::refine_rows "
					"before refine_rows_easy" << endl;
		}
		if (!refine_rows_easy(
				Output,
				cnt_second_system,
				verbose_level - 1)) {

			if (f_v) {
				cout << "tdo_refine_rows::refine_rows "
						"refine_rows_easy returns false" << endl;
			}
			return false;
		}
		if (f_v) {
			cout << "tdo_refine_rows::refine_rows "
					"after refine_rows_easy" << endl;
		}
	}

	if (Tdo_scheme_synthetic->Descr->f_do_the_geometric_test) {
		if (f_v) {
			cout << "tdo_refine_rows::refine_rows "
					"before geometric_test_for_row_scheme" << endl;
		}

		geometric_test_for_row_scheme(
				Output,
				Tdo_scheme_synthetic->Descr->f_omit1, Tdo_scheme_synthetic->Descr->omit1,
			verbose_level);

		if (f_v) {
			cout << "tdo_refine_rows::refine_rows "
					"after geometric_test_for_row_scheme" << endl;
		}
	}

	//FREE_OBJECT(Col_split);

	if (f_v) {
		cout << "tdo_refine_rows::refine_rows done" << endl;
	}
	return true;
}

int tdo_refine_rows::refine_rows_easy(
		tdo_refinement_output *&Output,
		int &cnt_second_system, int verbose_level)
// uses two temporary solvers::diophant objects
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_easy" << endl;
	}
	other_combinatorics::combinatorics_domain Combi;
	geometry::other_geometry::geometry_global Gg;

	int nb_rows;
	int i, j, J, S, l2, nb_eqns, nb_vars;
	int nb_eqns_joining, nb_eqns_upper_bound;
	int nb_sol, len, k, a2, a, b, ab;
	int f_used, j1, j2, len1, len2, cnt;
	int Nb_eqns, Nb_vars;



	Output = NEW_OBJECT(tdo_refinement_output);




	// Step 1: the point types

	// we only count the number of point types:




	int *point_types;
	int nb_point_types, point_type_len;

	l2 = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];

	//partitionstack &P = PB.P;
	nb_rows = Tdo_scheme_synthetic->Partition_refinement->startCell[1];
	S = nb_rows - 1;
	if (f_v) {
		cout << "nb_rows=" << nb_rows << endl;
	}

	nb_vars = l2 + 1; // 1 slack variable
	nb_eqns = 1;

	solvers::diophant D;

	D.open(nb_eqns, nb_vars, verbose_level - 1);

	// 1st equation: connections within the same row-partition
	for (J = 0; J < nb_vars; J++) {
		D.Aij(0, J) = Combi.minus_one_if_positive(
				Tdo_scheme_synthetic->the_col_scheme[0 * l2 + J]);
	}
	D.Aij(0, nb_vars - 1) = 0;
	if (f_vv) {
		cout << "nb_rows=" << nb_rows << endl;
	}
	D.RHS[0] = nb_rows - 1;
	if (f_vv) {
		cout << "RHS[0]=" << D.RHS[0] << endl;
	}

	for (j = 0; j < l2; j++) {
		D.x_min[j] = 0;
		D.x_max[j] = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][j];
	}
	D.x_min[nb_vars - 1] = 0;
	D.x_max[nb_vars - 1] = nb_rows - 1;

	//D.f_x_max = true;


	D.eliminate_zero_rows_quick(verbose_level);
	D.sum = S;
	if (f_vv) {
		cout << "tdo_refine_rows::refine_rows_easy "
				"The first system is" << endl;
		D.print();
	}
	if (f_vv) {
		string label;

		label = "first";
		D.write_xml(cout, label);
	}

	nb_sol = 0;
	point_type_len = nb_vars - 1;

	if (D.solve_first(verbose_level - 2)) {

		while (true) {
			if (f_vv) {
				cout << nb_sol << " : ";
				for (i = 0; i < nb_vars; i++) {
					cout << " " << D.x[i];
				}
				cout << endl;
			}
			nb_sol++;
			if (!D.solve_next()) {
				break;
			}
		}
	}
	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_easy "
				"found " << nb_sol << " point types" << endl;
	}
	if (nb_sol == 0) {
		return false;
	}
	nb_point_types = nb_sol;






	// Step 2, the distributions



	// Build the system of equations:



	nb_eqns_upper_bound = 0;
	for (j = 0; j < l2; j++) {
		len = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][j];
		if (len > 2) {
			nb_eqns_upper_bound += len - 2;
		}
	}
	nb_eqns_joining = l2 + ((l2 * (l2 - 1)) >> 1);


	Nb_eqns = l2 + nb_eqns_joining + nb_eqns_upper_bound;
	Nb_vars = nb_sol;

	solvers::diophant D2;

	D2.open(Nb_eqns, Nb_vars, verbose_level - 1);


	// we compute and store the actual point types:

	point_types = NEW_int(nb_point_types * point_type_len);



	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_easy: opening second "
			<< cnt_second_system << " system with "
			<< Nb_eqns << " equations and " << Nb_vars
			<< " variables" << endl;
	}


	// set up the counting-flags equations:
	// The labels will be set later


	nb_sol = 0;
	if (D.solve_first(verbose_level - 2)) {

		while (true) {
			if (f_vv) {
				cout << nb_sol << " : ";
				for (i = 0; i < nb_vars; i++) {
					cout << " " << D.x[i];
				}
				cout << endl;
			}
			for (i = 0; i < point_type_len; i++) {
				D2.Aij(i, nb_sol) = D.x[i];
				point_types[nb_sol * point_type_len + i] = D.x[i];
			}
			nb_sol++;
			if (!D.solve_next()) {
				break;
			}
		}
	}



	Output->types = point_types;
	Output->nb_types = nb_point_types;
	Output->type_len = point_type_len;



	// prepare the equations (actually, inequalities) for joining:


	// joining within a column block:

	for (j = 0; j < l2; j++) {
		len = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][j];
		for (i = 0; i < Nb_vars; i++) {
			a = point_types[i * point_type_len + j];
			a2 = Combi.binomial2(a);
			D2.Aij(l2 + j, i) = a2;
		}
		D2.RHS[l2 + j] = Combi.binomial2(len);
		D2.type[l2 + j] = t_LE;

		string label;
		label = "J_{" + std::to_string(j + 1) + "}";
		D2.init_eqn_label(l2 + j, label);
	}


	// joining between two column blocks:

	cnt = 0;
	for (j1 = 0; j1 < l2; j1++) {
		len1 = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][j1];
		for (j2 = j1 + 1; j2 < l2; j2++) {
			len2 = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][j2];
			for (i = 0; i < Nb_vars; i++) {
				a = point_types[i * point_type_len + j1];
				b = point_types[i * point_type_len + j2];
				ab = a * b;
				D2.Aij(l2 + l2 + cnt, i) = ab;
			}
			D2.RHS[l2 + l2 + cnt] = len1 * len2;
			D2.type[l2 + l2 + cnt] = t_LE;

			string label;

			label = "J_{" + std::to_string(j1 + 1) + "," + std::to_string(j2 + 1) + "}";
			D2.init_eqn_label(l2 + l2 + cnt, label);
			cnt++;
		}
	}

	// add upper bounds resulting from Gg.TDO_upper_bound:

	nb_eqns_upper_bound = 0;
	for (j = 0; j < l2; j++) {
		len = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][j];
		for (k = 3; k <= len; k++) {
			for (i = 0; i < Nb_vars; i++) {
				D2.Aij(l2 + nb_eqns_joining + nb_eqns_upper_bound, i) = 0;
			}
			f_used = false;
			for (i = 0; i < Nb_vars; i++) {
				a = point_types[i * point_type_len + j];
				if (a < k) {
					continue;
				}
				D2.Aij(l2 + nb_eqns_joining + nb_eqns_upper_bound, i) = 1;
				f_used = true;
			}
			if (f_used) {

				int bound = Gg.TDO_upper_bound(len, k);

				D2.RHS[l2 + nb_eqns_joining + nb_eqns_upper_bound] = bound;
				D2.type[l2 + nb_eqns_joining + nb_eqns_upper_bound] = t_LE;

				string label;


				label = "P_{" + std::to_string(j + 1) + "," + std::to_string(k) + "} \\,\\mbox{using}\\, "
						"P(" + std::to_string(len) + "," + std::to_string(k) + ")=" + std::to_string(bound);
				D2.init_eqn_label(
						l2 +
						nb_eqns_joining + nb_eqns_upper_bound, label);
				nb_eqns_upper_bound++;
			}
		} // next k
	} // next j


	Nb_eqns = l2 + nb_eqns_joining + nb_eqns_upper_bound;
	D2.m = Nb_eqns;


	// Now, the second system has been established



	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_easy "
				"second system " << cnt_second_system << " found "
			<< nb_sol << " point types" << endl;
	}
	cnt_second_system++;

	// check if the number of point types is zero:

	if (nb_sol == 0) {
		//FREE_int(point_types); // point_types is now in Output
		return false;
	}

	D2.sum = nb_rows;

	//label the counting-flags equations:

	for (i = 0; i < l2; i++) {
		D2.RHS[i] = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][i] * Tdo_scheme_synthetic->the_col_scheme[i];

		string label;

		label = "F_{" + std::to_string(i + 1) + "}";
		D2.init_eqn_label(i, label);
	}


	D2.eliminate_zero_rows_quick(verbose_level);

	if (f_vv) {
		cout << "The second system is" << endl;
		D2.print();
	}
	if (f_vv) {
		string label;

		label = "second";
		D2.write_xml(cout, label);
	}


	// we will solve it twice.
	// The first time we just count the number of solutions:


	nb_sol = 0;
	if (D2.solve_first(verbose_level - 2)) {
		while (true) {
			if (f_vv) {
				cout << nb_sol << " : ";
				for (i = 0; i < Nb_vars; i++) {
					cout << " " << D2.x[i];
				}
				cout << endl;
			}
			nb_sol++;
			if (!D2.solve_next()) {
				break;
			}
		}
	}

	int *distributions;
	int nb_distributions;


	// allocate data for the solutions:

	nb_distributions = nb_sol;
	distributions = NEW_int(nb_distributions * nb_point_types);


	// solve it again and store the solutions:

	nb_sol = 0;
	if (D2.solve_first(verbose_level - 2)) {
		while (true) {
			if (f_vv) {
				cout << nb_sol << " : ";
				for (i = 0; i < Nb_vars; i++) {
					cout << " " << D2.x[i];
				}
				cout << endl;
			}
			for (i = 0; i < Nb_vars; i++) {
				distributions[nb_sol * nb_point_types + i] = D2.x[i];
			}
			nb_sol++;
			if (!D2.solve_next()) {
				break;
			}
		}
	}

	// finished


	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_easy "
				"found " << nb_distributions
			<< " point type distributions." << endl;
	}


	Output->distributions = distributions;
	Output->nb_distributions = nb_distributions;


	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_easy done" << endl;
	}

	return true;
}



int tdo_refine_rows::refine_rows_hard(
		tdo_refinement_output *&Output,
		int &cnt_second_system,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_hard" << endl;
	}
	Tdo_scheme_synthetic->check_init();


	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, r, R, l1, /*l2,*/ L1, L2;
	int nb_sol;
	int h, u;
	tdo_data T;

	if (f_vv) {
		if (Tdo_scheme_synthetic->Descr->f_omit1) {
			cout << "omitting the last " << Tdo_scheme_synthetic->Descr->omit1
				<< " column blocks from the previous row-scheme" << endl;
		}
		if (Tdo_scheme_synthetic->Descr->f_omit2) {
			cout << "omitting the last " << Tdo_scheme_synthetic->Descr->omit2 << " row blocks" << endl;
		}
		cout << "f_use_packing_numbers=" << Tdo_scheme_synthetic->Descr->f_use_packing_numbers << endl;
		cout << "f_dual_is_linear_space=" << Tdo_scheme_synthetic->Descr->f_dual_is_linear_space << endl;
		cout << "f_use_mckay_solver=" << Tdo_scheme_synthetic->Descr->f_use_mckay_solver << endl;
	}
	R = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];
	l1 = Tdo_scheme_synthetic->nb_col_classes[ROW_SCHEME];
	//l2 = nb_col_classes[COL];

	if (f_vv) {
		cout << "tdo_refine_rows::refine_rows_hard the_row_scheme is:" << endl;
		int i, j;
		for (i = 0; i < R; i++) {
			for (j = 0; j < l1; j++) {
				cout << setw(4) << Tdo_scheme_synthetic->the_row_scheme[i * l1 + j];
			}
			cout << endl;
		}
	}

	row_refinement_L1_L2(
			Tdo_scheme_synthetic->Descr->f_omit1,
			Tdo_scheme_synthetic->Descr->omit1,
			L1, L2,
			verbose_level);

	T.allocate(R);

	T.types_first[0] = 0;



	int *point_types;
	int nb_point_types;
	int point_type_len;
	int *distributions;
	int nb_distributions;
	int point_types_allocated;




	point_types_allocated = 100;
	nb_point_types = 0;
	point_type_len = L2 + L1; // + slack variables
	point_types = NEW_int(point_types_allocated * point_type_len);
		// detected and corrected an error: Dec 6 2010
		// it was allocated to point_types_allocated * L2
		// which is not enough

		// when we are done, it is [point_types_allocated * L2]


	T.nb_only_one_type = 0;
	T.nb_multiple_types = 0;


	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_hard "
				"computing refined point types:" << endl;
		cout << "point_type_len = " << point_type_len << endl;
		cout << "L1 = " << L1 << endl;
		cout << "L2 = " << L2 << endl;
	}


	for (r = 0; r < R; r++) {

		if (f_v) {
			cout << "tdo_refine_rows::refine_rows_hard "
					"r=" << r << " / " << R << endl;
			cout << "T.types_first[r]=" << T.types_first[r] << endl;
		}

		tdo_rows_setup_first_system(
				T, r,
				point_types, nb_point_types,
				verbose_level - 1);

		if (f_vv) {
			cout << "tdo_refine_rows::refine_rows_hard "
					"r=" << r << " / " << R << " the system is:" << endl;
			T.D1->print();
		}

		if (f_vvv) {
			string label;

			label = "first_" + std::to_string(r);
			T.D1->write_xml(cout, label);
		}

		nb_sol = T.solve_first_system(
			point_types, nb_point_types, point_types_allocated,
			verbose_level - 1);

		if (f_v) {
			cout << "tdo_refine_rows::refine_rows_hard "
					"r = " << r << ", found " << nb_sol
					<< " refined point types" << endl;
		}
		if (f_vv) {
			cout << "tdo_refine_rows::refine_rows_hard "
					"r = " << r << ", found " << nb_sol
					<< " refined point types:" << endl;
			Int_vec_print_integer_matrix_width(
					cout,
				point_types + T.types_first[r] * point_type_len,
				nb_sol, point_type_len, point_type_len, 3);
		}

#if 0
		// MARUTA  Begin
		if (r == 1) {
			int h, a;

			for (h = nb_sol - 1; h >= 0; h--) {
				a = (point_types + (T.types_first[r] + h)
						* point_type_len)[0];
				if (a == 0) {
					cout << "removing last solution" << endl;
					nb_sol--;
					nb_point_types--;
					}
				}
			}
		// MARUTA   End
#endif

#if 0

		if (f_vv) {
			cout << "tdo_refine_rows::refine_rows_hard "
					"r = " << r << ", found " << nb_sol
					<< " refined point types:" << endl;
			Int_vec_print_integer_matrix_width(
					cout,
				point_types + T.types_first[r] * point_type_len,
				nb_sol, point_type_len, point_type_len, 3);
		}
#endif

		if (nb_sol == 0) {
			FREE_int(point_types);
			if (f_v) {
				cout << "tdo_refine_rows::refine_rows_hard "
						"no solution for this point type, we are done" << endl;
			}
			return false;
		}

		T.types_len[r] = nb_sol;
		T.types_first[r + 1] = T.types_first[r] + nb_sol;

		if (nb_sol == 1) {
			if (f_v) {
				cout << "tdo_refine_rows::refine_rows_hard "
						"only one solution in block r=" << r << endl;
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
		cout << "tdo_refine_rows::refine_rows_hard "
				"computing refined point types done" << endl;
		cout << "r : types_first[r] : types_len[r]" << endl;
		for (r = 0; r < R; r++) {
			cout << r << " : " << T.types_first[r] << " : " << T.types_len[r] << endl;
		}
	}



	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_hard "
				"eliminating slack variables" << endl;
	}
	// eliminate the slack variables from point_types:
	for (r = 0; r < nb_point_types; r++) {
		int f, l, a, j, J;

		for (i = 0; i < L1; i++) {
			f = Col_split->startCell[i];
			l = Col_split->cellSize[i];
			for (j = 0; j < l; j++) {
				J = f + i + j;
				a = point_types[r * point_type_len + J];
				point_types[r * L2 + f + j] = a;
			}
		}
	}
	point_type_len = L2;
	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_hard "
				"altogether, we found " << nb_point_types
				<< " refined point types" << endl;
	}
	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_hard "
				"after slack variables have been eliminated:" << endl;
	}
	if (f_vv) {
		cout << "tdo_refine_rows::refine_rows_hard "
				"altogether, we found " << nb_point_types
				<< " refined point types:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, point_types,
			nb_point_types, point_type_len, point_type_len, 3);
	}


	// now we compute the distributions:

	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_hard "
				"before tdo_rows_setup_second_system" << endl;
	}

	if (!tdo_rows_setup_second_system(
			T,
			point_types, nb_point_types,
			verbose_level)) {

		if (f_v) {
			cout << "tdo_refine_rows::refine_rows_hard "
					"tdo_rows_setup_second_system returns false, we are done" << endl;
		}

		FREE_int(point_types);
		return false;
	}
	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_hard "
				"after tdo_rows_setup_second_system" << endl;
	}
	if (f_vv) {
		string label;

		label = "second";
		T.D2->write_xml(cout, label);
	}


	if (T.D2->n == 0) {
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

#if 0
		// the output is here:
		int *point_types;
		int nb_point_types;
		int point_type_len;
		int *distributions;
		int nb_distributions;
		int point_types_allocated;
#endif

		Output = NEW_OBJECT(tdo_refinement_output);

		Output->types = point_types;
		Output->nb_types = nb_point_types;
		Output->type_len = point_type_len;
		Output->types_allocated = point_types_allocated;
		Output->distributions = distributions;
		Output->nb_distributions = nb_distributions;


		if (f_v) {
			cout << "tdo_refine_rows::refine_rows_hard done" << endl;
		}
		return true;
	}


#if 0
	if (cnt_second_system == 1) {
		int j;
		int x[] = {4,1,5,0,2,0,7,2,4,0,0,0,1,0,0,4,0,0};
		cout << "testing solution:" << endl;
		int_vec_print(cout, x, 18);
		cout << endl;
		if (T.D2->n != 18) {
			cout << "T.D2->n != 18" << endl;
			}
		for (j = 0; j < 18; j++) {
			T.D2->x[j] = x[j];
			}
		T.D2->multiply_A_x_to_RHS1();
		for (i = 0; i < T.D2->m; i++) {
			cout << i << " : " << T.D2->RHS1[i] << " : "
					<< T.D2->RHS[i] - T.D2->RHS1[i] << endl;
			}
		}
#endif

	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_hard "
				"solving second system "
				<< cnt_second_system << " which is " << T.D2->m
				<< " x " << T.D2->n << endl;
		cout << T.nb_multiple_types << " variable blocks:" << endl;
		int f, l;
		for (i = 0; i < T.nb_multiple_types; i++) {
			r = T.multiple_types[i];
			f = T.types_first2[i];
			l = T.types_len[r];
			cout << i << " : " << r << " : " << setw(3)
				<< Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][r] << " : " << setw(3) << f
				<< " : " << setw(3) << l << endl;
		}
	}


	// next we solve the second system:


	if (Tdo_scheme_synthetic->Descr->f_omit2) {
		if (f_v) {
			cout << "tdo_refine_rows::refine_rows_hard "
					"before T.solve_second_system_omit" << endl;
		}
		T.solve_second_system_omit(
				Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME],
			point_types, nb_point_types,
			distributions, nb_distributions,
			Tdo_scheme_synthetic->Descr->omit2,
			verbose_level - 1);
	}
	else {
		int f_scale = false;
		int scaling = 0;
		if (f_v) {
			cout << "tdo_refine_rows::refine_rows_hard "
					"before T.solve_second_system" << endl;
		}
		T.solve_second_system(
				Tdo_scheme_synthetic->Descr->f_use_mckay_solver,
				Tdo_scheme_synthetic->Descr->f_once,
				Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME],
				f_scale, scaling,
				point_types, nb_point_types,
				distributions, nb_distributions,
				verbose_level - 1);
	}



	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_hard second system "
			<< cnt_second_system
			<< " found " << nb_distributions
			<< " distributions." << endl;
	}
	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_hard "
				"number of point types total: " << nb_point_types << endl;
		cout << "tdo_scheme_synthetic::refine_rows_hard "
				"number of distributions: " << nb_distributions << endl;
	}

	if (f_vv) {
		cout << "tdo_refine_rows::refine_rows_hard "
				"refined point types:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, point_types,
				nb_point_types, point_type_len, point_type_len, 3);
	}

	if (f_vv) {
		cout << "tdo_refine_rows::refine_rows_hard "
				"distributions:" << endl;
		Int_vec_print_integer_matrix_width(
				cout, distributions,
				nb_distributions, nb_point_types, nb_point_types, 3);
	}


	cnt_second_system++;


	Output = NEW_OBJECT(tdo_refinement_output);

	Output->types = point_types;
	Output->nb_types = nb_point_types;
	Output->type_len = point_type_len;
	Output->types_allocated = point_types_allocated;
	Output->distributions = distributions;
	Output->nb_distributions = nb_distributions;




	if (f_v) {
		cout << "tdo_refine_rows::refine_rows_hard done" << endl;
	}
	return true;
}

void tdo_refine_rows::row_refinement_L1_L2(
		int f_omit, int omit,
		int &L1, int &L2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int l1, l2, omit2, i;

	l1 = Tdo_scheme_synthetic->nb_col_classes[ROW_SCHEME];
	l2 = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];

	omit2 = 0;
	if (f_omit) {
		for (i = l1 - omit; i < l1; i++) {
			omit2 += Col_split->cellSize[i];
		}
	}
	else {
		omit = 0;
	}
	L1 = l1 - omit;
	L2 = l2 - omit2;
	if (f_v) {
		cout << "tdo_refine_rows::row_refinement_L1_L2 "
				"l1 = " << l1 << " l2=" << l2
			<< " L1=" << L1 << " L2=" << L2 << endl;
	}
}

int tdo_refine_rows::tdo_rows_setup_first_system(
		tdo_data &T, int r,
		int *&point_types, int &nb_point_types,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "tdo_refine_rows::tdo_rows_setup_first_system r=" << r << endl;
	}
	Tdo_scheme_synthetic->check_init();

	int S, s_default, s_or_s_default, R, l1, l2, L1, L2;
	int J, r2, i, j, s, f, l;
	int nb_vars, nb_eqns;
	int omit_local;
	other_combinatorics::combinatorics_domain Combi;

	if (!Tdo_scheme_synthetic->Descr->f_omit1) {
		omit_local = 0;
	}
	else {
		omit_local = Tdo_scheme_synthetic->Descr->omit1;
	}

	if (f_v) {
		if (!Tdo_scheme_synthetic->Descr->f_omit1) {
			cout << "omit=" << omit_local << endl;
		}
	}
	R = Tdo_scheme_synthetic->nb_row_classes[ROW_SCHEME];
	l1 = Tdo_scheme_synthetic->nb_col_classes[ROW_SCHEME];
	l2 = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];

	row_refinement_L1_L2(Tdo_scheme_synthetic->Descr->f_omit1, omit_local, L1, L2, verbose_level);

	nb_vars = L2 + L1; // possible up to L1 slack variables
	nb_eqns = R + L1;
	if (f_v) {
		cout << "tdo_refine_rows::tdo_rows_setup_first_system L1=" << L1 << endl;
		cout << "tdo_refine_rows::tdo_rows_setup_first_system L2=" << L2 << endl;
	}

	T.D1->open(nb_eqns, nb_vars, verbose_level - 1);
	T.D1->fill_coefficient_matrix_with(0);
	S = 0;


	// make the ordinary equations based on connections of the point
	// to the other row classes or within the same row class:


	for (r2 = 0; r2 < R; r2++) {

		if (r2 == r) {


			// connections within the same row-partition


			// loop over all column classes, based on the partition
			// w.r.t. to previous, coarser column partition:

			for (i = 0; i < L1; i++) {
				f = Col_split->startCell[i];
				l = Col_split->cellSize[i];
				for (j = 0; j < l; j++) {
					J = f + i + j; // +i for the slack variables
					T.D1->Aij(r2, J) =
						Combi.minus_one_if_positive(
								Tdo_scheme_synthetic->the_col_scheme[r2 * l2 + f + j]);
				}
				T.D1->Aij(r2, f + i + l) = 0;
					// the slack variable is not needed
			}
#if 0
			for (J = 0; J < nb_vars; J++) {
				T.D1->Aij(r2, J) =
					minus_one_if_positive(the_col_scheme[r2 * l2 + J]);
			}
#endif
			T.D1->RHS[r] = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][r] - 1;
				// we must be connected to every point of row class r but one
				// (namely the point itself).
			if (Tdo_scheme_synthetic->Descr->f_omit1) {
				T.D1->type[r] = t_LE;
			}
		}
		else {

			// loop over all column classes, based on the partition
			// w.r.t. to previous, coarser column partition:

			for (i = 0; i < L1; i++) {
				f = Col_split->startCell[i];
				l = Col_split->cellSize[i];
				for (j = 0; j < l; j++) {
					J = f + i + j; // +i for the slack variables
					T.D1->Aij(r2, J) = Tdo_scheme_synthetic->the_col_scheme[r2 * l2 + f + j];
				}
				T.D1->Aij(r2, f + i + l) = 0;
					// the slack variable is not needed
			}
#if 0
			for (J = 0; J < nb_vars; J++) {
				T.D1->Aij(r2, J) = the_col_scheme[r2 * l2 + J];
			}
#endif
			T.D1->RHS[r2] = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][r2];
				// we must be connected to every point of row class r2.
			if (Tdo_scheme_synthetic->Descr->f_omit1) {
				T.D1->type[r2] = t_LE;
			}
		}
	}

	// make the slack equations:

	// loop over all column classes, based on the partition w.r.t.
	// to the previous, coarser column partition:

	// each class of the partition will give one equation.


	for (i = 0; i < L1; i++) {

		// define equation R + i:


		s = Tdo_scheme_synthetic->the_row_scheme[r * l1 + i];
		if (f_v) {
			cout << "tdo_refine_rows::tdo_rows_setup_first_system "
					"r=" << r << " i=" << i << " s=" << s << endl;
		}
		if (s == -1) {
			cout << "tdo_refine_rows::tdo_rows_setup_first_system "
					"row scheme entry " << r << "," << i
				<< " is -1, using slack variable" << endl;
			cout << "using " << Tdo_scheme_synthetic->col_classes_len[ROW_SCHEME][i]
				<< " as upper bound" << endl;
			s_default = Tdo_scheme_synthetic->col_classes_len[ROW_SCHEME][i];
			s_or_s_default = s_default;
		}
		else {
			s_default = 0; // not needed but compiler likes it
			s_or_s_default = s;
		}

		T.D1->RHS[R + i] = s_or_s_default;
		S += s_or_s_default;

		f = Col_split->startCell[i];
		l = Col_split->cellSize[i];
		if (f_v) {
			cout << "tdo_refine_rows::tdo_rows_setup_first_system "
					"r=" << r << " i=" << i << " f=" << f << " l=" << l << endl;
		}

		for (j = 0; j < l; j++) {
			J = f + i + j; // +i for the slack variables
			T.D1->Aij(R + i, J) = 1;
			T.D1->x_min[J] = 0;
			T.D1->x_max[J] = MINIMUM(Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][f + j],
					s_or_s_default);
			if (f_v) {
				cout << "tdo_refine_rows::tdo_rows_setup_first_system "
						"r=" << r << " i=" << i << " f=" << f << " j=" << j
						<< " T.D1->x_max[f + i + j]=" << T.D1->x_max[J] << endl;
			}
		}
		T.D1->Aij(R + i, f + i + l) = 1; // the slack variable
		if (s == -1) {
			T.D1->x_max[f + i + l] = s_default;
		}
		else {
			T.D1->x_max[f + i + l] = 0;
		}
		T.D1->x_min[f + i + l] = 0;
	}
	T.D1->f_has_sum = true;
	T.D1->sum = S;
	//T.D1->f_x_max = true;

	T.D1->eliminate_zero_rows_quick(verbose_level);

	if (f_vv) {
		cout << "tdo_refine_rows::tdo_rows_setup_first_system "
				"The first system for r=" << r << " is:" << endl;
		T.D1->print();
	}
	if (f_v) {
		cout << "tdo_refine_rows::tdo_rows_setup_first_system "
				"r=" << r << " finished" << endl;
	}
	return true;

}

int tdo_refine_rows::tdo_rows_setup_second_system(
		tdo_data &T,
		int *&point_types, int &nb_point_types,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "tdo_refine_rows::tdo_rows_setup_second_system" << endl;
	}
	Tdo_scheme_synthetic->check_init();

	int nb_eqns_joining, nb_eqns_counting, nb_eqns_packing, nb_eqns_used = 0;
	int Nb_vars, Nb_eqns;
	int l2, i, j, len, r, L1, L2;
	other_combinatorics::combinatorics_domain Combi;

	if (f_vv) {
		cout << "f_omit1=" << Tdo_scheme_synthetic->Descr->f_omit1
				<< " omit1=" << Tdo_scheme_synthetic->Descr->omit1 << endl;
		cout << "f_use_packing_numbers=" << Tdo_scheme_synthetic->Descr->f_use_packing_numbers << endl;
		cout << "f_dual_is_linear_space=" << Tdo_scheme_synthetic->Descr->f_dual_is_linear_space << endl;
	}

	l2 = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];

	row_refinement_L1_L2(
			Tdo_scheme_synthetic->Descr->f_omit1,
			Tdo_scheme_synthetic->Descr->omit1,
			L1, L2,
			verbose_level);

	nb_eqns_joining = L2 + Combi.binomial2(L2);
	nb_eqns_counting = T.nb_multiple_types * (L2 + 1);
	nb_eqns_packing = 0;
	if (Tdo_scheme_synthetic->Descr->f_use_packing_numbers) {
		for (j = 0; j < L2; j++) {
			len = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][j];
			if (len > 2) {
				nb_eqns_packing += len - 2;
			}
		}
	}

	Nb_eqns = nb_eqns_joining + nb_eqns_counting + nb_eqns_packing;
	Nb_vars = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		T.types_first2[i] = Nb_vars;
		Nb_vars += T.types_len[r];
	}


	T.D2->open(Nb_eqns, Nb_vars, verbose_level - 1);
	T.D2->fill_coefficient_matrix_with(0);

#if 0
	if (Nb_vars == 0) {
		return true;
	}
#endif

	if (f_v) {
		cout << "tdo_refine_rows::tdo_rows_setup_second_system: "
				"opening second system with "
			<< Nb_eqns << " equations and " << Nb_vars
			<< " variables" << endl;
		cout << "nb_eqns_joining=" << nb_eqns_joining << endl;
		cout << "nb_eqns_counting=" << nb_eqns_counting << endl;
		cout << "nb_eqns_packing=" << nb_eqns_packing << endl;
		cout << "l2=" << l2 << endl;
		cout << "L2=" << L2 << endl;
		cout << "T.nb_multiple_types=" << T.nb_multiple_types << endl;
	}

	if (!tdo_rows_setup_second_system_eqns_joining(
			T,
			point_types, nb_point_types,
			0 /*eqn_offset*/,
			verbose_level)) {

		if (f_v) {
			cout << "tdo_refine_rows::tdo_rows_setup_second_system "
					"tdo_rows_setup_second_system_eqns_joining returns false" << endl;
		}
		if (f_vv) {
			T.D2->print();
		}
		return false;
	}

	if (!tdo_rows_setup_second_system_eqns_counting(
			T,
			point_types, nb_point_types,
			nb_eqns_joining /*eqn_offset*/,
			verbose_level)) {

		if (f_v) {
			cout << "tdo_refine_rows::tdo_rows_setup_second_system "
					"tdo_rows_setup_second_system_eqns_counting returns false" << endl;
		}
		if (f_vv) {
			T.D2->print();
		}
		return false;
	}

	if (Tdo_scheme_synthetic->Descr->f_use_packing_numbers) {
		if (!tdo_rows_setup_second_system_eqns_packing(
				T,
				point_types, nb_point_types,
				nb_eqns_joining + nb_eqns_counting /* eqn_start */,
				nb_eqns_used,
				verbose_level)) {

			if (f_v) {
				cout << "tdo_refine_rows::tdo_rows_setup_second_system "
						"tdo_rows_setup_second_system_eqns_packing returns false" << endl;
			}
			if (f_vv) {
				T.D2->print();
			}
			return false;
		}
	}

	Nb_eqns = nb_eqns_joining + nb_eqns_counting + nb_eqns_used;
	T.D2->m = Nb_eqns;

	T.D2->eliminate_zero_rows_quick(verbose_level);


	// ToDo: where do we set x_max[] ? answer: in tdo_rows_setup_second_system_eqns_counting


	if (f_vv) {
		cout << "tdo_refine_rows::tdo_rows_setup_second_system "
				"The second system is:" << endl;
		T.D2->print();
	}
	if (f_v) {
		cout << "tdo_refine_rows::tdo_rows_setup_second_system done" << endl;
	}
	return true;
}

int tdo_refine_rows::tdo_rows_setup_second_system_eqns_joining(
		tdo_data &T,
		int *point_types, int nb_point_types,
		int eqn_offset,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "tdo_refine_rows::tdo_rows_setup_second_system_eqns_joining" << endl;
	}
	Tdo_scheme_synthetic->check_init();

	int l2, I1, I2, k, b, ab, i, j, r, I, J;
	int f, l, c, a, a2, rr, p, u, h, L1, L2;
	other_combinatorics::combinatorics_domain Combi;

	l2 = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];
	row_refinement_L1_L2(
			Tdo_scheme_synthetic->Descr->f_omit1,
			Tdo_scheme_synthetic->Descr->omit1,
			L1, L2,
			verbose_level);

	if (f_vv) {
		cout << "l2 = " << l2 << endl;
		cout << "L2 = " << L2 << endl;
		cout << "eqn_offset = " << eqn_offset << endl;
		cout << "T.nb_multiple_types = " << T.nb_multiple_types << endl;
	}

	for (I = 0; I < L2; I++) {

		string label;

		label = "J_{" + std::to_string(I + 1) + "}";
		T.D2->init_eqn_label(eqn_offset + I, label);
	}
	for (I1 = 0; I1 < L2; I1++) {
		for (I2 = I1 + 1; I2 < L2; I2++) {
			k = Combi.ij2k(I1, I2, L2);

			string label;

			label = "J_{" + std::to_string(I1 + 1) + "," + std::to_string(I2 + 1) + "}";
			T.D2->init_eqn_label(eqn_offset + L2 + k, label);
		}
	}
	if (f_vv) {
		cout << "tdo_refine_rows::tdo_rows_setup_second_system_eqns_joining "
				"filling coefficient matrix" << endl;
	}
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			for (I = 0; I < L2; I++) {
				a = point_types[c * L2 + I];
				a2 = Combi.binomial2(a);
				T.D2->Aij(eqn_offset + I, J) = a2;
			}
			for (I1 = 0; I1 < L2; I1++) {
				for (I2 = I1 + 1; I2 < L2; I2++) {
					k = Combi.ij2k(I1, I2, L2);
					a = point_types[c * L2 + I1];
					b = point_types[c * L2 + I2];
					ab = a * b;
					T.D2->Aij(eqn_offset + L2 + k, J) = ab;
				}
			}
		}
	}

	if (f_vv) {
		cout << "filling RHS" << endl;
	}
	for (I = 0; I < L2; I++) {
		a = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][I];
		a2 = Combi.binomial2(a);
		T.D2->RHS[eqn_offset + I] = a2;
		if (Tdo_scheme_synthetic->Descr->f_dual_is_linear_space) {
			T.D2->type[eqn_offset + I] = t_EQ;
		}
		else {
			T.D2->type[eqn_offset + I] = t_LE;
		}
	}
	for (I1 = 0; I1 < L2; I1++) {
		a = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][I1];
		for (I2 = I1 + 1; I2 < L2; I2++) {
			b = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][I2];
			k = Combi.ij2k(I1, I2, L2);
			T.D2->RHS[eqn_offset + L2 + k] = a * b;
			if (Tdo_scheme_synthetic->Descr->f_dual_is_linear_space) {
				T.D2->type[eqn_offset + L2 + k] = t_EQ;
			}
			else {
				T.D2->type[eqn_offset + L2 + k] = t_LE;
			}
		}
	}
	if (f_vv) {
		cout << "tdo_refine_rows::tdo_rows_setup_second_system_eqns_joining "
				"subtracting contribution from one-type blocks:" << endl;
	}
	// now subtract the contribution from one-type blocks:
	for (h = 0; h < T.nb_only_one_type; h++) {
		rr = T.only_one_type[h];
		p = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][rr];
		u = T.types_first[rr];
		for (I = 0; I < L2; I++) {
			a = point_types[u * L2 + I];
			a2 = Combi.binomial2(a);
			T.D2->RHS[eqn_offset + I] -= a2 * p;
			if (T.D2->RHS[eqn_offset + I] < 0) {
				if (f_vv) {
					cout << "tdo_refine_rows::tdo_rows_setup_second_system_eqns_joining "
							"RHS is negative, no solution for the "
							"distribution" << endl;
					cout << "h=" << h << endl;
					cout << "rr=T.only_one_type[h]=" << rr << endl;
					cout << "p=row_classes_len[ROW][rr]=" << p << endl;
					cout << "u=T.types_first[rr]="
							<< T.types_first[rr] << endl;
					cout << "I=" << I << endl;
					cout << "a=point_types[u * L2 + I]=" << a << endl;
					cout << "a2=binomial2(a)=" << a2 << endl;
					cout << "T.D2->RHS[eqn_offset + I]="
							<< T.D2->RHS[eqn_offset + I] << endl;
				}
				return false;
			}
		}
		for (I1 = 0; I1 < L2; I1++) {
			a = point_types[u * L2 + I1];
			for (I2 = I1 + 1; I2 < L2; I2++) {
				b = point_types[u * L2 + I2];
				k = Combi.ij2k(I1, I2, L2);
				ab = a * b * p;
				T.D2->RHS[eqn_offset + L2 + k] -= ab;
				if (T.D2->RHS[eqn_offset + L2 + k] < 0) {
					if (f_vv) {
						cout << "tdo_refine_rows::tdo_rows_setup_second_system_eqns_joining "
								"RHS is negative, no solution "
								"for the distribution" << endl;
						cout << "h=" << h << endl;
						cout << "rr=T.only_one_type[h]=" << rr << endl;
						cout << "p=row_classes_len[ROW][rr]=" << p << endl;
						cout << "u=T.types_first[rr]="
								<< T.types_first[rr] << endl;
						cout << "I1=" << I1 << endl;
						cout << "I2=" << I2 << endl;
						cout << "k=" << k << endl;
						cout << "a=point_types[u * L2 + I1]=" << a << endl;
						cout << "b=point_types[u * L2 + I2]=" << b << endl;
						cout << "ab=" << ab << endl;
						cout << "T.D2->RHS[eqn_offset + L2 + k]="
							<< T.D2->RHS[eqn_offset + L2 + k] << endl;
					}
					return false;
				}
			}
		}
	}
	if (f_v) {
		cout << "tdo_refine_rows::tdo_rows_setup_second_system_eqns_joining "
				"done" << endl;
	}
	return true;
}

int tdo_refine_rows::tdo_rows_setup_second_system_eqns_counting(
		tdo_data &T,
		int *point_types, int nb_point_types,
		int eqn_offset,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tdo_refine_rows::tdo_rows_setup_second_system_eqns_counting" << endl;
	}

	Tdo_scheme_synthetic->check_init();

	int l2, b, i, j, r, I, J, f, l, c, a, S, s, L1, L2;
	//int nb_vars = T.D1->n;

	l2 = Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME];
	row_refinement_L1_L2(
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

				label = "F_{" + std::to_string(I + 1) + "," + std::to_string(r + 1) + "}";
				T.D2->init_eqn_label(eqn_offset + i * (L2 + 1) + I, label);
			}
		}

		string label;
		label = "F_{" + std::to_string(r + 1) + "}";
		T.D2->init_eqn_label(eqn_offset + i * (L2 + 1) + l2, label);
	}

	// equations counting flags
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			for (I = 0; I < L2; I++) {
				a = point_types[c * L2 + I];
				T.D2->Aij(eqn_offset + i * (L2 + 1) + I, J) = a;
			}
			T.D2->Aij(eqn_offset + i * (L2 + 1) + L2, J) = 1;
		}
	}


	S = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		for (I = 0; I < L2; I++) {
			a = Tdo_scheme_synthetic->the_col_scheme[r * Tdo_scheme_synthetic->nb_col_classes[COL_SCHEME] + I];
			b = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][I];
			T.D2->RHS[eqn_offset + i * (L2 + 1) + I] = a * b;
		}
		s = Tdo_scheme_synthetic->row_classes_len[ROW_SCHEME][r];
		T.D2->RHS[eqn_offset + i * (L2 + 1) + L2] = s;
		S += s;
	}


	// set upper bound x_max:

	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		s = Tdo_scheme_synthetic->row_classes_len[COL_SCHEME][r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			T.D2->x_min[J] = 0;
			T.D2->x_max[J] = s;
			if (f_v) {
				cout << "tdo_refine_rows::tdo_rows_setup_second_system_eqns_counting "
						"setting x_max[" << J << "] = " << s << endl;
			}
		}
	}



	T.D2->f_has_sum = true;
	T.D2->sum = S;
	//T.D2->f_x_max = true;
	if (f_v) {
		cout << "tdo_refine_rows::tdo_rows_setup_second_system_eqns_counting "
				"done" << endl;
	}
	return true;
}

int tdo_refine_rows::tdo_rows_setup_second_system_eqns_packing(
		tdo_data &T,
		int *point_types, int nb_point_types,
		int eqn_start, int &nb_eqns_used,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tdo_refine_rows::tdo_rows_setup_second_system_eqns_packing" << endl;
	}
	Tdo_scheme_synthetic->check_init();


	int nb_eqns_packing;
	int /*l2,*/ i, r, f, l, j, c, J, JJ, k, h;
	int rr, p, u, a, len, f_used, L1, L2;
	//int nb_vars = T.D1->n;
	geometry::other_geometry::geometry_global Gg;


	//l2 = nb_col_classes[COL];
	row_refinement_L1_L2(
			Tdo_scheme_synthetic->Descr->f_omit1,
			Tdo_scheme_synthetic->Descr->omit1,
			L1, L2,
			verbose_level);

	nb_eqns_packing = 0;
	for (J = 0; J < L2; J++) {
		len = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][J];
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
					a = point_types[c * L2 + J];
					if (a < k) {
						continue;
					}
					JJ = T.types_first2[i] + j;
					f_used = true;
					T.D2->Aij(eqn_start + nb_eqns_packing, JJ) = 1;
				}
			} // next i
			if (f_used) {
				int bound;
				bound = Gg.TDO_upper_bound(len, k);
				T.D2->RHS[eqn_start + nb_eqns_packing] = bound;
				T.D2->type[eqn_start + nb_eqns_packing] = t_LE;
				for (h = 0; h < T.nb_only_one_type; h++) {
					rr = T.only_one_type[h];
					p = Tdo_scheme_synthetic->row_classes_len[COL_SCHEME][rr];
					u = T.types_first[rr];
					a = point_types[u * L2 + J];
					if (a < k) {
						continue;
					}
					T.D2->RHS[eqn_start + nb_eqns_packing] -= p;
					if (T.D2->RHS[eqn_start + nb_eqns_packing] < 0) {
						if (f_v) {
							cout << "tdo_refine_rows::tdo_rows_setup_second_system_eqns_packing "
									"RHS < 0" << endl;
						}
						return false;
					}
				}

				string label;

				label = "P_{" + std::to_string(J + 1) + "," + std::to_string(k) + "} \\,\\mbox{using}\\, "
						"P(" + std::to_string(len) + "," + std::to_string(k) + ")=" + std::to_string(bound);
				T.D2->init_eqn_label(eqn_start + nb_eqns_packing, label);
				if (f_v) {
					cout << "packing equation " << nb_eqns_packing
							<< " J=" << J << " k=" << k
							<< " len=" << len << endl;
				}
				nb_eqns_packing++;
			}
		} // next k
	}
	nb_eqns_used = nb_eqns_packing;
	if (f_v) {
		cout << "tdo_refine_rows::tdo_rows_setup_second_system_eqns_packing "
				"nb_eqns_used = " << nb_eqns_used << endl;
	}
	if (f_v) {
		cout << "tdo_refine_rows::tdo_rows_setup_second_system_eqns_packing "
				"done" << endl;
	}
	return true;
}



void tdo_refine_rows::geometric_test_for_row_scheme(
		tdo_refinement_output *Output,
	int f_omit1, int omit1, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_vvvv = (verbose_level >= 4);
	int f_v5 = (verbose_level >= 7);

	if (f_vvv) {
		cout << "tdo_refine_rows::geometric_test_for_row_scheme "
			"nb_distributions=" << Output->nb_distributions << endl;
	}

	int i, s, d, /*l2,*/ L1, L2, cnt, new_nb_distributions;
	int f_ruled_out;
	int *ruled_out_by;
	int *non_zero_blocks, nb_non_zero_blocks;


	//l2 = nb_col_classes[COL];
	row_refinement_L1_L2(
			f_omit1, omit1, L1, L2,
			verbose_level - 3);
	if (L2 != Output->type_len) {
		cout << "tdo_refine_rows::geometric_test_for_row_scheme "
				"L2 != point_type_len" << endl;
		exit(1);
	}

	ruled_out_by = NEW_int(Output->nb_types + 1);
	non_zero_blocks = NEW_int(Output->nb_types);
	for (i = 0; i <= Output->nb_types; i++) {
		ruled_out_by[i] = 0;
	}

	new_nb_distributions = 0;
	for (cnt = 0; cnt < Output->nb_distributions; cnt++) {

		nb_non_zero_blocks = 0;

		for (i = 0; i < Output->nb_types; i++) {
			d = Output->distributions[cnt * Output->nb_types + i];
			if (d == 0) {
				continue;
			}
			non_zero_blocks[nb_non_zero_blocks++] = i;
		}

		if (f_vvvv) {
			cout << "tdo_refine_rows::geometric_test_for_row_scheme "
					"testing distribution "
				<< cnt << " / " << Output->nb_distributions << " : ";
			Int_vec_print(
					cout,
					Output->distributions + cnt * Output->nb_types,
					Output->nb_types);
			cout << endl;
			if (f_v5) {
				cout << "that is" << endl;
				for (i = 0; i < nb_non_zero_blocks; i++) {
					d = Output->distributions[cnt * Output->nb_types + non_zero_blocks[i]];
					cout << setw(3) << i << " : " << setw(3) << d << " x ";
					Int_vec_print(
							cout,
							Output->types + non_zero_blocks[i] * Output->type_len,
							Output->type_len);
					cout << endl;
				}
			}
		}

		f_ruled_out = false;

		for (s = 1; s <= nb_non_zero_blocks; s++) {

			if (f_vvv) {
				cout << "tdo_refine_rows::geometric_test_for_row_scheme "
					"applying geometric test of strength s=" << s << endl;
			}

			if (!geometric_test_for_row_scheme_level_s(
					s,
					Output->types, Output->nb_types, Output->type_len,
				Output->distributions + cnt * Output->nb_types,
				non_zero_blocks, nb_non_zero_blocks,
				f_omit1, omit1, verbose_level - 4)) {

				f_ruled_out = true;
				ruled_out_by[s]++;
				if (f_vv) {
					cout << "tdo_refine_rows::geometric_test_for_row_scheme "
							"distribution "
						<< cnt << " / " << Output->nb_distributions
						<< " eliminated by test of strength " << s << endl;
				}
				if (f_vvv) {
					cout << "tdo_refine_rows::geometric_test_for_row_scheme "
							"the eliminated scheme is:" << endl;
					for (i = 0; i < nb_non_zero_blocks; i++) {
						d = Output->distributions[cnt * Output->nb_types +
										  non_zero_blocks[i]];
						cout << setw(3) << i << " : "
							<< setw(3) << d << " x ";
						Int_vec_print(
								cout,
								Output->types + non_zero_blocks[i] * Output->type_len,
								Output->type_len);
						cout << endl;
					}
					cout << "tdo_refine_rows::geometric_test_for_row_scheme "
							"we repeat the test with more printout:" << endl;
					geometric_test_for_row_scheme_level_s(
							s,
							Output->types, Output->nb_types, Output->type_len,
						Output->distributions + cnt * Output->nb_types,
						non_zero_blocks, nb_non_zero_blocks,
						f_omit1, omit1, verbose_level - 3);
				}
				break;
			}
		}



		if (!f_ruled_out) {
			for (i = 0; i < Output->nb_types; i++) {
				Output->distributions[new_nb_distributions * Output->nb_types + i] =
						Output->distributions[cnt * Output->nb_types + i];
			}
			new_nb_distributions++;
		}

	} // next cnt

	if (f_v) {
		cout << "tdo_refine_rows::geometric_test_for_row_scheme "
				"number of distributions has been reduced from "
				<< Output->nb_distributions << " to "
			<< new_nb_distributions << ", i.e. Eliminated "
			<< Output->nb_distributions - new_nb_distributions << " cases" << endl;
		cout << "# of ruled out by test of order ";
		Int_vec_print(cout, ruled_out_by, Output->nb_types + 1);
		cout << endl;
		//cout << "nb ruled out by first order test  = "
		//<< nb_ruled_out_by_order1 << endl;
		//cout << "nb ruled out by second order test = "
		//<< nb_ruled_out_by_order2 << endl;
		for (i = Output->nb_types; i >= 1; i--) {
			if (ruled_out_by[i]) {
				break;
			}
		}
		if (i) {
			cout << "tdo_refine_rows::geometric_test_for_row_scheme "
					"highest order test that was successfully "
					"applied is order " << i << endl;
		}
	}
	FREE_int(ruled_out_by);
	FREE_int(non_zero_blocks);

	Output->nb_distributions = new_nb_distributions;
	//return new_nb_distributions;
}


int tdo_refine_rows::geometric_test_for_row_scheme_level_s(
		int s,
	int *point_types, int nb_point_types, int point_type_len,
	int *distribution,
	int *non_zero_blocks, int nb_non_zero_blocks,
	int f_omit1, int omit1,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vvv = (verbose_level >= 3);

	if (f_vvv) {
		cout << "tdo_refine_rows::geometric_test_for_row_scheme_level_s s=" << s << endl;
	}

	int *set;
	int J, L1, L2, len, max, cur, u, D, d, c;
	int nb_inc, e, f, nb_ordererd_pairs;
	other_combinatorics::combinatorics_domain Combi;

	if (s >= 1000) {
		cout << "tdo_refine_rows::geometric_test_for_row_scheme_level_s level too deep" << endl;
		exit(1);
	}

	set = NEW_int(s);


	row_refinement_L1_L2(
			f_omit1, omit1, L1, L2, verbose_level - 3);

	Combi.first_k_subset(set, nb_non_zero_blocks, s);
	while (true) {
		D = 0;
		for (u = 0; u < s; u++) {
			d = distribution[non_zero_blocks[set[u]]];
			D += d;
		}
		max = D * (D - 1);
		cur = 0;
		for (J = 0; J < L2; J++) {
			len = Tdo_scheme_synthetic->col_classes_len[COL_SCHEME][J];
			nb_inc = 0;
			for (u = 0; u < s; u++) {
				c = point_types[non_zero_blocks[set[u]] * point_type_len + J];
				d = distribution[non_zero_blocks[set[u]]];
				// we have d rows with c incidences in len columns
				nb_inc += d * c;
			}

			e = nb_inc % len; // the number of incidences in the extra row
			f = nb_inc / len; // the number of full rows

			nb_ordererd_pairs = 0;
			if (Tdo_scheme_synthetic->n) {
				nb_ordererd_pairs = e * (f + 1) * f + (len - e) * f * (f - 1);
			}
			cur += nb_ordererd_pairs;
			if (cur > max) {
				if (f_v) {
					cout << "tdo_refine_rows::geometric_test_for_row_scheme_level_s "
							"s=" << s << " failure in point type ";
					Int_vec_print(cout, set, s);
					cout << endl;
					cout << "max=" << max << endl;
					cout << "J=" << J << endl;
					cout << "nb_inc=" << nb_inc << endl;
					cout << "nb_ordererd_pairs=" << nb_ordererd_pairs << endl;
					cout << "cur=" << cur << endl;
				}
				FREE_int(set);
				return false;
			}
		} // next J
		if (!Combi.next_k_subset(set, nb_non_zero_blocks, s)) {
			break;
		}
	}
	FREE_int(set);
	return true;
}


#if 0
int tdo_scheme_synthetic::test_row_distribution(
	int *point_types, int nb_point_types, int point_type_len,
	int *distributions, int nb_distributions, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int l2, cnt, J, len, k, i, d, c, new_nb_distributions, bound;
	int f_ruled_out, f_ruled_out_by_braun, f_ruled_out_by_packing;
	int nb_ruled_out_by_braun = 0, nb_ruled_out_by_packing = 0;
	int nb_ruled_out_by_both = 0;

	if (f_v) {
		cout << "tdo_scheme_synthetic::test_row_distribution "
				"nb_distributions=" << nb_distributions << endl;
		}
	l2 = nb_col_classes[COL];
	if (l2 != point_type_len) {
		cout << "tdo_scheme_synthetic::test_row_distribution "
				"l2 != point_type_len" << endl;
		exit(1);
		}

	new_nb_distributions = 0;

	for (cnt = 0; cnt < nb_distributions; cnt++) {
		if (f_vv) {
			cout << "testing distribution " << cnt << " : ";
			int_vec_print(cout,
				distributions + cnt * nb_point_types,
				nb_point_types);
			cout << endl;
			if (f_vvv) {
				cout << "that is" << endl;
				for (i = 0; i < nb_point_types; i++) {
					d = distributions[cnt * nb_point_types + i];
					if (d == 0)
						continue;
					cout << setw(3) << d << " x ";
					int_vec_print(cout,
						point_types + i * point_type_len,
						point_type_len);
					cout << endl;
					}
				}
			}
		f_ruled_out = false;
		f_ruled_out_by_braun = false;
		f_ruled_out_by_packing = false;

		for (J = 0; J < l2; J++) {
			len = col_classes_len[COL][J];
			int *type;

			if (f_vvv) {
				cout << "testing distribution " << cnt << " in block "
					<< J << " len=" << len << endl;
				}
			type = NEW_int(len + 1);
			for (k = 0; k <= len; k++)
				type[k] = 0;
			for (i = 0; i < nb_point_types; i++) {
				d = distributions[cnt * nb_point_types + i];
				c = point_types[i * point_type_len + J];
				type[c] += d;
				}
			if (f_vvv) {
				cout << "line type: ";
				int_vec_print(cout, type + 1, len);
				cout << endl;
				}
			if (!braun_test_on_line_type(len, type)) {
				if (f_vv) {
					cout << "distribution " << cnt << " is eliminated "
						"in block " << J << " using Braun test" << endl;
					}
				f_ruled_out = true;
				f_ruled_out_by_braun = true;
				FREE_int(type);
				break;
				}
			FREE_int(type);
			} // next J
		for (J = 0; J < l2; J++) {
			len = col_classes_len[COL][J];
			if (len == 1)
				continue;
			for (i = 0; i < nb_point_types; i++) {
				d = distributions[cnt * nb_point_types + i];
				if (d == 0)
					continue;
				c = point_types[i * point_type_len + J];
				// now we want d lines of size c on len points
				if (c > 1) {
					if (c > len) {
						cout << "c > len" << endl;
						cout << "J=" << J << " i=" << i << " d="
								<< d << " c=" << c << endl;
						exit(1);
						}
					bound = TDO_upper_bound(len, c);
					if (d > bound) {
						if (f_vv) {
							cout << "distribution " << cnt
								<< " is eliminated in block "
								<< J << " row-block " << i
								<< " using packing numbers" << endl;
							cout << "len=" << len << endl;
							cout << "d=" << d << endl;
							cout << "c=" << c << endl;
							cout << "bound=" << bound << endl;
							}
						f_ruled_out = true;
						f_ruled_out_by_packing = true;
						break;
						}
					}
				}
			if (f_ruled_out)
				break;
			}
		if (f_ruled_out) {
			if (f_ruled_out_by_braun)
				nb_ruled_out_by_braun++;
			if (f_ruled_out_by_packing)
				nb_ruled_out_by_packing++;
			if (f_ruled_out_by_braun && f_ruled_out_by_packing)
				nb_ruled_out_by_both++;
			}
		else {
			for (i = 0; i < nb_point_types; i++) {
				distributions[new_nb_distributions * nb_point_types + i] =
					distributions[cnt * nb_point_types + i];
				}
			new_nb_distributions++;
			}
		} // next cnt
	if (f_v) {
		cout << "number of distributions reduced from "
			<< nb_distributions << " to "
			<< new_nb_distributions << ", i.e. Eliminated "
			<< nb_distributions - new_nb_distributions << " cases" << endl;
		cout << "nb_ruled_out_by_braun = "
			<< nb_ruled_out_by_braun << endl;
		cout << "nb_ruled_out_by_packing = "
			<< nb_ruled_out_by_packing << endl;
		cout << "nb_ruled_out_by_both = "
			<< nb_ruled_out_by_both << endl;
		}
	return new_nb_distributions;
}
#endif


}}}}

