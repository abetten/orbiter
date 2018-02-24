// refine_rows.C
// Anton Betten
//
// started: December 2006
// moved away from inc_gen: 8/27/07
// separated out from refine: 1/24/08
// corrected a memory problem in refine_rows_hard: Dec 6 2010

#include "galois.h"
#include "incidence.h"

// #################################################################################
// parameter refinement (TDO) 
// #################################################################################

INT tdo_scheme::refine_rows(INT verbose_level,
	INT f_use_mckay, INT f_once, 
	partitionstack &P, 
	INT *&point_types, INT &nb_point_types, INT &point_type_len, 
	INT *&distributions, INT &nb_distributions, 
	INT &cnt_second_system, solution_file_data *Sol,
	INT f_omit1, INT omit1, INT f_omit2, INT omit2, 
	INT f_use_packing_numbers, INT f_dual_is_linear_space, INT f_do_the_geometric_test)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	//INT f_easy;
	INT l1, l2, R;
	
	if (f_v) {
		cout << "refine_rows" << endl;
		cout << "f_omit1=" << f_omit1 << " omit1=" << omit1 << endl;
		cout << "f_omit2=" << f_omit2 << " omit2=" << omit2 << endl;
		cout << "f_use_packing_numbers=" << f_use_packing_numbers << endl;
		cout << "f_dual_is_linear_space=" << f_dual_is_linear_space << endl;
		cout << "f_use_mckay=" << f_use_mckay << endl;
		}
	if (row_level >= 2) {
		R = nb_row_classes[ROW];
		l1 = nb_col_classes[ROW];
		l2 = nb_col_classes[COL];
		if (f_vv) {
			cout << "l1=" << l1 << " at level " << level[ROW] << endl;
			cout << "l2=" << l2 << " at level " << level[COL] << endl;
			cout << "R=" << R << endl;
			}
		get_column_split_partition(0 /*verbose_level*/, P);
		if (f_vv) {
			cout << "column split partition: " << P << endl;
			}
		if (P.ht != l1) {
			cout << "P.ht != l1" << endl;
			exit(1);
			}
		if ((R == 1) && (l1 == 1) && (the_row_scheme[0] == -1)) {
			if (!refine_rows_easy(verbose_level - 1, 
				point_types, nb_point_types, point_type_len, 
				distributions, nb_distributions, cnt_second_system)) {
				return FALSE;
				}
			}
		else {
			if (!refine_rows_hard(P, verbose_level - 1, f_use_mckay, f_once, 
				point_types, nb_point_types, point_type_len, 
				distributions, nb_distributions, cnt_second_system,
				f_omit1, omit1, f_omit2, omit2, 
				f_use_packing_numbers, f_dual_is_linear_space)) {
				return FALSE;
				}
			}
		}
	else {
		if (!refine_rows_easy(verbose_level - 1, 
			point_types, nb_point_types, point_type_len, 
			distributions, nb_distributions, cnt_second_system)) {
			return FALSE;
			}
		}

	if (f_do_the_geometric_test) {
		nb_distributions = geometric_test_for_row_scheme(P, 
			point_types, nb_point_types, point_type_len, 
			distributions, nb_distributions, 
			f_omit1, omit1, 
			verbose_level);
		}
	return TRUE;
}

INT tdo_scheme::refine_rows_easy(int verbose_level, 
	INT *&point_types, INT &nb_point_types, INT &point_type_len,  
	INT *&distributions, INT &nb_distributions, 
	INT &cnt_second_system)
{
	INT nb_rows;
	INT i, j, J, S, l2, nb_eqns, nb_vars, nb_eqns_joining, nb_eqns_upper_bound;
	INT nb_sol, len, k, a2, a, b, ab, f_used, j1, j2, len1, len2, cnt;
	INT Nb_eqns, Nb_vars;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	BYTE label[100];

	if (f_v) {
		cout << "refine_rows_easy" << endl;
		}
	l2 = nb_col_classes[COL];
	
	//partitionstack &P = PB.P;
	nb_rows = P->startCell[1];
	S = nb_rows - 1;
	if (f_v) {
		cout << "nb_rows=" << nb_rows << endl;
		}
	
	nb_vars = l2 + 1; // 1 slack variable
	nb_eqns = 1;
	
	diophant D;
	
	D.open(nb_eqns, nb_vars);
		
	// 1st equation: connections within the same row-partition
	for (J = 0; J < nb_vars; J++) {
		D.Aij(0, J) = minus_one_if_positive(the_col_scheme[0 * l2 + J]);
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
		D.x_max[j] = col_classes_len[COL][j];
		}
	D.x_max[nb_vars - 1] = nb_rows - 1;;
	
	D.f_x_max = TRUE;
		

	D.eliminate_zero_rows_quick(verbose_level);
	D.sum = S;
	if (f_vv) {
		cout << "The first system is" << endl;
		D.print();
		}
	if (f_vv) {
		BYTE label[1000];
			
		sprintf(label, "first");
		D.write_xml(cout, label);
		}

	nb_sol = 0;
	point_type_len = nb_vars - 1;
	
	if (D.solve_first(verbose_level - 2)) {
	
		while (TRUE) {
			if (f_vv) {
				cout << nb_sol << " : ";
				for (i = 0; i < nb_vars; i++) {
					cout << " " << D.x[i];
					}
				cout << endl;
				}
			nb_sol++;
			if (!D.solve_next())
				break;
			}
		}
	if (f_v) {
		cout << "found " << nb_sol << " point types" << endl;
		}
	if (nb_sol == 0) {
		return FALSE;
		}
	nb_point_types = nb_sol;
	
	nb_eqns_upper_bound = 0;
	for (j = 0; j < l2; j++) {
		len = col_classes_len[COL][j];
		if (len > 2) {
			nb_eqns_upper_bound += len - 2;
			}
		}
	nb_eqns_joining = l2 + ((l2 * (l2 - 1)) >> 1);
	Nb_eqns = l2 + nb_eqns_joining + nb_eqns_upper_bound;
	Nb_vars = nb_sol;
	
	dophant D2;
	
	D2.open(Nb_eqns, Nb_vars);
	point_types = NEW_INT(nb_point_types * point_type_len);
	if (f_v) {
		cout << "refine_rows_easy: opening second " << cnt_second_system << " system with " 
			<< Nb_eqns << " equations and " << Nb_vars << " variables" << endl;
		}
	
	nb_sol = 0;
	if (D.solve_first(verbose_level - 2)) {
	
		while (TRUE) {
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
			if (!D.solve_next())
				break;
			}
		}
	for (j = 0; j < l2; j++) {
		len = col_classes_len[COL][j];
		for (i = 0; i < Nb_vars; i++) {
			a = point_types[i * point_type_len + j];
			a2 = binomial2(a);
			D2.Aij(l2 + j, i) = a2;
			}
		D2.RHS[l2 + j] = binomial2(len);
		D2.type[l2 + j] = t_LE;
		sprintf(label, "J_{%ld}", j + 1);
		D2.init_eqn_label(l2 + j, label);
		}
	cnt = 0;
	for (j1 = 0; j1 < l2; j1++) {
		len1 = col_classes_len[COL][j1];
		for (j2 = j1 + 1; j2 < l2; j2++) {
			len2 = col_classes_len[COL][j2];
			for (i = 0; i < Nb_vars; i++) {
				a = point_types[i * point_type_len + j1];
				b = point_types[i * point_type_len + j2];
				ab = a * b;
				D2.Aij(l2 + l2 + cnt, i) = ab;
				}
			D2.RHS[l2 + l2 + cnt] = len1 * len2;
			D2.type[l2 + l2 + cnt] = t_LE;
			sprintf(label, "J_{%ld,%ld}", j1 + 1, j2 + 1);
			D2.init_eqn_label(l2 + l2 + cnt, label);
			cnt++;
			}
		}

	nb_eqns_upper_bound = 0;
	for (j = 0; j < l2; j++) {
		len = col_classes_len[COL][j];
		for (k = 3; k <= len; k++) {
			for (i = 0; i < Nb_vars; i++) {
				D2.Aij(l2 + nb_eqns_joining + nb_eqns_upper_bound, i) = 0;
				}
			f_used = FALSE;
			for (i = 0; i < Nb_vars; i++) {
				a = point_types[i * point_type_len + j];
				if (a < k)
					continue;
				D2.Aij(l2 + nb_eqns_joining + nb_eqns_upper_bound, i) = 1;
				f_used = TRUE;
				}
			if (f_used) {
				INT bound = TDO_upper_bound(len, k);
				D2.RHS[l2 + nb_eqns_joining + nb_eqns_upper_bound] = bound;
				D2.type[l2 + nb_eqns_joining + nb_eqns_upper_bound] = t_LE;
				sprintf(label, "P_{%ld,%ld} \\,\\mbox{using}\\, P(%ld,%ld)=%ld", j + 1, k, len, k, bound);
				D2.init_eqn_label(l2 + nb_eqns_joining + nb_eqns_upper_bound, label);
				nb_eqns_upper_bound++;
				}
			} // next k
		} // next j
	Nb_eqns = l2 + nb_eqns_joining + nb_eqns_upper_bound;
	D2.m = Nb_eqns;

	if (f_v) {
		cout << "second system " << cnt_second_system << " found " << nb_sol << " point types" << endl;
		}
	cnt_second_system++;
	if (nb_sol == 0) {
		FREE_INT(point_types);
		return FALSE;
		}
	D2.sum = nb_rows;
	for (i = 0; i < l2; i++) {
		D2.RHS[i] = col_classes_len[COL][i] * the_col_scheme[i];
		sprintf(label, "F_{%ld}", i + 1);
		D2.init_eqn_label(i, label);
		}
	D2.eliminate_zero_rows_quick(verbose_level);
	if (f_vv) {
		cout << "The second system is" << endl;
		D2.print();
		}
	if (f_vv) {
		BYTE label[1000];
			
		sprintf(label, "second");
		D2.write_xml(cout, label);
		}
	nb_sol = 0;
	if (D2.solve_first(verbose_level - 2)) {
		while (TRUE) {
			if (f_vv) {
				cout << nb_sol << " : ";
				for (i = 0; i < Nb_vars; i++) {
					cout << " " << D2.x[i];
					}
				cout << endl;
				}
			nb_sol++;
			if (!D2.solve_next())
				break;
			}
		}
	nb_distributions = nb_sol;
	distributions = NEW_INT(nb_distributions * nb_point_types);
	nb_sol = 0;
	if (D2.solve_first(verbose_level - 2)) {
		while (TRUE) {
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
			if (!D2.solve_next())
				break;
			}
		}
	if (f_v) {
		cout << "refine_rows_easy: found " << nb_distributions << " point type distributions." << endl;
		}
	return TRUE;
}

INT tdo_scheme::refine_rows_hard(partitionstack &P, int verbose_level, 
	INT f_use_mckay, INT f_once, 
	INT *&point_types, INT &nb_point_types, INT &point_type_len,  
	INT *&distributions, INT &nb_distributions, 
	INT &cnt_second_system, 
	INT f_omit1, INT omit1, INT f_omit, INT omit, 
	INT f_use_packing_numbers, INT f_dual_is_linear_space)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, r, R, l1, /*l2,*/ L1, L2;
	INT nb_sol;
	INT point_types_allocated;
	INT h, u;
	tdo_data T;

	if (f_v) {
		cout << "refine_rows_hard" << endl;
		if (f_omit1) {
			cout << "omitting the last " << omit1 << " column blocks from the previous row-scheme" << endl;
			}
		if (f_omit) {
			cout << "omitting the last " << omit << " row blocks" << endl;
			}
		cout << "f_use_packing_numbers=" << f_use_packing_numbers << endl;
		cout << "f_dual_is_linear_space=" << f_dual_is_linear_space << endl;
		cout << "f_use_mckay=" << f_use_mckay << endl;
		}
	R = nb_row_classes[ROW];
	l1 = nb_col_classes[ROW];
	//l2 = nb_col_classes[COL];

	if (f_v) {
		cout << "the_row_scheme is:" << endl;
		INT i, j;
		for (i = 0; i < R; i++) {
			for (j = 0; j < l1; j++) {
				cout << setw(4) << the_row_scheme[i * l1 + j];
				}
			cout << endl;
			}
		}
	
	row_refinement_L1_L2(P, f_omit1, omit1, L1, L2, verbose_level);

	T.allocate(R);
	
	T.types_first[0] = 0;
	
	point_types_allocated = 100;
	nb_point_types = 0;
	point_type_len = L2 + L1; // + slack variables
	point_types = NEW_INT(point_types_allocated * point_type_len);
		// detected and corrected an error: Dec 6 2010
		// it was allocated to point_types_allocated * L2
		// which is not enough
		
		// when we are done, it is [point_types_allocated * L2]
	

	T.nb_only_one_type = 0;
	T.nb_multiple_types = 0;
	
	for (r = 0; r < R; r++) {
		
		if (f_v) {
			cout << "r=" << r << endl;
			}
		
		tdo_rows_setup_first_system(verbose_level, 
			T, r, P, 
			f_omit1, omit1, 
			point_types, nb_point_types);
		
		if (f_vv) {
			BYTE label[1000];
			
			sprintf(label, "first_%ld", r);
			T.D1->write_xml(cout, label);
			}
		nb_sol = T.solve_first_system(verbose_level - 1, 
			point_types, nb_point_types, point_types_allocated);

		if (f_v) {
			cout << "r = " << r << ", found " << nb_sol << " refined point types" << endl;
			}
		if (f_vv) {
			print_integer_matrix_width(cout, point_types + T.types_first[r] * point_type_len, nb_sol, point_type_len, point_type_len, 3);
			}
		
#if 0
		// MARUTA  Begin
		if (r == 1) {
			INT h, a;

			for (h = nb_sol - 1; h >= 0; h--) {
				a = (point_types + (T.types_first[r] + h) * point_type_len)[0];
				if (a == 0) {
					cout << "removing last solution" << endl;
					nb_sol--;
					nb_point_types--;
					}
				}
			}
		// MARUTA   End
#endif


		if (f_vv) {
			print_integer_matrix_width(cout, point_types + T.types_first[r] * point_type_len, nb_sol, point_type_len, point_type_len, 3);
			}
		if (nb_sol == 0) {
			FREE_INT(point_types);
			return FALSE;
			}

		T.types_len[r] = nb_sol;
		T.types_first[r + 1] = T.types_first[r] + nb_sol;
		
		if (nb_sol == 1) {
			if (f_v) {
				cout << "only one solution in block r=" << r << endl;
				}
			T.only_one_type[T.nb_only_one_type++] = r;
			}
		else {
			T.multiple_types[T.nb_multiple_types++] = r;
			}
		
		T.D1->freeself();
		//diophant_close(T.D1);
		//T.D1 = NULL;

		} // next r
	
	// eliminate the slack variables from point_types:
	for (r = 0; r < nb_point_types; r++) {
		INT f, l, a, j, J;
		
		for (i = 0; i < L1; i++) {
			f = P.startCell[i];
			l = P.cellSize[i];
			for (j = 0; j < l; j++) {
				J = f + i + j;
				a = point_types[r * point_type_len + J];
				point_types[r * L2 + f + j] = a;
				}
			}
		}
	point_type_len = L2;
	if (f_v) {
		cout << "altogether, we found " << nb_point_types << " refined point types" << endl;
		}
	if (f_vv) {
		print_integer_matrix_width(cout, point_types, nb_point_types, point_type_len, point_type_len, 3);
		}
	

	// now we compute the distributions:

	if (!tdo_rows_setup_second_system(verbose_level, 
		T, P, 
		f_omit1, omit1, f_use_packing_numbers, f_dual_is_linear_space, 
		point_types, nb_point_types)) {
		FREE_INT(point_types);
		return FALSE;
		}
	if (f_vv) {
		BYTE label[1000];
			
		sprintf(label, "second");
		T.D2->write_xml(cout, label);
		}
	

	if (T.D2->n == 0) {
		distributions = NEW_INT(1 * nb_point_types);
		nb_distributions = 0;
		for (h = 0; h < T.nb_only_one_type; h++) {
			r = T.only_one_type[h];
			u = T.types_first[r];
			//cout << "only one type, r=" << r << " u=" << u << " row_classes_len[ROW][r]=" << row_classes_len[ROW][r] << endl;
			distributions[nb_distributions * nb_point_types + u] = 
				row_classes_len[ROW][r];
			}
		nb_distributions++;
		return TRUE;
		}
	

#if 0
	if (cnt_second_system == 1) {
		INT j;
		INT x[] = {4,1,5,0,2,0,7,2,4,0,0,0,1,0,0,4,0,0};
		cout << "testing solution:" << endl;
		INT_vec_print(cout, x, 18);
		cout << endl;
		if (T.D2->n != 18) {
			cout << "T.D2->n != 18" << endl;
			}
		for (j = 0; j < 18; j++) {
			T.D2->x[j] = x[j];
			}
		T.D2->multiply_A_x_to_RHS1();
		for (i = 0; i < T.D2->m; i++) {
			cout << i << " : " << T.D2->RHS1[i] << " : " << T.D2->RHS[i] - T.D2->RHS1[i] << endl;
			}
		}
#endif

	if (f_v) {
		cout << "refine_rows_hard: solving second system " << cnt_second_system << " which is " << T.D2->m << " x " << T.D2->n << endl;
		cout << T.nb_multiple_types << " variable blocks:" << endl;
		INT f, l;
		for (i = 0; i < T.nb_multiple_types; i++) {
			r = T.multiple_types[i];
			f = T.types_first2[i];
			l = T.types_len[r];
			cout << i << " : " << r << " : " << setw(3) << row_classes_len[ROW][r] << " : " << setw(3) << f << " : " << setw(3) << l << endl;
			}
		}
	if (f_omit) {
		T.solve_second_system_omit(verbose_level - 1, 
			row_classes_len[ROW], 
			point_types, nb_point_types, distributions, nb_distributions, omit);
		}
	else {
		int f_scale = FALSE;
		int scaling = 0;
		T.solve_second_system(verbose_level - 1, f_use_mckay, f_once, 
			row_classes_len[ROW], f_scale, scaling, 
			point_types, nb_point_types, distributions, nb_distributions);
		}



	if (f_v) {
		cout << "refine_rows_hard: second system " << cnt_second_system << " found " << nb_distributions << " distributions." << endl;
		}
	cnt_second_system++;
	return TRUE;
}

void tdo_scheme::row_refinement_L1_L2(partitionstack &P, INT f_omit, INT omit, 
	INT &L1, INT &L2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT l1, l2, omit2, i;
	l1 = nb_col_classes[ROW];
	l2 = nb_col_classes[COL];

	omit2 = 0;
	if (f_omit) {
		for (i = l1 - omit; i < l1; i++) {
			omit2 += P.cellSize[i];
			}
		}
	L1 = l1 - omit;
	L2 = l2 - omit2;
	if (f_v) {
		cout << "row_refinement_L1_L2: l1 = " << l1 << " l2=" << l2 << " L1=" << L1 << " L2=" << L2 << endl;
		}
}

INT tdo_scheme::tdo_rows_setup_first_system(INT verbose_level, 
	tdo_data &T, INT r, partitionstack &P, 
	INT f_omit, INT omit, 
	INT *&point_types, INT &nb_point_types)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT S, s_default, s_or_s_default, R, l1, l2, L1, L2;
	INT J, r2, i, j, s, f, l;
	INT nb_vars, nb_eqns;
	
	if (!f_omit)
		omit = 0;
	
	if (f_v) {
		cout << "tdo_rows_setup_first_system r=" << r << endl;
		if (f_omit) {
			cout << "omit=" << omit << endl;
			}
		}
	R = nb_row_classes[ROW];
	l1 = nb_col_classes[ROW];
	l2 = nb_col_classes[COL];

	row_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);
	
	nb_vars = L2 + L1; // possible up to L1 slack variables
	nb_eqns = R + L1;
		

	T.D1->open(nb_eqns, nb_vars);
	T.D1->fill_coefficient_matrix_with(0);
	S = 0;
		
	for (r2 = 0; r2 < R; r2++) {
		
		if (r2 == r) {
			// connections within the same row-partition
			for (i = 0; i < L1; i++) {
				f = P.startCell[i];
				l = P.cellSize[i];
				for (j = 0; j < l; j++) {
					J = f + i + j; // +i for the slack variables
					T.D1->Aij(r2, J) = minus_one_if_positive(the_col_scheme[r2 * l2 + f + j]);
					}
				T.D1->Aij(r2, f + i + l) = 0; // the slack variable is not needed
				}
#if 0
			for (J = 0; J < nb_vars; J++) {
				T.D1->Aij(r2, J) = minus_one_if_positive(the_col_scheme[r2 * l2 + J]);
				}
#endif
			T.D1->RHS[r] = row_classes_len[ROW][r] - 1;
			if (f_omit)
				T.D1->type[r] = t_LE;
			}
		else {
			// connections to the point from different row-partitions
			for (i = 0; i < L1; i++) {
				f = P.startCell[i];
				l = P.cellSize[i];
				for (j = 0; j < l; j++) {
					J = f + i + j; // +i for the slack variables
					T.D1->Aij(r2, J) = the_col_scheme[r2 * l2 + f + j];
					}
				T.D1->Aij(r2, f + i + l) = 0; // the slack variable is not needed
				}
#if 0
			for (J = 0; J < nb_vars; J++) {
				T.D1->Aij(r2, J) = the_col_scheme[r2 * l2 + J];
				}
#endif
			T.D1->RHS[r2] = row_classes_len[ROW][r2];
			if (f_omit)
				T.D1->type[r2] = t_LE;
			}
		}
		
	for (i = 0; i < L1; i++) {
		s = the_row_scheme[r * l1 + i];
		if (FALSE) {
			cout << "r=" << r << " i=" << i << " s=" << s << endl;
			}
		if (s == -1) {
			cout << "row scheme entry " << r << "," << i << " is -1, using slack variable" << endl;
			cout << "using " << col_classes_len[ROW][i] << " as upper bound" << endl;
			s_default = col_classes_len[ROW][i];
			s_or_s_default = s_default;
			}
		else {
			s_or_s_default = s;
			}
		
		T.D1->RHS[R + i] = s_or_s_default;
		S += s_or_s_default;
		
		f = P.startCell[i];
		l = P.cellSize[i];
		if (FALSE) {
			cout << "f=" << f << " l=" << l << endl;
			}
			
		for (j = 0; j < l; j++) {
			J = f + i + j; // +i for the slack variables
			T.D1->Aij(R + i, J) = 1;
			T.D1->x_max[J] = MINIMUM(col_classes_len[COL][f + j], s_or_s_default);
			}
		T.D1->Aij(R + i, f + i + l) = 1; // the slack variable
		if (s == -1) {
			T.D1->x_max[f + i + l] = s_default;
			}
		else {
			T.D1->x_max[f + i + l] = 0;
			}
		}
	T.D1->sum = S;
	T.D1->f_x_max = TRUE;
		
	T.D1->eliminate_zero_rows_quick(verbose_level);
	
	if (f_v) {
		cout << "tdo_rows_setup_first_system r=" << r << " finished" << endl;
		}
	if (f_vv) {
		T.D1->print();
		}
	return TRUE;
		
}

INT tdo_scheme::tdo_rows_setup_second_system(INT verbose_level, 
	tdo_data &T, partitionstack &P, 
	INT f_omit, INT omit, INT f_use_packing_numbers, INT f_dual_is_linear_space, 
	INT *&point_types, INT &nb_point_types)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT nb_eqns_joining, nb_eqns_counting, nb_eqns_packing, nb_eqns_used = 0;
	INT Nb_vars, Nb_eqns;
	INT l2, i, j, len, r, L1, L2;
	
	if (f_v) {
		cout << "tdo_rows_setup_second_system" << endl;
		cout << "f_omit=" << f_omit << " omit=" << omit << endl;
		cout << "f_use_packing_numbers=" << f_use_packing_numbers << endl;
		cout << "f_dual_is_linear_space=" << f_dual_is_linear_space << endl;
		}

	l2 = nb_col_classes[COL];

	row_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);

	nb_eqns_joining = L2 + binomial2(L2);
	nb_eqns_counting = T.nb_multiple_types * (L2 + 1);
	nb_eqns_packing = 0;
	if (f_use_packing_numbers) {
		for (j = 0; j < L2; j++) {
			len = col_classes_len[COL][j];
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
		

	T.D2->open(Nb_eqns, Nb_vars);
	T.D2->fill_coefficient_matrix_with(0);

#if 0
	if (Nb_vars == 0) {
		return TRUE;
		}
#endif

	if (f_v) {
		cout << "tdo_rows_setup_second_system: opening second system with " 
			<< Nb_eqns << " equations and " << Nb_vars << " variables" << endl;
		cout << "nb_eqns_joining=" << nb_eqns_joining << endl;
		cout << "nb_eqns_counting=" << nb_eqns_counting << endl;
		cout << "nb_eqns_packing=" << nb_eqns_packing << endl;
		cout << "l2=" << l2 << endl;
		cout << "L2=" << L2 << endl;
		cout << "T.nb_multiple_types=" << T.nb_multiple_types << endl;
		}

	if (!tdo_rows_setup_second_system_eqns_joining(verbose_level, 
		T, P, 
		f_omit, omit, f_dual_is_linear_space, 
		point_types, nb_point_types, 
		0 /*eqn_offset*/)) {
		if (f_v) {
			T.D2->print();
			}
		return FALSE;
		}
	if (!tdo_rows_setup_second_system_eqns_counting(verbose_level, 
		T, P, 
		f_omit, omit, 
		point_types, nb_point_types, 
		nb_eqns_joining /*eqn_offset*/)) {
		if (f_v) {
			T.D2->print();
			}
		return FALSE;
		}
	if (f_use_packing_numbers) {
		if (!tdo_rows_setup_second_system_eqns_packing(verbose_level, 
			T, P, 
			f_omit, omit, 
			point_types, nb_point_types,
			nb_eqns_joining + nb_eqns_counting /* eqn_start */, nb_eqns_used)) {
			if (f_v) {
				T.D2->print();
				}
			return FALSE;
			}
		}
	Nb_eqns = nb_eqns_joining + nb_eqns_counting + nb_eqns_used;
	T.D2->m = Nb_eqns;

	T.D2->eliminate_zero_rows_quick(verbose_level);


	
	if (f_v) {
		cout << "tdo_rows_setup_second_system finished" << endl;
		}
	if (f_vv) {
		T.D2->print();
		}
	return TRUE;
}

INT tdo_scheme::tdo_rows_setup_second_system_eqns_joining(INT verbose_level, 
	tdo_data &T, partitionstack &P, 
	INT f_omit, INT omit, INT f_dual_is_linear_space, 
	INT *point_types, INT nb_point_types, 
	INT eqn_offset)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT l2, I1, I2, k, b, ab, i, j, r, I, J, f, l, c, a, a2, rr, p, u, h, L1, L2;
	BYTE label[100];
	
	if (f_v) {
		cout << "tdo_scheme::tdo_rows_setup_second_system_eqns_joining" << endl;
		}
	l2 = nb_col_classes[COL];
	row_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);

	if (f_v) {
		cout << "l2 = " << l2 << endl;
		cout << "L2 = " << L2 << endl;
		cout << "eqn_offset = " << eqn_offset << endl;
		cout << "T.nb_multiple_types = " << T.nb_multiple_types << endl;
		}

	for (I = 0; I < L2; I++) {
		sprintf(label, "J_{%ld}", I + 1);
		T.D2->init_eqn_label(eqn_offset + I, label);
		}
	for (I1 = 0; I1 < L2; I1++) {
		for (I2 = I1 + 1; I2 < L2; I2++) {
			k = ij2k(I1, I2, L2);
			sprintf(label, "J_{%ld,%ld}", I1 + 1, I2 + 1);
			T.D2->init_eqn_label(eqn_offset + L2 + k, label);
			}
		}
	if (f_v) {
		cout << "filling coefficient matrix" << endl;
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
				a2 = binomial2(a);
				T.D2->Aij(eqn_offset + I, J) = a2;
				}
			for (I1 = 0; I1 < L2; I1++) {
				for (I2 = I1 + 1; I2 < L2; I2++) {
					k = ij2k(I1, I2, L2);
					a = point_types[c * L2 + I1];
					b = point_types[c * L2 + I2];
					ab = a * b;
					T.D2->Aij(eqn_offset + L2 + k, J) = ab;
					}
				}
			}
		}
		
	if (f_v) {
		cout << "filling RHS" << endl;
		}
	for (I = 0; I < L2; I++) {
		a = col_classes_len[COL][I];
		a2 = binomial2(a);
		T.D2->RHS[eqn_offset + I] = a2;
		if (f_dual_is_linear_space) {
			T.D2->type[eqn_offset + I] = t_EQ;
			}
		else {
			T.D2->type[eqn_offset + I] = t_LE;
			}
		}
	for (I1 = 0; I1 < L2; I1++) {
		a = col_classes_len[COL][I1];
		for (I2 = I1 + 1; I2 < L2; I2++) {
			b = col_classes_len[COL][I2];
			k = ij2k(I1, I2, L2);
			T.D2->RHS[eqn_offset + L2 + k] = a * b;
			if (f_dual_is_linear_space) {
				T.D2->type[eqn_offset + L2 + k] = t_EQ;
				}
			else {
				T.D2->type[eqn_offset + L2 + k] = t_LE;
				}
			}
		}
	if (f_v) {
		cout << "subtracting contribution from one-type blocks:" << endl;
		}
	// now subtract the contribution from one-type blocks:
	for (h = 0; h < T.nb_only_one_type; h++) {
		rr = T.only_one_type[h];
		p = row_classes_len[ROW][rr];
		u = T.types_first[rr];
		for (I = 0; I < L2; I++) {
			a = point_types[u * L2 + I];
			a2 = binomial2(a);
			T.D2->RHS[eqn_offset + I] -= a2 * p;
			if (T.D2->RHS[eqn_offset + I] < 0) {
				if (f_vv) {
					cout << "tdo_rows_setup_second_system_eqns_joining: RHS is negative, no solution for the distribution" << endl;
					cout << "h=" << h << endl;
					cout << "rr=T.only_one_type[h]=" << rr << endl;
					cout << "p=row_classes_len[ROW][rr]=" << p << endl;
					cout << "u=T.types_first[rr]=" << T.types_first[rr] << endl;
					cout << "I=" << I << endl;
					cout << "a=point_types[u * L2 + I]=" << a << endl;
					cout << "a2=binomial2(a)=" << a2 << endl;
					cout << "T.D2->RHS[eqn_offset + I]=" << T.D2->RHS[eqn_offset + I] << endl;
					}
				return FALSE;
				}
			}
		for (I1 = 0; I1 < L2; I1++) {
			a = point_types[u * L2 + I1];
			for (I2 = I1 + 1; I2 < L2; I2++) {
				b = point_types[u * L2 + I2];
				k = ij2k(I1, I2, L2);
				ab = a * b * p;
				T.D2->RHS[eqn_offset + L2 + k] -= ab;
				if (T.D2->RHS[eqn_offset + L2 + k] < 0) {
					if (f_vv) {
						cout << "tdo_rows_setup_second_system_eqns_joining: RHS is negative, no solution for the distribution" << endl;
						cout << "h=" << h << endl;
						cout << "rr=T.only_one_type[h]=" << rr << endl;
						cout << "p=row_classes_len[ROW][rr]=" << p << endl;
						cout << "u=T.types_first[rr]=" << T.types_first[rr] << endl;
						cout << "I1=" << I1 << endl;
						cout << "I2=" << I2 << endl;
						cout << "k=" << k << endl;
						cout << "a=point_types[u * L2 + I1]=" << a << endl;
						cout << "b=point_types[u * L2 + I2]=" << b << endl;
						cout << "ab=" << ab << endl;
						cout << "T.D2->RHS[eqn_offset + L2 + k]=" << T.D2->RHS[eqn_offset + L2 + k] << endl;
						}
					return FALSE;
					}
				}
			}
		}
	return TRUE;
}

INT tdo_scheme::tdo_rows_setup_second_system_eqns_counting(INT verbose_level, 
	tdo_data &T, partitionstack &P, 
	INT f_omit, INT omit, 
	INT *point_types, INT nb_point_types, 
	INT eqn_offset)
{
	INT l2, b, i, j, r, I, J, f, l, c, a, S, s, L1, L2;
	BYTE label[100];
	//INT nb_vars = T.D1->n;
	
	l2 = nb_col_classes[COL];
	row_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);

	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			for (I = 0; I < L2; I++) {
				sprintf(label, "F_{%ld,%ld}", I+1, r+1);
				T.D2->init_eqn_label(eqn_offset + i * (L2 + 1) + I, label);
				}
			}
		sprintf(label, "F_{%ld}", r+1);
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
			a = the_col_scheme[r * nb_col_classes[COL] + I];
			b = col_classes_len[COL][I];
			T.D2->RHS[eqn_offset + i * (L2 + 1) + I] = a * b;
			}
		s = row_classes_len[ROW][r];
		T.D2->RHS[eqn_offset + i * (L2 + 1) + L2] = s;
		S += s;
		}
	
	T.D2->sum = S;
	//T.D2->f_x_max = TRUE;
	return TRUE;
}

INT tdo_scheme::tdo_rows_setup_second_system_eqns_packing(INT verbose_level, 
	tdo_data &T, partitionstack &P, 
	INT f_omit, INT omit, 
	INT *point_types, INT nb_point_types,
	INT eqn_start, INT &nb_eqns_used)
{
	INT f_v = (verbose_level >= 1);
	INT nb_eqns_packing;
	INT /*l2,*/ i, r, f, l, j, c, J, JJ, k, h, rr, p, u, a, len, f_used, L1, L2;
	BYTE label[100];
	//INT nb_vars = T.D1->n;
	
	//l2 = nb_col_classes[COL];
	row_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);

	nb_eqns_packing = 0;
	for (J = 0; J < L2; J++) {
		len = col_classes_len[COL][J];
		if (len <= 2)
			continue;
		for (k = 3; k <= len; k++) {
			f_used = FALSE;
			for (i = 0; i < T.nb_multiple_types; i++) {
				r = T.multiple_types[i];
				f = T.types_first[r];
				l = T.types_len[r];
				for (j = 0; j < l; j++) {
					c = f + j;
					a = point_types[c * L2 + J];
					if (a < k)
						continue;
					JJ = T.types_first2[i] + j;
					f_used = TRUE;
					T.D2->Aij(eqn_start + nb_eqns_packing, JJ) = 1;
					}
				} // next i
			if (f_used) {
				INT bound;
				bound = TDO_upper_bound(len, k);
				T.D2->RHS[eqn_start + nb_eqns_packing] = bound;
				T.D2->type[eqn_start + nb_eqns_packing] = t_LE;
				for (h = 0; h < T.nb_only_one_type; h++) {
					rr = T.only_one_type[h];
					p = row_classes_len[COL][rr];
					u = T.types_first[rr];
					a = point_types[u * L2 + J];
					if (a < k)
						continue;
					T.D2->RHS[eqn_start + nb_eqns_packing] -= p;
					if (T.D2->RHS[eqn_start + nb_eqns_packing] < 0) {
						if (f_v) {
							cout << "tdo_scheme::tdo_rows_setup_second_system_eqns_packing RHS < 0" << endl;
							}
						return FALSE;
						}
					}
				sprintf(label, "P_{%ld,%ld} \\,\\mbox{using}\\, P(%ld,%ld)=%ld", J + 1, k, len, k, bound);
				T.D2->init_eqn_label(eqn_start + nb_eqns_packing, label);
				if (f_v) {
					cout << "packing equation " << nb_eqns_packing << " J=" << J << " k=" << k << " len=" << len << endl;
					}
				nb_eqns_packing++;
				}
			} // next k
		}
	nb_eqns_used = nb_eqns_packing;
	if (f_v) {
		cout << "tdo_rows_setup_second_system_eqns_packing nb_eqns_used = " << nb_eqns_used << endl;
		}
	return TRUE;
}


