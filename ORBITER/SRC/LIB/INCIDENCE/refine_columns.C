// refine_columns.C
// Anton Betten
//
// started: December 2006
// moved away from inc_gen: 8/27/07
// separated out from refine: 1/24/08

#include "galois.h"
#include "incidence.h"

INT tdo_scheme::refine_columns(INT verbose_level, INT f_once, partitionstack &P,
	INT *&line_types, INT &nb_line_types, INT &line_type_len, 
	INT *&distributions, INT &nb_distributions, 
	INT &cnt_second_system, solution_file_data *Sol, 
	INT f_omit1, INT omit1, INT f_omit, INT omit, 
	INT f_D1_upper_bound_x0, INT D1_upper_bound_x0, 
	INT f_use_mckay_solver, 
	INT f_use_packing_numbers)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT f_easy;
	INT l1, l2, R;
	INT ret = FALSE;
	
	if (f_v) {
		cout << "refine_columns" << endl;
		cout << "f_omit1=" << f_omit1 << " omit1=" << omit1 << endl;
		cout << "f_omit=" << f_omit << " omit=" << omit << endl;
		cout << "f_use_packing_numbers=" << f_use_packing_numbers << endl;
		cout << "f_D1_upper_bound_x0=" << f_D1_upper_bound_x0 << endl;
		cout << "f_use_mckay_solver=" << f_use_mckay_solver << endl;
		}
	R = nb_col_classes[COL];
	l1 = nb_row_classes[COL];
	l2 = nb_row_classes[ROW];
	if (f_vv) {
		cout << "l1=" << l1 << endl;
		cout << "l2=" << l2 << endl;
		cout << "R=" << R << endl;
		}
	
	get_row_split_partition(0 /*verbose_level*/, P);
	if (f_vv) {
		cout << "row split partition: " << P << endl;
		}
	if (P.ht != l1) {
		cout << "P.ht != l1" << endl;
		}

	if ((R == 1) && (l1 == 1) && (the_col_scheme[0] == -1)) {
		f_easy = TRUE;
		if (FALSE) {
			cout << "easy mode" << endl;
			}
		}
	else {
		f_easy = FALSE;
		if (FALSE) {
			cout << "full mode" << endl;
			}
		}


	if (f_easy) {
		cout << "refine_cols_easy nyi" << endl;
		exit(1);
		
		}
	else {
		ret = refine_cols_hard(P, verbose_level - 1, f_once, 
			line_types, nb_line_types, line_type_len, 
			distributions, nb_distributions, cnt_second_system, Sol, 
			f_omit1, omit1, f_omit, omit, 
			f_D1_upper_bound_x0, D1_upper_bound_x0, 
			f_use_mckay_solver, 
			f_use_packing_numbers);
		}
	if (f_v) {
		cout << "refine_columns finished" << endl;
		}
	return ret;
}

INT tdo_scheme::refine_cols_hard(partitionstack &P, INT verbose_level, INT f_once,
	INT *&line_types, INT &nb_line_types, INT &line_type_len,  
	INT *&distributions, INT &nb_distributions, 
	INT &cnt_second_system, solution_file_data *Sol, 
	INT f_omit1, INT omit1, INT f_omit, INT omit, 
	INT f_D1_upper_bound_x0, INT D1_upper_bound_x0, 
	INT f_use_mckay_solver, 
	INT f_use_packing_numbers)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT nb_eqns, nb_vars;
	INT R, /*l1,*/ l2, L1, L2, r;
	INT line_types_allocated;
	INT nb_sol, nb_sol1, f_survive;
	{
	tdo_data T;
	INT i, j, u;

	if (f_v) {
		cout << "refine_cols_hard" << endl;
		cout << "f_omit1=" << f_omit1 << " omit1=" << omit1 << endl;
		cout << "f_omit=" << f_omit << " omit=" << omit << endl;
		cout << "f_use_packing_numbers=" << f_use_packing_numbers << endl;
		cout << "f_D1_upper_bound_x0=" << f_D1_upper_bound_x0 << endl;
		cout << "f_use_mckay_solver=" << f_use_mckay_solver << endl;
		}
	R = nb_col_classes[COL];
	//l1 = nb_row_classes[COL];
	l2 = nb_row_classes[ROW];

	if (f_v) {
		cout << "the_row_scheme is:" << endl;
		for (i = 0; i < l2; i++) {
			for (j = 0; j < R; j++) {
				cout << setw(4) << the_row_scheme[i * R + j];
				}
			cout << endl;
			}
		}

	column_refinement_L1_L2(P, f_omit1, omit1, L1, L2, verbose_level);

	T.allocate(R);
	
	T.types_first[0] = 0;
	
	line_types_allocated = 100;
	nb_line_types = 0;
	line_types = NEW_INT(line_types_allocated * l2);
	line_type_len = l2;
	
	T.nb_only_one_type = 0;
	T.nb_multiple_types = 0;
	
	
	for (r = 0; r < R; r++) {
		
		if (f_v) {
			cout << "r=" << r << endl;
			}
		
		if (!tdo_columns_setup_first_system(verbose_level, 
			T, r, P, 
			f_omit1, omit1, 
			line_types, nb_line_types)) {
			FREE_INT(line_types);
			return FALSE;
			}
		
		if (f_D1_upper_bound_x0) {
			T.D1->x_max[0] = D1_upper_bound_x0;
			cout << "setting upper bound for D1->x[0] to " << T.D1->x_max[0] << endl;
			} 
		// ATTENTION, this is from a specific problem on arcs in a plane (MARUTA)
		
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
		
		
		
		if (f_vv) {
			BYTE label[1000];
			
			sprintf(label, "first_%ld", r);
			T.D1->write_xml(cout, label);
			}
				
		nb_sol = T.solve_first_system(verbose_level - 1, 
			line_types, nb_line_types, line_types_allocated);

		if (f_v) {
			cout << "r = " << r << ", found " << nb_sol << " refined line types" << endl;
			}
		if (f_vv) {
			print_integer_matrix_width(cout, line_types + T.types_first[r] * L2, nb_sol, L2, L2, 2);
			}
		nb_sol1 = 0;
		for (u = 0; u < nb_sol; u++) {
			f_survive = TRUE;
			for (i = 0; i < L2; i++) {
				INT len1, len2, flags;
				len1 = row_classes_len[ROW][i];
				if (len1 > 1)
					continue;
				len2 = col_classes_len[ROW][r];
				flags = the_row_scheme[i * R + r];
				if (flags == len2) {
					if (line_types[(T.types_first[r] + u) * L2 + i] == 0) {
						f_survive = FALSE;
						if (f_vv) {
							cout << "line type " << u << " eliminated, line_types[] = 0" << endl;
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
				cout << "eliminated " << nb_sol - nb_sol1 << " types" << endl;
				}
			nb_sol = nb_sol1;
			nb_line_types = T.types_first[r] + nb_sol1;
			if (f_v) {
				cout << "r = " << r << ", found " << nb_sol << " refined line types" << endl;
				}
			}
		
		if (f_vv) {
			print_integer_matrix_width(cout, line_types + T.types_first[r] * L2, nb_sol, L2, L2, 2);
			}
		if (nb_sol == 0) {
			FREE_INT(line_types);
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
	if (f_v) {
		cout << "R=" << R << endl;
		cout << "r : T.types_first[r] : T.types_len[r]" << endl;
		for (r = 0; r < R; r++) {
			cout << r << " : " << T.types_first[r] << " : " << T.types_len[r] << endl;
			}
		}
	if (f_vv) {
		print_integer_matrix_width(cout, line_types, nb_line_types, line_type_len, line_type_len, 3);
		}
	
	// now we compute the distributions:
	//
	INT f_scale = FALSE;
	INT scaling = 0;
	
	if (!tdo_columns_setup_second_system(verbose_level, 
		T, P, 
		f_omit1, omit1, 
		f_use_packing_numbers, 
		line_types, nb_line_types)) {
		FREE_INT(line_types);
		return FALSE;
		}


	// ATTENTION, this is for the classification of (42,6)_8 arcs where a_1 = 0 (MARUTA)
		
	//T.D2->x_max[5] = 0; // a_1 is known to be zero
		
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		



	if (f_vv) {
		BYTE label[1000];
			
		sprintf(label, "second");
		T.D2->write_xml(cout, label);
		}




	INT idx, /*f,*/ l;
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
		cout << "refine_cols_hard: solving second system " << cnt_second_system << " which is " << T.D2->m << " x " << T.D2->n << endl;
		cout << "variable blocks:" << endl;
		cout << "i : r : col_classes_len[COL][r] : types_first2[i] : types_len[r]" << endl;
		INT f, l;
		for (i = 0; i < T.nb_multiple_types; i++) {
			r = T.multiple_types[i];
			f = T.types_first2[i];
			l = T.types_len[r];
			cout << i << " : " << r << " : " << setw(3) << col_classes_len[COL][r] << " : " << setw(3) << f << " : " << setw(3) << l << endl;
			}
		}

	if (f_omit) {
		T.solve_second_system_omit(verbose_level, 
			col_classes_len[COL], 
			line_types, nb_line_types, distributions, nb_distributions, 
			omit);
		}
	else {
		T.solve_second_system_with_help(verbose_level, 
			f_use_mckay_solver, f_once, 
			col_classes_len[COL], f_scale, scaling, 
			line_types, nb_line_types, distributions, nb_distributions, 
			cnt_second_system, Sol);
		}

	if (f_v) {
		cout << "refine_cols_hard: second system " << cnt_second_system << " found " << nb_distributions << " distributions." << endl;
		}

#if 0
	// ATTENTION: this is from a specific problem of CHEON !!!!

	cout << "ATTENTION, we are running specific code for a problem of Cheon" << endl;
	INT cnt, h, x0;
	
	cnt = 0;
	for (h = 0; h < nb_distributions; h++) {
		x0 = distributions[h * nb_line_types + 0];
		if (x0 == 12) {
			for (j = 0; j < nb_line_types; j++) {
				distributions[cnt * nb_line_types + j] = distributions[h * nb_line_types + j];
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
	}
	if (f_v) {
		cout << "refine_cols_hard: after closing T." << endl;
		}
	return TRUE;
}

void tdo_scheme::column_refinement_L1_L2(partitionstack &P, INT f_omit, INT omit, 
	INT &L1, INT &L2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT l1, l2, omit2, i;
	l1 = nb_row_classes[COL];
	l2 = nb_row_classes[ROW]; // the finer scheme

	omit2 = 0;
	if (f_omit) {
		for (i = l1 - omit; i < l1; i++) {
			omit2 += P.cellSize[i];
			}
		}
	L1 = l1 - omit;
	L2 = l2 - omit2;
	if (f_v) {
		cout << "column_refinement_L1_L2: l1 = " << l1 << " l2=" << l2 << " L1=" << L1 << " L2=" << L2 << endl;
		}
}

INT tdo_scheme::tdo_columns_setup_first_system(INT verbose_level, 
	tdo_data &T, INT r, partitionstack &P, 
	INT f_omit, INT omit, 
	INT *&line_types, INT &nb_line_types)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, f, l, I, J, rr, R, S, a, a2, s, /*l1, l2,*/ L1, L2;
	INT h, u, d, d2, o, e, p, eqn_number, nb_vars, nb_eqns;
	
	// create all partitions which are refined line types

	if (!f_omit)
		omit = 0;

	if (f_v) {
		cout << "tdo_columns_setup_first_system r=" << r << endl;
		if (f_omit) {
			cout << "omit=" << omit << endl;
			}
		}
		
	R = nb_col_classes[COL];
	//l1 = nb_row_classes[COL];
	//l2 = nb_row_classes[ROW]; // the finer scheme

	column_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);

	nb_vars = L2;
	nb_eqns = L1 + 1 + (R - 1);
		
	
	T.D1->open(nb_eqns, nb_vars);
	S = 0;
		
	for (I = 0; I < nb_eqns; I++) 
		for (J = 0; J < nb_vars; J++) 
			T.D1->A[I * nb_vars + J] = 0;
	
	// the m equalities that come from the fact that the new type 
	// is a partition of the old type.
	
	// we are in the r-th column class (r is given)
	
	for (I = 0; I < L1; I++) {
		f = P.startCell[I];
		l = P.cellSize[I];
		for (j = 0; j < l; j++) {
			J = f + j;
			T.D1->Aij(I, J) = 1;
			a = the_row_scheme[J * R + r];
			if (a == 0)
				T.D1->x_max[J] = 0;
			else
				T.D1->x_max[J] = row_classes_len[ROW][J];
			}
		s = the_col_scheme[I * R + r];
		T.D1->RHS[I] = s;
		T.D1->type[I] = t_EQ;
		S += s;
		}
	
	eqn_number = L1;
	
	for (i = 0; i < L2; i++) {
		a = minus_one_if_positive(the_row_scheme[i * R + r]);
		T.D1->Aij(eqn_number, i) = a;
		}
	T.D1->RHS[eqn_number] = col_classes_len[ROW][r] - 1; // the -1 was missing!!!
	T.D1->type[eqn_number] = t_LE;
	eqn_number++;
	
	for (j = 0; j < R; j++) {
		if (j == r)
			continue;
		for (i = 0; i < L2; i++) {
			a = the_row_scheme[i * R + j];
			T.D1->Aij(eqn_number, i) = a;
			}
		T.D1->RHS[eqn_number] = col_classes_len[ROW][j];
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
			if (T.D1->x_max[j] == 0)
				continue;
			d = row_classes_len[ROW][j];
			p = col_classes_len[COL][rr];
			d2 = binomial2(d);
			a = line_types[u * nb_vars + j];
			a2 = binomial2(a);
			o = d2 - a2 * p;
			if (o < 0) {
				if (f_vv) {
					cout << "only one type, but no solution because of joining in row-class " << j << endl;
					//cout << "u=" << u << " j=" << j << endl;
					}
				return FALSE;
				}
			e = largest_binomial2_below(o);
			T.D1->x_max[j] = MINIMUM(T.D1->x_max[j], e);
			}
		}

		
	T.D1->sum = S;
	T.D1->f_x_max = TRUE;

	T.D1->eliminate_zero_rows_quick(verbose_level);
		
	if (f_v) {
		cout << "tdo_columns_setup_first_system r=" << r << " finished" << endl;
		}
	if (f_vv) {
		T.D1->print();
		}
	return TRUE;
}	

INT tdo_scheme::tdo_columns_setup_second_system(INT verbose_level,
	tdo_data &T, partitionstack &P, 
	INT f_omit, INT omit,  
	INT f_use_packing_numbers, 
	INT *&line_types, INT &nb_line_types)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, len, r, I, J, L1, L2, Nb_eqns, Nb_vars;

	if (f_v) {
		cout << "tdo_columns_setup_second_system" << endl;
		cout << "f_use_packing_numbers=" << f_use_packing_numbers << endl;
		}
		
	INT nb_eqns_joining, nb_eqns_counting, nb_eqns_upper_bound, nb_eqns_used;
	INT l2;
	
	l2 = nb_row_classes[ROW]; // the finer scheme

	column_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);
	
	nb_eqns_joining = L2 + binomial2(L2);
	nb_eqns_counting = T.nb_multiple_types * (L2 + 1);
	nb_eqns_upper_bound = 0;
	if (f_use_packing_numbers) {
		for (i = 0; i < l2; i++) {
			len = row_classes_len[ROW][i];
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

	T.D2->open(Nb_eqns, Nb_vars);
	if (f_v) {
		cout << "tdo_columns_setup_second_system: opening second system with " 
			<< Nb_eqns << " equations and " << Nb_vars << " variables" << endl;
		}
	if (f_vv) {
		cout << "l2=" << l2 << endl;
		cout << "L2=" << L2 << endl;
		cout << "nb_eqns_joining=" << nb_eqns_joining << endl;
		cout << "nb_eqns_counting=" << nb_eqns_counting << endl;
		cout << "nb_eqns_upper_bound=" << nb_eqns_upper_bound << endl;
		cout << "T.nb_multiple_types=" << T.nb_multiple_types << endl;
		cout << "i : r = T.multiple_types[i] : T.types_first2[i] : T.types_len[r]" << endl;
		for (i = 0; i < T.nb_multiple_types; i++) {
			r = T.multiple_types[i];
			cout << i << " : " << r << " : " << T.types_first2[i] << " : " << T.types_len[r] << endl;
			}
		}

	for (I = 0; I < Nb_eqns; I++) 
		for (J = 0; J < Nb_vars; J++) 
			T.D2->A[I * Nb_vars + J] = 0;

	if (!tdo_columns_setup_second_system_eqns_joining(verbose_level, 
		T, P, 
		f_omit, omit, 
		line_types, nb_line_types,
		0 /*eqn_start*/)) {
		if (f_v) {
			T.D2->print();
			}
		return FALSE;
		}
	tdo_columns_setup_second_system_eqns_counting(verbose_level, 
		T, P, 
		f_omit, omit, 
		line_types, nb_line_types,
		nb_eqns_joining /* eqn_start */);
	if (f_use_packing_numbers) {
		if (!tdo_columns_setup_second_system_eqns_upper_bound(verbose_level, 
			T, P, 
			f_omit, omit, 
			line_types, nb_line_types,
			nb_eqns_joining + nb_eqns_counting /* eqn_start */, nb_eqns_used)) {
			if (f_v) {
				T.D2->print();
				}
			return FALSE;
			}
		}
	
	
	
	T.D2->eliminate_zero_rows_quick(verbose_level);
	
	
	if (f_v) {
		cout << "tdo_columns_setup_second_system finished" << endl;
		}
	if (f_vv) {	
		cout << "tdo_columns_setup_second_system, The second system is" << endl;
		T.D2->print();
		}
	return TRUE;
}

INT tdo_scheme::tdo_columns_setup_second_system_eqns_joining(INT verbose_level,
	tdo_data &T, partitionstack &P, 
	INT f_omit, INT omit, 
	INT *line_types, INT nb_line_types,
	INT eqn_start)
{
	INT f_v = (verbose_level >= 1);
	INT l2, L1, L2, i, r, f, l, j, c, J, I, I1, I2, a, b, ab, a2, k, h, rr, p, u;
	BYTE label[100];
	
	l2 = nb_row_classes[ROW];
	column_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);
	
	for (I = 0; I < L2; I++) {
		sprintf(label, "J_{%ld}", I + 1);
		T.D2->init_eqn_label(eqn_start + I, label);
		}
	for (I1 = 0; I1 < L2; I1++) {
		for (I2 = I1 + 1; I2 < L2; I2++) {
			k = ij2k(I1, I2, L2);
			sprintf(label, "J_{%ld,%ld}", I1 + 1, I2 + 1);
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
				a2 = binomial2(a);
				T.D2->Aij(eqn_start + I, J) = a2;
				}
			for (I1 = 0; I1 < L2; I1++) {
				for (I2 = I1 + 1; I2 < L2; I2++) {
					k = ij2k(I1, I2, L2);
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
		a = row_classes_len[ROW][I];
		a2 = binomial2(a);
		T.D2->RHS[eqn_start + I] = a2;
		}
	for (I1 = 0; I1 < L2; I1++) {
		a = row_classes_len[ROW][I1];
		for (I2 = I1 + 1; I2 < L2; I2++) {
			b = row_classes_len[ROW][I2];
			k = ij2k(I1, I2, L2);
			T.D2->RHS[eqn_start + l2 + k] = a * b;
			}
		}
	
	// now subtract the contribution from one-type blocks:
	for (h = 0; h < T.nb_only_one_type; h++) {
		rr = T.only_one_type[h];
		p = col_classes_len[COL][rr];
		u = T.types_first[rr];
		for (I = 0; I < L2; I++) {
			a = line_types[u * L2 + I];
			a2 = binomial2(a);
			T.D2->RHS[eqn_start + I] -= a2 * p;
			if (T.D2->RHS[eqn_start + I] < 0) {
				if (f_v) {
					cout << "tdo_columns_setup_second_system_eqns_joining: RHS is negative, no solution for the distribution" << endl;
					}
				return FALSE;
				}
			}
		for (I1 = 0; I1 < L2; I1++) {
			a = line_types[u * L2 + I1];
			for (I2 = I1 + 1; I2 < L2; I2++) {
				b = line_types[u * L2 + I2];
				k = ij2k(I1, I2, L2);
				ab = a * b * p;
				T.D2->RHS[eqn_start + L2 + k] -= ab;
				if (T.D2->RHS[eqn_start + L2 + k] < 0) {
					if (f_v) {
						cout << "tdo_columns_setup_second_system_eqns_joining: RHS is negative, no solution for the distribution" << endl;
						}
					return FALSE;
					}
				}
			}
		}
	return TRUE;
}

void tdo_scheme::tdo_columns_setup_second_system_eqns_counting(INT verbose_level,
	tdo_data &T, partitionstack &P, 
	INT f_omit, INT omit, 
	INT *line_types, INT nb_line_types,
	INT eqn_start)
{
	INT /*l2,*/ L1, L2, i, r, f, l, j, c, J, I, a, b, S, s;
	BYTE label[100];

	//l2 = nb_row_classes[ROW];
	column_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);

	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		f = T.types_first[r];
		l = T.types_len[r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			for (I = 0; I < L2; I++) {
				sprintf(label, "F_{%ld,%ld}", r + 1, I + 1);
				T.D2->init_eqn_label(eqn_start + i * (L2 + 1) + I, label);
				}
			}
		sprintf(label, "F_{%ld}", r + 1);
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
		s = col_classes_len[ROW][r];
		for (j = 0; j < l; j++) {
			c = f + j;
			J = T.types_first2[i] + j;
			T.D2->x_max[J] = s;
			}
		}

	// prepare RHS:
	
	S = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		for (I = 0; I < L2; I++) {
			a = the_row_scheme[I * nb_col_classes[ROW] + r];
			b = row_classes_len[ROW][I];
			T.D2->RHS[eqn_start + i * (L2 + 1) + I] = a * b;
			}
		s = col_classes_len[ROW][r];
		T.D2->RHS[eqn_start + i * (L2 + 1) + L2] = s;
		S += s;
		}

	T.D2->sum = S;
	T.D2->f_x_max = TRUE;
}

INT tdo_scheme::tdo_columns_setup_second_system_eqns_upper_bound(INT verbose_level, 
	tdo_data &T, partitionstack &P, 
	INT f_omit, INT omit, 
	INT *line_types, INT nb_line_types,
	INT eqn_start, INT &nb_eqns_used)
{
	INT f_v = (verbose_level >= 1);
	INT nb_eqns_packing;
	INT /*l2,*/ L1, L2, i, r, f, l, j, c, J, I, k, h, rr, p, u, a, len, f_used;
	BYTE label[100];
	
	nb_eqns_packing = 0;
	//l2 = nb_row_classes[ROW];
	column_refinement_L1_L2(P, f_omit, omit, L1, L2, verbose_level);
	for (I = 0; I < L2; I++) {
		len = row_classes_len[ROW][I];
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
					J = T.types_first2[i] + j;
					a = line_types[c * L2 + I];
					if (a < k)
						continue;
					f_used = TRUE;
					T.D2->Aij(eqn_start + nb_eqns_packing, J) = 1;
					}
				} // next i
			if (f_used) {
				INT bound;
				
				bound = TDO_upper_bound(len, k);
				T.D2->RHS[eqn_start + nb_eqns_packing] = bound;
				T.D2->type[eqn_start + nb_eqns_packing] = t_LE;
				for (h = 0; h < T.nb_only_one_type; h++) {
					rr = T.only_one_type[h];
					p = col_classes_len[ROW][rr];
					u = T.types_first[rr];
					a = line_types[u * L2 + I];
					if (a < k)
						continue;
					T.D2->RHS[eqn_start + nb_eqns_packing] -= p;
					if (T.D2->RHS[eqn_start + nb_eqns_packing] < 0) {
						if (f_v) {
							cout << "tdo_scheme::tdo_columns_setup_second_system_eqns_upper_bound RHS < 0" << endl;
							}
						return FALSE;
						}
					}
				sprintf(label, "P_{%ld,%ld} \\,\\mbox{using}\\, P(%ld,%ld)=%ld", I + 1, k, len, k, bound);
				T.D2->init_eqn_label(eqn_start + nb_eqns_packing, label);
				nb_eqns_packing++;
				}
			} // next k
		}
	nb_eqns_used = nb_eqns_packing;
	if (f_v) {
		cout << "tdo_columns_setup_second_system_eqns_upper_bound nb_eqns_used = " << nb_eqns_used << endl;
		}
	return TRUE;
}

