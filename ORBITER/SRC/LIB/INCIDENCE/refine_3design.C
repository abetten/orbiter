// refine_3design.C
// Anton Betten
//
// started: January 17, 2007

#include "galois.h"
#include "incidence.h"

// #################################################################################
// TDO parameter refinement for 3-designs - row refinement
// #################################################################################


INT tdo_scheme::td3_refine_rows(INT verbose_level, INT f_once,
	INT lambda3, INT block_size,
	INT *&point_types, INT &nb_point_types, INT &point_type_len,  
	INT *&distributions, INT &nb_distributions)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
    INT R, l1, l2, r;
    INT nb_eqns, nb_vars = 0;
	INT point_types_allocated;
	partitionstack P;
	INT lambda2;
	INT nb_points;
	INT nb_sol;
	tdo_data T;

	if (f_v) {
		cout << "td3_refine_rows" << endl;
		}
	nb_points = m;
	lambda2 = lambda3 * (nb_points - 2) / (block_size - 2);
	if (f_v) {
		cout << "nb_points = " << nb_points << " lambda2 = " << lambda2 << endl;
		}
	if ((block_size - 2) * lambda2 != lambda3 * (nb_points - 2)) {
		cout << "parameters are wrong" << endl;
		exit(1);
		}

	get_column_split_partition(verbose_level, P);
	
	R = nb_row_classes[ROW];
	l1 = nb_col_classes[ROW];
	l2 = nb_col_classes[COL];

	
	T.allocate(R);
	
	T.types_first[0] = 0;
	
	point_types_allocated = 100;
	nb_point_types = 0;
	point_types = new INT[point_types_allocated * l2];
	point_type_len = l2;
	
	T.nb_only_one_type = 0;
	T.nb_multiple_types = 0;
	
	for (r = 0; r < R; r++) {
		
		if (f_vvv) {
			cout << "r=" << r << endl;
			}
		if (!td3_rows_setup_first_system(verbose_level - 1, 
			lambda3, block_size, lambda2, 
			T, r, P, 
			nb_vars, nb_eqns, 
			point_types, nb_point_types)) {
			delete [] point_types;
			return FALSE;
			}
		
		nb_sol = T.solve_first_system(verbose_level - 1, 
			point_types, nb_point_types, point_types_allocated);

		if (f_vv) {
			cout << "r = " << r << ", found " << nb_sol << " refined point types" << endl;
			}
		if (nb_sol == 0) {
			delete [] point_types;
			return FALSE;
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
		
		T.D1->freeself();
		//diophant_close(T.D1);
		//T.D1 = NULL;
		
		} // next r
	
	
	// now we compute the distributions:
	//
	INT Nb_vars, Nb_eqns;

	if (!td3_rows_setup_second_system(verbose_level, 
		lambda3, block_size, lambda2, 
		T, 
		nb_vars, Nb_vars, Nb_eqns, 
		point_types, nb_point_types)) {
		delete [] point_types;
		return FALSE;
		}
	
	if (Nb_vars == 0) {
		INT h, r, u;
		
		distributions = new INT[1 * nb_point_types];
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

	int f_scale = FALSE;
	int scaling = 0;
	
	T.solve_second_system(verbose_level - 1, FALSE /* f_use_mckay */,f_once, 
		row_classes_len[ROW], f_scale, scaling, 
		point_types, nb_point_types, distributions, nb_distributions);


	if (f_v) {
		cout << "td3_refine_rows: found " << nb_distributions << " distributions." << endl;
		}
	return TRUE;
}

INT tdo_scheme::td3_rows_setup_first_system(INT verbose_level,
	INT lambda3, INT block_size, INT lambda2,
	tdo_data &T, INT r, partitionstack &P,
	INT &nb_vars, INT &nb_eqns,
	INT *&point_types, INT &nb_point_types)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT i, j, R, l1, l2, r2, r3, S, I, J, f, l, s;
	INT eqn_offset, eqn_cnt;
	
	if (f_v) {
		cout << "td3_rows_setup_first_system r=" << r << endl;
		}
		

	// create all partitions which are refined point types of points in block r

	R = nb_row_classes[ROW];
	l1 = nb_col_classes[ROW];
	l2 = nb_col_classes[COL];
		
	nb_vars = l2;
	nb_eqns = R + R + (R - 1) + (((R - 1) * (R - 2)) >> 1) + l1;
		
	
	T.D1->open(nb_eqns, nb_vars);
	S = 0;
		
	for (I = 0; I < nb_eqns; I++) 
		for (J = 0; J < nb_vars; J++) 
			T.D1->A[I * nb_vars + J] = 0;
	for (I = 0; I < nb_eqns; I++) 
		T.D1->RHS[I] = 9999;
	
	// pair joinings
	for (r2 = 0; r2 < R; r2++) {
		if (r2 == r) {
			// connections within the same row-partition
			for (J = 0; J < nb_vars; J++) {
				T.D1->A[r2 * nb_vars + J] = minus_one_if_positive(the_col_scheme[r2 * l2 + J]);
				}
			T.D1->RHS[r2] = (row_classes_len[ROW][r2] - 1) * lambda2;
			}
		else {
			// connections to the point from different row-partitions
			for (J = 0; J < nb_vars; J++) {
				T.D1->A[r2 * nb_vars + J] = the_col_scheme[r2 * l2 + J];
				}
			T.D1->RHS[r2] = row_classes_len[ROW][r2] * lambda2;
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
				T.D1->A[(eqn_offset + r2) * nb_vars + J] = binomial2(minus_one_if_positive(the_col_scheme[r2 * l2 + J]));
				}
			T.D1->RHS[eqn_offset + r2] = binomial2((row_classes_len[ROW][r2] - 1)) * lambda3;
			}
		else {
			// connections to pairs with one in the same and one in the other part
			for (J = 0; J < nb_vars; J++) {
				T.D1->A[(eqn_offset + r2) * nb_vars + J] = minus_one_if_positive(the_col_scheme[r * l2 + J]) * the_col_scheme[r2 * l2 + J];
				}
			T.D1->RHS[eqn_offset + r2] = (row_classes_len[ROW][r] - 1) * row_classes_len[ROW][r2] * lambda3;
			}
		}
	if (f_vv) {
		cout << "r=" << r << " after triple joining, the system is" << endl;
		T.D1->print();
		}
	
	eqn_offset += R;
	eqn_cnt = 0;
	for (r2 = 0; r2 < R; r2++) {
		if (r2 == r)
			continue;
		// connections to pairs from one different row-partition
		for (J = 0; J < nb_vars; J++) {
			T.D1->A[(eqn_offset + eqn_cnt) * nb_vars + J] = binomial2(the_col_scheme[r2 * l2 + J]);
			}
		T.D1->RHS[eqn_offset + eqn_cnt] = binomial2(row_classes_len[ROW][r2]) * lambda3;
		eqn_cnt++;
		}
	if (f_vv) {
		cout << "r=" << r << " after connections to pairs from one different row-partition, the system is" << endl;
		T.D1->print();
		}

	eqn_offset += (R - 1);
	eqn_cnt = 0;
	for (r2 = 0; r2 < R; r2++) {
		if (r2 == r)
			continue;
		for (r3 = r2 + 1; r3 < R; r3++) {
			if (r3 == r)
				continue;
			// connections to pairs from two different row-partitions
			for (J = 0; J < nb_vars; J++) {
				T.D1->A[(eqn_offset + eqn_cnt) * nb_vars + J] = the_col_scheme[r2 * l2 + J] * the_col_scheme[r3 * l2 + J];
				}
			T.D1->RHS[eqn_offset + eqn_cnt] = row_classes_len[ROW][r2] * row_classes_len[ROW][r3] * lambda3;
			eqn_cnt++;
			}
		}
	eqn_offset += eqn_cnt;
	if (f_vv) {
		cout << "r=" << r << " after connections to pairs from two different row-partitions, the system is" << endl;
		T.D1->print();
		}
	
	S = 0;
	for (i = 0; i < l1; i++) {
		s = the_row_scheme[r * l1 + i];
		if (f_vvv) {
			cout << "r=" << r << " i=" << i << " s=" << s << endl;
			}
		T.D1->RHS[eqn_offset + i] = s;
		S += s;
		f = P.startCell[i];
		l = P.cellSize[i];
		if (f_vvv) {
			cout << "f=" << f << " l=" << l << endl;
			}
			
		for (j = 0; j < l; j++) {
			T.D1->A[(eqn_offset + i) * nb_vars + f + j] = 1;
			T.D1->x_max[f + j] = s;
			}
		}
	if (f_vv) {
		cout << "r=" << r << " after adding extra equations, the system is" << endl;
		T.D1->print();
		}
	
	T.D1->sum = S;
	T.D1->f_x_max = TRUE;
		
		
	if (f_vv) {
		cout << "r=" << r << " the system is" << endl;
		T.D1->print();
		}
	return TRUE;
}		

INT tdo_scheme::td3_rows_setup_second_system(INT verbose_level,
	INT lambda3, INT block_size, INT lambda2,
	tdo_data &T, 
	INT nb_vars, INT &Nb_vars, INT &Nb_eqns, 
	INT *&point_types, INT &nb_point_types)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	INT l2, i, r, I, J, nb_eqns_counting;
	INT S;
	
	l2 = nb_col_classes[COL];
	
	nb_eqns_counting = T.nb_multiple_types * (l2 + 1);
	Nb_eqns = nb_eqns_counting;
	Nb_vars = 0;
	for (i = 0; i < T.nb_multiple_types; i++) {
		r = T.multiple_types[i];
		T.types_first2[i] = Nb_vars;
		Nb_vars += T.types_len[r];
		}


	T.D2->open(Nb_eqns, Nb_vars);
	if (f_v) {
		cout << "td3_rows_setup_second_system: opening second system with " 
			<< Nb_eqns << " equations and " << Nb_vars << " variables" << endl;
		}

	for (I = 0; I < Nb_eqns; I++) 
		for (J = 0; J < Nb_vars; J++) 
			T.D2->A[I * Nb_vars + J] = 0;
	for (I = 0; I < Nb_eqns; I++) 
		T.D2->RHS[I] = 9999;


	if (!td3_rows_counting_flags(verbose_level, 
		lambda3, block_size, lambda2, S, 
		T, 
		nb_vars, Nb_vars, 
		point_types, nb_point_types, 0)) {
		return FALSE;
		}
	

	T.D2->sum = S;
	

	if (f_vv) {	
		cout << "The second system is" << endl;
	
		T.D2->print();
		}
	
	return TRUE;

}

INT tdo_scheme::td3_rows_counting_flags(INT verbose_level,
	INT lambda3, INT block_size, INT lambda2, INT &S,
	tdo_data &T, 
	INT nb_vars, INT Nb_vars, 
	INT *&point_types, INT &nb_point_types, INT eqn_offset)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT I, i, r, f, l, j, c, J, a, b, rr, p, u, l2, h, s;

	l2 = nb_col_classes[COL];
	
	if (f_v) {
		cout << "td3_rows_counting_flags: eqn_offset=" << eqn_offset 
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
			a = the_col_scheme[r * nb_col_classes[COL] + I];
			b = col_classes_len[COL][I];
			T.D2->RHS[eqn_offset + i * (l2 + 1) + I] = a * b;
			for (h = 0; h < T.nb_only_one_type; h++) {
				rr = T.only_one_type[h];
				p = col_classes_len[COL][rr];
				u = T.types_first[rr];
				a = point_types[u * nb_vars + I];
				T.D2->RHS[eqn_offset + i * (l2 + 1) + I] -= a * p;
				if (T.D2->RHS[eqn_offset + i * (l2 + 1) + I] < 0) {
					if (f_v) {
						cout << "td3_rows_counting_flags: RHS[nb_eqns_joining + i * (l2 + 1) + I] is negative, no solution for the distribution" << endl;
						}
					return FALSE;
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
		
		s = row_classes_len[COL][r];
		T.D2->RHS[eqn_offset + i * (l2 + 1) + l2] = s;
		S += s;
		}
	if (f_vvv) {
		cout << "td3_rows_counting_flags, the system is" << endl;
		T.D2->print();
		}

	
	return TRUE;
}

// #################################################################################
// TDO parameter refinement for 3-designs - column refinement
// #################################################################################



INT tdo_scheme::td3_refine_columns(INT verbose_level, INT f_once,
	INT lambda3, INT block_size, INT f_scale, INT scaling,
	INT *&line_types, INT &nb_line_types, INT &line_type_len,  
	INT *&distributions, INT &nb_distributions)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT R, l1, l2, r, nb_eqns, nb_vars = 0;
	INT line_types_allocated;
	partitionstack P;
	INT lambda2;
	INT nb_points;
	INT nb_sol;
	tdo_data T;

	if (f_v) {
		cout << "td3_refine_columns" << endl;
		}
	nb_points = m;
	lambda2 = lambda3 * (nb_points - 2) / (block_size - 2);
	if (f_v) {
		cout << "nb_points = " << nb_points << " lambda2 = " << lambda2 << endl;
		}
	if ((block_size - 2) * lambda2 != lambda3 * (nb_points - 2)) {
		cout << "parameter are wrong" << endl;
		exit(1);
		}

	get_row_split_partition(verbose_level, P);
	
	R = nb_col_classes[COL];
	l1 = nb_row_classes[COL];
	l2 = nb_row_classes[ROW];
	
	T.allocate(R);
	
	T.types_first[0] = 0;
	
	line_types_allocated = 100;
	nb_line_types = 0;
	line_types = new INT[line_types_allocated * l2];
	line_type_len = l2;
	
	T.nb_only_one_type = 0;
	T.nb_multiple_types = 0;
	
	for (r = 0; r < R; r++) {
		
		if (f_vvv) {
			cout << "r=" << r << endl;
			}
		if (!td3_columns_setup_first_system(verbose_level - 1, 
			lambda3, block_size, lambda2, 
			T, r, P, 
			nb_vars, nb_eqns, 
			line_types, nb_line_types)) {
			delete [] line_types;
			return FALSE;
			}
		
		nb_sol = T.solve_first_system(verbose_level - 1, 
			line_types, nb_line_types, line_types_allocated);

		if (f_vv) {
			cout << "r = " << r << ", found " << nb_sol << " refine line types" << endl;
			}
		if (nb_sol == 0) {
			delete [] line_types;
			return FALSE;
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
		
		T.D1->freeself();
		//diophant_close(T.D1);
		//T.D1 = NULL;
		
		} // next r
	
	
	// now we compute the distributions:
	//
	INT Nb_vars, Nb_eqns;

	if (!td3_columns_setup_second_system(verbose_level, 
		lambda3, block_size, lambda2, f_scale, scaling, 
		T, 
		nb_vars, Nb_vars, Nb_eqns,
		line_types, nb_line_types)) {
		delete [] line_types;
		return FALSE;
		}

	T.solve_second_system(verbose_level - 1, FALSE /* f_use_mckay */, f_once, 
		col_classes_len[COL], f_scale, scaling, 
		line_types, nb_line_types, distributions, nb_distributions);


	if (f_v) {
		cout << "td3_refine_columns: found " << nb_distributions << " distributions." << endl;
		}
	return TRUE;
}

INT tdo_scheme::td3_columns_setup_first_system(INT verbose_level,
	INT lambda3, INT block_size, INT lambda2,
	tdo_data &T, INT r, partitionstack &P,
	INT &nb_vars,INT &nb_eqns,
	INT *&line_types, INT &nb_line_types)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT j, R, l1, l2, S, I, J, f, l, a, a2, s, d, d2, d3, o, h, rr, p, u, a3, e;
	
	if (f_v) {
		cout << "td3_columns_setup_first_system r=" << r << endl;
		}
		

	// create all partitions which are refined line types

	R = nb_col_classes[COL];
	l1 = nb_row_classes[COL];
	l2 = nb_row_classes[ROW];
		
	nb_vars = l2; // P.n
	nb_eqns = l1; // = P.ht
		
	
	T.D1->open(nb_eqns, nb_vars);
	S = 0;
		
	for (I = 0; I < nb_eqns; I++) 
		for (J = 0; J < nb_vars; J++) 
			T.D1->A[I * nb_vars + J] = 0;
			
	for (I = 0; I < nb_eqns; I++) {
		f = P.startCell[I];
		l = P.cellSize[I];
		for (j = 0; j < l; j++) {
			J = f + j;
			T.D1->A[I * nb_vars + J] = 1;
			a = the_row_scheme[J * R + r];
			if (a == 0)
				T.D1->x_max[J] = 0;
			else
				T.D1->x_max[J] = row_classes_len[ROW][J];
			}
		s = the_col_scheme[I * R + r];
		T.D1->RHS[I] = s;
		S += s;
		}

	// try to reduce the upper bounds:
		
	for (j = 0; j < nb_vars; j++) {
		//cout << "j=" << j << endl;
		if (T.D1->x_max[j] == 0)
			continue;
		d = row_classes_len[ROW][j];
		d2 = binomial2(d) * lambda2;
		o = d2;
		for (h = 0; h < T.nb_only_one_type; h++) {
			rr = T.only_one_type[h];
			p = col_classes_len[COL][rr];

			u = T.types_first[rr];
			//cout << "u=" << u << endl;
				
			a = line_types[u * nb_vars + j];
			a2 = binomial2(a);
			o -= a2 * p;
			if (o < 0) {
				if (f_vvv) {
					cout << "only one type, but no solution because of joining in row-class " << j << endl;
					//cout << "u=" << u << " j=" << j << endl;
					}
				return FALSE;
				}
			}
		e = largest_binomial2_below(o);
		T.D1->x_max[j] = MINIMUM(T.D1->x_max[j], e);
		}
	for (j = 0; j < nb_vars; j++) {
		//cout << "j=" << j << endl;
		if (T.D1->x_max[j] == 0)
			continue;
		d = row_classes_len[ROW][j];
		d3 = binomial3(d) * lambda3;
		o = d3;
		for (h = 0; h < T.nb_only_one_type; h++) {
			rr = T.only_one_type[h];
			p = col_classes_len[COL][rr];
				u = T.types_first[rr];
			//cout << "u=" << u << endl;
			
			a = line_types[u * nb_vars + j];
			a3 = binomial3(a);
			o -= a3 * p;
			if (o < 0) {
				if (f_vvv) {
					cout << "only one type, but no solution because of joining in row-class " << j << endl;
					//cout << "u=" << u << " j=" << j << endl;
					}
				return FALSE;
				}
			}
		e = largest_binomial3_below(o);
		T.D1->x_max[j] = MINIMUM(T.D1->x_max[j], e);
		}
		
	T.D1->sum = S;
	T.D1->f_x_max = TRUE;
		
	if (f_vv) {
		cout << "r=" << r << " the system is" << endl;
		T.D1->print();
		}
	return TRUE;
}		


INT tdo_scheme::td3_columns_setup_second_system(INT verbose_level,
	INT lambda3, INT block_size, INT lambda2, INT f_scale, INT scaling,
	tdo_data &T, 
	INT nb_vars, INT &Nb_vars, INT &Nb_eqns, 
	INT *&line_types, INT &nb_line_types)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT l2, i, r, I, J, a;
	INT S;
	INT nb_eqns_joining, nb_eqns_joining_pairs, nb_eqns_joining_triples, nb_eqns_counting;

	l2 = nb_row_classes[ROW];
	
	nb_eqns_joining_triples = l2 + l2 * (l2 - 1) + binomial3(l2);
		// l2 times: triples within a given class
		// l2 * (l2 - 1) times (ordered pairs from an l2 set): 
		//     triples with 2 in a given class, 1 in another given class
		// binomial3(l2) triples from different classes
	nb_eqns_joining_pairs = l2 + binomial2(l2);
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

	T.D2->open(Nb_eqns, Nb_vars);
	if (f_v) {
		cout << "td3_columns_setup_second_system: opening second system with " 
			<< Nb_eqns << " equations and " << Nb_vars << " variables" << endl;
		}

	for (I = 0; I < Nb_eqns; I++) 
		for (J = 0; J < Nb_vars; J++) 
			T.D2->A[I * Nb_vars + J] = 0;
	for (I = 0; I < Nb_eqns; I++) 
		T.D2->RHS[I] = 9999;


	if (!td3_columns_triples_same_class(verbose_level, 
		lambda3, block_size, 
		T, 
		nb_vars, Nb_vars, 
		line_types, nb_line_types, 0)) {
		return FALSE;
		}
	
	if (!td3_columns_pairs_same_class(verbose_level, 
		lambda3, block_size, lambda2, 
		T, 
		nb_vars, Nb_vars, 
		line_types, nb_line_types, nb_eqns_joining_triples)) {
		return FALSE;
		}
	
	if (!td3_columns_counting_flags(verbose_level, 
		lambda3, block_size, lambda2, S, 
		T, 
		nb_vars, Nb_vars, 
		line_types, nb_line_types, nb_eqns_joining)) {
		return FALSE;
		}
	
	if (!td3_columns_lambda2_joining_pairs_from_different_classes(verbose_level, 
		lambda3, block_size, lambda2,  
		T, 
		nb_vars, Nb_vars, 
		line_types, nb_line_types, nb_eqns_joining_triples + l2)) {
		return FALSE;
		}
	
	if (!td3_columns_lambda3_joining_triples_2_1(verbose_level, 
		lambda3, block_size, lambda2,  
		T, 
		nb_vars, Nb_vars, 
		line_types, nb_line_types, l2)) {
		return FALSE;
		}
	
	if (!td3_columns_lambda3_joining_triples_1_1_1(verbose_level, 
		lambda3, block_size, lambda2,  
		T, 
		nb_vars, Nb_vars, 
		line_types, nb_line_types, l2 + l2 * (l2 - 1))) {
		return FALSE;
		}
	
	if (f_scale) {
		if (S % scaling) {
			cout << "cannot scale by " << scaling << " b/c S=" << S << endl;
			exit(1);
			}
		S /= scaling;
		for (I = 0; I < Nb_eqns; I++) {
			a = T.D2->RHS[I];
			if (a % scaling) {
				if (a % scaling) {
					cout << "cannot scale by " << scaling << " b/c RHS[" << I << "]=" << a << endl;
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

	

	T.D2->sum = S;
	

	if (f_vv) {	
		cout << "The second system is" << endl;
	
		T.D2->print();
		}
	
	return TRUE;

}


INT tdo_scheme::td3_columns_triples_same_class(INT verbose_level,
	INT lambda3, INT block_size,
	tdo_data &T, 
	INT nb_vars, INT Nb_vars, 
	INT *&line_types, INT &nb_line_types, INT eqn_offset)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT I, i, r, f, l, j, c, J, a, a3, rr, p, u, l2, h;

	l2 = nb_row_classes[ROW];
	
	if (f_v) {
		cout << "td3_columns_triples_same_class: eqn_offset=" << eqn_offset << endl;
		}
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
				a3 = binomial3(a);
				// joining triples from the same class:
				T.D2->A[(eqn_offset + I) * Nb_vars + J] = a3;
				}
			}
		a = row_classes_len[ROW][I];
		a3 = binomial3(a);
		T.D2->RHS[eqn_offset + I] = a3 * lambda3;
		for (h = 0; h < T.nb_only_one_type; h++) {
			rr = T.only_one_type[h];
			p = col_classes_len[COL][rr];
			u = T.types_first[rr];
			a = line_types[u * nb_vars + I];
			a3 = binomial3(a);
			T.D2->RHS[eqn_offset + I] -= a3 * p;
			if (T.D2->RHS[eqn_offset + I] < 0) {
				if (f_v) {
					cout << "td3_refine_columns: RHS[I] is negative, no solution for the distribution" << endl;
					}
				return FALSE;
				}
			}
		}
	if (f_vvv) {
		cout << "triples from the same class, the system is" << endl;
		T.D2->print();
		}
	return TRUE;
}

INT tdo_scheme::td3_columns_pairs_same_class(INT verbose_level,
	INT lambda3, INT block_size, INT lambda2,
	tdo_data &T, 
	INT nb_vars, INT Nb_vars, 
	INT *&line_types, INT &nb_line_types, INT eqn_offset)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT I, i, r, f, l, j, c, J, a, a2, rr, p, u, l2, h;

	l2 = nb_row_classes[ROW];
	
	if (f_v) {
		cout << "td3_columns_pairs_same_class: eqn_offset=" << eqn_offset << endl;
		}
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
				a2 = binomial2(a);
				// joining pairs from the same class:
				T.D2->A[(eqn_offset + I) * Nb_vars + J] = a2;
				}
			}
		a = row_classes_len[ROW][I];
		a2 = binomial2(a);
		T.D2->RHS[eqn_offset + I] = a2 * lambda2;
		for (h = 0; h < T.nb_only_one_type; h++) {
			rr = T.only_one_type[h];
			p = col_classes_len[COL][rr];
			u = T.types_first[rr];
			a = line_types[u * nb_vars + I];
			a2 = binomial2(a);
			T.D2->RHS[eqn_offset + I] -= a2 * p;
			if (T.D2->RHS[eqn_offset + I] < 0) {
				if (f_v) {
					cout << "td3_refine_columns: RHS[eqn_offset + I] is negative, no solution for the distribution" << endl;
					}
				return FALSE;
				}
			}
		}
	if (f_vvv) {
		cout << "pairs from the same class, the system is" << endl;
		T.D2->print();
		}
	
	return TRUE;
}

INT tdo_scheme::td3_columns_counting_flags(INT verbose_level,
	INT lambda3, INT block_size, INT lambda2, INT &S,
	tdo_data &T, 
	INT nb_vars, INT Nb_vars, 
	INT *&line_types, INT &nb_line_types, INT eqn_offset)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT I, i, r, f, l, j, c, J, a, b, rr, p, u, l2, h, s;

	l2 = nb_row_classes[ROW];
	
	if (f_v) {
		cout << "td3_columns_counting_flags: eqn_offset=" << eqn_offset << endl;
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
			a = the_row_scheme[I * nb_col_classes[ROW] + r];
			b = row_classes_len[ROW][I];
			T.D2->RHS[eqn_offset + i * (l2 + 1) + I] = a * b;
			for (h = 0; h < T.nb_only_one_type; h++) {
				rr = T.only_one_type[h];
				p = col_classes_len[COL][rr];
				u = T.types_first[rr];
				a = line_types[u * nb_vars + I];
				T.D2->RHS[eqn_offset + i * (l2 + 1) + I] -= a * p;
				if (T.D2->RHS[eqn_offset + i * (l2 + 1) + I] < 0) {
					if (f_v) {
						cout << "td3_columns_counting_flags: RHS[nb_eqns_joining + i * (l2 + 1) + I] is negative, no solution for the distribution" << endl;
						}
					return FALSE;
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
		
		s = col_classes_len[ROW][r];
		T.D2->RHS[eqn_offset + i * (l2 + 1) + l2] = s;
		S += s;
		}
	if (f_vvv) {
		cout << "td3_columns_counting_flags, the system is" << endl;
		T.D2->print();
		}

	
	return TRUE;
}

INT tdo_scheme::td3_columns_lambda2_joining_pairs_from_different_classes(INT verbose_level,
	INT lambda3, INT block_size, INT lambda2,
	tdo_data &T, 
	INT nb_vars, INT Nb_vars, 
	INT *&line_types, INT &nb_line_types, INT eqn_offset)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT I1, I2, i, r, f, l, j, c, J, a, b, ab, k, rr, p, u, l2, h;

	l2 = nb_row_classes[ROW];
	
	if (f_v) {
		cout << "td3_columns_lambda2_joining_pairs_from_different_classes: eqn_offset=" << eqn_offset << endl;
		}
	// lambda2: joining pairs from different classes
	for (I1 = 0; I1 < l2; I1++) {
		for (I2 = I1 + 1; I2 < l2; I2++) {
			k = ij2k(I1, I2, l2);
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
			a = row_classes_len[ROW][I1];
			b = row_classes_len[ROW][I2];
			T.D2->RHS[eqn_offset + k] = a * b * lambda2;
			for (h = 0; h < T.nb_only_one_type; h++) {
				rr = T.only_one_type[h];
				p = col_classes_len[COL][rr];
				u = T.types_first[rr];
				a = line_types[u * nb_vars + I1];
				b = line_types[u * nb_vars + I2];
				T.D2->RHS[eqn_offset + k] -= a * b * p;
				if (T.D2->RHS[eqn_offset + k] < 0) {
					if (f_v) {
						cout << "td3_columns_lambda2_joining_pairs_from_different_classes: RHS[eqn_offset + k] is negative, no solution for the distribution" << endl;
						}
					return FALSE;
					}
				} // next h
			}
		}
	if (f_vvv) {
		cout << "td3_columns_lambda2_joining_pairs_from_different_classes, the system is" << endl;
		T.D2->print();
		}
	
	return TRUE;
}

INT tdo_scheme::td3_columns_lambda3_joining_triples_2_1(INT verbose_level,
	INT lambda3, INT block_size, INT lambda2,
	tdo_data &T, 
	INT nb_vars, INT Nb_vars, 
	INT *&line_types, INT &nb_line_types, INT eqn_offset)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT I1, I2, i, r, f, l, j, c, J, a, a2, ab, b, k, rr, p, u, l2, h;
	INT length_first, length_first2, length_second;

	l2 = nb_row_classes[ROW];
	
	if (f_v) {
		cout << "td3_columns_lambda3_joining_triples_2_1: eqn_offset=" << eqn_offset << endl;
		}
	// lambda3: joining triples with two in the first class and one in the second class
	for (I1 = 0; I1 < l2; I1++) {
		length_first = row_classes_len[ROW][I1];
		length_first2 = binomial2(length_first);
		for (I2 = 0; I2 < l2; I2++) {
			if (I2 == I1)
				continue;
			length_second = row_classes_len[ROW][I2];
			k = ordered_pair_rank(I1, I2, l2);
			for (i = 0; i < T.nb_multiple_types; i++) {
				r = T.multiple_types[i];
				f = T.types_first[r];
				l = T.types_len[r];
				for (j = 0; j < l; j++) {
					c = f + j;
					J = T.types_first2[i] + j;
					a = line_types[c * nb_vars + I1];
					b = line_types[c * nb_vars + I2];
					ab = binomial2(a) * b;
					T.D2->A[(l2 + k) * Nb_vars + J] = ab;
					}
				}
			T.D2->RHS[l2 + k] = length_first2 * length_second * lambda3;
			for (h = 0; h < T.nb_only_one_type; h++) {
				rr = T.only_one_type[h];
				p = col_classes_len[COL][rr];
				u = T.types_first[rr];
				a = line_types[u * nb_vars + I1];
				a2 = binomial2(a);
				b = line_types[u * nb_vars + I2];
				T.D2->RHS[l2 + k] -= a2 * b * p;
				if (T.D2->RHS[l2 + k] < 0) {
					if (f_v) {
						cout << "td3_columns_lambda3_joining_triples_2_1: RHS[l2 + k] is negative, no solution for the distribution" << endl;
						}
					return FALSE;
					}
				} // next h
			}
		}
	if (f_vvv) {
		cout << "td3_columns_lambda3_joining_triples_2_1, the system is" << endl;
		T.D2->print();
		}
	
	return TRUE;
}

INT tdo_scheme::td3_columns_lambda3_joining_triples_1_1_1(INT verbose_level,
	INT lambda3, INT block_size, INT lambda2,
	tdo_data &T, 
	INT nb_vars, INT Nb_vars, 
	INT *&line_types, INT &nb_line_types, INT eqn_offset)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT I1, I2, I3, i, r, f, l, j, c, J, a, b, k, rr, p, u, l2, h, g;
	INT length_first, length_second, length_third;

	l2 = nb_row_classes[ROW];
	
	if (f_v) {
		cout << "td3_columns_lambda3_joining_triples_1_1_1: eqn_offset=" << eqn_offset << endl;
		}
	// lambda3: joining triples with all in different classes
	for (I1 = 0; I1 < l2; I1++) {
		length_first = row_classes_len[ROW][I1];
		for (I2 = I1 + 1; I2 < l2; I2++) {
			length_second = row_classes_len[ROW][I2];
			for (I3 = I2 + 1; I3 < l2; I3++) {
				length_third = row_classes_len[ROW][I3];
				k = ijk_rank(I1, I2, I3, l2);
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
						T.D2->A[(l2 + l2 * (l2 - 1) + k) * Nb_vars + J] = a * b * g;
						}
					}
				T.D2->RHS[l2 + l2 * (l2 - 1) + k] = length_first * length_second * length_third * lambda3;
				for (h = 0; h < T.nb_only_one_type; h++) {
					rr = T.only_one_type[h];
					p = col_classes_len[COL][rr];
					u = T.types_first[rr];
					a = line_types[u * nb_vars + I1];
					b = line_types[u * nb_vars + I2];
					g = line_types[u * nb_vars + I3];
					T.D2->RHS[l2 + l2 * (l2 - 1) + k] -= a * b * g * p;
					if (T.D2->RHS[l2 + l2 * (l2 - 1) + k] < 0) {
						if (f_v) {
							cout << "td3_columns_lambda3_joining_triples_1_1_1: RHS[l2 + l2 * (l2 - 1) + k] is negative, no solution for the distribution" << endl;
							}
						return FALSE;
						}
					} // next h
				}
			}
		}
	if (f_vvv) {
		cout << "td3_columns_lambda3_joining_triples_1_1_1, the system is" << endl;
		T.D2->print();
		}
	
	return TRUE;
}

