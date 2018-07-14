// object_in_projective_space.C
// 
// Anton Betten
//
// December 23, 2017
//
//
// 
//
//

#include "galois.h"


object_in_projective_space::object_in_projective_space()
{
	null();
}

object_in_projective_space::~object_in_projective_space()
{
	freeself();
}

void object_in_projective_space::null()
{
	P = NULL;
	set = NULL;
	sz = 0;
	SoS = NULL;
	C = NULL;
}

void object_in_projective_space::freeself()
{
	if (set) {
		FREE_INT(set);
		}
	if (SoS) {
		delete SoS;
		}
	if (C) {
		delete C;
		}
	null();
}

void object_in_projective_space::print(ostream &ost)
{
	if (type == t_PTS) {
		ost << "set of points of size " << sz << ": ";
		INT_vec_print(ost, set, sz);
		ost << endl;
		}
	else if (type == t_LNS) {
		ost << "set of lines of size " << sz << ": ";
		INT_vec_print(ost, set, sz);
		ost << endl;
		}
	else if (type == t_PAC) {
		ost << "packing:" << endl;
		SoS->print_table_tex(ost);
		ost << endl;
		}
}

void object_in_projective_space::print_tex(ostream &ost)
{
	if (type == t_PTS) {
		ost << "set of points of size " << sz << ": $\\{";
		INT_vec_print(ost, set, sz);
		ost << "\\}$" << endl;
		}
	else if (type == t_LNS) {
		ost << "set of lines of size " << sz << ": $\\{";
		INT_vec_print(ost, set, sz);
		ost << "\\}$" << endl;
		}
	else if (type == t_PAC) {
		ost << "packing: " << endl;
		SoS->print_table_tex(ost);
		ost << endl;
		}
}

void object_in_projective_space::init_point_set(projective_space *P, INT *set, INT sz, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space::init_point_set" << endl;
		}
	object_in_projective_space::P = P;
	type = t_PTS;
	object_in_projective_space::set = NEW_INT(sz);
	INT_vec_copy(set, object_in_projective_space::set, sz);
	object_in_projective_space::sz = sz;
	if (f_v) {
		cout << "object_in_projective_space::init_point_set done" << endl;
		}
}

void object_in_projective_space::init_line_set(projective_space *P, INT *set, INT sz, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space::init_line_set" << endl;
		}
	object_in_projective_space::P = P;
	type = t_LNS;
	object_in_projective_space::set = NEW_INT(sz);
	INT_vec_copy(set, object_in_projective_space::set, sz);
	object_in_projective_space::sz = sz;
	if (f_v) {
		cout << "object_in_projective_space::init_line_set done" << endl;
		}
}

void object_in_projective_space::init_packing_from_set(projective_space *P, INT *packing, INT sz, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, q, size_of_spread, size_of_packing;

	if (f_v) {
		cout << "object_in_projective_space::init_packing_from_set" << endl;
		}
	object_in_projective_space::P = P;
	type = t_PAC;
	q = P->q;
	size_of_spread = q * q + 1;
	size_of_packing = q * q + q + 1;
	if (sz != size_of_packing * size_of_spread) {
		cout << "object_in_projective_space::init_packing_from_set sz != size_of_packing * size_of_spread" << endl;
		exit(1);
		}
	SoS = new set_of_sets;

	SoS->init_basic_constant_size(P->N_lines, 
		size_of_packing /* nb_sets */, 
		size_of_spread /* constant_size */, 
		0 /* verbose_level */);

	for (i = 0; i < size_of_packing; i++) {
		INT_vec_copy(packing + i * size_of_spread, SoS->Sets[i], size_of_spread);
		}
#if 0
	if (f_v) {
		cout << "object_in_projective_space::init_packing_from_set it is" << endl;
		SoS->print_table();
		}
#endif
	
	
	if (f_v) {
		cout << "object_in_projective_space::init_packing_from_set done" << endl;
		}
}

void object_in_projective_space::init_packing_from_set_of_sets(projective_space *P, set_of_sets *SoS, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space::init_packing_from_set_of_sets" << endl;
		}
	object_in_projective_space::P = P;
	type = t_PAC;
	//object_in_projective_space::set = NEW_INT(sz);
	//INT_vec_copy(set, object_in_projective_space::set, sz);
	//object_in_projective_space::sz = sz;

	object_in_projective_space::SoS = SoS->copy();

	if (f_v) {
		cout << "object_in_projective_space::init_packing_from_set_of_sets done" << endl;
		}
}

void object_in_projective_space::init_packing_from_spread_table(projective_space *P, 
	INT *data, INT *Spread_table, INT nb_spreads, INT spread_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT a, i, q, size_of_spread, size_of_packing;

	if (f_v) {
		cout << "object_in_projective_space::init_packing_from_spread_table" << endl;
		}
	object_in_projective_space::P = P;
	type = t_PAC;
	q = P->q;
	size_of_spread = q * q + 1;
	size_of_packing = q * q + q + 1;
	if (spread_size != size_of_spread) {
		cout << "object_in_projective_space::init_packing_from_spread_table spread_size != size_of_spread" << endl;
		exit(1);
		}
	SoS = new set_of_sets;

	SoS->init_basic_constant_size(P->N_lines, 
		size_of_packing /* nb_sets */, 
		size_of_spread /* constant_size */, 
		0 /* verbose_level */);

	for (i = 0; i < size_of_packing; i++) {
		a = data[i];
		INT_vec_copy(Spread_table + a * size_of_spread, SoS->Sets[i], size_of_spread);
		}

	if (f_v) {
		cout << "object_in_projective_space::init_packing_from_spread_table done" << endl;
		}
}

void object_in_projective_space::encode_incma(INT *&Incma, INT &nb_rows, INT &nb_cols, INT *&partition, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space::encode_incma" << endl;
		}
	if (type == t_PTS) {
		
		encode_point_set(Incma, nb_rows, nb_cols, partition, verbose_level);

		}
	else if (type == t_LNS) {
		
		encode_line_set(Incma, nb_rows, nb_cols, partition, verbose_level);

		}
	else if (type == t_PAC) {
		
		encode_packing(Incma, nb_rows, nb_cols, partition, verbose_level);

		}
	else {
		cout << "object_in_projective_space::encode_incma unknown type" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "object_in_projective_space::encode_incma done" << endl;
		}
}

void object_in_projective_space::encode_point_set(INT *&Incma, INT &nb_rows, INT &nb_cols, INT *&partition, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space::encode_point_set" << endl;
		}
	INT i, j;
	INT f_vvv = (verbose_level >= 3);
	
	C = new classify;

	C->init(set, sz, TRUE, 0);
	if (C->second_nb_types > 1) {
		cout << "object_in_projective_space::encode_point_set The set is a multiset:" << endl;
		C->print(FALSE /*f_backwards*/);
		}

	if (f_v) {
		cout << "The type of the set is:" << endl;
		C->print(FALSE /*f_backwards*/);
		cout << "C->second_nb_types = " << C->second_nb_types << endl;
		}

	nb_rows = P->N_points + 1;
	nb_cols = P->N_lines + C->second_nb_types;

	INT N, L;
	
	N = nb_rows + nb_cols;
	L = nb_rows * nb_cols;

	Incma = NEW_INT(L);
	partition = NEW_INT(N);

	INT_vec_zero(Incma, L);

	for (i = 0; i < P->N_points; i++) {
		for (j = 0; j < P->N_lines; j++) {
			Incma[i * nb_cols + j] = P->is_incident(i, j);
			}
		}

#if 0
	// last columns, make zero:
	for (j = 0; j < C->second_nb_types; j++) {
		for (i = 0; i < P->N_points; i++) {
			Incma[i * nb_cols + P->N_lines + j] = 0;
			}
		}

	// last row, make zero:
	for (j = 0; j < nb_cols; j++) {
		Incma[P->N_points * nb_cols + j] = 0;
		}
#endif

	// last columns:
	for (j = 0; j < C->second_nb_types; j++) {
		INT h, f2, l2, m, idx, f, l;

		f2 = C->second_type_first[j];
		l2 = C->second_type_len[j];
		m = C->second_data_sorted[f2 + 0];
		if (f_vvv) {
			cout << "j=" << j << " f2=" << f2 << " l2=" << l2 << " multiplicity=" << m << endl;
			}
		for (h = 0; h < l2; h++) {
			idx = C->second_sorting_perm_inv[f2 + h];
			f = C->type_first[idx];
			l = C->type_len[idx];
			i = C->data_sorted[f + 0];
			if (f_vvv) {
				cout << "h=" << h << " idx=" << idx << " f=" << f << " l=" << l << " i=" << i << endl;
				}
			Incma[i * nb_cols + P->N_lines + j] = 1;
			}	
#if 0
		for (h = 0; h < set_size; h++) {
			i = set[h];
			Incma[i * nb_cols + N_lines + j] = 1;
			}
#endif
		}
	// bottom right entries:
	for (j = 0; j < C->second_nb_types; j++) {
		Incma[P->N_points * nb_cols + P->N_lines + j] = 1;
		}

	for (i = 0; i < N; i++) {
		partition[i] = 1;
		}
	partition[P->N_points - 1] = 0;
	partition[P->N_points] = 0;
	partition[nb_rows + P->N_lines - 1] = 0;
	for (j = 0; j < C->second_nb_types; j++) {
		partition[nb_rows + P->N_lines + j] = 0;
		}
	if (f_vvv) {
		cout << "object_in_projective_space::encode_point_set partition:" << endl;
		for (i = 0; i < N; i++) {
			//cout << i << " : " << partition[i] << endl;
			cout << partition[i];
			}
		cout << endl;
		}
	if (f_v) {
		cout << "object_in_projective_space::encode_point_set done" << endl;
		}
}

void object_in_projective_space::encode_line_set(INT *&Incma, INT &nb_rows, INT &nb_cols, INT *&partition, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space::encode_line_set" << endl;
		}
	INT i, j;
	INT f_vvv = (verbose_level >= 3);
	

	nb_rows = P->N_points + 1;
	nb_cols = P->N_lines + 1;

	INT N, L;
	
	N = nb_rows + nb_cols;
	L = nb_rows * nb_cols;

	partition = NEW_INT(N);
	Incma = NEW_INT(L);


	INT_vec_zero(Incma, L);

	for (i = 0; i < P->N_points; i++) {
		for (j = 0; j < P->N_lines; j++) {
			Incma[i * nb_cols + j] = P->is_incident(i, j);
			}
		}

	// last rows:
	for (i = 0; i < 1; i++) {
		INT h;

		for (h = 0; h < sz; h++) {
			j = set[h];
			Incma[(P->N_points + i) * nb_cols + j] = 1;
			}	
		}
	// bottom right entries:
	for (i = 0; i < 1; i++) {
		Incma[(P->N_points + i) * nb_cols + P->N_lines] = 1;
		}

	for (i = 0; i < N; i++) {
		partition[i] = 1;
		}
	partition[P->N_points - 1] = 0;
	partition[nb_rows - 1] = 0;
	partition[nb_rows + P->N_lines - 1] = 0;
	partition[nb_rows + P->N_lines + 1 - 1] = 0;
	if (f_vvv) {
		cout << "object_in_projective_space::encode_line_set partition:" << endl;
		for (i = 0; i < N; i++) {
			//cout << i << " : " << partition[i] << endl;
			cout << partition[i];
			}
		cout << endl;
		}
	if (f_v) {
		cout << "object_in_projective_space::encode_line_set done" << endl;
		}
}


void object_in_projective_space::encode_packing(INT *&Incma, INT &nb_rows, INT &nb_cols, INT *&partition, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space::encode_packing" << endl;
		}
	INT i, j;
	INT f_vvv = (verbose_level >= 3);
	

	nb_rows = P->N_points + SoS->nb_sets;
	nb_cols = P->N_lines + 1;

	INT N, L;
	
	N = nb_rows + nb_cols;
	L = nb_rows * nb_cols;

	Incma = NEW_INT(L);
	partition = NEW_INT(N);


	INT_vec_zero(Incma, L);

	for (i = 0; i < P->N_points; i++) {
		for (j = 0; j < P->N_lines; j++) {
			Incma[i * nb_cols + j] = P->is_incident(i, j);
			}
		}

	// last rows:
	for (i = 0; i < SoS->nb_sets; i++) {
		INT h;

		for (h = 0; h < SoS->Set_size[i]; h++) {
			j = SoS->Sets[i][h];
			Incma[(P->N_points + i) * nb_cols + j] = 1;
			}	
		}
	// bottom right entries:
	for (i = 0; i < SoS->nb_sets; i++) {
		Incma[(P->N_points + i) * nb_cols + P->N_lines] = 1;
		}

	for (i = 0; i < N; i++) {
		partition[i] = 1;
		}
	partition[P->N_points - 1] = 0;
	partition[nb_rows - 1] = 0;
	partition[nb_rows + P->N_lines - 1] = 0;
	partition[nb_rows + P->N_lines + 1 - 1] = 0;
	if (f_vvv) {
		cout << "object_in_projective_space::encode_packing partition:" << endl;
		for (i = 0; i < N; i++) {
			//cout << i << " : " << partition[i] << endl;
			cout << partition[i];
			}
		cout << endl;
		}
	if (f_v) {
		cout << "object_in_projective_space::encode_packing done" << endl;
		}
}

void object_in_projective_space::encode_incma_and_make_decomposition(
	INT *&Incma, INT &nb_rows, INT &nb_cols, INT *&partition, 
	incidence_structure *&Inc, 
	partitionstack *&Stack, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space::encode_incma_and_make_decomposition" << endl;
		}
	if (type == t_PTS) {
		
		encode_point_set(Incma, nb_rows, nb_cols, partition, verbose_level);

		}
	else if (type == t_LNS) {
		
		encode_line_set(Incma, nb_rows, nb_cols, partition, verbose_level);

		}
	else if (type == t_PAC) {
		
		encode_packing(Incma, nb_rows, nb_cols, partition, verbose_level);

		}
	else {
		cout << "object_in_projective_space::encode_incma_and_make_decomposition unknown type" << endl;
		exit(1);
		}

	Inc = new incidence_structure;
	Inc->init_by_matrix(nb_rows, nb_cols, Incma, verbose_level - 2);




	Stack = new partitionstack;
	Stack->allocate(nb_rows + nb_cols, 0);
	Stack->subset_continguous(Inc->nb_points(), Inc->nb_lines());
	Stack->split_cell(0);

	if (type == t_PTS) {
		
		if (f_v) {
			cout << "object_in_projective_space::encode_incma_and_make_decomposition t_PTS split1" << endl;
			}
		Stack->subset_continguous(Inc->nb_points() + P->N_lines, nb_cols - P->N_lines);
		Stack->split_cell(0);
		if (f_v) {
			cout << "object_in_projective_space::encode_incma_and_make_decomposition t_PTS split2" << endl;
			}
		if (nb_rows - Inc->nb_points()) {
			Stack->subset_continguous(Inc->nb_points(), nb_rows - Inc->nb_points());
			Stack->split_cell(0);
			}

		}
	
	else if (type == t_LNS) {
		
		if (f_v) {
			cout << "object_in_projective_space::encode_incma_and_make_decomposition t_LNS" << endl;
			}
		Stack->subset_continguous(P->N_points, 1);
		Stack->split_cell(0);
		Stack->subset_continguous(Inc->nb_points() + P->N_lines, nb_cols - P->N_lines);
		Stack->split_cell(0);

		}
	else if (type == t_PAC) {
		
		if (f_v) {
			cout << "object_in_projective_space::encode_incma_and_make_decomposition t_PAC" << endl;
			}
		Stack->subset_continguous(P->N_points, nb_rows - P->N_points);
		Stack->split_cell(0);
		Stack->subset_continguous(Inc->nb_points() + P->N_lines, nb_cols - P->N_lines);
		Stack->split_cell(0);

		}
	
	if (f_v) {
		cout << "object_in_projective_space::encode_incma_and_make_decomposition done" << endl;
		}
}

void object_in_projective_space::encode_object(INT *&encoding, INT &encoding_sz, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space::encode_object" << endl;
		}
	if (type == t_PTS) {
		
		encode_object_points(encoding, encoding_sz, verbose_level);

		}
	else if (type == t_LNS) {
		
		encode_object_lines(encoding, encoding_sz, verbose_level);

		}
	else if (type == t_PAC) {
		
		encode_object_packing(encoding, encoding_sz, verbose_level);

		}
	else {
		cout << "object_in_projective_space::encode_object unknown type" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "object_in_projective_space::encode_object encoding_sz=" << encoding_sz << endl;
		}
	if (f_v) {
		cout << "object_in_projective_space::encode_object done" << endl;
		}
}

void object_in_projective_space::encode_object_points(INT *&encoding, INT &encoding_sz, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space::encode_object_points" << endl;
		}
	encoding_sz = sz;
	encoding = NEW_INT(sz);
	INT_vec_copy(set, encoding, sz);
}

void object_in_projective_space::encode_object_lines(INT *&encoding, INT &encoding_sz, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space::encode_object_lines" << endl;
		}
	encoding_sz = sz;
	encoding = NEW_INT(sz);
	INT_vec_copy(set, encoding, sz);
}

void object_in_projective_space::encode_object_packing(INT *&encoding, INT &encoding_sz, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space::encode_object_packing" << endl;
		}
	INT i, h;
	
	encoding_sz = SoS->total_size();
	encoding = NEW_INT(encoding_sz);
	h = 0;
	for (i = 0; i < SoS->nb_sets; i++) {
		INT_vec_copy(SoS->Sets[i], encoding + h, SoS->Set_size[i]);
		h += SoS->Set_size[i];
		}
	if (h != encoding_sz) {
		cout << "object_in_projective_space::encode_object_packing h != encoding_sz" << endl;
		exit(1);
		}
}

void object_in_projective_space::klein(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space::klein" << endl;
		}
	if (type != t_LNS) {
		if (f_v) {
			cout << "object_in_projective_space::klein not of type t_LNS" << endl;
			}
		return;
		}
	if (P->n != 3) {
		if (f_v) {
			cout << "object_in_projective_space::klein not in three space" << endl;
			}
		return;
		}


	projective_space *P5;
	grassmann *Gr;
	INT *pts_klein;
	INT i, N;
	
	longinteger_object *R;
	INT **Pts_on_plane;
	INT *nb_pts_on_plane;
	INT nb_planes;



	P5 = new projective_space;
	
	P5->init(5, P->F, 
		FALSE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	pts_klein = NEW_INT(sz);
	
	if (f_v) {
		cout << "object_in_projective_space::klein before P3->klein_correspondence" << endl;
		}
	P->klein_correspondence(P5, 
		set, sz, pts_klein, 0/*verbose_level*/);


	N = P5->nb_rk_k_subspaces_as_INT(3);
	if (f_v) {
		cout << "object_in_projective_space::klein N = " << N << endl;
		}

	

	Gr = new grassmann;

	Gr->init(6, 3, P->F, 0 /* verbose_level */);

	if (f_v) {
		cout << "object_in_projective_space::klein before plane_intersection_type_fast" << endl;
		}
	P5->plane_intersection_type_slow(Gr, pts_klein, sz, 
		R, Pts_on_plane, nb_pts_on_plane, nb_planes, 
		verbose_level /*- 3*/);

	if (f_v) {
		cout << "object_in_projective_space::klein We found " << nb_planes << " planes." << endl;

		classify C;

		C.init(nb_pts_on_plane, nb_planes, FALSE, 0);
		cout << "plane types are: ";
		C.print(TRUE /* f_backwards*/);
		cout << endl;
#if 0
		for (i = 0; i < nb_planes; i++) {
			if (nb_pts_on_plane[i] >= 3) {
				cout << setw(3) << i << " / " << nb_planes << " : " << R[i] 
					<< " : " << setw(5) << nb_pts_on_plane[i] << " : ";
				INT_vec_print(cout, Pts_on_plane[i], nb_pts_on_plane[i]);
				cout << endl;
				}
			}
#endif
		}
	cout << "before delete [] R;" << endl;
	delete [] R;
	cout << "before FREE_INT(Pts_on_plane[i]);" << endl;
	for (i = 0; i < nb_planes; i++) {
		FREE_INT(Pts_on_plane[i]);
		}
	cout << "before FREE_PINT(Pts_on_plane);" << endl;
	FREE_PINT(Pts_on_plane);
	cout << "before FREE_INT(nb_pts_on_plane);" << endl;
	FREE_INT(nb_pts_on_plane);

	
	
	FREE_INT(pts_klein);
	delete P5;
	delete Gr;
	if (f_v) {
		cout << "object_in_projective_space::klein done" << endl;
		}
}


