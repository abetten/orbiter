// pentomino_5x5.C
//
// Anton Betten
// June 17, 2015
//

#include "orbiter.h"

#define NB_PIECES 18

	INT *S[NB_PIECES];
	INT S_length[NB_PIECES];
	INT *O[NB_PIECES];
	INT O_length[NB_PIECES];
	INT *T[NB_PIECES];
	INT T_length[NB_PIECES];
	INT *R[NB_PIECES];
	INT R_length[NB_PIECES];
	INT Rotate[4 * 25];
	INT Rotate6[4 * 36];
	INT var_start[NB_PIECES + 1];
	INT var_length[NB_PIECES + 1];

INT has_interlocking_Ps(INT *set);
INT has_interlocking_Pprime(INT *set);
INT has_interlocking_Ls(INT *set);
INT test_if_interlocking_Ps(INT a1, INT a2);
INT has_interlocking_Lprime(INT *set);
INT test_if_interlocking_Ls(INT a1, INT a2);
INT number_of_pieces_of_type(INT t, INT *set);
INT does_it_contain_an_I(INT *set);
void decode_assembly(INT *set);
// input set[5]
void decode_piece(INT j, INT &h, INT &r, INT &t);
// h is the kind of piece
// r is the rotation index
// t is the translation index
// to get the actual rotation and translation, use 
// R[h][r] and T[h][t].
INT code_piece(INT h, INT r, INT t);
void draw_it(ostream &ost, INT *sol);
void compute_image_function(set_of_sets *S, void *compute_image_data, INT elt_idx, INT gen_idx, INT &idx_of_image, INT verbose_level);
INT compare_func(void *vec, void *a, INT b, void *data_for_compare);
void turn_piece(INT &h, INT &r, INT &t, INT verbose_level);
void flip_piece(INT &h, INT &r, INT &t, INT verbose_level);
void setup_pieces();
void setup_rotate();
void setup_var_start();
void make_coefficient_matrix(diophant *D);




int main(void)
{
	INT verbose_level = 0;
	INT i;
	
	setup_pieces();
	setup_rotate();
	setup_var_start();

	
	
	INT nb_eqns;
	INT nb_vars;
	INT nb_eqn1;
	INT nb_eqn2;

	nb_vars = var_start[NB_PIECES];
	nb_eqn1 = 5 * 5;
	nb_eqn2 = 0;
	nb_eqns = nb_eqn1 + nb_eqn2;
	
	cout << "nb_vars=" << nb_vars << endl;
	cout << "nb_eqn1=" << nb_eqn1 << endl;
	cout << "nb_eqn2=" << nb_eqn2 << endl;
	cout << "nb_eqns=" << nb_eqns << endl;

	diophant *D;

	D = new diophant;

	D->open(nb_eqns, nb_vars);
	D->fill_coefficient_matrix_with(0);
	
	make_coefficient_matrix(D);


	INT f_write_tree = FALSE;
	const BYTE *fname_tree = "";
	
	D->solve_all_DLX_with_RHS(f_write_tree, fname_tree, verbose_level);
	cout << "After solve, we found " << D->_resultanz << " solutions" << endl;

	INT *Sol;
	INT nb_sol, sol_length = 5;

	D->get_solutions(Sol, nb_sol, verbose_level);


	set_of_sets *L;
	INT l;

	L = new set_of_sets;
	L->init_basic_constant_size(nb_vars, 
		nb_sol, sol_length, 0 /* verbose_level */);

	for (l = 0; l < nb_sol; l++) {
		INT_vec_copy(Sol + l * sol_length, L->Sets[l], sol_length);
		}

	L->sort_all(0);

	L->sort_big(0 /*verbose_level */);

	{

	const BYTE *fname = "solutions.csv";

	L->save_csv(fname, TRUE /* f_make_heading */, verbose_level);
	}

	cout << "Solution 8:" << endl;
	decode_assembly(L->Sets[8]);
	cout << endl;

	cout << "Solution 10:" << endl;
	decode_assembly(L->Sets[10]);
	cout << endl;

	cout << "Solution 90:" << endl;
	decode_assembly(L->Sets[90]);
	cout << endl;

	cout << "Solution 94:" << endl;
	decode_assembly(L->Sets[94]);
	cout << endl;

	cout << "Solution 163:" << endl;
	decode_assembly(L->Sets[163]);
	cout << endl;

	



	{
	const BYTE *fname = "pentomino_all.tex";
	ofstream fp(fname);

	latex_head_easy(fp);

	fp << "\\noindent" << endl;
	for (l = 0; l < nb_sol; l++) {


		cout << "Solution " << l << " : ";
#if 1
		INT_vec_print(cout, L->Sets[l], sol_length);
		cout << "\\\\" << endl;
#endif

		draw_it(fp, L->Sets[l]);
		cout << "\\\\" << endl;

#if 0
		for (u = 0; u < sol_length; u++) {
			j = Sol[l * sol_length + u];
			decode_piece(j, h, r, t);
			cout << "j=" << j << " h=" << h << " r=" << r << " t=" << t << endl;
			}
#endif

		}
	latex_foot(fp);
	}

	INT nb_orbits;
	INT *orbit;
	INT *orbit_inv;
	INT *orbit_first;
	INT *orbit_len;

	L->compute_orbits(nb_orbits, orbit, orbit_inv, orbit_first, orbit_len, 
		compute_image_function, 
		NULL /* void *compute_image_data */, 
		2 /* nb_gens */, 
		verbose_level + 2);

	cout << "We found " << nb_orbits << " orbits \\\\" << endl;
	
	INT o, f, p;
	
	{
	const BYTE *fname = "pentomino_orbits.tex";
	ofstream fp(fname);

	latex_head_easy(fp);

	fp << "\\noindent" << endl;
	for (o = 0; o < nb_orbits; o++) {
		f = orbit_first[o];
		i = orbit[f];
#if 1
		cout << "Representative of orbit " << o << " is solution " << i << " : ";
		INT_vec_print(cout, L->Sets[i], sol_length);
		cout << "\\\\" << endl;
#endif

		draw_it(fp, L->Sets[i]);
		cout << "\\\\" << endl;

		if (((o + 1) % 35) == 0) {
			cout << endl << "\\clearpage" << endl << endl << "\\noindent" << endl;
			}
		
		}
	latex_foot(fp);
	}

	INT nb_orbits_without_I;
	INT *orbits_without_I;
	
	nb_orbits_without_I = 0;
	orbits_without_I = NEW_INT(nb_orbits);
	for (o = 0; o < nb_orbits; o++) {
		f = orbit_first[o];
		i = orbit[f];
		if (does_it_contain_an_I(L->Sets[i])) {
			continue;
			}
		if (has_interlocking_Ls(L->Sets[i])) {
			continue;
			}
		if (has_interlocking_Lprime(L->Sets[i])) {
			continue;
			}
		if (has_interlocking_Ps(L->Sets[i])) {
			continue;
			}
		if (has_interlocking_Pprime(L->Sets[i])) {
			continue;
			}
		orbits_without_I[nb_orbits_without_I++] = o;
		}

	cout << "We found " << nb_orbits_without_I << " orbits without I and with no interlocking L's and P's\\\\" << endl;



	{
	const BYTE *fname = "pentomino_orbits_reduced.tex";
	ofstream fp(fname);


	latex_head_easy(fp);

	fp << "\\noindent" << endl;
	for (p = 0; p < nb_orbits_without_I; p++) {
		o = orbits_without_I[p];
		f = orbit_first[o];
		i = orbit[f];

#if 1
		cout << p << " / " << nb_orbits_without_I << " Representative of orbit " << o << " is solution " << i << " : ";
		INT_vec_print(cout, L->Sets[i], sol_length);
		cout << "\\\\" << endl;
#endif

		draw_it(fp, L->Sets[i]);
		cout << "\\\\" << endl;

		if (((o + 1) % 35) == 0) {
			cout << endl << "\\clearpage" << endl << endl << "\\noindent" << endl;
			}
		}
	latex_foot(fp);
	}

#if 0

	cout << "Orbits with interlocking Ps:\\\\" << endl;
	INT cnt = 0;
	for (p = 0; p < nb_orbits_without_I; p++) {
		o = orbits_without_I[p];
		f = orbit_first[o];
		i = orbit[f];

		if (has_interlocking_Ps(L->Sets[i])) {
			cout << "With interlocking P's, orbit " << o << " without I is solution " << i << " : ";
			INT_vec_print(cout, L->Sets[i], sol_length);
			cout << "\\\\" << endl;

			draw_it(L->Sets[i]);
			cout << "\\\\" << endl;

			cnt++;
			}
		}
#endif
	delete D;
}

INT has_interlocking_Ps(INT *set)
{
	INT i, j, a;
	INT L[5];
	INT nb_L = 0;

	for (i = 0; i < 5; i++) {
		a = set[i];
		if (a >= var_start[6] && a < var_start[6 + 1]) {
			L[nb_L++] = a;
			}
		}
	if (nb_L <= 1) {
		return FALSE;
		}
	for (i = 0; i < nb_L; i++) {
		for (j = i + 1; j < nb_L; j++) {
			if (test_if_interlocking_Ps(L[i], L[j])) {
				return TRUE;
				}
			}
		}
	return FALSE;
}

INT has_interlocking_Pprime(INT *set)
{
	INT i, j, a;
	INT L[5];
	INT nb_L = 0;

	for (i = 0; i < 5; i++) {
		a = set[i];
		if (a >= var_start[15] && a < var_start[15 + 1]) {
			L[nb_L++] = a;
			}
		}
	if (nb_L <= 1) {
		return FALSE;
		}
	for (i = 0; i < nb_L; i++) {
		for (j = i + 1; j < nb_L; j++) {
			if (test_if_interlocking_Ps(L[i], L[j])) {
				return TRUE;
				}
			}
		}
	return FALSE;
}

INT has_interlocking_Ls(INT *set)
{
	INT i, j, a;
	INT L[5];
	INT nb_L = 0;

	for (i = 0; i < 5; i++) {
		a = set[i];
		if (a >= var_start[4] && a < var_start[4 + 1]) {
			L[nb_L++] = a;
			}
		}
	if (nb_L <= 1) {
		return FALSE;
		}
	for (i = 0; i < nb_L; i++) {
		for (j = i + 1; j < nb_L; j++) {
			if (test_if_interlocking_Ls(L[i], L[j])) {
				return TRUE;
				}
			}
		}
	return FALSE;
}

INT has_interlocking_Lprime(INT *set)
{
	INT i, j, a;
	INT L[5];
	INT nb_L = 0;

	for (i = 0; i < 5; i++) {
		a = set[i];
		if (a >= var_start[13] && a < var_start[13 + 1]) {
			L[nb_L++] = a;
			}
		}
	if (nb_L <= 1) {
		return FALSE;
		}
	for (i = 0; i < nb_L; i++) {
		for (j = i + 1; j < nb_L; j++) {
			if (test_if_interlocking_Ls(L[i], L[j])) {
				return TRUE;
				}
			}
		}
	return FALSE;
}

INT test_if_interlocking_Ps(INT a1, INT a2)
{
	INT h1, r1, t1, tt1, x1, y1, rr1;
	INT h2, r2, t2, tt2, x2, y2, rr2;

	decode_piece(a1, h1, r1, t1);
	tt1 = T[h1][t1];
	x1 = tt1 % 5;
	y1 = tt1 / 5;
	rr1 = R[h1][r1];
	decode_piece(a2, h2, r2, t2);
	tt2 = T[h2][t2];
	x2 = tt2 % 5;
	y2 = tt2 / 5;
	rr2 = R[h2][r2];

	if (((rr1 + 2) % 4) != rr2) {
		return FALSE;
		}
	if (y1 != 0) {
		return FALSE;
		}
	if (y2 != 0) {
		return FALSE;
		}
	if (x1 + x2 != 3) {
		return FALSE;
		}
	return TRUE;
}

INT test_if_interlocking_Ls(INT a1, INT a2)
{
	INT h1, r1, t1, tt1, x1, y1, rr1;
	INT h2, r2, t2, tt2, x2, y2, rr2;

	decode_piece(a1, h1, r1, t1);
	tt1 = T[h1][t1];
	x1 = tt1 % 5;
	y1 = tt1 / 5;
	rr1 = R[h1][r1];
	decode_piece(a2, h2, r2, t2);
	tt2 = T[h2][t2];
	x2 = tt2 % 5;
	y2 = tt2 / 5;
	rr2 = R[h2][r2];

	if (((rr1 + 2) % 4) != rr2) {
		return FALSE;
		}
	if (y1 != 1) {
		return FALSE;
		}
	if (y2 != 1) {
		return FALSE;
		}
	if (x1 + x2 != 3) {
		return FALSE;
		}
	return TRUE;
}

INT number_of_pieces_of_type(INT t, INT *set)
{
	INT i, a, cnt = 0;

	for (i = 0; i < 5; i++) {
		a = set[i];
		if (a >= var_start[t] && a < var_start[t + 1]) {
			cnt++;
			}
		}
	return cnt;
}

INT does_it_contain_an_I(INT *set)
{
	INT i, a;

	for (i = 0; i < 5; i++) {
		a = set[i];
		if (a >= var_start[3] && a < var_start[4]) {
			return TRUE;
			}
		}
	return FALSE;
}

void decode_assembly(INT *set)
// input set[5]
{
	INT i, h, r, t, tt, x, y, rr;

	cout << "Set ";
	INT_vec_print(cout, set, 5);
	cout << endl;

	for (i = 0; i < 5; i++) {
		decode_piece(set[i], h, r, t);
		tt = T[h][t];
		x = tt % 5;
		y = tt / 5;
		rr = R[h][r];
		cout << "h=" << h << " r=" << r << " t=" << t << " tt=" << tt << " x=" << x << " y=" << y << " rr=" << rr << endl;
		}
}

void decode_piece(INT j, INT &h, INT &r, INT &t)
// h is the kind of piece
// r is the rotation index
// t is the translation index
// to get the actual rotation rr and translation tt, use 
// rr = R[h][r] and tt = T[h][t].
// To get the x and y shift from tt, use:
// x = tt % 5;
// y = tt / 5;

{
	INT j0;
	
	for (h = 0; h < NB_PIECES; h++) {
		j0 = var_start[h + 1];
		if (j0 > j) {
			j -= var_start[h];
			t = j % T_length[h];
			j -= t;
			r = j / T_length[h];
			break;
			}
		}
}

INT code_piece(INT h, INT r, INT t)
{
	INT j;

	j = var_start[h] + r * T_length[h] + t;
	return j;
}


void draw_it(ostream &ost, INT *sol)
{
	INT sol_length = 5;
	INT u, h, r, rr, t, tt, tx, ty, s, a, b, x, y, j;
	INT *O1;

	ost << "\\begin{tikzpicture}[x=1cm, y=1cm, semitransparent, scale=0.5]" << endl;
	ost << "\\draw[step=1cm, line width=0.3mm, black!30!white] (0,0) grid (5cm,5cm);" << endl;
	for (u = 0; u < sol_length; u++) {
		j = sol[u];
		decode_piece(j, h, r, t);
		//cout << "% j=" << j << " h=" << h << " r=" << r << " t=" << t << endl;
		tt = T[h][t];
		tx = tt % 5;
		ty = tt / 5;
		rr = R[h][r];

		O1 = NEW_INT(O_length[h]);
		for (s = 0; s < O_length[h]; s++) {
			a = O[h][s];
			a += tx;
			a += 6 * ty;
			b = Rotate6[rr * 36 + a];
			O1[s] = b;
			}
		ost << "\\draw [very thick] ";
		for (s = 0; s < O_length[h]; s++) {
			a = O1[s];
			x = a % 6;
			y = 5 - a / 6;
			ost << "(" << x << "," << y << ")";
			if (s < O_length[h] - 1) {
				ost << " -- ";
				}
			}
		ost << ";" << endl;
		FREE_INT(O1);
		}
	ost << "\\end{tikzpicture}" << endl;
}

void compute_image_function(set_of_sets *S, void *compute_image_data, INT elt_idx, INT gen_idx, INT &idx_of_image, INT verbose_level)
// implements a rotation by 90 degree:
{
	//INT verbose_level = 0;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *set1;
	INT *set2;
	INT sz, i, a, b, h, r, t, idx;

	set1 = S->Sets[elt_idx];
	sz = S->Set_size[elt_idx];
	set2 = NEW_INT(sz);


	if (f_v) {
		cout << "compute_image_function computing image of solution " << elt_idx << " = ";
		INT_vec_print(cout, set1, sz);
		cout << " under generator " << gen_idx << endl;
		}

	for (i = 0; i < sz; i++) {
		a = set1[i];
		decode_piece(a, h, r, t);
		if (f_vv) {
			cout << "a=" << a << " h=" << h << " r=" << r << " t=" << t << " -> ";
			}
		if (gen_idx == 0) {
			turn_piece(h, r, t, verbose_level - 1);
			}
		else if (gen_idx == 1) {
			flip_piece(h, r, t, verbose_level - 1);
			}
		else {
			cout << "compute_image_function gen_idx unrecognized" << endl;
			exit(1);
			}
		b = code_piece(h, r, t);
		if (f_vv) {
			cout << "b=" << b << " h=" << h << " r=" << r << " t=" << t << endl;
			}
		set2[i] = b;
		}
	INT_vec_heapsort(set2, sz);
	if (!vec_search_general(S, 
		compare_func, 
		NULL /* void *data_for_compare */, 
		S->nb_sets, set2, idx, 0 /*verbose_level*/)) {
		cout << "compute_image_function cannot find image" << endl;
		INT_vec_print(cout, set2, sz);
		cout << endl;
		exit(1);
		}
	idx_of_image = idx;
	if (f_v) {
		cout << "compute_image_function image is ";
		INT_vec_print(cout, set2, sz);
		cout << " which is solution " << idx_of_image << endl;
		}
	FREE_INT(set2);

}

INT compare_func(void *vec, void *a, INT b, void *data_for_compare)
{
	set_of_sets *S = (set_of_sets *) vec;
	INT sz, c;

	sz = S->Set_size[b];
	c = INT_vec_compare((INT *) a, S->Sets[b], sz);
#if 0
	cout << "compare ";
	INT_vec_print(cout, (INT *) a, sz);
	cout << " : ";
	INT_vec_print(cout, S->Sets[b], sz);
	cout << " yields " << c << endl;
#endif
	return -c;	
}
 
void turn_piece(INT &h, INT &r, INT &t, INT verbose_level)
{
	INT tx, ty, txx = 0, tyy = 0, tt;

	tt = T[h][t];
	tx = tt % 5;
	ty = tt / 5;
	txx = tx;
	tyy = ty;
	if (h == 0) { // X
		txx = 2 - ty;
		tyy = tx;
		}
	else if (h == 3 && r == 1) { // I
		txx = 4 - tx;
		tyy = ty;
		}
	else if (h == 12 && r == 1) { // Z
		txx = 2 - tx;
		tyy = 2 - ty;
		}
	else if (h == 17 && r == 1) { // Z'
		txx = 2 - tx;
		tyy = 2 - ty;
		}
	r++;
	r %= R_length[h];
	tt = tyy * 5 + txx;
	if (!INT_vec_search_linear(T[h], T_length[h], tt, t)) {
		cout << "turn_piece cannot find tt=" << tt << " for h=" << h << endl;
		exit(1);
		}
}

void flip_piece(INT &h, INT &r, INT &t, INT verbose_level)
{
	//INT verbose_level = 0;
	INT f_v = (verbose_level >= 1);
	INT tx, ty, txx = 0, tyy = 0, tt;

	if (f_v) {
		cout << "flip_piece" << endl;
		}
	tt = T[h][t];
	tx = tt % 5;
	ty = tt / 5;
	txx = tx;
	tyy = ty;
	if (f_v) {
		cout << "r=" << r << " tt=" << tt << " x=" << tx << " y=" << ty << endl;
		}
	if (h == 0) { // X
		txx = 2 - tx;
		tyy = ty;
		}
	else if (h == 3 && r == 0) { // I
		txx = 4 - tx;
		tyy = ty;
		}
	else if (h == 7 || h == 8) { // T or U 
		if (r == 1) {
			r = 3;
			}
		else if (r == 3) {
			r = 1;
			}
		txx = 2 - tx;
		tyy = ty;
		}
	else if (h == 9 || h == 10) { // V or W
		if (r == 0) {
			r = 3;
			}
		else if (r == 1) {
			r = 2;
			}
		else if (r == 2) {
			r = 1;
			}
		else if (r == 3) {
			r = 0;
			}
		txx = 2 - ty;
		tyy = 2 - tx;
		}
	else if (h == 12 || h == 17) { // Z (12) or Z' (17)
		if (h == 12) {
			h = 17;
			}
		else {
			h = 12;
			}
		if (r == 1) {
			txx = tx;
			tyy = 2 - ty;
			}
		else {
			txx = 2 - tx;
			tyy = ty;
			}
		}
	else if (h == 11 || h == 16) { // Y (11) or Y' (16)
		if (h == 11) {
			h = 16;
			}
		else {
			h = 11;
			}
		if (r == 1) {
			r = 3;
			}
		else if (r == 3) {
			r = 1;
			}
		txx = 3 - tx;
		tyy = ty;
		}
	else if (h == 4 || h == 13) { // L (4) or L' (13)
		if (h == 4) {
			h = 13;
			}
		else {
			h = 4;
			}
		if (r == 1) {
			r = 3;
			}
		else if (r == 3) {
			r = 1;
			}
		txx = 3 - tx;
		tyy = ty;
		}
	else if (h == 5 || h == 14) { // N (5) or N' (14)
		if (h == 5) {
			h = 14;
			}
		else {
			h = 5;
			}
		if (r == 1) {
			r = 3;
			}
		else if (r == 3) {
			r = 1;
			}
		txx = 3 - tx;
		tyy = ty;
		}
	else if (h == 6 || h == 15) { // P (6) or P' (15)
		if (h == 6) {
			h = 15;
			}
		else {
			h = 6;
			}
		if (r == 1) {
			r = 3;
			}
		else if (r == 3) {
			r = 1;
			}
		txx = 3 - tx;
		tyy = ty;
		}
	else if (h == 1 || h == 2) { // F (1) or F' (2)
		if (h == 1) {
			h = 2;
			}
		else {
			h = 1;
			}
		if (r == 1) {
			r = 3;
			}
		else if (r == 3) {
			r = 1;
			}
		txx = 2 - tx;
		tyy = ty;
		}
	tt = tyy * 5 + txx;
	if (f_v) {
		cout << "r=" << r << " x'=" << txx << " y'=" << tyy << " tt=" << tt << endl;
		}
	if (!INT_vec_search_linear(T[h], T_length[h], tt, t)) {
		cout << "flip_piece cannot find tt=" << tt << " for h=" << h << endl;
		exit(1);
		}
	if (f_v) {
		cout << "h'=" << h << " r'=" << r << " t'=" << t << endl;
		}
	if (f_v) {
		cout << "flip_piece done" << endl;
		}
}


void setup_pieces()
{
	// pieces on a 5 x 5 grid:
	INT S1[] = {1,5,6,7,11,-1}; // X, Plus
	INT S2[] = {1,2,5,6,11,-1}; // F
	INT S3[] = {0,1,6,7,11,-1}; // F'
	INT S4[] = {0,5,10,15,20,-1}; // I
	INT S5[] = {0,5,10,15,16,-1}; // L
	INT S6[] = {1,6,10,11,15,-1}; // N
	INT S7[] = {0,1,5,6,10,-1}; // P
	INT S8[] = {0,1,2,6,11,-1}; // T
	INT S9[] = {0,2,5,6,7,-1}; // U
	INT S10[] = {0,5,10,11,12,-1}; // V
	INT S11[] = {0,5,6,11,12,-1}; // W
	INT S12[] = {1,5,6,11,16,-1}; // Y
	INT S13[] = {0,1,6,11,12,-1}; // Z
	INT S14[] = {1,6,11,15,16,-1}; // L'
	INT S15[] = {0,5,10,11,16,-1}; // N'
	INT S16[] = {0,1,5,6,11,-1}; // P'
	INT S17[] = {0,5,6,10,15,-1}; // Y'
	INT S18[] = {1,2,6,10,11,-1}; // Z'

	//outline on a 6 x 6 grid:
	INT O1[] = {1,2,8,9,15,14,20,19,13,12,6,7,1,-1}; // X, Plus
	INT O2[] = {1,3,9,8,20,19,13,12,6,7,1,-1}; // F
	INT O3[] = {0,2,8,9,15,14,20,19,7,6,0,-1}; // F'
	INT O4[] = {0,1,31,30,0,-1}; // I
	INT O5[] = {0,1,19,20,26,24,0,-1}; // L
	INT O6[] = {1,2,20,19,25,24,12,13,1,-1}; // N
	INT O7[] = {0,2,14,13,19,18,0,-1}; // P
	INT O8[] = {0,3,9,8,20,19,7,6,0,-1}; // T
	INT O9[] = {0,1,7,8,2,3,15,12,0,-1}; // U
	INT O10[] = {0,1,13,15,21,18,0,-1}; // V
	INT O11[] = {0,1,7,8,14,15,21,19,13,12,0,-1}; // W
	INT O12[] = {1,2,26,25,13,12,6,7,1,-1}; // Y
	INT O13[] = {0,2,14,15,21,19,7,6,0,-1}; // Z
	INT O14[] = {1,2,26,24,18,19,1,-1}; // L'
	INT O15[] = {0,1,13,14,26,25,19,18,0,-1}; // N'
	INT O16[] = {0,2,20,19,13,12,0,-1}; // P'
	INT O17[] = {0,1,7,8,14,13,25,24,0,-1}; // Y'
	INT O18[] = {1,3,9,8,20,18,12,13,1,-1}; // Z'
	

	// translations:
	INT T1[] = {0,1,2,5,6,7,10,11,12,-1}; // X, Plus
	INT T2[] = {0,1,2,5,6,7,10,11,12,-1}; // F
	INT T3[] = {0,1,2,5,6,7,10,11,12,-1}; // F'
	INT T4[] = {0,1,2,3,4,-1}; // I
	INT T5[] = {0,1,2,3,5,6,7,8,-1}; // L
	INT T6[] = {0,1,2,3,5,6,7,8,-1}; // N
	INT T7[] = {0,1,2,3,5,6,7,8,10,11,12,13,-1}; // P
	INT T8[] = {0,1,2,5,6,7,10,11,12,-1}; // T
	INT T9[] = {0,1,2,5,6,7,10,11,12,15,16,17,-1}; // U
	INT T10[] = {0,1,2,5,6,7,10,11,12,-1}; // V
	INT T11[] = {0,1,2,5,6,7,10,11,12,-1}; // W
	INT T12[] = {0,1,2,3,5,6,7,8,-1}; // Y
	INT T13[] = {0,1,2,5,6,7,10,11,12,-1}; // Z
	INT T14[] = {0,1,2,3,5,6,7,8,-1}; // L'
	INT T15[] = {0,1,2,3,5,6,7,8,-1}; // N'
	INT T16[] = {0,1,2,3,5,6,7,8,10,11,12,13,-1}; // P'
	INT T17[] = {0,1,2,3,5,6,7,8,-1}; // Y'
	INT T18[] = {0,1,2,5,6,7,10,11,12,-1}; // Z'

	//rotations:
	INT R1[] = {0,-1}; // X, Plus
	INT R2[] = {0,1,2,3,-1}; // F
	INT R3[] = {0,1,2,3,-1}; // F'
	INT R4[] = {0,1,-1}; // I
	INT R5[] = {0,1,2,3,-1}; // L
	INT R6[] = {0,1,2,3,-1}; // N
	INT R7[] = {0,1,2,3,-1}; // P
	INT R8[] = {0,1,2,3,-1}; // T
	INT R9[] = {0,1,2,3,-1}; // U
	INT R10[] = {0,1,2,3,-1}; // V
	INT R11[] = {0,1,2,3,-1}; // W
	INT R12[] = {0,1,2,3,-1}; // Y
	INT R13[] = {0,1,-1}; // Z
	INT R14[] = {0,1,2,3,-1}; // L
	INT R15[] = {0,1,2,3,-1}; // N'
	INT R16[] = {0,1,2,3,-1}; // P'
	INT R17[] = {0,1,2,3,-1}; // Y'
	INT R18[] = {0,1,-1}; // Z'


	INT i, j;

	S[0] = S1;
	S[1] = S2;
	S[2] = S3;
	S[3] = S4;
	S[4] = S5;
	S[5] = S6;
	S[6] = S7;
	S[7] = S8;
	S[8] = S9;
	S[9] = S10;
	S[10] = S11;
	S[11] = S12;
	S[12] = S13;
	S[13] = S14;
	S[14] = S15;
	S[15] = S16;
	S[16] = S17;
	S[17] = S18;

	O[0] = O1;
	O[1] = O2;
	O[2] = O3;
	O[3] = O4;
	O[4] = O5;
	O[5] = O6;
	O[6] = O7;
	O[7] = O8;
	O[8] = O9;
	O[9] = O10;
	O[10] = O11;
	O[11] = O12;
	O[12] = O13;
	O[13] = O14;
	O[14] = O15;
	O[15] = O16;
	O[16] = O17;
	O[17] = O18;

	T[0] = T1;
	T[1] = T2;
	T[2] = T3;
	T[3] = T4;
	T[4] = T5;
	T[5] = T6;
	T[6] = T7;
	T[7] = T8;
	T[8] = T9;
	T[9] = T10;
	T[10] = T11;
	T[11] = T12;
	T[12] = T13;
	T[13] = T14;
	T[14] = T15;
	T[15] = T16;
	T[16] = T17;
	T[17] = T18;

	R[0] = R1;
	R[1] = R2;
	R[2] = R3;
	R[3] = R4;
	R[4] = R5;
	R[5] = R6;
	R[6] = R7;
	R[7] = R8;
	R[8] = R9;
	R[9] = R10;
	R[10] = R11;
	R[11] = R12;
	R[12] = R13;
	R[13] = R14;
	R[14] = R15;
	R[15] = R16;
	R[16] = R17;
	R[17] = R18;

	for (i = 0; i < NB_PIECES; i++) {
		for (j = 0; ; j++) {
			if (S[i][j] == -1) {
				S_length[i] = j;
				break;
				}
			}
		for (j = 0; ; j++) {
			if (O[i][j] == -1) {
				O_length[i] = j;
				break;
				}
			}
		for (j = 0; ; j++) {
			if (T[i][j] == -1) {
				T_length[i] = j;
				break;
				}
			}
		for (j = 0; ; j++) {
			if (R[i][j] == -1) {
				R_length[i] = j;
				break;
				}
			}
		}
}

void setup_rotate()
{
	INT i, j, ii, jj, h;
	
	for (i = 0; i < 5; i++) {
		for (j = 0; j < 5; j++) {
			Rotate[0 * 25 + i * 5 + j] = i * 5 + j;
			}
		}
	for (i = 0; i < 5; i++) {
		jj = 4 - i;
		for (j = 0; j < 5; j++) {
			ii = j;
			Rotate[1 * 25 + i * 5 + j] = ii * 5 + jj;
			}
		}
	for (i = 0; i < 25; i++) {
		Rotate[2 * 25 + i] = Rotate[1 * 25 + Rotate[1 * 25 + i]];
		}
	for (i = 0; i < 25; i++) {
		Rotate[3 * 25 + i] = Rotate[2 * 25 + Rotate[1 * 25 + i]];
		}

	cout << "Rotate:" << endl;
	for (h = 0; h < 4; h++) {
		for (j = 0; j < 25; j++) {
			cout << setw(3) << Rotate[h * 25 + j] << " ";
			}
		cout << endl;
		}


	for (i = 0; i < 6; i++) {
		for (j = 0; j < 6; j++) {
			Rotate6[0 * 36 + i * 6 + j] = i * 6 + j;
			}
		}
	for (i = 0; i < 6; i++) {
		jj = 5 - i;
		for (j = 0; j < 6; j++) {
			ii = j;
			Rotate6[1 * 36 + i * 6 + j] = ii * 6 + jj;
			}
		}
	for (i = 0; i < 36; i++) {
		Rotate6[2 * 36 + i] = Rotate6[1 * 36 + Rotate6[1 * 36 + i]];
		}
	for (i = 0; i < 36; i++) {
		Rotate6[3 * 36 + i] = Rotate6[2 * 36 + Rotate6[1 * 36 + i]];
		}

	cout << "Rotate6:" << endl;
	for (h = 0; h < 4; h++) {
		for (j = 0; j < 36; j++) {
			cout << setw(3) << Rotate6[h * 36 + j] << " ";
			}
		cout << endl;
		}


}

void setup_var_start()
{
	INT h;
	
	var_start[0] = 0;
	for (h = 0; h < NB_PIECES; h++) {
		var_length[h] = R_length[h] * T_length[h];
		var_start[h + 1] = var_start[h] + var_length[h];
		}
	cout << "i : var_start[i] : var_length[i]" << endl;
	for (h = 0; h < NB_PIECES; h++) {
		cout << h << " : " << var_start[h] << " : " << var_length[h] << endl;
		}
}


void make_coefficient_matrix(diophant *D)
{
	INT i, h, j0, r, t, rr, tt, s, x, y, z;

	for (h = 0; h < NB_PIECES; h++) {
		j0 = var_start[h];

		//cout << "h=" << h << "/" << NB_PIECES << " j0=" << j0 << ":" << endl;
		for (r = 0; r < R_length[h]; r++) {
			rr = R[h][r];
			//cout << "h=" << h << "/" << NB_PIECES << " r=" << r << "/" << R_length[h] << " rr=" << rr << ":" << endl;
			for (t = 0; t < T_length[h]; t++) {
				tt = T[h][t];
				//cout << "h=" << h << "/" << NB_PIECES << " r=" << r << "/" << R_length[h] << " rr=" << rr << " t=" << t << "/" << T_length[h] << " tt=" << tt << ":" << endl;
				for (s = 0; s < S_length[h]; s++) {
					x = S[h][s];
					y = x + tt;
					z = Rotate[rr * 25 + y];
					
					//cout << "h=" << h << "/" << 6 << " r=" << r << "/" << R_length[h] << " rr=" << rr << " t=" << t << "/" << T_length[h] << " tt=" << tt << " s=" << s << "/" << S_length[h] << " x=" << x << " y=" << y << " z=" << z << " entry=(" << z << "," << j0 + r * T_length[h] + t << ")" << endl;
					D->Aij(z, j0 + r * T_length[h] + t) = 1;
					//M[z * nb_vars + j0 + r * T_length[h] + t] = 1;
					}
				}
			}
		}

#if 0
	// make the bottom rows to ensure that each piece is chosen exactly once:
	for (h = 0; h < 6; h++) {
		j0 = var_start[h];

		for (r = 0; r < R_length[h]; r++) {
			for (t = 0; t < T_length[h]; t++) {
				D->Aij(nb_eqn1 + h, j0 + r * T_length[h] + t) = 1;
				//M[(nb_eqn1 + h) * nb_vars + j0 + r * T_length[h] + t] = 1;
				}
			}
		}
#endif

	for (i = 0; i < D->m; i++) {
		D->RHS[i] = 1;
		}
	D->sum = 5;
}

