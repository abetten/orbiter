// test.C
// 
// Anton Betten
// Sept 2, 2008
//
//
// 
//
//

#include "orbiter.h"


// global data:

INT t0; // the system time when the program started
INT nb_sol = 0;

void Wilson();
void SOLS(INT n);
void SOLS_recursion(INT n, INT i, INT j, INT *A, INT *pairs);
void MacNeish();
INT is_self_orthogonal(INT *A, INT n);
void MacNeish_product(INT *A, INT m, INT *B, INT n, INT *C);
void test1();
void test2();
void Niketa();
void Eric();
void NateBurch();
void PatrickWahl();
INT count(INT n);
void count_recursion(INT n, INT *digits, INT depth);
void draw_action(char *fname_base, INT xmax, INT ymax, INT *Adjacency, INT m, INT n, INT f_cartesian);
void draw_action_(mp_graphics &G, INT *Adjacency, INT m, INT n, INT f_cartesian);
void Halls_theorem();
void draw_matrix(matrix &H, char *fname, INT xmax, INT ymax, INT f_cartesian);
void test_hall(INT n, INT nb_inc, INT verbose_level);
void matchmaker(matrix &H, Vector &matching, Vector &matching_inv, INT verbose_level);
INT search_new_boy(matrix &H, Vector &x, Vector &y, INT l, INT &prev);
void get_random_matrix(INT n, INT nb_inc, matrix &H, INT verbose_level);
void get_random_Hall_matrix(INT n, INT nb_inc, matrix &H, INT verbose_level);
INT test_hall_condition(matrix &A, Vector &S, INT n, INT &k);
INT test_hall_condition1(matrix &A, Vector &S, INT n, INT k);
void one_by_one();
void MISSISSIPPI();
void print_letter(int i);
void palindromic_problem();
INT distinct_digits(INT i);
INT palindrom(INT i);
void rank_set();



int main(int argc, char **argv)
{
	//INT verbose_level = 2;
	t0 = os_ticks();
#if 0
	if (argc <= 1) {
		usage(argc, argv);
		exit(1);
		}
#endif
	//test1();
	//Niketa();
	//Eric();
	//NateBurch();
	//PatrickWahl();
	//test2();
	//MacNeish();
	//SOLS(5);
	//Wilson();
	//test_hall(10, 60, verbose_level);
	//one_by_one();
	//MISSISSIPPI();
	//find_primitive_root(31);
	//palindromic_problem();
	rank_set();
	
	the_end_quietly(t0);
}

void Wilson()
{
	INT b, i, j, u, v, a1, a2;
	INT *line;
	
	INT A[13*13];
	INT lines[13 * 4*4] = {
		0,1,2,3,
		0,4,5,6,
		0,7,8,9,
		0,10,11,12,
		1,4,7,10,
		1,5,8,11,
		1,6,9,12,
		2,4,9,11,
		2,5,7,12,
		2,6,8,10,
		3,4,8,12,
		3,5,9,10,
		3,6,7,11,
		};
	INT L4[4*4] = {
		0,3,1,2,
		2,1,3,0,
		3,0,2,1,
		1,2,0,3
		};
	INT L[13][4*4];
	for (b = 0; b < 13; b++) {
		line = lines + b * 4;
		for (i = 0; i < 4; i++) {
			for (j = 0; j < 4; j++) {
				L[b][i * 4 + j] = line[L4[i * 4 + j]];
				}
			}
		cout << "L[" << b << "]:" << endl;
		for (i = 0; i < 4; i++) {
			for (j = 0; j < 4; j++) {
				cout << setw(3) << L[b][i * 4 + j];
				}
			cout << endl;
			}
		
		}
	for (b = 0; b < 13; b++) {
		A[b * 13 + b] = b;
		}
	for (b = 0; b < 13; b++) {
		line = lines + b * 4;
		for (u = 0; u < 4; u++) {
			i = line[u];
			for (v = u + 1; v < 4; v++) {
				j = line[v];
				a1 = L[b][u * 4 + v];
				a2 = L[b][v * 4 + u];
				A[i * 13 + j] = a1;
				A[j * 13 + i] = a2;
				}
			}
		}

	cout << "A:" << endl;
	for (i = 0; i < 13; i++) {
		for (j = 0; j < 13; j++) {
			cout << setw(3) << A[i * 13 + j];
			}
		cout << endl;
		}
	if (is_self_orthogonal(A, 13)) {
		cout << "is self-orthogonal" << endl;
		}
	else {
		cout << "is NOT self-orthogonal" << endl;
		}	
}

void SOLS(INT n)
{
	INT *A;
	INT *pairs;
	INT i;
	
	A = new INT[n * n];
	pairs = new INT[n * n];
	for (i = 0; i < n * n; i++) {
		pairs[i] = FALSE;
		A[i] = 0;
		}
	SOLS_recursion(n, 0, 0, A, pairs);
}


void SOLS_recursion(INT n, INT i, INT j, INT *A, INT *pairs)
{
	INT prev[100];
	INT alphabet[100];
	INT u, h, k, l, p1, p2, a, b, ii, jj;
	
	if (FALSE) {
		cout << "SOLS_recursion i=" << i << " j=" << j << endl;
		for (ii = 0; ii < n; ii++) {
			for (jj = 0; jj < n; jj++) {
				cout << setw(2) << A[ii * n + jj];
				}
			cout << endl;
			}
		}
	if (i == n) {
		nb_sol++;
		cout << "solution " << nb_sol << " is:" << endl;
		for (ii = 0; ii < n; ii++) {
			for (jj = 0; jj < n; jj++) {
				cout << setw(2) << A[ii * n + jj];
				}
			cout << endl;
			}
		return;
		}
	for (ii = 0; ii < i; ii++) {
		prev[ii] = A[ii * n + j];
		}
	l = i;
	for (jj = 0; jj < j; jj++) {
		prev[l++] = A[i * n + jj];
		}
	INT_vec_sort(l, prev);
	//cout << "prev=";
	//INT_vec_print(cout, prev, l);
	//cout << endl;
	h = 0;
	u = 0;
	for (k = 0; k < n; k++) {
		if (h < l && k == prev[h]) {
			while (prev[h] == k)
				h++;
			continue;
			}
		alphabet[u++] = k;
		}
	//cout << "alphabet=";
	//INT_vec_print(cout, alphabet, u);
	//cout << endl;
	for (k = 0; k < u; k++) {
		a = alphabet[k];
		A[i * n + j] = a;
		p1 = -1;
		p2 = -1;
		if (i == j) {
			p1 = a * n + a;
			if (pairs[p1])
				continue;
			}
		if (i > j) {
			b = A[j * n + i];
			p1 = a * n + b;
			if (pairs[p1])
				continue;
			p2 = b * n + a;
			if (pairs[p2])
				continue;
			}
		if (p1 >= 0)
			pairs[p1] = TRUE;
		if (p2 >= 0)
			pairs[p2] = TRUE;
		if (j < n - 1) {
			SOLS_recursion(n, i, j + 1, A, pairs);
			}
		else {
			SOLS_recursion(n, i + 1, 0, A, pairs);
			}
		
		if (p1 >= 0)
			pairs[p1] = FALSE;
		if (p2 >= 0)
			pairs[p2] = FALSE;
		}
}

void MacNeish()
{
	INT A[] = {
		0,3,1,2,
		2,1,3,0,
		3,0,2,1,
		1,2,0,3};
	INT C[16 * 16];
	INT i, j;
	
	
	MacNeish_product(A, 4, A, 4, C);
	for (i = 0; i < 16; i++) {
		for (j = 0; j < 16; j++) {
			cout << setw(4) << C[i * 16 + j];
			}
		cout << endl;
		}
	if (is_self_orthogonal(C, 16)) {
		cout << "is self-orthogonal" << endl;
		}
	else {
		cout << "is NOT self-orthogonal" << endl;
		}	
}

INT is_self_orthogonal(INT *A, INT n)
{
	INT *vec;
	INT i, j, a, b, c;
	
	vec = new INT[n * n];
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			a = A[i * n + j];
			b = A[j * n + i];
			c = a * n + b;
			vec[i * n + j] = c;
			}
		}
	INT_vec_sort(n * n, vec);
	for (i = 0; i < n * n; i++) {
		if (vec[i] != i)
			return FALSE;
		}
	return TRUE;
}


void MacNeish_product(INT *A, INT m, INT *B, INT n, INT *C)
{
	INT i, j, u, v, mn, a, b, c, ii, jj;
	
	mn = m * n;
	for (u = 0; u < n; u++) {
		for (v = 0; v < n; v++) {
			b = B[u * n + v];
			for (i = 0; i < m; i++) {
				ii = u * m + i;
				for (j = 0; j < m; j++) {
					jj = v * m + j;
					a = A[i * m + j];
					c = a + m * b;
					C[ii * mn + jj] = c;
					}
				}
			}
		}
}

// find the number of positive integers less than 10^n whose digits are nondecreasing:

void test1()
{
	INT n, k, a, b, S;
	
	cout << "test1" << endl;
	for (n = 1; n <= 15; n++) {
		S = 0;
		for (k = 1; k <= 10; k++) {
			a = INT_n_choose_k(n - 1, k - 1);
			//cout << n - 1 << " choose " << k - 1 << " = " << a << endl;
			b = INT_n_choose_k(10, k);
			//cout << 10 << " choose " << k << " = " << b << endl;
			//cout << "together = " << a * b << endl;
			S += a * b;
			}
		S--;
		cout << setw(3) << n << " : " << setw(10) << S << endl;
		}
}

void test2()
{
	INT n, S;
	
	cout << "test2" << endl;
	for (n = 1; n <= 15; n++) {
		S = count(n) - 1;
		cout << setw(3) << n << " : " << setw(10) << S << endl;
		}
}

void Niketa()
{
	INT n, k, a, S;
	
	cout << "Niketa" << endl;
	for (n = 1; n <= 15; n++) {
		S = 0;
		for (k = 0; k <= 9; k++) {
			a = INT_n_choose_k(n + k - 1, n - 1);
			S += a;
			}
		S--;
		cout << setw(3) << n << " : " << setw(10) << S << endl;
		}
}

void Eric()
{
	INT n, S;
	
	cout << "Eric" << endl;
	for (n = 1; n <= 15; n++) {
		S = INT_n_choose_k(n + 10 - 1, n) - 1;
		cout << setw(3) << n << " : " << setw(10) << S << endl;
		}
}


void NateBurch()
{
	INT n, S;
	
	cout << "NateBurch" << endl;
	for (n = 1; n <= 15; n++) {
		S = INT_n_choose_k(n + 9, 9) - 1;
		cout << setw(3) << n << " : " << setw(10) << S << endl;
		}
}


void PatrickWahl()
{
	INT n, i, j, a, S;
	
	cout << "PatrickWahl" << endl;
	for (n = 1; n <= 15; n++) {
		S = 1;
		for (i = 0; i <= n - 1; i++) {
			for (j = 0; j <= 8; j++) {
				a = INT_n_choose_k(i + j, i);
				//cout << i + j << " choose " << i << " = " << a << endl;
				S += a;
				}
			}
		S--;
		cout << setw(3) << n << " : " << setw(10) << S << endl;
		}
}



INT count(INT n)
{
	INT *digits;
	
	digits = new INT[n];
	nb_sol = 0;
	count_recursion(n, digits, 0);
	return nb_sol;
}

void count_recursion(INT n, INT *digits, INT depth)
{
	INT i, f;
	
	if (depth == n) {
		for (i = 0; i < n; i++) {
			if (digits[i])
				break;
			}
		if (i < n) {
			nb_sol++;
#if 1
			cout << setw(10) << nb_sol << " & ";
			for (i = 0; i < depth; i++) {
				cout << digits[i];
				}
			cout << "\\\\" << endl;
#endif
			}
		return;
		}
	if (depth == 0)
		f = 0;
	else
		f = digits[depth - 1];
	for (i = f; i <= 9; i++) {
		digits[depth] = i;
		count_recursion(n, digits, depth + 1);
		}

}

void draw_action(char *fname_base, INT xmax, INT ymax, INT *Adjacency, INT m, INT n, INT f_cartesian)
{
	INT x_min = -1500, x_max = 1500;
	INT y_min = -1500, y_max = 1500;
	INT factor_1000 = 1000;
	char fname_full[1000];
	INT f_embedded = TRUE;
	INT f_sideways = FALSE;
	
	sprintf(fname_full, "%s.mp", fname_base);
	{
	mp_graphics G(fname_full, x_min, y_min, x_max, y_max, f_embedded, f_sideways);
	G.out_xmin() = 0;
	G.out_ymin() = 0;
	G.out_xmax() = xmax;
	G.out_ymax() = ymax;
	cout << "xmax/ymax = " << xmax << " / " << ymax << endl;
	
	G.header();
	G.begin_figure(factor_1000);
	
	draw_action_(G, Adjacency, m, n, f_cartesian);


	G.end_figure();
	G.footer();
	}
	cout << "written file " << fname_full << " of size " << file_size(fname_full) << endl;
	
}


void draw_action_(mp_graphics &G, INT *Adjacency, INT m, INT n, INT f_cartesian)
{
	INT PX[100], PY[100];
	double delta_phi1;
	double delta_phi2;
	double phi0, phi1, d_phi, phi_t, phi_tp1;
	double r0, r1, r0a, r1a, dr, r_t, r_tp1;
	double x1, y1, x2, y2;
	double offset;
	INT step;
	INT i, j, t;
	INT rad;
	char str[1000];
	
	if (f_cartesian) {
		delta_phi1 = 2000 / m;
		delta_phi2 = 2000 / n;
		step = 1;
		offset = 70.;
		rad = 15;
		}
	else {
		delta_phi1 = (2. * M_PI) / m;
		delta_phi2 = (2. * M_PI) / n;
		step = 25;
		offset = 30.;
		rad = 5;
		}
	r0 = 400;
	r1 = 1200;
	r0a = r0 - offset;
	r1a = r1 + offset;
	dr = (r1 - r0) / step;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (Adjacency[i * n+ j] == 0)
				continue;
			phi0 = i * delta_phi1;
			phi1 = j * delta_phi2;
			d_phi = (phi1 - phi0) / step;
			r_t = r0;
			phi_t = phi0;
			for (t = 0; t < step; t++) {
				phi_tp1 = phi_t + d_phi;
				r_tp1 = r_t + dr;
				if (f_cartesian) {
					x1 = phi_t;
					y1 = r_t;
					x2 = phi_tp1;
					y2 = r_tp1;
					}
				else {
					x1 = cos(phi_t) * r_t;
					y1 = sin(phi_t) * r_t;
					x2 = cos(phi_tp1) * r_tp1;
					y2 = sin(phi_tp1) * r_tp1;
					}
				PX[0] = (INT)x1;
				PY[0] = (INT)y1;
				PX[1] = (INT)x2;
				PY[1] = (INT)y2;
				G.polygon2(PX, PY, 0, 1);
				r_t = r_tp1;
				phi_t = phi_tp1;
				}
			}
		}

	for (i = 0; i < m; i++) {
		phi0 = i * delta_phi1;
		if (f_cartesian) {
			x1 = phi0;
			y1 = r0;
			}
		else {
			x1 = cos(phi0) * r0;
			y1 = sin(phi0) * r0;
			}
		PX[0] = (INT)x1;
		PY[0] = (INT)y1;
		G.sf_interior(100);
		G.sf_color(0);
		G.circle(PX[0], PY[0], rad);
		G.sf_interior(0);
		G.sf_color(0);
		G.circle(PX[0], PY[0], rad);

		phi0 = i * delta_phi1;
		if (f_cartesian) {
			x1 = phi0;
			y1 = r0a;
			}
		else {
			x1 = cos(phi0) * r0a;
			y1 = sin(phi0) * r0a;
			}
		PX[0] = (INT)x1;
		PY[0] = (INT)y1;
		sprintf(str, "$x_{%ld}$", (INT) i + 1); 
		G.aligned_text(PX[0], PY[0], "", str);
		}
	for (i = 0; i < n; i++) {
		phi1 = i * delta_phi2;
		if (f_cartesian) {
			x1 = phi1;
			y1 = r1;
			}
		else {
			x1 = cos(phi1) * r1;
			y1 = sin(phi1) * r1;
			}
		PX[0] = (INT)x1;
		PY[0] = (INT)y1;
		G.sf_interior(100);
		G.sf_color(0);
		G.circle(PX[0], PY[0], rad);
		G.sf_interior(0);
		G.sf_color(0);
		G.circle(PX[0], PY[0], rad);

		phi1 = i * delta_phi2;
		if (f_cartesian) {
			x1 = phi1;
			y1 = r1a;
			}
		else {
			x1 = cos(phi1) * r1a;
			y1 = sin(phi1) * r1a;
			}
		PX[0] = (INT)x1;
		PY[0] = (INT)y1;
		sprintf(str, "$y_{%ld}$", (INT) i + 1); 
		G.aligned_text(PX[0], PY[0], "", str);
		}
		
	
}

void Halls_theorem()
{
	char fname[1000];
	INT xmax = 1000;
	INT ymax = 1000;
	INT Adjacency[] = {
1,1,1,0,0,0,0,0,0,0,0,0,0,
1,0,0,1,1,0,0,0,0,0,0,0,0,
1,0,0,0,0,1,1,0,0,0,0,0,0,
0,1,0,1,0,1,0,0,0,0,0,0,0,
0,1,0,0,1,0,1,0,0,0,0,0,0,
0,0,1,1,0,0,1,0,0,0,0,0,0,
0,0,1,0,0,0,0,1,1,0,0,0,0,
0,0,0,0,1,0,0,0,0,1,1,0,0,
0,0,0,0,0,1,0,0,0,0,0,1,1,
0,0,0,0,0,0,0,1,0,1,0,1,0,
0,0,0,0,0,0,0,1,0,0,1,0,1,
0,0,0,0,0,0,0,0,1,1,0,0,1,
0,0,0,0,0,0,0,0,1,0,1,1,0,
		};
	INT m = 13;
	INT n = 13;
	INT k, a;
	INT f_cartesian = FALSE;
	
	strcpy(fname, "Hall");
	for (k = 0; k < 10; k++) {
		a = random_integer(m * n);
		Adjacency[a] = 1 - Adjacency[a];
		}
	draw_action(fname, xmax, ymax, Adjacency, m, n, f_cartesian);
}

void draw_matrix(matrix &H, char *fname, INT xmax, INT ymax, INT f_cartesian)
{
	INT *Adjacency;
	INT m, n, i, j;
	
	m = H.s_m();
	n = H.s_n();
	Adjacency = new INT[m * n];
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			Adjacency[i * n + j] = H.s_iji(i, j);
			}
		}
	draw_action(fname, xmax, ymax, Adjacency, m, n, f_cartesian);
}

void test_hall(INT n, INT nb_inc, INT verbose_level)
{
	matrix H;
	Vector S;
	Vector matching, matching_inv;
	INT i, k;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT xmax = 1000;
	INT ymax = 1000;
	INT f_cartesian = TRUE;

#if 0
		get_random_Hall_matrix(n, nb_inc, H, verbose_level - 2);
		get_random_Hall_matrix(n, nb_inc, H, verbose_level - 2);
		get_random_Hall_matrix(n, nb_inc, H, verbose_level - 2);
		get_random_Hall_matrix(n, nb_inc, H, verbose_level - 2);
		get_random_Hall_matrix(n, nb_inc, H, verbose_level - 2);
		matchmaker(H, matching, matching_inv, verbose_level);

#else
	for (i = 0; i < 20; i++) {
		char fname[1000];
		if (f_v) {
			cout << "i=" << i << " ";
			}
		get_random_matrix(n, nb_inc, H, verbose_level);
		
		if (!test_hall_condition(H, S, n, k)) {
			if (f_vv) {
				cout << "H=\n" << H << endl;
				cout << "Hall condition not satisfied for selection" << S << endl;
				}
			}
		else {
			matchmaker(H, matching, matching_inv, verbose_level);
			if (f_vv) {
				cout << "H=" << endl << H << endl;
				cout << "matching=" << matching << endl;
				cout << "matching_inv=" << matching_inv << endl;
				}
			}
		sprintf(fname, "Hall_%ld", i);
		draw_matrix(H, fname, xmax, ymax, f_cartesian);
		}
#endif
}

void matchmaker(matrix &H, Vector &matching, Vector &matching_inv, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT n, i, j, size, l, nb_complications = 0;
	Vector x, y, p;
	INT prev;
	
	if (f_v) {
		cout << "matchmaker() H =" << endl;
		cout << H << endl;
		}
	n = H.s_m();
	matching.m_l_n(n);
	matching_inv.m_l_n(n);
	x.m_l_n(n + 1);
	y.m_l_n(n + 1);
	p.m_l_n(n + 1);
	for (i = 0; i < n; i++) {
		matching.m_ii(i, -1);
		matching_inv.m_ii(i, -1);
		}
	size = 0;
	while (size < n) {
		// pick first free girl:
		for (i = 0; i < n; i++)
			if (matching.s_ii(i) == -1)
				break;
		if (i == n) {
			cout << "matchmaker() error: no free girl" << endl;
			exit(1);
			}
		// there must be a free boy:
		for (j = 0; j < n; j++) {
			if (H.s_iji(i, j) == 0)
				continue;
			if (matching_inv.s_ii(j) == -1)
				break;
			}
		if (j < n) {
			if (f_vv) {
				cout << "matchmaker() extend matching: girl " 
					<< i << " gets boy " << j << endl;
				}
			matching.m_ii(i, j);
			matching_inv.m_ii(j, i);
			size++;
			continue;
			}
		// now: all friends of girl i are occupied
		for (j = 0; j < n; j++) {
			if (H.s_iji(i, j) != 0)
				break;
			}
		if (f_vv) {
			cout << "matchmaker() all friends of girl " 
				<< i << " busy, picking the first boy " << j << endl;
			}
		nb_complications++;
		x.m_ii(0, i);
		y.m_ii(1, j);
		p.m_ii(1, 0);
		x.m_ii(1, matching_inv.s_ii(j));
		l = 2;
		while (TRUE) {
			j = search_new_boy(H, x, y, l, prev);
			if (f_vv) {
				cout << "boy " << j << " prev = " << prev << endl;
				}
			y.m_ii(l, j);
			p.m_ii(l, prev);
			i = matching_inv.s_ii(j);
			if (i < 0)
				break;
			if (l == n) {
				cout << "matchmaker() error: l == n" << endl;
				exit(1);
				}
			x.m_ii(l, i);
			l++;
			if (FALSE) {
				cout << "l=" << l << endl;
				cout << "x=" << x << endl;
				cout << "y=" << y << endl;
				cout << "p=" << p << endl;
				}
			}
		if (f_vv) {
			cout << "x=" << x << endl;
			cout << "y=" << y << endl;
			cout << "p=" << p << endl;
			cout << "matching    =" << matching << endl;
			cout << "matching_inv=" << matching_inv << endl;
			}
		for ( ; l > 0; ) {
			j = y.s_ii(l);
			l = p.s_ii(l);
			i = x.s_ii(l);
			matching.m_ii(i, j);
			matching_inv.m_ii(j, i);
			}
		size++;
		if (f_vv) {
			cout << "new matching of size " << size << endl;
			cout << "matching    =" << matching << endl;
			cout << "matching_inv=" << matching_inv << endl;
			}
		}
	if (f_v) {
		cout << "nb_complications = " << nb_complications << endl;
		cout << "matching    =" << matching << endl;
		cout << "matching_inv=" << matching_inv << endl;
		}
}

INT search_new_boy(matrix &H, Vector &x, Vector &y, INT l, INT &prev)
{
	INT n, i, j, h;
	
	n = H.s_m();
	for (j = 0; j < n; j++) {
		// check boy j
		for (prev = 0; prev < l; prev++) {
			i = x.s_ii(prev);
			if (H.s_iji(i, j) != 0)
				break;
			}
		if (prev == l)
			continue;
		// cout << "testing boy " << j << ", prev " << prev << endl;
		
		// found boy which is friend to girl prev in the list x
		// test if that boy in not already contained in y[1..l-1]:
		for (h = 1; h < l; h++) {
			if (y.s_ii(h) == j)
				break;
			}
		if (h == l) {
			// no, we can return boy j
			return j;
			}
		// cout << "no, already there" << endl;
		// yes, we must search for another boy:
		}
	cout << "search_new_boy() fails, error" << endl;
	exit(1);
}

void get_random_matrix(INT n, INT nb_inc, matrix &H, INT verbose_level)
{
	integer a, b;
	INT i, j, h;
	
	H.m_mn_n(n, n);
	for (h = 0; h < nb_inc; h++) {
		a.rand(0, n - 1);
		b.rand(0, n - 1);
		i = a.s_i();
		j = b.s_i();
		if (H.s_iji(i, j)) {
			H.m_iji(i, j, 0);
			}
		else {
			H.m_iji(i, j, 1);
			}
		}
}

void get_random_Hall_matrix(INT n, INT nb_inc, matrix &H, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	Vector S;
	integer a, b;
	INT i, j, h, k, nb = 0, c;
	
	H.m_mn_n(n, n);
	for (h = 0; h < nb_inc; h++) {
		a.rand(0, n - 1);
		b.rand(0, n - 1);
		i = a.s_i();
		j = b.s_i();
		if (H.s_iji(i, j)) {
			H.m_iji(i, j, 0);
			}
		else {
			H.m_iji(i, j, 1);
			}
		}
	while (!test_hall_condition(H, S, n, k)) {
		if (f_vv) {
			cout << "H_" << nb << "=\n" << H << endl;
			cout << "Hall condition not satisfied for selection" << S << endl;
			}
		
		a.rand(0, k - 1);
		i = S.s_ii(a.s_i());
		b.rand(0, n - 1);
		j = b.s_i();
		c = H.s_iji(i, j);
		if (c) {
			c = 0;
			}
		else
			c = 1;
		H.m_iji(i, j, c);
		nb++;
		}
	if (f_v) {
		cout << "get_random_Hall_matrix():" << endl;
		cout << "H=\n" << H << endl;
		cout << "satisfies Hall condition after " << nb << " alterations:" << endl;
		}
}

INT test_hall_condition(matrix &A, Vector &S, INT n, INT &k)
{
	for (k = 1; k <= n; k++) {
		S.m_l_n(k);
		S.n_choose_k_first(n, k); 
		do {
			if (!test_hall_condition1(A, S, n, k))
				return FALSE;
			} while (S.n_choose_k_next(n, k));
		}
	return TRUE;
}

INT test_hall_condition1(matrix &A, Vector &S, INT n, INT k)
{
	Vector F;
	INT h, i, j, m;
	
	F.m_l_n(n);
	for (h = 0; h < k; h++) {
		i = S.s_ii(h);
		for (j = 0; j < n; j++) {
			if (A.s_iji(i, j)) {
				F.m_ii(j, 1);
				}
			}
		}
	m = 0;
	for (j = 0; j < n; j++) {
		if (F.s_ii(j)) {
			m++;
			}
		}
	if (m >= k)
		return TRUE;
	else
		return FALSE;
}

void one_by_one()
{
	INT n = 20;
	INT np1 = n + 1;
	INT *M;
	INT i, j;
	char fname[1000];
	
	M = new INT[np1 * np1];
	for (i = 1; i <= n; i++) {
		M[i * np1 + 0] = i;
		M[0 * np1 + i] = i;
		}
	M[0] = 0;
	for (i = 1; i <= n; i++) {
		for (j = 1; j <= n; j++) {
			M[i * np1 + j] = i * j;
			}
		}

	sprintf(fname, "one_by_one.tex");
	{
	ofstream file(fname);
	cout << "opening file " << fname << endl;

	latex_head(file, FALSE/* f_book*/, FALSE /* f_title */, NULL /*title*/, 
		"Anton Betten" /*char *author*/, FALSE/* f_toc*/, 
		FALSE /* f_landscape*/,
			FALSE /* f_12pt */, 
			TRUE /* f_enlarged_page */, 
			TRUE /* f_pagenumbers */, 
			NULL /* extra_praeamble */);
	
	file << "\\pagestyle{empty}" << endl;
	//file << "\\section{BLT set " << cnt << " over GF$(" << q << ")$}" << endl;
	
	//file << "\\Huge" << endl;
	file << "$$" << endl;
	file << "\\begin{array}{|r||*{" << n << "}{r|}}" << endl;
	file << "\\hline" << endl;
	for (i = 0; i <= n; i++) {
		for (j = 0; j <= n; j++) {
			file << setw(5) /*<< "\\,"*/ << M[i * np1 + j] /*<< "\\,"*/;
			if (j < n)
				file << " & ";
			}
		file << "\\\\" << endl;
		if (i == 0)
			file << "\\hline" << endl;
		file << "\\hline" << endl;
		}
	file << "\\end{array}" << endl;
	file << "$$" << endl; 
	latex_foot(file);
	}

	cout << "written file " << fname << " of size " << file_size(fname) << endl;
}

void MISSISSIPPI()
{
	INT word[4];
	INT m[4];
	INT i0, i1, i2, i3, i, cnt = 0;
	
	
	for (i0 = 0; i0 < 4; i0++) {
		word[0] = i0;
		for (i1 = 0; i1 < 4; i1++) {
			word[1] = i1;
			for (i2 = 0; i2 < 4; i2++) {
				word[2] = i2;
				for (i3 = 0; i3 < 4; i3++) {
					word[3] = i3;
					for (i = 0; i < 4; i++) {
						m[i] = 0;
						}
					for (i = 0; i < 4; i++) {
						m[word[i]]++;
						}
					if (m[0] > 1)
						continue;
					if (m[1] > 2)
						continue;
					INT_vec_quicksort_decreasingly(m, 4);
					if (m[0] != 2)
						continue;
					if (m[1] != 2)
						continue;
					
					cnt++;
					cout << setw(5) << cnt << " & ";
					for (i = 0; i < 4; i++) {
						print_letter(word[i]);
						}
					cout << "\\\\" << endl;
					}
				}
			}
		}
}

void print_letter(int i)
{
	if (i == 0)
		cout << "M";
	if (i == 1)
		cout << "P";
	if (i == 2)
		cout << "S";
	if (i == 3)
		cout << "I";
}

void palindromic_problem()
{
	INT i, j, a, b, year = 2012;

#if 0
	for (i = 100; i < 999; i++) {
		j = palindrom(i);
		cout << "2 * " << i << " = 9 * " << j << "?" << endl;
		if (2 * i == 9 * j) {
			cout << "yes" << endl;
			exit(1);
			}
		}
#endif
	for (i = 100; i < 999; i++) {
		if (!distinct_digits(i)) {
			continue;
			}
		j = palindrom(i);
		a = year * i;
		if ((a % j) == 0) {
			b = a / j;
			if (b % year != 0) {
				cout << year << " * " << i << " = " << b << " * " << j << endl;
				}
			}
		}

}

INT distinct_digits(INT i)
{
	INT a, b, c;

	c = i % 10;
	i -= c;
	i /= 10;
	b = i % 10;
	i -= b;
	i /= 10;
	a = i;
	if (a == b || a == c || b == c) {
		return FALSE;
		}
	return TRUE;
}

INT palindrom(INT i)
{
	INT a, b, c, j;

	c = i % 10;
	i -= c;
	i /= 10;
	b = i % 10;
	i -= b;
	i /= 10;
	a = i;
	j = c * 100 + b * 10 + a;
	return j;
}

void rank_set()
{
	INT n = 72;
	INT k = 3;
	INT set[] = {11,50,51};
	INT *set1;
	INT rk;

	set1 = NEW_INT(k);
	rk = rank_k_subset(set, n, k);
	cout << "the rank of the set ";
	INT_vec_print(cout, set, k);
	cout << " with n=" << n;
	cout << " is " << rk << endl;

	unrank_k_subset(rk, set1, n, k);
	cout << "after unrank: ";
	INT_vec_print(cout, set1, k);
	cout << endl;
	
}




