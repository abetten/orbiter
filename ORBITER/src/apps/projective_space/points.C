// points.C
// 
// Anton Betten
// 2/18/2011
//
// ranks and unranks the points in PG(n-1,q)
// 
// extended to more types of objects
// March 16, 2018
//

#include "orbiter.h"
#include "discreta.h"


// global data:

INT t0; // the system time when the program started

void orthogonal_points(INT epsilon, INT n, INT q, INT f_lines, INT verbose_level);
void orthogonal_lines(finite_field *F, INT epsilon, INT n, INT *c123, INT verbose_level);
void points(INT n, INT q, INT verbose_level);

int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	INT f_d = FALSE;
	INT d = 0;
	INT f_q = FALSE;
	INT q;
	INT f_n = FALSE;
	INT n = 0;
	INT f_orthogonal = FALSE;
	INT f_epsilon = FALSE;
	INT epsilon = 0;
	INT f_lines = FALSE;
	
 	t0 = os_ticks();
	
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
			d = atoi(argv[++i]);
			cout << "-d " << d << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-orthogonal") == 0) {
			f_orthogonal = TRUE;
			cout << "-orthogonal " << endl;
			}
		else if (strcmp(argv[i], "-epsilon") == 0) {
			f_epsilon = TRUE;
			epsilon = atoi(argv[++i]);
			cout << "-epsilon " << epsilon << endl;
			}
		else if (strcmp(argv[i], "-lines") == 0) {
			f_lines = TRUE;
			cout << "-lines " << endl;
			}
		}
	if (!f_d && !f_n) {
		cout << "please use either the -d <d> or the -n <n> option" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use -q option" << endl;
		exit(1);
		}
	if (f_n) {
		d = n + 1;
		}
	else {
		n = d - 1;
		}
	if (f_orthogonal) {
		cout << "orthogonal" << endl;
		if (ODD(n)) {
			if (!f_epsilon) {
				cout << "please use -epsilon <epsilon> option if n is odd." << endl;
				exit(1);
				}
			}
		orthogonal_points(epsilon, n, q, f_lines, verbose_level);
		}
	else {
		cout << "projective" << endl;
		points(d, q, verbose_level);
		}
	
	INT dt;
	dt = delta_time(t0);

	cout << "time in ticks " << dt << " tps=" << os_ticks_per_second() << endl;

	the_end(t0);
}

void orthogonal_points(INT epsilon, INT n, INT q, INT f_lines, INT verbose_level)
{
	INT i, j;
	INT d;
	INT N;
	INT c123[3] = {0, 0, 0};
	INT *v;
	INT *v2;
	INT *G;
	
	d = n + 1;
	N = nb_pts_Qepsilon(epsilon, n, q);
	
	cout << "number of points = " << N << endl;


	v = new INT[d];
	v2 = new INT[d];
	G = new INT[d * d];

	
	finite_field *F;
	
	F = new finite_field;
	F->init(q, verbose_level - 1);
	F->print(TRUE);
	
	if (epsilon == 0)
		c123[0] = 1;
	else if (epsilon == -1) {
		choose_anisotropic_form(*F, c123[0], c123[1], c123[2], verbose_level - 2);
		//cout << "incma.C: epsilon == -1, need irreducible polynomial" << endl;
		//exit(1);
		}
	Gram_matrix(*F, epsilon, n, c123[0], c123[1], c123[2], G);
	cout << "Gram matrix" << endl;
	print_integer_matrix_width(cout, G, d, d, d, 2);
	
	for (i = 0; i < N; i++) {
		Q_epsilon_unrank(*F, v, 1, epsilon, n, c123[0], c123[1], c123[2], i);
		cout << "P_{" << i << "} & ";
		INT_vec_print(cout, v, n + 1);
		j = Q_epsilon_rank(*F, v, 1, epsilon, n, c123[0], c123[1], c123[2]);
		cout << "\\\\" << endl;
		if (j != i) {
			cout << "orthogonal_points j != i" << endl;
			exit(1);
			}
		}


	if (f_lines) {
		orthogonal_lines(F, epsilon, n, c123, verbose_level);
		}
	
	delete F;
	FREE_INT(v);
	FREE_INT(v2);
	FREE_INT(G);
}

void orthogonal_lines(finite_field *F, INT epsilon, INT n, INT *c123, INT verbose_level)
{
	orthogonal O;
	INT p1, p2, i, j, a, len;
	INT d, q;
	INT *L;
	
	d = n + 1;
	q = F->q;
	O.init(epsilon, d, F, verbose_level);
	L = NEW_INT(2 * d);
	
	cout << "O^" << epsilon << "(" << d << "," << q << ") with " 
		<< O.nb_points << " points and " << O.nb_lines << " lines" << endl << endl;
	

#if 0
	if (f_points) {
		cout << "points:" << endl;
		for (i = 0; i < O.T1_m; i++) {
			O.unrank_point(O.v1, 1, i, 0);
			cout << i << " : ";
			INT_vec_print(cout, O.v1, n);
			j = O.rank_point(O.v1, 1, 0);
			cout << " : " << j << endl;
			}
		cout << endl;
		}
#endif

	cout << "lines:" << endl;
	len = O.nb_lines; // O.L[0] + O.L[1] + O.L[2];
	cout << "len=" << len << endl;
	for (i = 0; i < len; i++) {
		cout << "L_{" << i << "} &= ";
		O.unrank_line(p1, p2, i, 0 /* verbose_level - 1*/);
		//cout << "(" << p1 << "," << p2 << ") : ";
	
		O.unrank_point(O.v1, 1, p1, 0);
		O.unrank_point(O.v2, 1, p2, 0);

		INT_vec_copy(O.v1, L, d);
		INT_vec_copy(O.v2, L + d, d);

		cout << "\\left[" << endl;
		print_integer_matrix_tex(cout, L, 2, d);
		cout << "\\right]" << endl;

		a = O.evaluate_bilinear_form(O.v1, O.v2, 1);
		if (a) {
			cout << "not orthogonal" << endl;
			exit(1);
			}

		cout << " & ";
		j = O.rank_line(p1, p2, 0 /*verbose_level - 1*/);
		if (i != j) {
			cout << "error: i != j" << endl;
			exit(1);
			}

#if 1
		O.points_on_line(p1, p2, O.line1, 0 /*verbose_level - 1*/);
		INT_vec_sort(q + 1, O.line1);
		
		INT_set_print_masked_tex(cout, O.line1, q + 1, "P_{", "}");
		cout << "\\\\" << endl;
#if 0
		for (r1 = 0; r1 <= q; r1++) {
			for (r2 = 0; r2 <= q; r2++) {
				if (r1 == r2)
					continue;
				//p3 = p1;
				//p4 = p2;
				p3 = O.line1[r1];
				p4 = O.line1[r2];
				cout << p3 << "," << p4 << " : ";
				j = O.rank_line(p3, p4, verbose_level - 1);
				cout << " : " << j << endl;
				if (i != j) {
					cout << "error: i != j" << endl;
					exit(1);
					}
				}
			}
		cout << endl;
#endif
#endif
		}

}


void points(INT d, INT q, INT verbose_level)
{
	INT N_points, n, i, j;
	finite_field F;
	INT *v;
	
	n = d + 1;
	v = NEW_INT(n);

	F.init(q, 0);
	
	N_points = generalized_binomial(n, 1, q);
	cout << "number of points = " << N_points << endl;
	for (i = 0; i < N_points; i++) {
		PG_element_unrank_modified(F, v, 1, n, i);
#if 0
		cout << "point " << i << " : ";
		INT_vec_print(cout, v, n);
		cout << " = ";
		PG_element_normalize_from_front(F, v, 1, n);
		INT_vec_print(cout, v, n);
		cout << endl;
#endif
		PG_element_rank_modified(F, v, 1, n, j);
		if (j != i) {
			cout << "error: i != j" << endl;
			exit(1);
			}
		}

	FREE_INT(v);
}

