// conic.C
// 
// Anton Betten
// March 4, 2010
//
//
// creates the Lunelli Sce Hyperoval in F_{16} 
// as the symmetric difference of two cubics.
//
// Computes the variety of a conic,
// chooses the first 5 points, 
// recomputes the equation from the 5 points
// computes external lines
// computes a graph whose vertices are the external lines.
// Two vertices are adjacent if the external lines intersect in an 
// external point
// Computes the induced action on the external lines
//

#include "orbiter.h"

// global data:

INT t0; // the system time when the program started

void draw_empty_grid(INT q, INT f_include_line_at_infinity, INT verbose_level);
void LunelliSce(INT verbose_level);
void conic(INT q, INT *six_coeffs, INT xmax, INT ymax, INT f_do_stabilizer, INT verbose_level);
void find_collinear_triple(projective_space *P, INT *pts, INT sz);
INT analyze_color_graph(INT C, INT *colors, INT n, INT *M, INT *Pijk, INT verbose_level);
void draw_beginning(char *fname, mp_graphics *&G, INT xmax, INT ymax, INT verbose_level);
void draw_end(char *fname, mp_graphics *G, INT xmax, INT ymax, INT verbose_level);
void draw_grid_(mp_graphics &G, INT q, INT f_include_line_at_infinity, INT verbose_level);
void draw_points(mp_graphics &G, projective_space *P, INT *pts, INT nb_points, INT verbose_level);
void get_ab(INT q, INT x1, INT x2, INT x3, INT &a, INT &b);
void prepare_latex(BYTE *fname_base, projective_space *P, INT *pts, INT nb_points, 
	strong_generators *Aut_gens, INT verbose_level);
void prepare_latex_simple(BYTE *fname_base, INT verbose_level);


int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	
 	t0 = os_ticks();
	
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		}

	INT q = 8;
	INT five_pts[] = {48,55,38,29,60};

	determine_conic(q, NULL /* override_poly */, five_pts, 5, verbose_level);


	//draw_empty_grid(5, TRUE /* INT f_include_line_at_infinity  */,  0); exit(1);
	//LunelliSce(verbose_level);


	//INT six_coeffs[] = { 3, 1, 2, 4, 1, 4 }; INT f_do_stabilizer = FALSE;

	//INT six_coeffs[] = { 0, 1, 0, 0, 6, 0 }; INT f_do_stabilizer = FALSE;



	//INT six_coeffs[] = { 0, 1, 0, 0, 1, 0 }; INT q = 4; INT f_do_stabilizer = FALSE; 
	// Y^2 = XZ


	//INT six_coeffs[] = { 0, 1, 0, 0, 1, 0 }; INT q = 8; INT f_do_stabilizer = TRUE;
	// Y^2 = XZ PROBLEM in set_stabilizer !!!
	
	//INT six_coeffs[] = { 0, 1, 0, 0, 4, 0 }; INT q = 5; INT f_do_stabilizer = FALSE;

	//INT six_coeffs[] = { 1, 1, 0, 1, 1, 0 }; INT q = 5; INT f_do_stabilizer = FALSE;
	
	// Osaka homework problem:
	//INT six_coeffs[] = { 1, 6, 1, 0, 0, 1 }; INT q = 7; INT f_do_stabilizer = FALSE;
	//INT xmax = 270; INT ymax = 270;


	// Osaka homework problem:
	// X^2 + Y^2 = Z^2 mod 13 (affine: x^2 + y^2 = 1, ellipse, 2 pts at infinity)
	//INT six_coeffs[] = { 1, 1, 12, 0, 0, 0  }; INT q = 13; INT f_do_stabilizer = FALSE;
	//INT xmax = 400; INT ymax = 400;


	//INT six_coeffs[] = { 0, 1, 0, 0, 12, 0  }; INT q = 13; INT f_do_stabilizer = FALSE;
	// Y^2 = XZ mod 13 

	//INT six_coeffs[] = { 1, 0, 0, 0, 0, 12  }; INT q = 13; INT f_do_stabilizer = FALSE;
	// X^2 = YZ mod 13 (affine: y = x^2, parabola, 1 pt at infinity)
	
	//INT six_coeffs[] = { 0, 0, 1, 12, 0, 0  }; INT q = 13; INT f_do_stabilizer = FALSE;
	// XY = Z^2 mod 13 (affine: y = 1/x, hyperbola, 2 pts at infinity)

	//INT six_coeffs[] = { 1, 1, 6, 0, 0, 0  }; INT q = 7; INT f_do_stabilizer = FALSE;
	// X^2 + Y^2 = Z^2 mod 7 (affine: x^2 + y^2 = 1, ellipse, 0 pts at infinity)
	
	//INT six_coeffs[] = { 1, 1, 10, 0, 0, 0  }; INT q = 11; INT f_do_stabilizer = FALSE;
	// X^2 + Y^2 = Z^2 mod 11 (affine: x^2 + y^2 = 1, ellipse, 0 pts at infinity)
	
	//INT six_coeffs[] = { 1, 1, 12, 0, 0, 0  }; INT q = 13; INT f_do_stabilizer = FALSE;
	// X^2 + Y^2 = Z^2 mod 13 (affine: x^2 + y^2 = 1, ellipse, 2 pts at infinity)

	//INT six_coeffs[] = { 1, 0, 0, 0, 0, 2  }; INT q = 27; INT f_do_stabilizer = FALSE;
	// X^2 = YZ in GF(27) (affine: y = x^2, parabola, 1 pt at infinity)
	
	//INT six_coeffs[] = { 1, 2, 3, 4, 5, 6  }; INT q = 17; INT f_do_stabilizer = FALSE; 
	// 
	
	//conic(q, six_coeffs, xmax, ymax, f_do_stabilizer, verbose_level);
	
	the_end(t0);
}

void draw_empty_grid(INT q, INT f_include_line_at_infinity, INT verbose_level)
{
	{
	BYTE fname[1000];
	{
	INT xmax = 300;
	INT ymax = 300;
	mp_graphics *G;

	sprintf(fname, "grid_%ld_%ld", q, f_include_line_at_infinity);
	draw_beginning(fname, G, xmax, ymax, verbose_level);

	draw_grid_(*G, q, f_include_line_at_infinity, verbose_level);
	draw_end(fname, G, xmax, ymax, verbose_level);
	}
	prepare_latex_simple(fname, verbose_level);
	}
		
}

void LunelliSce(INT verbose_level) 
{
	const BYTE *override_poly = "19";
	finite_field F;
	projective_space *P;
	action *A;
	//matrix_group *Mtx;
	INT n = 3;
	INT q = 16;
	//INT f_with_group = TRUE;
	//INT f_basis = TRUE;
	//INT f_semilinear = TRUE;
	INT v[3];
	//INT w[3];

	//F.init(q), verbose_level - 2);
	F.init_override_polynomial(q, override_poly, verbose_level);

	P = NEW_OBJECT(projective_space);
	cout << "before P->init" << endl;
	P->init(n - 1, &F, 
		FALSE /* f_init_incidence_structure */, 
		MINIMUM(2, verbose_level));

	cout << "after P->init" << endl;

	A = NEW_OBJECT(action);
	A->init_general_linear_group(n, &F, FALSE /* f_semilinear */, TRUE /* f_basis */, verbose_level - 2);
	
	
	//Mtx = A->G.matrix_grp;
	INT cubic1[100];
	INT cubic1_size = 0;
	INT cubic2[100];
	INT cubic2_size = 0;
	INT hoval[100];
	INT hoval_size = 0;
	INT a, b, i;

	for (i = 0; i < P->N_points; i++) {
		P->unrank_point(v, i);
		a = LunelliSce_evaluate_cubic1(&F, v);
		b = LunelliSce_evaluate_cubic2(&F, v);
		if (a == 0) {
			cubic1[cubic1_size++] = i;
			}
		if (b == 0) {
			cubic2[cubic2_size++] = i;
			}
		if ((a == 0 && b) || (b == 0 && a)) {
			hoval[hoval_size++] = i;
			}
		}
	cout << "the size of the hyperoval is " << hoval_size << endl;
	cout << "the hyperoval is:" << endl;
	INT_vec_print(cout, hoval, hoval_size);
	cout << endl;
	cout << "the size of cubic1 is " << cubic1_size << endl;
	cout << "the cubic1 is:" << endl;
	INT_vec_print(cout, cubic1, cubic1_size);
	cout << endl;
	cout << "the size of cubic2 is " << cubic2_size << endl;
	cout << "the cubic2 is:" << endl;
	INT_vec_print(cout, cubic2, cubic2_size);
	cout << endl;
	
}

void conic(INT q, INT *six_coeffs, INT xmax, INT ymax, INT f_do_stabilizer, INT verbose_level) 
{
	const BYTE *override_poly = NULL;
	finite_field F;
	projective_space *P;
	action *A;
	INT n = 3;
	//INT f_with_group = TRUE;
	
	INT v[3];
	//INT w[3];

	//F.init(q), verbose_level - 2);
	F.init_override_polynomial(q, override_poly, verbose_level);

	P = NEW_OBJECT(projective_space);
	cout << "before P->init" << endl;
	P->init(n - 1, &F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level/*MINIMUM(2, verbose_level)*/);

	cout << "after P->init" << endl;
	A = NEW_OBJECT(action);
	A->init_general_linear_group(n, &F, FALSE /* f_semilinear */, TRUE /* f_basis */, verbose_level - 2);
	
	INT variety[100];
	INT variety_size = 0;
	INT a, i, j, h;

	for (i = 0; i < P->N_points; i++) {
		P->unrank_point(v, i);
		a = F.evaluate_conic_form(six_coeffs, v);
		cout << i << " : ";
		INT_vec_print(cout, v, 3);
		cout << " : " << a << endl;
		if (a == 0) {
			variety[variety_size++] = i;
			}
		}
	cout << "the size of the variety is " << variety_size << endl;
	cout << "the variety is:" << endl;
	INT_vec_print(cout, variety, variety_size);
	cout << endl;
	for (i = 0; i < variety_size; i++) {
		P->unrank_point(v, variety[i]);
		a = F.evaluate_conic_form(six_coeffs, v);
		cout << i << " : ";
		INT_vec_print(cout, v, 3);
		cout << " : " << a << endl;
		}
	
#if 0
	INT pts[] = {2, 22, 6, 18, 10};
	find_collinear_triple(P, pts, 5);
	
	INT five_pts[] = {3, 6, 7, 28, 30};
#endif
	//INT six_coeffs[6];
	INT five_pts[5];
	five_pts[0] = variety[0];
	five_pts[1] = variety[1];
	five_pts[2] = variety[2];
	five_pts[3] = variety[3];
	five_pts[4] = variety[4];
	
	P->determine_conic_in_plane(five_pts, 5, six_coeffs, verbose_level);

	INT points[1000];
	INT tangents[1000];
	INT *exterior_points;
	INT *secants;
	INT nb_secants, nb_exterior_points;
	INT nb_points;
	//INT v[3];
	
	P->conic_points(five_pts, six_coeffs, points, nb_points, verbose_level);
	cout << "the " << nb_points << " conic points are: ";
	INT_vec_print(cout, points, nb_points);
	cout << endl;
	for (i = 0; i < nb_points; i++) {
		P->unrank_point(v, points[i]);
		cout << i << " : " << points[i] << " : ";
		INT_vec_print(cout, v, 3);
		cout << endl;
		}


	strong_generators *Aut_gens;


	if (f_do_stabilizer) {
		// computing stabilizer:

		cout << "computing stabilizer" << endl;
	
		set_stabilizer_compute STAB;
		sims *Stab;
		INT nb_backtrack_nodes;
	
		cout << "computing stabilizer of conic:" << endl;
		STAB.init(A, variety, variety_size /* points, nb_points*/ , verbose_level);
		STAB.compute_set_stabilizer(t0, nb_backtrack_nodes, Aut_gens, verbose_level + 10);
		longinteger_object go, go2;
		Stab = Aut_gens->create_sims(verbose_level - 1);
		Stab->group_order(go);
		cout << "computing stabilizer of conic done, found a group of order " << go << endl;

		FREE_OBJECT(Stab);
		}
	else {
		Aut_gens = NULL;
		}
	{
	BYTE fname[1000];
	{
	mp_graphics *G;
	INT f_include_line_at_infinity = TRUE;

	sprintf(fname, "conic_%ld", q);
	draw_beginning(fname, G, xmax, ymax, verbose_level);

	//variety_size = 0;
	draw_grid_(*G, q, f_include_line_at_infinity, verbose_level);
	draw_points(*G, P, /*variety + 3, 5*/ variety, variety_size /*points, nb_points*/, verbose_level);
	draw_end(fname, G, xmax, ymax, verbose_level);
	}
	if (f_do_stabilizer) {
		prepare_latex(fname, P, points, nb_points, Aut_gens, verbose_level);
		}
	}
		
	FREE_OBJECT(Aut_gens);
	return;
	
	P->find_tangent_lines_to_conic(six_coeffs, 
		points, nb_points, 
		tangents, verbose_level);

	cout << "the " << nb_points << " tangent lines are: ";
	INT_vec_print(cout, tangents, nb_points);
	cout << endl;

	nb_exterior_points = (nb_points * (nb_points - 1)) >> 1;
	nb_secants = nb_exterior_points;
	exterior_points = NEW_INT(nb_exterior_points);
	h = 0;
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			exterior_points[h++] = P->Line_intersection[
				tangents[i] * P->N_lines + tangents[j]];
			}
		}
	INT_vec_heapsort(exterior_points, nb_exterior_points);
	cout << "the " << nb_exterior_points << " exterior points are: ";
	INT_vec_print(cout, exterior_points, nb_exterior_points);
	cout << endl;

	secants = NEW_INT(nb_secants);
	h = 0;
	for (i = 0; i < nb_points; i++) {
		for (j = i + 1; j < nb_points; j++) {
			secants[h++] = P->Line_through_two_points[
				points[i] * P->N_points + points[j]];
			}
		}
	INT_vec_heapsort(secants, nb_secants);
	cout << "the " << nb_secants << " secants are: ";
	INT_vec_print(cout, secants, nb_secants);
	cout << endl;

	INT *external_lines;
	INT nb_external_lines;
	
	external_lines = NEW_INT(P->N_lines);
	for (i = 0; i < P->N_lines; i++) {
		external_lines[i] = i;
		}
	nb_external_lines = P->N_lines;
	set_delete_elements(external_lines, nb_external_lines, tangents, nb_points);
	set_delete_elements(external_lines, nb_external_lines, secants, nb_secants);

	cout << "the " << nb_external_lines << " external lines are: ";
	INT_vec_print(cout, external_lines, nb_external_lines);
	cout << endl;
	
	INT *adjacency;
	INT idx;

	adjacency = NEW_INT(nb_external_lines * nb_external_lines);
	for (i = 0; i < nb_external_lines; i++) {
		adjacency[i * nb_external_lines + i] = 0;
		for (j = i + 1; j < nb_external_lines; j++) {
			a = P->Line_intersection[
				external_lines[i] * P->N_lines + external_lines[j]];
			if (INT_vec_search(exterior_points, nb_exterior_points, a, idx)) {
				adjacency[i * nb_external_lines + j] = 1;
				adjacency[j * nb_external_lines + i] = 1;
				}
			else {
				adjacency[i * nb_external_lines + j] = 0;
				adjacency[j * nb_external_lines + i] = 0;
				}
			}
		}
	cout << "adjacency matrix:" << endl;
	print_integer_matrix_width(cout, adjacency, 
		nb_external_lines, nb_external_lines, nb_external_lines, 
		1);
	INT *Edges1;
	INT *Edges2;
	INT *Incidence;
	INT nb_e = 0;
	for (i = 0; i < nb_external_lines; i++) {
		for (j = i + 1; j < nb_external_lines; j++) {
			if (adjacency[i * nb_external_lines + j])
				nb_e++;
			}
		}
	Edges1 = NEW_INT(nb_e);
	Edges2 = NEW_INT(nb_e);
	Incidence = NEW_INT(nb_external_lines * nb_e);
	for (i = 0; i < nb_external_lines * nb_e; i++) {
		Incidence[i] = 0;
		}

	nb_e = 0;
	for (i = 0; i < nb_external_lines; i++) {
		for (j = i + 1; j < nb_external_lines; j++) {
			if (adjacency[i * nb_external_lines + j]) {
				Edges1[nb_e] = i;
				Edges2[nb_e] = j;
				nb_e ++;
				}
			}
		}
	for (j = 0; j < nb_e; j++) {
		Incidence[Edges1[j] * nb_e + j] = 1;
		Incidence[Edges2[j] * nb_e + j] = 1;
		}

	BYTE fname[1000];
	sprintf(fname, "ext_lines_%ld.inc", P->F->q);
	{
	ofstream f(fname);

	f << nb_external_lines << " " << nb_e << " " << nb_e * 2 << endl;
	for (i = 0; i < nb_external_lines * nb_e; i++) {
		if (Incidence[i]) {
			f << i << " ";
			}
		}
	f << endl;
	f << -1 << endl;
	
	}
	cout << "written file " << fname << " of size " << file_size(fname) << endl;

	INT colors[] = {0,1};
	INT C = 2;
	INT *Pijk;
	
	Pijk = NEW_INT(C * C * C);
	if (analyze_color_graph(C, colors, nb_external_lines, adjacency, Pijk, verbose_level)) {
		cout << "is association scheme" << endl;
		}
	else {
		cout << "not an association scheme" << endl;
		}

	//exit(1);
	

	{
	set_stabilizer_compute STAB;
	strong_generators *Aut_gens;
	sims *Stab;
	//vector_ge gens;
	//INT *tl;
	INT nb_backtrack_nodes;

	cout << "computing stabilizer of conic:" << endl;
	STAB.init(A, points, nb_points, verbose_level);
	STAB.compute_set_stabilizer(t0, nb_backtrack_nodes, Aut_gens, verbose_level + - 2);
	Stab = Aut_gens->create_sims(verbose_level - 1);
	longinteger_object go, go2;
	Stab->group_order(go);
	cout << "computing stabilizer of conic done, found a group of order " << go << endl;

	
	action *A2;
	action A2r;
	INT f_induce_action = TRUE;

	projective_space_init_line_action(P, A, A2, verbose_level);
	A2r.induced_action_by_restriction(*A2, 
		f_induce_action, Stab, 
		nb_external_lines, external_lines, verbose_level);
	A2r.group_order(go2);
	cout << "induced action on external lines has order " << go2 << endl;

	for (i = 0; i < Aut_gens->gens->len; i++) {
		A2r.element_print_quick(Aut_gens->gens->ith(i), cout);
		A2->element_print_as_permutation(Aut_gens->gens->ith(i), cout);
		A2r.element_print_as_permutation(Aut_gens->gens->ith(i), cout);
		}
	FREE_OBJECT(Aut_gens);
	FREE_OBJECT(Stab);
	FREE_OBJECT(A2);
	}
}

void find_collinear_triple(projective_space *P, INT *pts, INT sz)
{
	INT i1, i2, i3, rk;
	INT M[9];
	INT base_cols[3];
	
	for (i1 = 0; i1 < sz; i1++) {
		for (i2 = i1 + 1; i2 < sz; i2++) {
			for (i3 = i2 + 1; i3 < sz; i3++) {
				P->unrank_point(M, pts[i1]);
				P->unrank_point(M + 3, pts[i2]);
				P->unrank_point(M + 6, pts[i3]);
				rk = P->F->Gauss_simple(M, 3, 3, base_cols, 0);
				if (rk < 3) {
					cout << "collinear points " << pts[i1] << ", " << pts[i2] << ", " << pts[i3] << endl;
					}
				}
			}
		}
}

INT analyze_color_graph(INT C, INT *colors, INT n, INT *M, INT *Pijk, INT verbose_level)
{
	INT k, i, j, u, v, w, pijk, pijk1;
	
	for (k = 0; k < C; k++) {
		for (i = 0; i < C; i++) {
			for (j = 0; j < C; j++) {
				pijk = -1;
				for (u = 0; u < n; u++) {
					for (v = 0; v < n; v++) {
						if (v == u)
							continue;
						if (M[u * n + v] != colors[k])
							continue;
						// now: edge (u,v) is colored k
						pijk1 = 0;
						for (w = 0; w < n; w++) {
							if (w == u)
								continue;
							if (w == v)
								continue;
							if (M[u * n + w] != colors[i])
								continue;
							if (M[v * n + w] != colors[j])
								continue;
							//cout << "i=" << i << " j=" << j << " k=" << k << " u=" << u << " v=" << v << " w=" << w << " increasing pijk" << endl;
							pijk1++;
							} // next w
						//cout << "i=" << i << " j=" << j << " k=" << k << " u=" << u << " v=" << v << " pijk1=" << pijk1 << endl;
						if (pijk == -1) {
							pijk = pijk1;
							//cout << "u=" << u << " v=" << v << " p_{" << i << "," << j << "," << k << "}=" << pijk << endl;
							}
						else {
							if (pijk1 != pijk) {
								cout << "not an association scheme" << endl;
								cout << "k=" << k << endl;
								cout << "i=" << i << endl;
								cout << "j=" << j << endl;
								cout << "u=" << u << endl;
								cout << "v=" << v << endl;
								cout << "pijk=" << pijk << endl;
								cout << "pijk1=" << pijk1 << endl;
								return FALSE;
								}
							}
						} // next v
					} // next u
				Pijk[i * C * C + j * C + k] = pijk;
				} // next j
			} // next i
		} // next k
	for (k = 0; k < C; k++) {
		for (i = 0; i < C; i++) {
			for (j = 0; j < C; j++) {
				cout << "p_{" << i << "," << j << "," << k << "}=" << 
					Pijk[i * C * C + j * C + k] << endl;
				}
			}
		}
	return TRUE;
}


void draw_beginning(char *fname, mp_graphics *&G, INT xmax, INT ymax, INT verbose_level)
{
	INT x_min = 0, x_max = 1000;
	INT y_min = 0, y_max = 1000;
	INT factor_1000 = 1000;
	BYTE fname_full[1000];
	INT f_embedded = TRUE;
	INT f_sideways = FALSE;
	
	//cout << "draw_grid q=" << q << endl;
	sprintf(fname_full, "%s.mp", fname);
	//{
	G = NEW_OBJECT(mp_graphics);
	G->init(fname_full, x_min, y_min, x_max, y_max, f_embedded, f_sideways);
	G->out_xmin() = 0;
	G->out_ymin() = 0;
	G->out_xmax() = xmax;
	G->out_ymax() = ymax;
	cout << "xmax/ymax = " << xmax << " / " << ymax << endl;
	
	G->header();
	G->begin_figure(factor_1000);
	
	//draw_grid_(G, q, verbose_level);
}

void draw_end(char *fname, mp_graphics *G, INT xmax, INT ymax, INT verbose_level)
{
	//INT x_min = 0, x_max = 1000;
	//INT y_min = 0, y_max = 1000;
	//INT factor_1000 = 1000;
	BYTE fname_full[1000];
	
	sprintf(fname_full, "%s.mp", fname);
	G->end_figure();
	G->footer();
	FREE_OBJECT(G);
	
	cout << "written file " << fname_full << " of size " << file_size(fname_full) << endl;
	
}

#define Xcoord(x) (2 + 2 * (x))
#define Ycoord(y) (2 + 2 * (y))

void draw_grid_(mp_graphics &G, INT q, INT f_include_line_at_infinity, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT x, y, Q, a, b, c, d;
	INT *Px, *Py;
	INT u;
	//INT x1, x2, x3;
	//INT y1, y2, y3;
	//INT rad = 20;
	//INT i, j;
	
	if (f_v) {
		cout << "draw_grid_" << endl;
		}
	u = 500 / q;
	if (q == 4) {
		u = 400 / q;
		}
	
	Q = 2 * (q + 3) + 1;
	if (f_v) {
		cout << "u=" << u << endl;
		cout << "Q=" << Q << endl;
		}
	Px = NEW_INT(Q * Q);
	Py = NEW_INT(Q * Q);
	
	for (x = 0; x < Q; x++) {
		for (y = 0; y < Q; y++) {
			Px[x * Q + y] = x * u;
			Py[x * Q + y] = y * u;
			}
		}
	
		


	if (f_v) {
		cout << "drawing grid" << endl;
		}
	//G.polygon2(Px, Py, qq, n - 1);
	for (x = 0; x < q; x++) {
		a = Xcoord(x);
		b = Ycoord(0);
		c = Xcoord(x);
		d = Ycoord(q - 1);
		cout << a << "," << b << "," << c << "," << d << endl;
		G.polygon2(Px, Py, a * Q + b, c * Q + d);
		}
	for (y = 0; y < q; y++) {
		a = Xcoord(0);
		b = Ycoord(y);
		c = Xcoord(q - 1);
		d = Ycoord(y);
		cout << a << "," << b << "," << c << "," << d << endl;
		G.polygon2(Px, Py, a * Q + b, c * Q + d);
		}


	if (f_include_line_at_infinity) {
		if (f_v) {
			cout << "drawing line at infinity" << endl;
			}
		a = Xcoord(0);
		b = Ycoord(q);
		c = Xcoord(q);
		d = Ycoord(q);
		cout << a << "," << b << "," << c << "," << d << endl;
		G.polygon2(Px, Py, a * Q + b, c * Q + d);
		}


	if (f_v) {
		cout << "drawing text" << endl;
		}
	for (x = 0; x < q; x++) {
		BYTE str[1000];
		sprintf(str, "$%ld$", x);
		a = Xcoord(x);
		b = Ycoord(-1);
		G.aligned_text(Px[a * Q + b], Py[a * Q + b], "t", str);
		}
	for (y = 0; y < q; y++) {
		BYTE str[1000];
		sprintf(str, "$%ld$", y);
		a = Xcoord(-1);
		b = Ycoord(y);
		G.aligned_text(Px[a * Q + b], Py[a * Q + b], "r", str);
		}


	if (f_include_line_at_infinity) {
		BYTE str[1000];
		sprintf(str, "$\\infty$");
		a = Xcoord(-1);
		b = Ycoord(q);
		G.aligned_text(Px[a * Q + b], Py[a * Q + b], "r", str);
		}

//done:
	FREE_INT(Px);
	FREE_INT(Py);
}

void draw_points(mp_graphics &G, projective_space *P, INT *pts, INT nb_points, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT x, y, Q, a, b; //, c, d;
	INT *Px, *Py;
	INT u;
	INT q;
	INT v[3];
	INT x1, x2, x3;
	//INT y1, y2, y3;
	INT rad = 20;
	INT i; //, j;
	
	if (f_v) {
		cout << "draw_points" << endl;
		}
	q = P->q;
	u = 500 / q;
	if (q == 4) {
		u = 400 / q;
		}
	
	Q = 2 * (q + 3) + 1;
	if (f_v) {
		cout << "u=" << u << endl;
		cout << "Q=" << Q << endl;
		}
	Px = NEW_INT(Q * Q);
	Py = NEW_INT(Q * Q);
	
	for (x = 0; x < Q; x++) {
		for (y = 0; y < Q; y++) {
			Px[x * Q + y] = x * u;
			Py[x * Q + y] = y * u;
			}
		}

	if (f_v) {
		cout << "drawing points" << endl;
		}
	for (i = 0; i < nb_points; i++) {
		P->unrank_point(v, pts[i]);
		cout << "point " << i << " : ";
		INT_vec_print(cout, v, 3);
		cout << endl;
		x1 = v[0];
		x2 = v[1];
		x3 = v[2];
		get_ab(q, x1, x2, x3, a, b);
		G.nice_circle(Px[a * Q + b], Py[a * Q + b], rad);
		}
	
	FREE_INT(Px);
	FREE_INT(Py);
}
	

void get_ab(INT q, INT x1, INT x2, INT x3, INT &a, INT &b)
{
	if (x3 == 0) {
		if (x2 == 0) {
			a = Xcoord(q);
			b = Ycoord(q);
			}
		else {
			a = Xcoord(x1);
			b = Ycoord(q);
			}
		}
	else {
		a = Xcoord(x1);
		b = Ycoord(x2);
		}
}


void prepare_latex(BYTE *fname_base, projective_space *P, INT *pts, INT nb_points, 
	strong_generators *Aut_gens, INT verbose_level)
{
	char tex_file_name[1000];
	char dvi_file_name[1000];
	char ps_file_name[1000];
	int i, j, h;
	
	sprintf(tex_file_name, "%s.tex", fname_base);
	sprintf(dvi_file_name, "%s.dvi", fname_base);
	sprintf(ps_file_name, "%s.ps", fname_base);
	
	{
	ofstream f(tex_file_name);
	
	f << "\\documentclass[]{article}" << endl;
	f << "\\usepackage{amsmath}" << endl;
	f << "\\usepackage{amssymb}" << endl;
	f << "\\usepackage{latexsym}" << endl;
	f << "\\usepackage{epsfig}" << endl;
	f << "%%\\usepackage{supertabular}" << endl;
	f << "\\evensidemargin 0in" << endl;
	f << "\\oddsidemargin 0in" << endl;
	f << "\\marginparwidth 0pt" << endl;
	f << "\\marginparsep 0pt" << endl;
	f << "\\topmargin -1in" << endl;
	f << "\\headheight 0.7cm" << endl;
	f << "\\headsep 1.8cm" << endl;
	f << "%%\\footheight 0.7cm" << endl;
	f << "\\footskip 2cm" << endl;
	f << "\\textheight 22cm" << endl;
	f << "\\textwidth 6.2in" << endl;
	f << "\\marginparpush 0pt" << endl;
	f << "%%\\newcommand{\\dominowidth}{167mm}" << endl;
	f << "\\newcommand{\\dominowidth}{190mm}" << endl;
	f << "\\newcommand{\\dominowidthsmall}{90mm}" << endl;
	//f << "\\title{" << photo_label_tex << "}" << endl;
	//f << "%%\\author{{\\sc }}" << endl;
	//f << "\\date{\\today}" << endl;
	f << "\\pagestyle{empty}" << endl;
	f << "\\begin{document}" << endl;
	//f << "\\maketitle" << endl;
	f << "\\begin{center}" << endl;
	f << "\\hspace*{-15mm}\\epsfig{file=" << fname_base << ".1,width=\\dominowidth}" << endl; 
	
	if (Aut_gens) {
		longinteger_object go;

		Aut_gens->group_order(go);
		f << "Group order " << go << endl;
		f << "$$" << endl;
		for (h = 0; h < Aut_gens->gens->len; h++) {
			INT *Elt;

			Elt = Aut_gens->gens->ith(h);
			f << "\\left(" << endl;
			f << "\\begin{array}{ccc}" << endl;
			for (i = 0; i < 3; i++) {
				for (j = 0; j < 3; j++) {
					f << Elt[i * 3 + j];
					if (j < 2) {
						f << "&";
						}
					else {
						f << "\\\\" << endl;
						}
					}
				}
			f << "\\end{array}" << endl;
			f << "\\right)" << endl;
			if (Aut_gens->A->G.matrix_grp->f_semilinear) {
				f << "_{" << Elt[9] << "}";
				}
		
			}
		f << "$$" << endl;
		}
	f << "\\end{center}" << endl;
	

	f << "\\end{document}" << endl;
	}
	char cmd0[1000];
	char cmd1[1000];
	char cmd2[1000];
	char cmd3[1000];
	char cmd4[1000];
	
	sprintf(cmd0, "mpost %s.mp", fname_base);
	sprintf(cmd1, "latex %s", tex_file_name);
	sprintf(cmd2, "dvips %s -o", dvi_file_name);
	sprintf(cmd4, "convert -trim -density 300 %s.ps %s.png", fname_base, fname_base);
	sprintf(cmd3, "open %s &", ps_file_name);
	system(cmd0);
	system(cmd1);
	system(cmd2);
	system(cmd3);
	system(cmd4);
	//system("latex view.tex");
	//system("dvips view.dvi -o");
	//system("open view.ps &");


}

void prepare_latex_simple(BYTE *fname_base, INT verbose_level)
{
	char tex_file_name[1000];
	char dvi_file_name[1000];
	char ps_file_name[1000];
	
	sprintf(tex_file_name, "%s.tex", fname_base);
	sprintf(dvi_file_name, "%s.dvi", fname_base);
	sprintf(ps_file_name, "%s.ps", fname_base);
	
	{
	ofstream f(tex_file_name);
	
	f << "\\documentclass[]{article}" << endl;
	f << "\\usepackage{amsmath}" << endl;
	f << "\\usepackage{amssymb}" << endl;
	f << "\\usepackage{latexsym}" << endl;
	f << "\\usepackage{epsfig}" << endl;
	f << "%%\\usepackage{supertabular}" << endl;
	f << "\\evensidemargin 0in" << endl;
	f << "\\oddsidemargin 0in" << endl;
	f << "\\marginparwidth 0pt" << endl;
	f << "\\marginparsep 0pt" << endl;
	f << "\\topmargin -1in" << endl;
	f << "\\headheight 0.7cm" << endl;
	f << "\\headsep 1.8cm" << endl;
	f << "%%\\footheight 0.7cm" << endl;
	f << "\\footskip 2cm" << endl;
	f << "\\textheight 22cm" << endl;
	f << "\\textwidth 6.2in" << endl;
	f << "\\marginparpush 0pt" << endl;
	f << "%%\\newcommand{\\dominowidth}{167mm}" << endl;
	f << "\\newcommand{\\dominowidth}{190mm}" << endl;
	f << "\\newcommand{\\dominowidthsmall}{90mm}" << endl;
	//f << "\\title{" << photo_label_tex << "}" << endl;
	//f << "%%\\author{{\\sc }}" << endl;
	//f << "\\date{\\today}" << endl;
	f << "\\pagestyle{empty}" << endl;
	f << "\\begin{document}" << endl;
	//f << "\\maketitle" << endl;
	f << "\\begin{center}" << endl;
	f << "\\hspace*{-15mm}\\epsfig{file=" << fname_base << ".1,width=\\dominowidth}" << endl; 
	
	f << "\\end{center}" << endl;
	

	f << "\\end{document}" << endl;
	}
	char cmd0[1000];
	char cmd1[1000];
	char cmd2[1000];
	char cmd3[1000];
	char cmd4[1000];
	
	sprintf(cmd0, "mpost %s.mp", fname_base);
	sprintf(cmd1, "latex %s", tex_file_name);
	sprintf(cmd2, "dvips %s -o", dvi_file_name);
	sprintf(cmd4, "convert -trim -density 300 %s.ps %s.png", fname_base, fname_base);
	sprintf(cmd3, "open %s &", ps_file_name);
	system(cmd0);
	system(cmd1);
	system(cmd2);
	system(cmd4);
	system(cmd3);
	//system("latex view.tex");
	//system("dvips view.dvi -o");
	//system("open view.ps &");


}





