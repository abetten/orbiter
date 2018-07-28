// singer_cycle.C
// 
// Anton Betten
// March 19, 2013
//
//
// 
//
//

#include "orbiter.h"



singer_cycle::singer_cycle()
{
	null();
}

singer_cycle::~singer_cycle()
{
	freeself();
}

void singer_cycle::null()
{
	n = 0;
	F = NULL;
	A = NULL;
	A2 = NULL;
	poly_coeffs = NULL;
	Singer_matrix = NULL;
	Elt = NULL;
	gens = NULL;
	P = NULL;
	singer_point_list = NULL;
	singer_point_list_inv = NULL;
	Sch = NULL;
	line_orbit_reps = NULL;
	line_orbit_len = NULL;
	line_orbit_first = NULL;
	line_orbit_label = NULL;
	line_orbit_label_tex = NULL;
	line_orbit = NULL;
	line_orbit_inv = NULL;
}

void singer_cycle::freeself()
{
	if (poly_coeffs) {
		FREE_INT(poly_coeffs);
		}
	if (Singer_matrix) {
		FREE_INT(Singer_matrix);
		}
	if (Elt) {
		FREE_INT(Elt);
		}
	if (gens) {
		delete gens;
		}
	if (singer_point_list) {
		FREE_INT(singer_point_list);
		}
	if (singer_point_list_inv) {
		FREE_INT(singer_point_list_inv);
		}
	if (Sch) {
		delete Sch;
		}
	if (line_orbit_reps) {
		FREE_INT(line_orbit_reps);
		}
	if (line_orbit_len) {
		FREE_INT(line_orbit_len);
		}
	if (line_orbit_first) {
		FREE_INT(line_orbit_first);
		}
	if (line_orbit_label) {
		INT i;
		for (i = 0; i < P->N_lines; i++) {
			FREE_BYTE(line_orbit_label[i]);
			}
		FREE_PBYTE(line_orbit_label);
		}
	if (line_orbit_label_tex) {
		INT i;
		for (i = 0; i < P->N_lines; i++) {
			FREE_BYTE(line_orbit_label_tex[i]);
			}
		FREE_PBYTE(line_orbit_label_tex);
		}
	if (line_orbit) {
		FREE_INT(line_orbit);
		}
	if (line_orbit_inv) {
		FREE_INT(line_orbit_inv);
		}
			// P must be deleted last:
	if (P) {
		delete P;
		}
	null();
}

void singer_cycle::init(INT n, finite_field *F, action *A, action *A2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	const BYTE *poly;
	INT i, j, a;

	if (f_v) {
		cout << "singer_cycle::init" << endl;
		}
	singer_cycle::n = n;
	singer_cycle::q = F->q;
	singer_cycle::F = F;
	singer_cycle::A = A;
	singer_cycle::A2 = A2;
	if (F->e > 1) {
		cout << "singer_cycle::init field must be prime field" << endl;
		exit(1);
		}
	poly = get_primitive_polynomial(q, n, verbose_level);
	poly_coeffs = NEW_INT(n + 1);
	{
	//finite_field GFp;
	
	//GFp.init(p, 0);

	unipoly_domain FX(F);
	unipoly_object m;

	FX.create_object_by_rank_string(m, poly, 0);
	INT *rep = (INT *) m;
	INT *coeffs = rep + 1;
	
	for (i = 0; i <= n; i++) {
		poly_coeffs[i] = coeffs[i];
		}

	}
	
	if (f_v) {
		cout << "singer_cycle::init coefficients: ";
		INT_vec_print(cout, poly_coeffs, n + 1);
		cout << endl;
		}

	Singer_matrix = NEW_INT(n* n);
	for (i = 0; i < n - 1; i++) {
		for (j = 0; j < n; j++) {
			if (j == i + 1) {
				a = 1;
				}
			else {
				a = 0;
				}
			Singer_matrix[i * n + j] = a;
			}
		}
	for (j = 0; j < n; j++) {
		Singer_matrix[(n - 1) * n + j] = F->negate(poly_coeffs[j]);
		}
	if (f_v) {
		cout << "singer_cycle::init Singer_matrix: " << endl;
		INT_matrix_print(Singer_matrix, n, n);
		}
	Elt = NEW_INT(A->elt_size_in_INT);
	A->make_element(Elt, Singer_matrix, verbose_level);
	gens = new vector_ge;
	gens->init(A);
	gens->allocate(1);
	A->element_move(Elt, gens->ith(0), 0);
	if (f_v) {
		cout << "singer_cycle::init created Singer cycle:" << endl;
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
		cout << "singer_cycle::init Singer cycle on lines:" << endl;
		A2->element_print_as_permutation(Elt, cout);
		cout << endl;
		}
	if (f_v) {
		cout << "singer_cycle::init done" << endl;
		}
}

void singer_cycle::init_lines(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a, b, c, h;
	INT *v;
	INT *line;

	if (f_v) {
		cout << "singer_cycle::init_lines" << endl;
		}

	v = NEW_INT(n);

	P = new projective_space;

	P->init(n - 1, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level);


	singer_point_list = NEW_INT(P->N_points);
	singer_point_list_inv = NEW_INT(P->N_points);
	a = 0;
	singer_point_list[0] = 0;
	singer_point_list_inv[0] = 0;
	for (i = 0; i < P->N_points - 1; i++) {
		b = A->element_image_of(a, Elt, 0);
		singer_point_list[1 + i] = b;
		singer_point_list_inv[b] = i + 1;
		a = b;
		}

	line = NEW_INT(P->k);

	if (f_v) {
		cout << "singer_cycle::init_lines singer_point_list:" << endl;
		for (i = 0; i < P->N_points; i++) {
			cout << i << " : " << singer_point_list[i] << " : ";
			P->unrank_point(v, singer_point_list[i]);
			INT_vec_print(cout, v, n);
			cout << endl;
			}
		}

	if (f_v) {
		cout << "Lines on point P_0:" << endl;
		for (i = 0; i < P->r; i++) {
			a = P->Lines_on_point[0 * P->r + i];
			cout << "Line " <<  i << " has rank " << a << ":" << endl;
			P->Grass_lines->unrank_INT(a, 0);
			INT_matrix_print(P->Grass_lines->M, 2, n);
			h = 0;
			for (j = 0; j < P->k; j++) {
				b = P->Lines[a * P->k + j];
				c = singer_point_list_inv[b];
				if (c != 0) {
					line[h++] = c;
					}
				}
			cout << "points on this line in powers of singer cycle: ";
			INT_vec_print(cout, line, h);
			cout << endl;
			}
		}



	Sch = new schreier;
	Sch->init(A2);
	Sch->initialize_tables();
	Sch->init_single_generator(Elt);
	Sch->compute_all_point_orbits(0);
	if (f_v) {
		cout << "Found " << Sch->nb_orbits << " orbits on lines" << endl;
		for (i = 0; i < Sch->nb_orbits; i++) {
			cout << "Orbit " << i << " of length " << Sch->orbit_len[i] << endl;
			}
		}
	nb_line_orbits = Sch->nb_orbits;
	line_orbit_reps = NEW_INT(nb_line_orbits);
	line_orbit_len = NEW_INT(nb_line_orbits);
	line_orbit_first = NEW_INT(nb_line_orbits);

	line_orbit_label = NEW_PBYTE(P->N_lines);
	line_orbit_label_tex = NEW_PBYTE(P->N_lines);
	line_orbit = NEW_INT(P->N_lines);
	line_orbit_inv = NEW_INT(P->N_lines);
	for (i = 0; i < Sch->nb_orbits; i++) {
		line_orbit_reps[i] = Sch->orbit[Sch->orbit_first[i]];
		line_orbit_len[i] = Sch->orbit_len[i];
		line_orbit_first[i] = Sch->orbit_first[i];
		}
	if (f_v) {
		cout << "line_orbit_reps:";
		INT_vec_print(cout, line_orbit_reps, nb_line_orbits);
		cout << endl;
		cout << "line_orbit_len:";
		INT_vec_print(cout, line_orbit_len, nb_line_orbits);
		cout << endl;
		cout << "line_orbit_first:";
		INT_vec_print(cout, line_orbit_first, nb_line_orbits);
		cout << endl;
		}
	h = 0;
	for (i = 0; i < nb_line_orbits; i++) {
		a = line_orbit_reps[i];
		if (f_v) {
			cout << "computing orbit of line " << a << " of length " << line_orbit_len[i] << ":" << endl;
			}
		for (j = 0; j < line_orbit_len[i]; j++) {
			line_orbit[h] = a;
			line_orbit_inv[a] = h;
			BYTE str[1000];
			sprintf(str, "A%ld", j);
			str[0] += i;
			if (f_v) {
				cout << "label " << j << " is " << str << endl;
				}
			line_orbit_label[h] = NEW_BYTE(strlen(str) + 1);
			sprintf(str, "A_{%ld}", j);
			str[0] += i;
			if (f_v) {
				cout << "label " << j << " in tex is " << str << endl;
				}
			line_orbit_label_tex[h] = NEW_BYTE(strlen(str) + 1);
			strcpy(line_orbit_label_tex[h], str);
			b = A2->element_image_of(a, Elt, 0);
			a = b;
			h++;
			}
		}
	if (f_v) {
		cout << "h=" << h << endl;
		for (i = 0; i < P->N_lines; i++) {
			cout << i << " : " << line_orbit_label[i] << " : " << line_orbit[i] << endl;
			}
		}
	
	partitionstack *Stack;
	incidence_structure *Inc;
	INT f_combined_action = FALSE;
	INT f_write_tda_files = FALSE;
	INT f_include_group_order = FALSE;
	INT f_pic = FALSE;
	INT f_include_tda_scheme = FALSE;
	
	Inc = new incidence_structure;
	
	Inc->init_by_matrix_as_bitvector(P->N_points, P->N_lines, P->incidence_bitvec, 0);

	INT set_size = P->N_points;
	INT nb_blocks = P->N_lines;
		
	Stack = new partitionstack;
	Stack->allocate(set_size + nb_blocks, 0 /* verbose_level */);
	Stack->subset_continguous(set_size, nb_blocks);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();

	if (f_v) {
		cout << "before incidence_structure_compute_TDA_general" << endl;
		}
	incidence_structure_compute_TDA_general(*Stack, 
		Inc, 
		f_combined_action, 
		NULL, A, A2, 
		gens, 
		f_write_tda_files, 
		f_include_group_order, 
		f_pic, 
		f_include_tda_scheme, 
		verbose_level);
	if (f_v) {
		cout << "after incidence_structure_compute_TDA_general" << endl;
		}

#if 0
	//Stack->isolate_point(set_size + 11);
	Stack->isolate_point(0);

	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();

	INT TDO_depth = INT_MAX;
	
	cout << "before compute_TDO_safe" << endl;
	Inc->compute_TDO_safe(*Stack, TDO_depth, verbose_level - 3);
	cout << "after compute_TDO_safe" << endl;
	Inc->get_and_print_row_tactical_decomposition_scheme_tex(cout, FALSE /* f_enter_math */, *Stack);
	Inc->get_and_print_column_tactical_decomposition_scheme_tex(cout, FALSE /* f_enter_math */, *Stack);
#endif

	delete Inc;
	delete Stack;
	
	FREE_INT(line);
	FREE_INT(v);
	if (f_v) {
		cout << "singer_cycle::init_lines done" << endl;
		}
}


