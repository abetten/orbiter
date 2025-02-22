// singer_cycle.cpp
// 
// Anton Betten
// March 19, 2013
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_geometry {


singer_cycle::singer_cycle()
{
	Record_birth();
	F = NULL;
	A = NULL;
	A2 = NULL;
	n = q = 0;
	poly_coeffs = NULL;
	Singer_matrix = NULL;
	//Elt = NULL;
	//gens = NULL;
	nice_gens = NULL;
	SG = NULL;
	// target_go;
	P = NULL;
	singer_point_list = NULL;
	singer_point_list_inv = NULL;
	Sch = NULL;
	nb_line_orbits = 0;
	line_orbit_reps = NULL;
	line_orbit_len = NULL;
	line_orbit_first = NULL;
	line_orbit_label = NULL;
	line_orbit_label_tex = NULL;
	line_orbit = NULL;
	line_orbit_inv = NULL;
	Inc = NULL;
	//T = NULL;
}

singer_cycle::~singer_cycle()
{
	Record_death();
	if (poly_coeffs) {
		FREE_int(poly_coeffs);
	}
	if (Singer_matrix) {
		FREE_int(Singer_matrix);
	}
	if (nice_gens) {
		FREE_OBJECT(nice_gens);
	}
	if (SG) {
		FREE_OBJECT(SG);
	}
	if (singer_point_list) {
		FREE_int(singer_point_list);
	}
	if (singer_point_list_inv) {
		FREE_int(singer_point_list_inv);
	}
	if (Sch) {
		FREE_OBJECT(Sch);
	}
	if (line_orbit_reps) {
		FREE_int(line_orbit_reps);
	}
	if (line_orbit_len) {
		FREE_int(line_orbit_len);
	}
	if (line_orbit_first) {
		FREE_int(line_orbit_first);
	}
	if (line_orbit_label) {
		delete [] line_orbit_label;
	}
	if (line_orbit_label_tex) {
		delete [] line_orbit_label_tex;
	}
	if (line_orbit) {
		FREE_int(line_orbit);
	}
	if (line_orbit_inv) {
		FREE_int(line_orbit_inv);
	}
	if (Inc) {
		FREE_OBJECT(Inc);
	}
#if 0
	if (T) {
		FREE_OBJECT(T);
	}
#endif
			// P must be deleted last:
	if (P) {
		FREE_OBJECT(P);
	}
}

void singer_cycle::init(
		int n,
		algebra::field_theory::finite_field *F,
		actions::action *A,
		actions::action *A2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string poly;
	int i, j, a;

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
	combinatorics::knowledge_base::knowledge_base K;

	K.get_primitive_polynomial(poly, q, n, verbose_level);
	poly_coeffs = NEW_int(n + 1);
	{
		//finite_field GFp;

		//GFp.init(p, 0);
	
		algebra::ring_theory::unipoly_domain FX(F);
		algebra::ring_theory::unipoly_object m;
	
		FX.create_object_by_rank_string(m, poly, 0);
		int *rep = (int *) m;
		int *coeffs = rep + 1;

		for (i = 0; i <= n; i++) {
			poly_coeffs[i] = coeffs[i];
		}

	}
	
	if (f_v) {
		cout << "singer_cycle::init coefficients: ";
		Int_vec_print(cout, poly_coeffs, n + 1);
		cout << endl;
	}

	Singer_matrix = NEW_int(n * n);
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
		Int_matrix_print(Singer_matrix, n, n);
	}
	//Elt = NEW_int(A->elt_size_in_int);
	//A->make_element(Elt, Singer_matrix, verbose_level);

	algebra::number_theory::number_theory_domain NT;

	target_go.create((NT.i_power_j(q, n) - 1) / (q - 1));

	SG = NEW_OBJECT(groups::strong_generators);
	SG->init_from_data_with_target_go(A,
			Singer_matrix,
			n * n, 1 /* nb_gens*/,
			target_go,
			nice_gens,
			verbose_level);

	//gens = NEW_OBJECT(vector_ge);
	//gens->init(A);
	//gens->allocate(1);
	//A->element_move(Elt, gens->ith(0), 0);
	if (f_v) {
		cout << "singer_cycle::init created Singer cycle:" << endl;
		A->Group_element->element_print_as_permutation(nice_gens->ith(0), cout);
		cout << endl;
		cout << "singer_cycle::init Singer cycle on lines:" << endl;
		A2->Group_element->element_print_as_permutation(nice_gens->ith(0), cout);
		cout << endl;
	}
	if (f_v) {
		cout << "singer_cycle::init done" << endl;
	}
}

void singer_cycle::init_lines(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, h;
	int *v;
	int *line;

	if (f_v) {
		cout << "singer_cycle::init_lines" << endl;
	}

	v = NEW_int(n);

	P = NEW_OBJECT(geometry::projective_geometry::projective_space);

	P->projective_space_init(n - 1, F,
		false /* f_init_incidence_structure */, 
		verbose_level);


	singer_point_list = NEW_int(P->Subspaces->N_points);
	singer_point_list_inv = NEW_int(P->Subspaces->N_points);
	a = 0;
	singer_point_list[0] = 0;
	singer_point_list_inv[0] = 0;
	for (i = 0; i < P->Subspaces->N_points - 1; i++) {
		b = A->Group_element->element_image_of(a, nice_gens->ith(0), 0);
		singer_point_list[1 + i] = b;
		singer_point_list_inv[b] = i + 1;
		a = b;
	}

	line = NEW_int(P->Subspaces->k);

	if (f_v) {
		cout << "singer_cycle::init_lines singer_point_list:" << endl;
		for (i = 0; i < P->Subspaces->N_points; i++) {
			cout << i << " : " << singer_point_list[i] << " : ";
			P->unrank_point(v, singer_point_list[i]);
			Int_vec_print(cout, v, n);
			cout << endl;
		}
	}

	if (f_v) {
		cout << "Lines on point P_0:" << endl;
		for (i = 0; i < P->Subspaces->r; i++) {
			a = P->Subspaces->Implementation->lines_on_point(0, i);
			cout << "Line " <<  i << " has rank " << a << ":" << endl;
			P->Subspaces->Grass_lines->unrank_lint(a, 0);
			Int_matrix_print(P->Subspaces->Grass_lines->M, 2, n);
			h = 0;
			for (j = 0; j < P->Subspaces->k; j++) {
				b = P->Subspaces->Implementation->lines(a, j);
				c = singer_point_list_inv[b];
				if (c != 0) {
					line[h++] = c;
				}
			}
			cout << "points on this line in powers of singer cycle: ";
			Int_vec_print(cout, line, h);
			cout << endl;
		}
	}



	Sch = NEW_OBJECT(groups::schreier);
	Sch->init(A2, verbose_level - 2);
	Sch->initialize_tables();
	Sch->init_single_generator(nice_gens->ith(0), verbose_level - 2);
	Sch->compute_all_point_orbits(0);
	if (f_v) {
		cout << "Found " << Sch->nb_orbits << " orbits on lines" << endl;
		for (i = 0; i < Sch->nb_orbits; i++) {
			cout << "Orbit " << i << " of length " << Sch->orbit_len[i] << endl;
		}
	}
	nb_line_orbits = Sch->nb_orbits;
	line_orbit_reps = NEW_int(nb_line_orbits);
	line_orbit_len = NEW_int(nb_line_orbits);
	line_orbit_first = NEW_int(nb_line_orbits);

	line_orbit_label = new string[P->Subspaces->N_lines];
	line_orbit_label_tex = new string[P->Subspaces->N_lines];
	line_orbit = NEW_int(P->Subspaces->N_lines);
	line_orbit_inv = NEW_int(P->Subspaces->N_lines);
	for (i = 0; i < Sch->nb_orbits; i++) {
		line_orbit_reps[i] = Sch->orbit[Sch->orbit_first[i]];
		line_orbit_len[i] = Sch->orbit_len[i];
		line_orbit_first[i] = Sch->orbit_first[i];
	}
	if (f_v) {
		cout << "line_orbit_reps:";
		Int_vec_print(cout, line_orbit_reps, nb_line_orbits);
		cout << endl;
		cout << "line_orbit_len:";
		Int_vec_print(cout, line_orbit_len, nb_line_orbits);
		cout << endl;
		cout << "line_orbit_first:";
		Int_vec_print(cout, line_orbit_first, nb_line_orbits);
		cout << endl;
	}
	h = 0;
	for (i = 0; i < nb_line_orbits; i++) {
		a = line_orbit_reps[i];
		if (f_v) {
			cout << "computing orbit of line " << a << " of length "
					<< line_orbit_len[i] << ":" << endl;
		}
		for (j = 0; j < line_orbit_len[i]; j++) {
			line_orbit[h] = a;
			line_orbit_inv[a] = h;

			line_orbit_label[h] = "A" + std::to_string(j);
			if (f_v) {
				cout << "label " << j << " is " << line_orbit_label[h] << endl;
			}

			line_orbit_label_tex[h] = "A_{" + std::to_string(j) + "}";
			if (f_v) {
				cout << "label " << j << " in tex is " << line_orbit_label_tex[h] << endl;
			}

			b = A2->Group_element->element_image_of(a, nice_gens->ith(0), 0);
			a = b;
			h++;
		}
	}
	if (f_v) {
		cout << "h=" << h << endl;
		for (i = 0; i < P->Subspaces->N_lines; i++) {
			cout << i << " : " << line_orbit_label[i] << " : " << line_orbit[i] << endl;
		}
	}

	//int f_combined_action = false;

	Inc = NEW_OBJECT(geometry::other_geometry::incidence_structure);

	Inc->init_by_matrix_as_bitmatrix(
			P->Subspaces->N_points,
			P->Subspaces->N_lines,
			P->Subspaces->Implementation->get_Bitmatrix(),
			0);


#if 0
	T = NEW_OBJECT(apps_combinatorics::tactical_decomposition);
	T->init(P->Subspaces->N_points, P->Subspaces->N_lines,
			Inc,
			f_combined_action,
			NULL /* Aut */,
			A /* A_on_points */,
			A2 /*A_on_lines*/,
			SG /* Aut->strong_generators*/,
			verbose_level - 1);
#endif

	
#if 0
	partitionstack *Stack;
	incidence_structure *Inc;
	int f_combined_action = false;
	int f_write_tda_files = false;
	int f_include_group_order = false;
	int f_pic = false;
	int f_include_tda_scheme = false;
	

	int set_size = P->N_points;
	int nb_blocks = P->N_lines;
		
	Stack = NEW_OBJECT(partitionstack);
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
#endif
	if (f_v) {
		cout << "after incidence_structure_compute_TDA_general" << endl;
	}

#if 0
	//Stack->isolate_point(set_size + 11);
	Stack->isolate_point(0);

	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();

	int TDO_depth = INT_MAX;
	
	cout << "before compute_TDO_safe" << endl;
	Inc->compute_TDO_safe(*Stack, TDO_depth, verbose_level - 3);
	cout << "after compute_TDO_safe" << endl;
	Inc->get_and_print_row_tactical_decomposition_scheme_tex(cout, false /* f_enter_math */, *Stack);
	Inc->get_and_print_column_tactical_decomposition_scheme_tex(cout, false /* f_enter_math */, *Stack);
#endif

	//FREE_OBJECT(Inc);
	//FREE_OBJECT(Stack);
	//FREE_OBJECT(T);
	
	FREE_int(line);
	FREE_int(v);
	if (f_v) {
		cout << "singer_cycle::init_lines done" << endl;
	}
}

}}}


