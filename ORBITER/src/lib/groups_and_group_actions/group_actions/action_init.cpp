// action_init.C
//
// Anton Betten
// 1/1/2009

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"
#include <cstring>
	// for memcpy

void action::init_direct_product_group_and_restrict(
		matrix_group *M1, matrix_group *M2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A_direct_product;
	action *Adp;
	direct_product *P;
	int *points;
	int nb_points;
	int i;

	if (f_v) {
		cout << "action::init_direct_product_group_and_restrict" << endl;
		cout << "M1=" << M1->label << endl;
		cout << "M2=" << M2->label << endl;
		}
	A_direct_product = NEW_OBJECT(action);
	A_direct_product->init_direct_product_group(M1, M2, verbose_level);
	if (f_v) {
		cout << "action::init_direct_product_group_and_restrict "
				"after A_direct_product->init_direct_product_group" << endl;
	}

	P = A_direct_product->G.direct_product_group;
	nb_points = P->degree_of_product_action;
	points = NEW_int(nb_points);
	for (i = 0; i < nb_points; i++) {
		points[i] = P->perm_offset_i[2] + i;
	}

	if (f_v) {
		cout << "action::init_direct_product_group_and_restrict "
				"before A_direct_product->restricted_action" << endl;
	}
	Adp = A_direct_product->restricted_action(points, nb_points,
			verbose_level);
	Adp->f_is_linear = FALSE;
	if (f_v) {
		cout << "action::init_direct_product_group_and_restrict "
				"after A_direct_product->restricted_action" << endl;
	}

	memcpy(this, Adp, sizeof(action));
	Adp->null();
	delete Adp;
}

void action::init_direct_product_group(
		matrix_group *M1, matrix_group *M2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	direct_product *P;

	if (f_v) {
		cout << "action::init_direct_product_group" << endl;
		cout << "M1=" << M1->label << endl;
		cout << "M2=" << M2->label << endl;
		}

	P = NEW_OBJECT(direct_product);



	type_G = direct_product_t;
	G.direct_product_group = P;
	f_allocated = TRUE;

	if (f_v) {
		cout << "action::init_direct_product_group "
				"before P->init" << endl;
		}
	P->init(M1, M2, verbose_level);
	if (f_v) {
		cout << "action::init_direct_product_group "
				"after P->init" << endl;
		}

	f_is_linear = FALSE;
	dimension = 0;


	low_level_point_size = 0;
	if (f_v) {
		cout << "action::init_direct_product_group low_level_point_size="
			<< low_level_point_size<< endl;
		}
	strcpy(label, P->label);
	strcpy(label_tex, P->label_tex);
	if (f_v) {
		cout << "action::init_direct_product_group "
				"label=" << label << endl;
		}

	degree = P->degree_overall;
	make_element_size = P->make_element_size;

	init_function_pointers_direct_product_group();

	elt_size_in_int = P->elt_size_int;
	coded_elt_size_in_char = P->char_per_elt;
	allocate_element_data();

	strcpy(group_prefix, label);



	degree = P->degree_overall;
	if (f_v) {
		cout << "action::init_direct_product_group "
				"degree=" << degree << endl;
		}
	base_len = P->base_length;
	if (f_v) {
		cout << "action::init_direct_product_group "
				"base_len=" << base_len << endl;
		}

	allocate_base_data(base_len);

	int_vec_copy(P->the_base, base, base_len);
	int_vec_copy(P->the_transversal_length,
			transversal_length, base_len);

	int *gens_data;
	int gens_size;
	int gens_nb;

	if (f_v) {
		cout << "action::init_direct_product_group "
				"before W->make_strong_generators_data" << endl;
		}
	P->make_strong_generators_data(gens_data,
			gens_size, gens_nb, verbose_level - 1);
	if (f_v) {
		cout << "action::init_direct_product_group "
				"after W->make_strong_generators_data" << endl;
		}
	Strong_gens = NEW_OBJECT(strong_generators);
	if (f_v) {
		cout << "action::init_direct_product_group "
				"before Strong_gens->init_from_data" << endl;
		}
	Strong_gens->init_from_data(this, gens_data, gens_nb, gens_size,
			transversal_length, verbose_level - 1);
	if (f_v) {
		cout << "action::init_direct_product_group "
				"after Strong_gens->init_from_data" << endl;
		}
	f_has_strong_generators = TRUE;
	FREE_int(gens_data);

	sims *S;

	S = NEW_OBJECT(sims);

	S->init(this);
	if (f_v) {
		cout << "action::init_direct_product_group "
				"before S->init_generators" << endl;
		}
	S->init_generators(*Strong_gens->gens, verbose_level);
	if (f_v) {
		cout << "action::init_direct_product_group "
				"after S->init_generators" << endl;
		}
	if (f_v) {
		cout << "action::init_direct_product_group "
				"before S->compute_base_orbits_known_length" << endl;
		}
	S->compute_base_orbits_known_length(transversal_length, verbose_level);
	if (f_v) {
		cout << "action::init_direct_product_group "
				"after S->compute_base_orbits_known_length" << endl;
		}


	if (f_v) {
		cout << "action::init_direct_product_group "
				"before init_sims" << endl;
		}

	init_sims(S, verbose_level);

	if (f_v) {
		cout << "action::init_direct_product_group "
				"after init_sims" << endl;
		}

	if (f_v) {
		cout << "action::init_direct_product_group, finished setting up "
				<< group_prefix;
		cout << ", a permutation group of degree " << degree << " ";
		cout << "and of order ";
		print_group_order(cout);
		cout << endl;
		//cout << "make_element_size=" << make_element_size << endl;
		//cout << "base_len=" << base_len << endl;
		//cout << "f_semilinear=" << f_semilinear << endl;
		}
}

void action::init_wreath_product_group_and_restrict(
		int nb_factors, int n,
		finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A_wreath;
	action *Awr;
	wreath_product *W;
	int *points;
	int nb_points;
	int i;

	if (f_v) {
		cout << "action::init_wreath_product_group_and_restrict" << endl;
		cout << "nb_factors=" << nb_factors
				<< " n=" << n << " q=" << F->q << endl;
		}
	A_wreath = NEW_OBJECT(action);
	A_wreath->init_wreath_product_group(nb_factors, n, F, verbose_level);
	if (f_v) {
		cout << "action::init_wreath_product_group_and_restrict "
				"after A_wreath->init_wreath_product_group" << endl;
	}

	W = A_wreath->G.wreath_product_group;
	nb_points = W->degree_of_tensor_action;
	points = NEW_int(nb_points);
	for (i = 0; i < nb_points; i++) {
		points[i] = W->perm_offset_i[nb_factors] + i;
	}

	if (f_v) {
		cout << "action::init_wreath_product_group_and_restrict "
				"before A_wreath->restricted_action" << endl;
	}
	Awr = A_wreath->restricted_action(points, nb_points,
			verbose_level);
	Awr->f_is_linear = TRUE;
	if (f_v) {
		cout << "action::init_wreath_product_group_and_restrict "
				"after A_wreath->restricted_action" << endl;
	}

	memcpy(this, Awr, sizeof(action));
	Awr->null();
	FREE_OBJECT(Awr);
}


void action::init_wreath_product_group(int nb_factors, int n,
		finite_field *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A_mtx;
	wreath_product *W;
	matrix_group *M;

	if (f_v) {
		cout << "action::init_wreath_product_group" << endl;
		cout << "nb_factors=" << nb_factors
				<< " n=" << n << " q=" << F->q << endl;
		}

	A_mtx = NEW_OBJECT(action);
	M = NEW_OBJECT(matrix_group);
	W = NEW_OBJECT(wreath_product);



	type_G = wreath_product_t;
	G.wreath_product_group = W;
	f_allocated = TRUE;

	if (f_v) {
		cout << "action::init_wreath_product_group "
				"before M->init_general_linear_group" << endl;
		}
	M->init_general_linear_group(n,
			F, FALSE /* f_semilinear */, A_mtx,
			verbose_level - 1);
	if (f_v) {
		cout << "action::init_wreath_product_group "
				"after M->init_general_linear_group" << endl;
		}

	if (f_v) {
		cout << "action::init_wreath_product_group "
				"before W->init_tensor_wreath_product" << endl;
		}
	W->init_tensor_wreath_product(M, A_mtx, nb_factors, verbose_level);
	if (f_v) {
		cout << "action::init_wreath_product_group "
				"after W->init_tensor_wreath_product" << endl;
		}

	f_is_linear = TRUE;
	dimension = W->dimension_of_tensor_action;


	low_level_point_size = W->low_level_point_size;
	if (f_v) {
		cout << "action::init_wreath_product_group "
			"low_level_point_size="
			<< low_level_point_size<< endl;
		}
	strcpy(label, W->label);
	strcpy(label_tex, W->label_tex);
	if (f_v) {
		cout << "action::init_wreath_product_group "
				"label=" << label << endl;
		}

	degree = W->degree_overall;
	make_element_size = W->make_element_size;

	init_function_pointers_wreath_product_group();

	elt_size_in_int = W->elt_size_int;
	coded_elt_size_in_char = W->char_per_elt;
	allocate_element_data();

	strcpy(group_prefix, label);



	degree = W->degree_overall;
	if (f_v) {
		cout << "action::init_wreath_product_group "
				"degree=" << degree << endl;
		}
	base_len = W->base_length;
	if (f_v) {
		cout << "action::init_wreath_product_group "
				"base_len=" << base_len << endl;
		}

	allocate_base_data(base_len);

	int_vec_copy(W->the_base, base, base_len);
	int_vec_copy(W->the_transversal_length,
			transversal_length, base_len);

	int *gens_data;
	int gens_size;
	int gens_nb;

	if (f_v) {
		cout << "action::init_wreath_product_group "
				"before W->make_strong_generators_data" << endl;
		}
	W->make_strong_generators_data(gens_data,
			gens_size, gens_nb, verbose_level - 1);
	if (f_v) {
		cout << "action::init_wreath_product_group "
				"after W->make_strong_generators_data" << endl;
		}
	Strong_gens = NEW_OBJECT(strong_generators);
	if (f_v) {
		cout << "action::init_wreath_product_group "
				"before Strong_gens->init_from_data" << endl;
		}
	Strong_gens->init_from_data(this, gens_data, gens_nb, gens_size,
			transversal_length, verbose_level - 1);
	if (f_v) {
		cout << "action::init_wreath_product_group "
				"after Strong_gens->init_from_data" << endl;
		}
	f_has_strong_generators = TRUE;
	FREE_int(gens_data);

	sims *S;

	S = NEW_OBJECT(sims);

	S->init(this);
	if (f_v) {
		cout << "action::init_wreath_product_group "
				"before S->init_generators" << endl;
		}
	S->init_generators(*Strong_gens->gens, verbose_level);
	if (f_v) {
		cout << "action::init_wreath_product_group "
				"after S->init_generators" << endl;
		}
	if (f_v) {
		cout << "action::init_wreath_product_group "
				"before S->compute_base_orbits_known_length" << endl;
		}
	S->compute_base_orbits_known_length(transversal_length, verbose_level);
	if (f_v) {
		cout << "action::init_wreath_product_group "
				"after S->compute_base_orbits_known_length" << endl;
		}


	if (f_v) {
		cout << "action::init_wreath_product_group "
				"before init_sims" << endl;
		}

	init_sims(S, verbose_level);

	if (f_v) {
		cout << "action::init_wreath_product_group "
				"after init_sims" << endl;
		}

	if (f_v) {
		cout << "action::init_wreath_product_group, finished setting up "
				<< group_prefix;
		cout << ", a permutation group of degree " << degree << " ";
		cout << "and of order ";
		print_group_order(cout);
		cout << endl;
		//cout << "make_element_size=" << make_element_size << endl;
		//cout << "base_len=" << base_len << endl;
		//cout << "f_semilinear=" << f_semilinear << endl;
		}
}

void action::init_orthogonal_group(int epsilon, 
	int n, finite_field *F, 
	int f_on_points, int f_on_lines, int f_on_points_and_lines, 
	int f_semilinear, 
	int f_basis, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	action *A;
	orthogonal *O;
	action_on_orthogonal *AO;
	int q = F->q;

	if (f_v) {
		cout << "action::init_orthogonal_group "
				"verbose_level=" << verbose_level << endl;
		}
	A = NEW_OBJECT(action);
	if (f_vv) {
		cout << "action::init_orthogonal_group "
				"before A->init_projective_group" << endl;
		}
	A->init_projective_group(n, F, f_semilinear,
			TRUE /* f_basis */, verbose_level - 2);

	O = NEW_OBJECT(orthogonal);
	if (f_vv) {
		cout << "action::init_orthogonal_group before O->init" << endl;
		}
	O->init(epsilon, n, F, verbose_level);
	if (f_vv) {
		cout << "action::init_orthogonal_group after O->init" << endl;
		}

	AO = NEW_OBJECT(action_on_orthogonal);
	if (f_vv) {
		cout << "action::init_orthogonal_group before AO->init" << endl;
		}
	AO->init(A, O, f_on_points, f_on_lines,
			f_on_points_and_lines, verbose_level - 2);
	
	type_G = action_on_orthogonal_t;
	G.AO = AO;

	f_has_subaction = TRUE;
	subaction = A;
	degree = AO->degree;
	low_level_point_size = A->low_level_point_size;
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	allocate_element_data();
	init_function_pointers_induced_action();
	make_element_size = A->make_element_size;

	sprintf(group_prefix, "O%d_%d_%d", epsilon, n, q);
	sprintf(label, "O^%s(%d,%d)", plus_minus_string(epsilon), n, q);
	sprintf(label_tex, "O^{%s}(%d,%d)", plus_minus_string(epsilon), n, q);



	if (f_basis) {
		longinteger_object target_go;

		if (f_vv) {
			cout << "action::init_orthogonal_group "
					"we will create the orthogonal group now" << endl;
			}
		
		if (get_orthogonal_group_type_f_reflection()) {
			if (f_vv) {
				cout << "action::init_orthogonal_group "
						"with reflections, before order_PO_epsilon" << endl;
				}
			order_PO_epsilon(f_semilinear, epsilon, n - 1, F->q,
					target_go, verbose_level);
			}
		else {
			if (f_vv) {
				cout << "action::init_orthogonal_group "
						"without reflections, before order_POmega_epsilon"
						<< endl;
				}
			order_POmega_epsilon(epsilon, n - 1,
					F->q, target_go, verbose_level);
			}

		if (f_vv) {
			cout << "action::init_orthogonal_group "
					"the target group order is " << target_go << endl;
			}

		if (f_vv) {
			cout << "action::init_orthogonal_group "
					"before create_orthogonal_group" << endl;
			}
		create_orthogonal_group(A /*subaction*/, 
			TRUE /* f_has_target_go */, target_go, 
			callback_choose_random_generator_orthogonal, 
			verbose_level - 2);
		if (f_vv) {
			cout << "action::init_orthogonal_group "
					"after create_orthogonal_group" << endl;
			}
		}

	if (f_v) {
		cout << "action::init_orthogonal_group done" << endl;
		}
}

void action::init_BLT(finite_field *F, int f_basis,
		int f_init_hash_table, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int p, hh, epsilon, n;
	int f_semilinear = FALSE;
	
	if (f_v) {
		cout << "action::init_BLT q=" << F->q
				<< " f_init_hash_table=" << f_init_hash_table << endl;
		cout << "f_basis=" << f_basis << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	is_prime_power(F->q, p, hh);
	if (hh > 1)
		f_semilinear = TRUE;
	else
		f_semilinear = FALSE;
	epsilon = 0;
	n = 5;


	if (f_v) {
		cout << "action::init_BLT before init_orthogonal_group" << endl;
		}
	init_orthogonal_group(epsilon, n, F, 
		TRUE /* f_on_points */, 
		FALSE /* f_on_lines */, 
		FALSE /* f_on_points_and_lines */, 
		f_semilinear, 
		f_basis, 
		verbose_level - 2);
	if (f_v) {
		cout << "action::init_BLT after init_orthogonal_group" << endl;
		}



	if (f_v) {
		cout << "action::init_BLT computing lex least base" << endl;
		}
	lex_least_base_in_place(verbose_level - 2);
	if (f_v) {
		cout << "action::init_BLT computing lex least base done" << endl;
		cout << "base: ";
		int_vec_print(cout, base, base_len);
		cout << endl;
		}

	if (f_v) {
		print_base();
		}


	if (f_has_strong_generators) {
		if (f_v) {
			cout << "action::init_BLT strong "
					"generators have been computed" << endl;
			}
		if (f_vv) {
			Strong_gens->print_generators();
			}
		}
	else {
		cout << "action::init_BLT we don't have strong generators" << endl;
		exit(1);
		}

#if 0
	if (f_init_hash_table) {
		matrix_group *M;
		orthogonal *O;

		M = subaction->G.matrix_grp;
		O = M->O;
	
		if (f_v) {
			cout << "calling init_hash_table_parabolic" << endl;
			}
		init_hash_table_parabolic(*O->F, 4, 0 /* verbose_level */);
		}
#endif

	if (f_v) {
		print_info();
		}
	if (f_v) {
		cout << "action::init_BLT done" << endl;
		}
}

void action::init_group_from_strong_generators(
	vector_ge *gens, sims *K,
	int given_base_length, int *given_base,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	sims *G;
	longinteger_object G_order;
	int i;
	
	
	if (f_v) {
		cout << "action::init_group_from_strong_generators" << endl;
		}
	strcpy(label, "from sgs");
	strcpy(label_tex, "from sgs");
	if (f_vv) {
		cout << "generators are" << endl;
		gens->print(cout);
		cout << endl;
		}
	

	if (f_vv) {
		cout << "calling allocate_base_data, initial base:";
		int_vec_print(cout, given_base, given_base_length);
		cout << " of length " << given_base_length << endl;
		}
	allocate_base_data(given_base_length);
	base_len = given_base_length;
	
	for (i = 0; i < base_len; i++) {
		base[i] = given_base[i];
		}



	if (f_vv) {
		cout << "action::init_group_from_strong_generators, "
				"now trying to set up the group "
				"from the given generators" << endl;
		}
	
	G = NEW_OBJECT(sims);
	
	G->init(this);
	G->init_trivial_group(verbose_level - 1);
	G->group_order(G_order);
	
	G->build_up_group_from_generators(K, gens, 
		FALSE, NULL, /* target_go */
		FALSE /* f_override_choose_next_base_point */,
		NULL, 
		verbose_level);
	
	G->group_order(G_order);


	if (f_vvv) {
		//G.print(TRUE);
		//G.print_generator_depth_and_perm();
		}

	if (f_v) {
		cout << "init_group_from_strong_generators: "
				"found a group of order " << G_order << endl;
		if (f_vv) {
			cout << "transversal lengths:" << endl;
			int_vec_print(cout, G->orbit_len, base_len);
			cout << endl;
			}
		}
	
	if (f_vv) {
		cout << "init_sims" << endl;
		}
	init_sims(G, 0/*verbose_level - 1*/);
	if (f_vv) {
		cout << "after init_sims" << endl;
		}
	compute_strong_generators_from_sims(0/*verbose_level - 2*/);

	if (f_v) {
		print_info();
		}
}

void action::init_projective_special_group(
	int n, finite_field *F,
	int f_semilinear, int f_basis, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action::init_projective_special_group" << endl;
		cout << "n=" << n << " q=" << F->q << endl;
		cout << "f_semilinear=" << f_semilinear << endl;
		cout << "f_basis=" << f_basis << endl;
		}

	if (f_v) {
		cout << "action::init_projective_special_group "
				"before init_projective_group" << endl;
	}
	init_projective_group(n, F,
			f_semilinear, f_basis, verbose_level);
	if (f_v) {
		cout << "action::init_projective_special_group "
				"after init_projective_group" << endl;
	}

	{
		action A_on_det;
		longinteger_object go;
		strong_generators *gens;
		sims *Sims2;
		
		gens = NEW_OBJECT(strong_generators);
		
		if (f_v) {
			cout << "action::init_projective_special_group "
					"computing intersection with special linear group" << endl;
			}
		A_on_det.induced_action_on_determinant(
				Sims, verbose_level);
		if (f_v) {
			cout << "action::init_projective_special_group "
					"induced_action_on_determinant finished" << endl;
			A_on_det.Kernel->group_order(go);
			cout << "action::init_projective_special_group "
					"intersection has order " << go << endl;
			}
		gens->init_from_sims(A_on_det.Kernel, verbose_level - 1);
		
		
		Sims2 = gens->create_sims(verbose_level - 1);

		FREE_OBJECT(gens);
		init_sims(Sims2, verbose_level);
	}

	if (f_v) {
		cout << "action::init_projective_special_group done" << endl;
		}
}

void action::init_projective_group(int n, finite_field *F, 
	int f_semilinear, int f_basis, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	matrix_group *M;
	
	if (f_v) {
		cout << "action::init_projective_group" << endl;
		cout << "n=" << n << " q=" << F->q << endl;
		cout << "f_semilinear=" << f_semilinear << endl;
		cout << "f_basis=" << f_basis << endl;
		}

	M = NEW_OBJECT(matrix_group);



	type_G = matrix_group_t;
	G.matrix_grp = M;
	f_allocated = TRUE;

	f_is_linear = TRUE;
	dimension = n;
		
	if (f_v) {
		cout << "action::init_projective_group "
				"before M->init_projective_group" << endl;
		}
	M->init_projective_group(n,
			F, f_semilinear, this, verbose_level - 1);
	if (f_v) {
		cout << "action::init_projective_group "
				"after M->init_projective_group" << endl;
		}


	low_level_point_size = M->low_level_point_size;
	if (f_v) {
		cout << "action::init_projective_group low_level_point_size=" 
			<< low_level_point_size<< endl;
		}
	strcpy(label, M->label);
	strcpy(label_tex, M->label_tex);
	if (f_v) {
		cout << "action::init_projective_group label=" << label << endl;
		}

	degree = M->degree;
	make_element_size = M->elt_size_int_half;

	init_function_pointers_matrix_group();

	elt_size_in_int = M->elt_size_int;
	coded_elt_size_in_char = M->char_per_elt;
	allocate_element_data();

	strcpy(group_prefix, label);

	if (f_basis) {
		if (f_v) {
			cout << "action::init_projective_group "
					"before setup_linear_group_from_strong_generators"
					<< endl;
			}
		setup_linear_group_from_strong_generators(M, verbose_level);
		if (f_v) {
			cout << "action::init_projective_group "
					"after setup_linear_group_from_strong_generators"
					<< endl;
			}
		}
	if (f_v) {
		cout << "action::init_projective_group, finished setting up "
				<< group_prefix;
		cout << ", a permutation group of degree " << degree << " ";
		cout << "and of order ";
		print_group_order(cout);
		cout << endl;
		//cout << "make_element_size=" << make_element_size << endl;
		//cout << "base_len=" << base_len << endl;
		//cout << "f_semilinear=" << f_semilinear << endl;
		}	
}

void action::init_affine_group(int n, finite_field *F, 
	int f_semilinear, 
	int f_basis, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	matrix_group *M;
	
	if (f_v) {
		cout << "action::init_affine_group" << endl;
		cout << "n=" << n << " q=" << F->q << endl;
		cout << "f_semilinear=" << f_semilinear << endl;
		cout << "f_basis=" << f_basis << endl;
		}

	M = NEW_OBJECT(matrix_group);



	type_G = matrix_group_t;
	G.matrix_grp = M;
	f_allocated = TRUE;

	f_is_linear = TRUE;
	dimension = n;
		
	M->init_affine_group(n, F, f_semilinear, this, verbose_level - 1);


	low_level_point_size = M->low_level_point_size;
	if (f_v) {
		cout << "action::init_affine_group low_level_point_size=" 
		<< low_level_point_size<< endl;
		}
	strcpy(label, M->label);
	strcpy(label_tex, M->label_tex);
	if (f_v) {
		cout << "action::init_affine_group label=" << label << endl;
		}

	degree = M->degree;
	make_element_size = M->elt_size_int_half;

	init_function_pointers_matrix_group();

	elt_size_in_int = M->elt_size_int;
	coded_elt_size_in_char = M->char_per_elt;
	allocate_element_data();

	strcpy(group_prefix, label);

	if (f_basis) {
		setup_linear_group_from_strong_generators(M, verbose_level);
		}
	if (f_v) {
		cout << "action::init_affine_group, finished setting up "
				<< group_prefix;
		cout << ", a permutation group of degree " << degree << " ";
		cout << "and of order ";
		print_group_order(cout);
		cout << endl;
		//cout << "make_element_size=" << make_element_size << endl;
		//cout << "base_len=" << base_len << endl;
		//cout << "f_semilinear=" << f_semilinear << endl;
		}	
}

void action::init_general_linear_group(int n, finite_field *F, 
	int f_semilinear, 
	int f_basis, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	matrix_group *M;
	
	if (f_v) {
		cout << "action::init_general_linear_group" << endl;
		cout << "n=" << n << " q=" << F->q << endl;
		cout << "f_semilinear=" << f_semilinear << endl;
		cout << "f_basis=" << f_basis << endl;
		}

	M = NEW_OBJECT(matrix_group);



	type_G = matrix_group_t;
	G.matrix_grp = M;
	f_allocated = TRUE;

	f_is_linear = TRUE;
	dimension = n;
		
	M->init_general_linear_group(n, F, 
		f_semilinear, this, verbose_level - 1);


	low_level_point_size = M->low_level_point_size;
	if (f_v) {
		cout << "action::init_general_linear_group low_level_point_size=" 
			<< low_level_point_size<< endl;
		}
	strcpy(label, M->label);
	strcpy(label_tex, M->label_tex);
	if (f_v) {
		cout << "action::init_general_linear_group label=" << label << endl;
		}

	degree = M->degree;
	make_element_size = M->elt_size_int_half;

	init_function_pointers_matrix_group();

	elt_size_in_int = M->elt_size_int;
	coded_elt_size_in_char = M->char_per_elt;
	allocate_element_data();

	strcpy(group_prefix, label);

	if (f_basis) {
		setup_linear_group_from_strong_generators(M, verbose_level);
		}
	if (f_v) {
		cout << "action::init_general_linear_group, "
				"finished setting up " << group_prefix;
		cout << ", a permutation group of degree " << degree << " ";
		cout << "and of order ";
		print_group_order(cout);
		cout << endl;
		//cout << "make_element_size=" << make_element_size << endl;
		//cout << "base_len=" << base_len << endl;
		//cout << "f_semilinear=" << f_semilinear << endl;
		}	
}

void action::setup_linear_group_from_strong_generators(matrix_group *M, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "action::setup_linear_group_from_strong_generators "
				"setting up a basis" << endl;
		cout << "action::setup_linear_group_from_strong_generators "
				"before init_matrix_group_strong_generators_builtin" << endl;
		}
	init_matrix_group_strong_generators_builtin(M, verbose_level - 2);
		// see below
	if (f_v) {
		cout << "action::setup_linear_group_from_strong_generators "
				"after init_matrix_group_strong_generators_builtin" << endl;
		}

	if (f_v) {
		cout << "action::setup_linear_group_from_strong_generators "
				"before S->compute_base_orbits_known_length" << endl;
		}
	sims *S;

	S = NEW_OBJECT(sims);

	S->init(this);
	S->init_generators(*Strong_gens->gens, 0/*verbose_level*/);
	if (f_v) {
		cout << "action::setup_linear_group_from_strong_generators "
				"before S->compute_base_orbits_known_length" << endl;
		}
	S->compute_base_orbits_known_length(transversal_length, verbose_level);
	if (f_v) {
		cout << "action::setup_linear_group_from_strong_generators "
				"after S->compute_base_orbits_known_length" << endl;
		}


	if (f_v) {
		cout << "action::setup_linear_group_from_strong_generators "
				"before init_sims" << endl;
		}

	init_sims(S, verbose_level);

	if (f_v) {
		cout << "action::setup_linear_group_from_strong_generators "
				"after init_sims" << endl;
		}
}

void action::init_matrix_group_strong_generators_builtin(
	matrix_group *M,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int n, q;
	finite_field *F;
	int *data;
	int size, nb_gens;
	
	if (f_v) {
		cout << "action::init_matrix_group_strong_generators_builtin" << endl;
		}
	F = M->GFq;
	n = M->n;
	q = F->q;
	if (f_v) {
		cout << "action::init_matrix_group_strong_generators_builtin "
				"computing strong generators builtin group" << endl;
		cout << "n=" << n << endl;
		cout << "q=" << q << endl;
		cout << "p=" << F->p << endl;
		cout << "e=" << F->e << endl;
		cout << "f_semilinear=" << M->f_semilinear << endl;
		}

	if (M->f_projective) {
		strong_generators_for_projective_linear_group(n, F, 
			M->f_semilinear, 
			data, size, nb_gens, 
			0 /*verbose_level - 1*/);
			// in GALOIS/projective.C
		}
	else if (M->f_affine) {
		strong_generators_for_affine_linear_group(n, F, 
			M->f_semilinear, 
			data, size, nb_gens, 
			0 /*verbose_level - 1*/);
			// in GALOIS/projective.C
		}
	else if (M->f_general_linear) {
		strong_generators_for_general_linear_group(n, F, 
			M->f_semilinear, 
			data, size, nb_gens, 
			0 /*verbose_level - 1*/);
			// in GALOIS/projective.C
		}
	else {
		cout << "action::init_matrix_group_strong_generators_builtin "
				"unknown group type" << endl;
		exit(1);
		}



	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->init_from_data(this, data, nb_gens, size,
			transversal_length, 0 /*verbose_level - 1*/);

	FREE_int(data);
	
	if (f_v) {
		cout << "action::init_matrix_group_strong_generators_builtin "
				"computing strong generators builtin group finished" << endl;
		}
}

void action::init_permutation_group(int degree, int verbose_level)
{
	int page_length_log = PAGE_LENGTH_LOG;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	perm_group *P;
	
	if (f_v) {
		cout << "action::init_permutation_group, degree=" << degree << endl;
		}
	sprintf(group_prefix, "Perm%d", degree);
	sprintf(label, "Perm%d", degree);
	sprintf(label_tex, "Perm%d", degree);
	P = NEW_OBJECT(perm_group);
	type_G = perm_group_t;
	G.perm_grp = P;
	f_allocated = TRUE;
	
	if (f_v) {
		cout << "action::init_permutation_group before P->init" << endl;
		}

	P->init(degree, page_length_log, verbose_level);

	if (f_v) {
		cout << "action::init_permutation_group after P->init" << endl;
		}
	
	init_function_pointers_permutation_group();

	
	elt_size_in_int = P->elt_size_int;
	coded_elt_size_in_char = P->char_per_elt;
	if (f_vv) {
		cout << "elt_size_in_int = " << elt_size_in_int << endl;
		cout << "coded_elt_size_in_char = " << coded_elt_size_in_char << endl;
		}
	allocate_element_data();
	action::degree = degree;
	make_element_size = degree;
	
	if (f_v) {
		cout << "action::init_permutation_group, finished" << endl;
		cout << "a permutation group of degree " << action::degree << endl;
		cout << "and of order ";
		print_group_order(cout);
		cout << endl;
		print_info();
		}
	
}

void action::init_permutation_group_from_generators(int degree, 
	int f_target_go, longinteger_object &target_go, 
	int nb_gens, int *gens, 
	int given_base_length, int *given_base,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "action::init_permutation_group_from_generators "
				"degree=" << degree << " nb_gens=" << nb_gens
				<< " given_base_length=" << given_base_length << endl;
		}
	sprintf(group_prefix, "Perm%d", degree);
	sprintf(label, "Perm%d", degree);
	sprintf(label_tex, "Perm%d", degree);
	if (f_vv) {
		cout << "action::init_permutation_group_from_generators "
				"the " << nb_gens << " generators are" << endl;
		for (i = 0; i < nb_gens; i++) {
			cout << i << " : ";
			if (degree < 20) {
				perm_print(cout, gens + i * degree, degree);
				}
			else {
				cout << "too large to print";
				}
			cout << endl;
			}
		}
	
	if (f_vv) {
		cout << "action::init_permutation_group_from_generators "
				"calling init_permutation_group" << endl;
		}
	init_permutation_group(degree, verbose_level /* - 2 */);
	if (f_vv) {
		cout << "action::init_permutation_group_from_generators "
				"after init_permutation_group" << endl;
		}

	if (f_vv) {
		cout << "action::init_permutation_group_from_generators "
				"calling allocate_base_data" << endl;
		cout << "initial base:";
		int_vec_print(cout, given_base, given_base_length);
		cout << " of length " << given_base_length << endl;
		}
	allocate_base_data(given_base_length);
	base_len = given_base_length;
	
	// init base:
	for (i = 0; i < base_len; i++) {
		base[i] = given_base[i];
		}



	if (f_vv) {
		cout << "action::init_permutation_group_from_generators, "
				"now trying to set up the group from the given generators"
				<< endl;
		}
	
	vector_ge *generators;
	strong_generators *Strong_gens;

	generators = NEW_OBJECT(vector_ge);
	generators->init(this);
	generators->allocate(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		make_element(generators->ith(i), 
			gens + i * degree, 
			0 /*verbose_level*/);
		}
	

	generators_to_strong_generators(this, 
		f_target_go, target_go, 
		generators, Strong_gens, 
		verbose_level);

	sims *G;

	G = Strong_gens->create_sims(verbose_level);

	
	if (f_vv) {
		cout << "action::init_permutation_group_from_generators "
				"init_sims" << endl;
		}
	init_sims(G, 0/*verbose_level - 1*/);

	if (f_vv) {
		cout << "action::init_permutation_group_from_generators "
				"after init_sims" << endl;
		}



	if (f_vv) {
		cout << "action::init_permutation_group_from_generators "
				"compute_strong_generators_from_sims" << endl;
		}
	compute_strong_generators_from_sims(0 /*verbose_level - 2*/);
	if (f_vv) {
		cout << "action::init_permutation_group_from_generators "
				"after_strong_generators_from_sims" << endl;
		}


	FREE_OBJECT(generators);
	FREE_OBJECT(Strong_gens);

	if (f_v) {
		print_info();
		}
	if (f_v) {
		cout << "action::init_permutation_group_from_generators done" << endl;
		}
}

void action::init_affine_group(int n, int q, 
	int f_translations, 
	int f_semilinear, int frobenius_power, 
	int f_multiplication, 
	int multiplication_order, 
	int verbose_level)
{
	int nb_gens, degree;
	int *gens;
	int given_base_length;
	int *given_base;
	finite_field F;
	longinteger_object go;
	
	sprintf(label, "AGL_%d_%d", n, q);
	sprintf(label_tex, "AGL(%d,%d)", n, q);

	F.init(q, verbose_level - 1);
	
	
	F.affine_generators(n, f_translations, 
		f_semilinear, frobenius_power, 
		f_multiplication, multiplication_order, 
		nb_gens, degree, gens, 
		given_base_length, given_base);

	init_permutation_group_from_generators(degree, 
		FALSE, go, 
		nb_gens, gens, 
		given_base_length, given_base,
		verbose_level);

	FREE_int(gens);
	FREE_int(given_base);
}

#if 0
void action::init_affine_grid_group(int q1, int q2, 
	int f_translations1, int f_translations2, 
	int f_semilinear1, int frobenius_power1, 
	int f_semilinear2, int frobenius_power2, 
	int f_multiplication1, int multiplication_order1, 
	int f_multiplication2, int multiplication_order2, 
	int f_diagonal, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int n = 1;
	int nb_gens1, degree1;
	int nb_gens2, degree2;
	int nb_gens3, degree3;
	int *gens1;
	int *gens2;
	int *gens3;
	int given_base_length1;
	int given_base_length2;
	int given_base_length3;
	int *given_base1;
	int *given_base2;
	int *given_base3;
	finite_field F1, F2;
	longinteger_object go;
	
	sprintf(label, "AGL_1_%d_x_AGL_1_%d", q1, q2);
	sprintf(label_tex, "AGL(1,%d) x AGL(1,%d)", q1, q2);

	F1.init(q1, verbose_level - 1);
	F2.init(q2, verbose_level - 1);
	
	if (f_v) {
		cout << "computing generators first group" << endl;
		}
	F1.affine_generators(n, f_translations1, 
		f_semilinear1, frobenius_power1, 
		f_multiplication1, multiplication_order1, 
		nb_gens1, degree1, gens1, 
		given_base_length1, given_base1);
	if (f_v) {
		cout << "computing generators second group" << endl;
		}
	F2.affine_generators(n, f_translations2, 
		f_semilinear2, frobenius_power2, 
		f_multiplication2, multiplication_order2, 
		nb_gens2, degree2, gens2, 
		given_base_length2, given_base2);

	if (f_v) {
		cout << "computing generators for the direct product" << endl;
		}
	if (f_diagonal) {
		perm_group_generators_direct_product(1, 
			degree1, degree2, degree3, 
			nb_gens1, nb_gens2, nb_gens3, 
			gens1, gens2, gens3, 
			given_base_length1, given_base_length2, given_base_length3, 
			given_base1, given_base2, given_base3);
		}
	else {
		perm_group_generators_direct_product(0, 
			degree1, degree2, degree3, 
			nb_gens1, nb_gens2, nb_gens3, 
			gens1, gens2, gens3, 
			given_base_length1, given_base_length2, given_base_length3, 
			given_base1, given_base2, given_base3);
		}
	
	init_permutation_group_from_generators(degree3, 
		FALSE, go, 
		nb_gens3, gens3, 
		given_base_length3, given_base3,
		verbose_level);
	if (f_v) {
		cout << "elt_size_in_int=" << elt_size_in_int << endl;
		}

	FREE_int(gens1);
	FREE_int(gens2);
	FREE_int(gens3);
	FREE_int(given_base1);
	FREE_int(given_base2);
	FREE_int(given_base3);
}
#endif

void action::init_symmetric_group(int degree, int verbose_level)
{
	int nb_gens, *gens;
	int given_base_length, *given_base;
	int i, j;
	longinteger_object go;
	longinteger_domain D;
	
	sprintf(label, "Sym_%d", degree);
	sprintf(label_tex, "Sym(%d)", degree);

	D.factorial(go, degree);

	make_element_size = degree;
	nb_gens = degree - 1;
	given_base_length = degree - 1;
	gens = NEW_int(nb_gens * degree);
	given_base = NEW_int(given_base_length);
	for (i = 0; i < nb_gens; i++) {
		for (j = 0; j < degree; j++) {
			gens[i * degree + j] = j;
			}
		gens[i * degree + i] = i + 1;
		gens[i * degree + i + 1] = i;
		}
	for (i = 0; i < given_base_length; i++) {
		given_base[i] = i;
		}
	init_permutation_group_from_generators(degree, 
		TRUE, go,
		nb_gens, gens, 
		given_base_length, given_base,
		verbose_level);
	FREE_int(gens);
	FREE_int(given_base);
}

void action::null_function_pointers()
{
	ptr_element_image_of = NULL;
	ptr_element_image_of_low_level = NULL;
	ptr_element_linear_entry_ij = NULL;
	ptr_element_linear_entry_frobenius = NULL;
	ptr_element_one = NULL;
	ptr_element_is_one = NULL;
	ptr_element_unpack = NULL;
	ptr_element_pack = NULL;
	ptr_element_retrieve = NULL;
	ptr_element_store = NULL;
	ptr_element_mult = NULL;
	ptr_element_invert = NULL;
	ptr_element_transpose = NULL;
	ptr_element_move = NULL;
	ptr_element_dispose = NULL;
	ptr_element_print = NULL;
	ptr_element_print_quick = NULL;
	ptr_element_print_latex = NULL;
	ptr_element_print_verbose = NULL;
	ptr_element_code_for_make_element = NULL;
	ptr_element_print_for_make_element = NULL;
	ptr_element_print_for_make_element_no_commas = NULL;
	ptr_print_point = NULL;
}

void action::init_function_pointers_matrix_group()
{
	ptr_element_image_of = matrix_group_element_image_of;
	ptr_element_image_of_low_level = matrix_group_element_image_of_low_level;
	ptr_element_linear_entry_ij = matrix_group_element_linear_entry_ij;
	ptr_element_linear_entry_frobenius = matrix_group_element_linear_entry_frobenius;
	ptr_element_one = matrix_group_element_one;
	ptr_element_is_one = matrix_group_element_is_one;
	ptr_element_unpack = matrix_group_element_unpack;
	ptr_element_pack = matrix_group_element_pack;
	ptr_element_retrieve = matrix_group_element_retrieve;
	ptr_element_store = matrix_group_element_store;
	ptr_element_mult = matrix_group_element_mult;
	ptr_element_invert = matrix_group_element_invert;
	ptr_element_transpose = matrix_group_element_transpose;
	ptr_element_move = matrix_group_element_move;
	ptr_element_dispose = matrix_group_element_dispose;
	ptr_element_print = matrix_group_element_print;
	ptr_element_print_quick = matrix_group_element_print_quick;
	ptr_element_print_latex = matrix_group_element_print_latex;
	ptr_element_print_verbose = matrix_group_element_print_verbose;
	ptr_element_code_for_make_element = matrix_group_element_code_for_make_element;
	ptr_element_print_for_make_element = matrix_group_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas = matrix_group_element_print_for_make_element_no_commas;
	ptr_print_point = matrix_group_print_point;
}

void action::init_function_pointers_wreath_product_group()
{
	ptr_element_image_of = wreath_product_group_element_image_of;
	ptr_element_image_of_low_level = wreath_product_group_element_image_of_low_level;
	ptr_element_linear_entry_ij = wreath_product_group_element_linear_entry_ij;
	ptr_element_linear_entry_frobenius = wreath_product_group_element_linear_entry_frobenius;
	ptr_element_one = wreath_product_group_element_one;
	ptr_element_is_one = wreath_product_group_element_is_one;
	ptr_element_unpack = wreath_product_group_element_unpack;
	ptr_element_pack = wreath_product_group_element_pack;
	ptr_element_retrieve = wreath_product_group_element_retrieve;
	ptr_element_store = wreath_product_group_element_store;
	ptr_element_mult = wreath_product_group_element_mult;
	ptr_element_invert = wreath_product_group_element_invert;
	ptr_element_transpose = wreath_product_group_element_transpose;
	ptr_element_move = wreath_product_group_element_move;
	ptr_element_dispose = wreath_product_group_element_dispose;
	ptr_element_print = wreath_product_group_element_print;
	ptr_element_print_quick = wreath_product_group_element_print_quick;
	ptr_element_print_latex = wreath_product_group_element_print_latex;
	ptr_element_print_verbose = wreath_product_group_element_print_verbose;
	ptr_element_code_for_make_element = wreath_product_group_element_code_for_make_element;
	ptr_element_print_for_make_element = wreath_product_group_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas = wreath_product_group_element_print_for_make_element_no_commas;
	ptr_print_point = wreath_product_group_print_point;
}

void action::init_function_pointers_direct_product_group()
{
	ptr_element_image_of = direct_product_group_element_image_of;
	ptr_element_image_of_low_level = direct_product_group_element_image_of_low_level;
	ptr_element_linear_entry_ij = direct_product_group_element_linear_entry_ij;
	ptr_element_linear_entry_frobenius = direct_product_group_element_linear_entry_frobenius;
	ptr_element_one = direct_product_group_element_one;
	ptr_element_is_one = direct_product_group_element_is_one;
	ptr_element_unpack = direct_product_group_element_unpack;
	ptr_element_pack = direct_product_group_element_pack;
	ptr_element_retrieve = direct_product_group_element_retrieve;
	ptr_element_store = direct_product_group_element_store;
	ptr_element_mult = direct_product_group_element_mult;
	ptr_element_invert = direct_product_group_element_invert;
	ptr_element_transpose = direct_product_group_element_transpose;
	ptr_element_move = direct_product_group_element_move;
	ptr_element_dispose = direct_product_group_element_dispose;
	ptr_element_print = direct_product_group_element_print;
	ptr_element_print_quick = direct_product_group_element_print_quick;
	ptr_element_print_latex = direct_product_group_element_print_latex;
	ptr_element_print_verbose = direct_product_group_element_print_verbose;
	ptr_element_code_for_make_element = direct_product_group_element_code_for_make_element;
	ptr_element_print_for_make_element = direct_product_group_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas = direct_product_group_element_print_for_make_element_no_commas;
	ptr_print_point = direct_product_group_print_point;
}

void action::init_function_pointers_permutation_group()
{
	ptr_element_image_of = perm_group_element_image_of;
	ptr_element_image_of_low_level = NULL;
	ptr_element_linear_entry_ij = NULL;
	ptr_element_linear_entry_frobenius = NULL;
	ptr_element_one = perm_group_element_one;
	ptr_element_is_one = perm_group_element_is_one;
	ptr_element_unpack = perm_group_element_unpack;
	ptr_element_pack = perm_group_element_pack;
	ptr_element_retrieve = perm_group_element_retrieve;
	ptr_element_store = perm_group_element_store;
	ptr_element_mult = perm_group_element_mult;
	ptr_element_invert = perm_group_element_invert;
	ptr_element_transpose = NULL;
	ptr_element_move = perm_group_element_move;
	ptr_element_dispose = perm_group_element_dispose;
	ptr_element_print = perm_group_element_print;
	ptr_element_print_quick = perm_group_element_print; // no quick version here!
	ptr_element_print_latex = perm_group_element_print_latex;
	ptr_element_print_verbose = perm_group_element_print_verbose;
	ptr_element_code_for_make_element = perm_group_element_code_for_make_element;
	ptr_element_print_for_make_element = perm_group_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas = perm_group_element_print_for_make_element_no_commas;
	ptr_print_point = perm_group_print_point;
}

void action::init_function_pointers_induced_action()
{
	//ptr_get_transversal_rep = induced_action_get_transversal_rep;
	ptr_element_image_of = induced_action_element_image_of;
	ptr_element_image_of_low_level = induced_action_element_image_of_low_level;
	ptr_element_linear_entry_ij = NULL;
	ptr_element_linear_entry_frobenius = NULL;
	ptr_element_one = induced_action_element_one;
	ptr_element_is_one = induced_action_element_is_one;
	ptr_element_unpack = induced_action_element_unpack;
	ptr_element_pack = induced_action_element_pack;
	ptr_element_retrieve = induced_action_element_retrieve;
	ptr_element_store = induced_action_element_store;
	ptr_element_mult = induced_action_element_mult;
	ptr_element_invert = induced_action_element_invert;
	ptr_element_transpose = induced_action_element_transpose;
	ptr_element_move = induced_action_element_move;
	ptr_element_dispose = induced_action_element_dispose;
	ptr_element_print = induced_action_element_print;
	ptr_element_print_quick = induced_action_element_print_quick;
	ptr_element_print_latex = induced_action_element_print_latex;
	ptr_element_print_verbose = induced_action_element_print_verbose;
	ptr_element_code_for_make_element = induced_action_element_code_for_make_element;
	ptr_element_print_for_make_element = induced_action_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas = induced_action_element_print_for_make_element_no_commas;
	ptr_print_point = induced_action_print_point;
}

void action::create_sims(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sims *S;
	
	if (f_v) {
		cout << "action::create_sims" << endl;
		}
	if (!f_has_strong_generators) {
		cout << "action::create_sims we need strong generators" << endl;
		exit(1);
		}

	S = Strong_gens->create_sims(verbose_level - 1);
	
	init_sims(S, verbose_level);
	if (f_v) {
		cout << "action::create_sims done" << endl;
		}
}



void action::create_orthogonal_group(action *subaction, 
	int f_has_target_group_order, longinteger_object &target_go, 
	void (* callback_choose_random_generator)(int iteration, 
		int *Elt, void *data, int verbose_level), 
	int verbose_level)
{
	//verbose_level = 10;
	
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "action::create_orthogonal_group" << endl;
		}
	if (f_v) {
		cout << "action::create_orthogonal_group "
				"before allocating a schreier_sims object" << endl;
		}
	
	{
	schreier_sims ss;

	if (f_v) {
		cout << "action::create_orthogonal_group "
				"before ss.init" << endl;
		}
	ss.init(this, verbose_level - 1);
	if (f_v) {
		cout << "action::create_orthogonal_group "
				"after ss.init" << endl;
		}

	ss.interested_in_kernel(subaction, verbose_level - 1);
	
	if (f_has_target_group_order) {
		ss.init_target_group_order(target_go, verbose_level - 1);
		}
	
	if (f_v) {
		cout << "action::create_orthogonal_group "
				"before ss.init_random_process" << endl;
		}
	ss.init_random_process(
		callback_choose_random_generator,   // see action_global.C
		&ss, 
		verbose_level - 1);
	if (f_v) {
		cout << "action::create_orthogonal_group "
				"after ss.init_random_process" << endl;
		}
	
	if (f_v) {
		cout << "action::create_orthogonal_group "
				"before ss.create_group" << endl;
		}
	ss.create_group(verbose_level - 1);
	if (f_v) {
		cout << "action::create_orthogonal_group "
				"after ss.create_group" << endl;
		}
	
	init_sims(ss.G, verbose_level);
	f_has_kernel = TRUE;
	Kernel = ss.K;

	ss.K = NULL;
	ss.G = NULL;
	
	//init_transversal_reps_from_stabilizer_chain(G, verbose_level - 2);
	if (f_v) {
		cout << "action::create_orthogonal_group after init_sims, "
				"calling compute_strong_generators_from_sims" << endl;
		}
	compute_strong_generators_from_sims(verbose_level - 2);
	if (f_v) {
		cout << "action::create_orthogonal_group done, "
				"freeing schreier_sims object" << endl;
		}
	}
	if (f_v) {
		cout << "action::create_orthogonal_group "
				"done" << endl;
		}
}


