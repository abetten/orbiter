// strong_generators_groups.cpp
//
// Anton Betten

// started: December 4, 2013
// moved here: Dec 21, 2015


#include "layer1_foundations/foundations.h"
#include "group_actions.h"


using namespace std;



namespace orbiter {
namespace layer3_group_actions {
namespace groups {


void strong_generators::prepare_from_generator_data(
		actions::action *A,
		int *data,
		int nb_gens,
		int data_size,
		std::string &ascii_target_go,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::prepare_from_generator_data" << endl;
	}

	if (f_v) {
		cout << "strong_generators::prepare_from_generator_data "
				"data_size=" << data_size << endl;
		cout << "strong_generators::prepare_from_generator_data "
				"nb_gens=" << nb_gens << endl;
	}

	data_structures_groups::vector_ge *gens;

	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init_from_data(A, data,
			nb_gens, data_size, 0 /*verbose_level*/);

	ring_theory::longinteger_object target_go;

	target_go.create_from_base_10_string(ascii_target_go);


	if (f_v) {
		cout << "strong_generators::prepare_from_generator_data "
				"generators are:" << endl;
		gens->print_quick(cout);
	}



	strong_generators *Strong_gens2;

	if (f_v) {
		cout << "strong_generators::prepare_from_generator_data "
				"before generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		gens, Strong_gens2,
		0 /* verbose_level */);

	if (f_v) {
		cout << "strong_generators::prepare_from_generator_data "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(Strong_gens2, 0 /* verbose_level */);

	FREE_OBJECT(Strong_gens2);
	FREE_OBJECT(gens);


	if (f_v) {
		cout << "strong_generators::prepare_from_generator_data done" << endl;
	}
}



void strong_generators::init_linear_group_from_scratch(
		actions::action *&A,
	field_theory::finite_field *F, int n,
	group_constructions::linear_group_description *Descr,
	data_structures_groups::vector_ge *&nice_gens,
	std::string &label,
	std::string &label_tex,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "strong_generators::init_linear_group_from_scratch" << endl;
	}


	A = NEW_OBJECT(actions::action);
	strong_generators::A = A;

	int f_basis = true;
	int f_init_sims = false;
	
	if (Descr->f_projective) {
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch "
					"before A->Known_groups->init_projective_group" << endl;
		}
		A->Known_groups->init_projective_group(
				n, F, Descr->f_semilinear,
			f_basis, f_init_sims,
			nice_gens,
			verbose_level - 1);
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch "
					"after A->Known_groups->init_projective_group" << endl;
		}
	}
	else if (Descr->f_general) {
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch "
					"before A->Known_groups->init_general_linear_group" << endl;
		}
		A->Known_groups->init_general_linear_group(
				n, F, Descr->f_semilinear,
			f_basis, f_init_sims,
			nice_gens,
			verbose_level - 1);
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch "
					"after A->Known_groups->init_general_linear_group" << endl;
		}
	}
	else if (Descr->f_affine) {
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch "
					"before A->Known_groups->init_affine_group" << endl;
		}
		A->Known_groups->init_affine_group(
				n, F, Descr->f_semilinear,
			f_basis, f_init_sims,
			nice_gens,
			verbose_level - 1);
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch "
					"after A->Known_groups->init_affine_group" << endl;
		}
	}
	else if (Descr->f_GL_d_q_wr_Sym_n) {
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch "
					"before A->Known_groups->init_wreath_product_group" << endl;
		}
		A->Known_groups->init_wreath_product_group(
				Descr->GL_wreath_Sym_n /* nb_factors */,
				Descr->GL_wreath_Sym_d /* n */, F, nice_gens,
				verbose_level);
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch "
					"after A->Known_groups->init_wreath_product_group" << endl;
		}
	}
	else if (Descr->f_orthogonal || Descr->f_orthogonal_p || Descr->f_orthogonal_m) {
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch "
					"detected orthogonal group" << endl;
		}
		int epsilon;
		if (Descr->f_orthogonal) {
			epsilon = 0;
		}
		else if (Descr->f_orthogonal_p) {
			epsilon = 1;
		}
		else if (Descr->f_orthogonal_m) {
			epsilon = -1;
		}
		else {
			cout << "cannot reach this" << endl;
			exit(1);
		}
		orthogonal_geometry::orthogonal *O;

		if (f_v) {
			cout << "known_groups::init_orthogonal_group "
					"verbose_level=" << verbose_level << endl;
		}
		O = NEW_OBJECT(orthogonal_geometry::orthogonal);
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch "
					"before O->init" << endl;
		}
		O->init(epsilon, n, F, verbose_level);
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch "
					"after O->init" << endl;
		}
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch "
					"before A->Known_groups->init_orthogonal_group_with_O" << endl;
		}
		A->Known_groups->init_orthogonal_group_with_O(
				O,
			true /* f_on_points */, false /* f_on_lines */,
			false /* f_on_points_and_lines */,
			Descr->f_semilinear,
			true /* f_basis */, verbose_level);
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch "
					"after A->Known_groups->init_orthogonal_group_with_O" << endl;
		}


	}
	else {
		cout << "strong_generators::init_linear_group_from_scratch "
				"the type of group is not specified" << endl;
		exit(1);
	}


	if (!A->f_has_strong_generators) {
		cout << "strong_generators::init_linear_group_from_scratch "
				"fatal: !A->f_has_strong_generators" << endl;
	}

	label.assign(A->label);
	label_tex.assign(A->label_tex);

	if (Descr->f_special) {


		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch "
					"special linear group" << endl;
		}

		special_subgroup(verbose_level);

		if (Descr->f_projective) {

			data_structures::string_tools String;

			String.name_of_group_projective(
					label,
					label_tex,
					n, F->q, Descr->f_semilinear, true /* f_special */,
					verbose_level - 1);

		}
		else if (Descr->f_affine) {

			data_structures::string_tools String;

			String.name_of_group_affine(
					label,
					label_tex,
					n, F->q, Descr->f_semilinear, true /* f_special */,
					verbose_level - 1);

		}
		else if (Descr->f_general) {

			data_structures::string_tools String;

			String.name_of_group_general_linear(
					label,
					label_tex,
					n, F->q, Descr->f_semilinear, true /* f_special */,
					verbose_level - 1);


		}
		else {
			cout << "strong_generators::init_linear_group_from_scratch "
					"name of group in case f_special for this "
					"type of group not implemented." << endl;
			exit(1);
		}

		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch "
					"special linear group done" << endl;
			cout << "label=" << label << endl;
		}
	}
	else {

		if (f_init_sims) {
			if (f_v) {
				cout << "strong_generators::init_linear_group_from_scratch "
						"creating sims and collecting generators" << endl;
			}
			sims *S;
			S = A->Strong_gens->create_sims(0 /* verbose_level */);
			init_from_sims(S, verbose_level);
			FREE_OBJECT(S);
		}
		else {
			if (f_v) {
				cout << "strong_generators::init_linear_group_from_scratch "
						"before init_copy" << endl;
			}
			init_copy(A->Strong_gens, verbose_level);
			if (f_v) {
				cout << "strong_generators::init_linear_group_from_scratch "
						"before init_copy" << endl;
			}
		}
	}
	if (f_v) {
		cout << "strong_generators::init_linear_group_from_scratch "
				"strong generators have been created" << endl;
	}

	if (false) {
		cout << "strong_generators::init_linear_group_from_scratch "
				"we found the following generators:" << endl;
		print_generators(cout, verbose_level - 1);
		print_generators_tex();
	}
	if (f_v) {
		cout << "strong_generators::init_linear_group_from_scratch "
				"label=" << label << endl;
		cout << "strong_generators::init_linear_group_from_scratch go=";
		ring_theory::longinteger_object go;
		group_order(go);
		cout << go << endl;
	}


	if (f_v) {
		cout << "strong_generators::init_linear_group_from_scratch "
				"done" << endl;
	}
}

void strong_generators::special_subgroup(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	actions::action *A_on_det;
	ring_theory::longinteger_object go;
		
	if (f_v) {
		cout << "strong_generators::special_subgroup "
				"setting up action on determinant" << endl;
	}
	if (A->Sims == NULL) {
		if (f_v) {
			cout << "strong_generators::special_subgroup "
					"before A->Known_groups->init_sims_from_generators" << endl;
		}
		A->Known_groups->init_sims_from_generators(verbose_level);
		if (f_v) {
			cout << "strong_generators::special_subgroup "
					"after A->Known_groups->init_sims_from_generators" << endl;
		}
	}
	if (f_v) {
		cout << "strong_generators::special_subgroup "
				"before A->Induced_action->induced_action_on_determinant" << endl;
	}
	A_on_det = A->Induced_action->induced_action_on_determinant(
			A->Sims, verbose_level);
	if (f_v) {
		cout << "strong_generators::special_subgroup "
				"after A->Induced_action->induced_action_on_determinant" << endl;
	}
	A_on_det->Kernel->group_order(go);
	if (f_v) {
		cout << "strong_generators::special_subgroup "
				"kernel has order " << go << endl;
	}


	if (f_v) {
		cout << "strong_generators::special_subgroup "
				"before init_from_sims" << endl;
	}
	init_from_sims(A_on_det->Kernel, verbose_level);
	if (f_v) {
		cout << "strong_generators::special_subgroup "
				"after init_from_sims" << endl;
	}
	
	FREE_OBJECT(A_on_det);

	if (f_v) {
		cout << "strong_generators::special_subgroup done" << endl;
	}
}

void strong_generators::projectivity_subgroup(
		sims *S, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::projectivity_subgroup" << endl;
	}
	actions::action *A_on_Galois;
	if (f_v) {
		cout << "strong_generators::projectivity_subgroup A->A=" << endl;
		S->A->print_info();
	}
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "strong_generators::projectivity_subgroup "
				"setting up action on Galois group" << endl;
	}
	A_on_Galois = S->A->Induced_action->induced_action_on_Galois_group(
			S, verbose_level - 2);
	if (f_v) {
		cout << "strong_generators::projectivity_subgroup "
				"induced_action_on_Galois_group finished" << endl;
	}
	A_on_Galois->Kernel->group_order(go);
	if (f_v) {
		cout << "strong_generators::projectivity_subgroup "
				"kernel has order " << go << endl;
	}


	if (f_v) {
		cout << "strong_generators::projectivity_subgroup "
				"before init_from_sims" << endl;
	}
	init_from_sims(A_on_Galois->Kernel, verbose_level - 2);
	if (f_v) {
		cout << "strong_generators::projectivity_subgroup "
				"after init_from_sims" << endl;
	}
	FREE_OBJECT(A_on_Galois);

	if (f_v) {
		cout << "strong_generators::projectivity_subgroup done" << endl;
	}
}

void strong_generators::even_subgroup(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	actions::action *A_on_sign;
	ring_theory::longinteger_object go;
		
	if (f_v) {
		cout << "strong_generators::even_subgroup "
				"setting up action on sign" << endl;
	}
	A_on_sign = A->Induced_action->induced_action_on_sign(
			A->Sims, verbose_level);
	if (f_v) {
		cout << "strong_generators::even_subgroup "
				"induced_action_on_sign finished" << endl;
	}
	A_on_sign->Kernel->group_order(go);
	if (f_v) {
		cout << "strong_generators::even_subgroup "
				"kernel has order " << go << endl;
	}


	init_from_sims(A_on_sign->Kernel, verbose_level);

	FREE_OBJECT(A_on_sign);
	
	if (f_v) {
		cout << "strong_generators::even_subgroup done" << endl;
	}
}

void strong_generators::Sylow_subgroup(
		sims *S, int p, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sims *P;
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "strong_generators::Sylow_subgroup " << endl;
	}

	P = NEW_OBJECT(sims);
	S->sylow_subgroup(p, P, verbose_level);
	init_from_sims(P, verbose_level);
	FREE_OBJECT(P);

	if (f_v) {
		cout << "strong_generators::Sylow_subgroup done" << endl;
	}
}

void strong_generators::init_single(
		actions::action *A,
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sims *S;

	if (f_v) {
		cout << "strong_generators::init_single" << endl;
	}
	S = A->create_sims_from_single_generator_without_target_group_order(
		Elt, verbose_level);
	init_from_sims(S, verbose_level);
	FREE_OBJECT(S);

	if (f_v) {
		cout << "strong_generators::init_single "
				"done" << endl;
	}
}

void strong_generators::init_single_with_target_go(
		actions::action *A,
		int *Elt, int target_go, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sims *S;

	if (f_v) {
		cout << "strong_generators::init_single_with_target_go" << endl;
	}
	S = A->create_sims_from_single_generator_without_target_group_order(
		Elt, verbose_level);
	init_from_sims(S, verbose_level);
	FREE_OBJECT(S);

	if (f_v) {
		cout << "strong_generators::init_single_with_target_go "
				"done" << endl;
	}
}

void strong_generators::init_trivial_group(
		actions::action *A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "strong_generators::init_trivial_group" << endl;
	}
	strong_generators::A = A;
	tl = NEW_int(A->base_len());
	Int_vec_one(tl, A->base_len());

	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init(A, verbose_level - 2);
	gens->allocate(0, verbose_level - 2);
	//S->extract_strong_generators_in_order(*gens,
	// tl, 0 /*verbose_level*/);
	if (f_v) {
		cout << "strong_generators::init_trivial_group done" << endl;
	}
}

void strong_generators::generators_for_the_monomial_group(
		actions::action *A,
		algebra::matrix_group *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Elt1;
	sims *S;
	field_theory::finite_field *F;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object target_go;
	int *go_factored;
	int n, q, pos_frobenius;
	data_structures_groups::vector_ge *my_gens;
	int *data;
	int i, h, hh, h1, j, a, b, nb_gens;
	number_theory::number_theory_domain NT;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_monomial_group "
				"initializing monomial group" << endl;
	}
	strong_generators::A = A;
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (f_v) {
		cout << "n=" << n << " q=" << q << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	go_factored = NEW_int(3 * n + 1);
	data = NEW_int(n * n + n + 1);

	pos_frobenius = 0;
	if (Mtx->f_projective) {
		if (f_v) {
			cout << "strong_generators::generators_for_the_monomial_group "
					"type is projective" << endl;
		}
		pos_frobenius = n * n;
	}

	if (Mtx->f_affine) {
		if (f_v) {
			cout << "strong_generators::generators_for_the_monomial_group "
					"type is affine" << endl;
		}
		pos_frobenius = n * n + n;
		//exit(1);
	}

	if (Mtx->f_general_linear) {
		if (f_v) {
			cout << "strong_generators::generators_for_the_monomial_group "
					"type is general_linear" << endl;
		}
		pos_frobenius = n * n;
	}


	// group order 
	// = n! * (q - 1)^(n-1) * e if projective
	// = n! * (q - 1)^n * e if general linear
	// = n! * (q - 1)^n * q^n * e if affine
	// where e is the degree of the field if f_semilinear is true
	// and e = 1 otherwise
	
	for (i = 0; i < n; i++) {
		go_factored[i] = n - i;
	}
	for (i = 0; i < n; i++) {
		if (i == n - 1) {
			go_factored[n + i] = 1; // because it is projective
		}
		else {
			go_factored[n + i] = q - 1;
		}
	}
	for (i = 0; i < n; i++) {
		if (Mtx->f_affine) {
			go_factored[2 * n + i] = q;
		}
		else {
			go_factored[2 * n + i] = 1;
		}
	}
	if (Mtx->f_semilinear) {
		go_factored[3 * n] = F->e;
	}
	else {
		go_factored[3 * n] = 1;
	}
	D.multiply_up(target_go,
			go_factored, 3 * n + 1,
			0 /* verbose_level */);
	if (f_v) {
		cout << "group order factored: ";
		Int_vec_print(cout, go_factored, 3 * n + 1);
		cout << endl;
		cout << "target_go=" << target_go << endl;
	}
	my_gens = NEW_OBJECT(data_structures_groups::vector_ge);
	my_gens->init(A, verbose_level - 2);
	nb_gens = n - 1 + 1 + 1;
	if (Mtx->f_affine) {
		nb_gens += n * F->e;
	}
	my_gens->allocate(nb_gens, verbose_level - 2);
	for (h = 0; h < nb_gens; h++) {

		if (f_v) {
			cout << "strong_generators::generators_for_the_monomial_group "
					"generator " << h << " / " << nb_gens << ":" << endl;
		}
		F->Linear_algebra->identity_matrix(data, n);
		if (Mtx->f_affine) {
			Int_vec_zero(data + n * n, n);
		}

		if (h < n - 1) {
			// swap basis vector h and h + 1:
			hh = h + 1;
			data[h * n + h] = 0;
			data[hh * n + hh] = 0;
			data[h * n + hh] = 1;
			data[hh * n + h] = 1;
		}
		else if (h == n - 1) {
			data[0] = F->alpha_power(1);
		}
		else if (h == n) {
			if (Mtx->f_semilinear) {
				data[pos_frobenius] = 1;
			}
		}
		else if (Mtx->f_affine) {
			h1 = h - n - 1;
			a = h1 / F->e;
			b = h1 % F->e;
			for (j = 0; j < n; j++) {
				data[n * n + j] = 0;
			}
			data[n * n + a] = NT.i_power_j(F->p, b);
				// elements of a field basis of F_q over F_p
		}
		if (f_v) {
			cout << "strong_generators::generators_for_the_monomial_group "
					"generator " << h << " / "
					<< nb_gens << ", before A->make_element" << endl;
			cout << "data = ";
			Int_vec_print(cout, data, Mtx->elt_size_int_half);
			cout << endl;
			cout << "in action " << A->label << endl;
		}
		A->Group_element->make_element(
				Elt1, data, verbose_level - 1);
		if (f_vv) {
			cout << "generator " << h << ":" << endl;
			A->Group_element->element_print_quick(Elt1, cout);
		}
		my_gens->copy_in(h, Elt1);
	}
	if (f_v) {
		cout << "strong_generators::generators_for_the_monomial_group "
				"creating group" << endl;
	}
	S = A->create_sims_from_generators_randomized(
		my_gens, true /* f_target_go */, 
		target_go, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "strong_generators::generators_for_the_monomial_group "
				"after creating group" << endl;
	}
	init_from_sims(S, 0);
	if (f_v) {
		cout << "strong_generators::generators_for_the_monomial_group "
				"after extracting strong generators" << endl;
	}
	if (f_vv) {
		int f_print_as_permutation = false;
		int f_offset = false;
		int offset = 0;
		int f_do_it_anyway_even_for_big_degree = false;
		int f_print_cycles_of_length_one = false;
		
		ring_theory::longinteger_object go;
	
		cout << "computing the group order:" << endl;
		group_order(go);
		cout << "The group order is " << go << endl;
		
		cout << "strong generators are:" << endl;
		gens->print(
				cout, f_print_as_permutation,
			f_offset, offset,
			f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0 /* verbose_level */);
	}
	FREE_OBJECT(S);
	FREE_OBJECT(my_gens);
	FREE_int(data);
	FREE_int(go_factored);
	FREE_int(Elt1);
	if (f_v) {
		cout << "strong_generators::generators_for_the_monomial_group done" << endl;
	}
}

void strong_generators::generators_for_the_diagonal_group(
		actions::action *A,
		algebra::matrix_group *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Elt1;
	sims *S;
	field_theory::finite_field *F;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object target_go;
	int *go_factored;
	int n, q;
	data_structures_groups::vector_ge *my_gens;
	int *data;
	int i, h;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_diagonal_group "
				"initializing diagonal group" << endl;
	}
	strong_generators::A = A;
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (f_v) {
		cout << "n=" << n << " q=" << q << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	go_factored = NEW_int(n + 1);
	data = NEW_int(n * n + 1);

	// group order 
	// = q^n * e if not projective
	// = q^(n-1) * e if projective
	// where e is the degree of the field if f_semilinear is true
	// and e = 1 otherwise
	
	for (i = 0; i < n; i++) {
		if (i == n - 1) {
			go_factored[i] = 1; // because it is projective
		}
		else {
			go_factored[i] = q - 1;
		}
	}

	if (Mtx->f_projective) {
		cout << "strong_generators::generators_for_the_diagonal_group "
				"type is projective" << endl;
	}

	if (Mtx->f_affine) {
		cout << "strong_generators::generators_for_the_diagonal_group "
				"type should not be affine" << endl;
		exit(1);
	}

	if (Mtx->f_general_linear) {
		cout << "strong_generators::generators_for_the_diagonal_group "
				"type is general_linear" << endl;
	}

	if (Mtx->f_semilinear) {
		go_factored[n] = F->e;
	}
	else {
		go_factored[n] = 1;
	}
	D.multiply_up(target_go, go_factored, n + 1, 0 /* verbose_level */);
	if (f_v) {
		cout << "group order factored: ";
		Int_vec_print(cout, go_factored, n + 1);
		cout << endl;
		cout << "target_go=" << target_go << endl;
	}
	my_gens = NEW_OBJECT(data_structures_groups::vector_ge);
	my_gens->init(A, verbose_level - 2);
	my_gens->allocate(n + 1, verbose_level - 2);
	for (h = 0; h < n + 1; h++) {

		F->Linear_algebra->identity_matrix(data, n);

		if (h < n) {
			data[h * n + h] = F->alpha_power(1);
		}
		else if (h == n) {
			if (Mtx->f_semilinear) {
				data[n * n] = 1;
			}
		}
		A->Group_element->make_element(
				Elt1, data, 0 /*verbose_level - 1*/);
		if (f_vv) {
			cout << "generator " << h << ":" << endl;
			A->Group_element->element_print_quick(Elt1, cout);
		}
		my_gens->copy_in(h, Elt1);
	}
	if (f_v) {
		cout << "strong_generators::generators_for_the_diagonal_group "
				"creating group" << endl;
	}
	S = A->create_sims_from_generators_randomized(
		my_gens, true /* f_target_go */, 
		target_go, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "strong_generators::generators_for_the_diagonal_group "
				"after creating group" << endl;
	}
	init_from_sims(S, 0);
	if (f_v) {
		cout << "strong_generators::generators_for_the_diagonal_group "
				"after extracting strong generators" << endl;
	}
	if (f_vv) {
		int f_print_as_permutation = false;
		int f_offset = false;
		int offset = 0;
		int f_do_it_anyway_even_for_big_degree = false;
		int f_print_cycles_of_length_one = false;
		
		ring_theory::longinteger_object go;
	
		cout << "computing the group order:" << endl;
		group_order(go);
		cout << "The group order is " << go << endl;
		
		cout << "strong generators are:" << endl;
		gens->print(
				cout, f_print_as_permutation,
			f_offset, offset,
			f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0 /* verbose_level */);
	}
	FREE_OBJECT(S);
	FREE_OBJECT(my_gens);
	FREE_int(data);
	FREE_int(go_factored);
	FREE_int(Elt1);
	if (f_v) {
		cout << "strong_generators::generators_for_the_diagonal_group done" << endl;
	}
}

void strong_generators::generators_for_the_singer_cycle(
		actions::action *A,
		algebra::matrix_group *Mtx, int power_of_singer,
	data_structures_groups::vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Elt1;
	sims *S;
	field_theory::finite_field *F;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object target_go;
	long int *go_factored;
	int n, q;
	//vector_ge *my_gens;
	int *data;
	int i;
	number_theory::number_theory_domain NT;
	geometry::geometry_global Gg;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_singer_cycle "
				"initializing singer group "
				"power_of_singer=" << power_of_singer << endl;
	}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (f_v) {
		cout << "n=" << n << " q=" << q << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	go_factored = NEW_lint(1);
	data = NEW_int(n * n + 1);

	// group order 
	// = (q^n - 1) / (q - 1) if projective
	// = q^n - 1 if general_linear
	
	go_factored[0] = Gg.nb_PG_elements(n - 1, q);
	long int g;
	g = NT.gcd_lint(go_factored[0], power_of_singer);
	go_factored[0] = go_factored[0] / g;

	D.multiply_up_lint(target_go, go_factored, 1, 0 /* verbose_level */);
	if (f_v) {
		cout << "group order factored: ";
		Lint_vec_print(cout, go_factored, 1);
		cout << endl;
		cout << "target_go=" << target_go << endl;
	}
	nice_gens = NEW_OBJECT(data_structures_groups::vector_ge);
	nice_gens->init(A, verbose_level - 2);
	nice_gens->allocate(1, verbose_level - 2);

	

	{
		field_theory::finite_field Fq;

#if 0
		if (!NT.is_prime(q)) {
			cout << "strong_generators::generators_for_the_singer_cycle "
					"field order must be a prime" << endl;
			exit(1);
		}
#endif
	
		Fq.finite_field_init_small_order(q,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				0 /*verbose_level*/);
		ring_theory::unipoly_domain FX(&Fq);

		ring_theory::unipoly_object m;
		ring_theory::longinteger_object rk;

		FX.create_object_by_rank(m, 0, verbose_level);

		if (f_v) {
			cout << "strong_generators::generators_for_the_singer_cycle "
					"before FX.get_a_primitive_polynomial "
					"q=" << q << " degree=" << n << endl;
		}
		FX.get_a_primitive_polynomial(m, n, verbose_level - 1);
		if (f_v) {
			cout << "strong_generators::generators_for_the_singer_cycle "
					"after FX.get_a_primitive_polynomial" << endl;
			cout << "m=";
			FX.print_object(m, cout);
			cout << endl;
		}
	
		Int_vec_zero(data, n * n);
	
		// create companion matrix of the polynomial:

		// create upper diagonal:
		for (i = 0; i < n - 1; i++) {
			data[i * n + i + 1] = 1;
		}
	
		int a, b;

		// create the lower row:
		for (i = 0; i < n; i++) {
			a = FX.s_i(m, i);
			b = F->negate(a);
			data[(n - 1) * n + i] = b;
		}
	
		if (Mtx->f_semilinear) {
			data[n * n] = 0;
		}
	}

	
	A->Group_element->make_element(Elt1, data, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "generator :" << endl;
		A->Group_element->element_print_quick(Elt1, cout);
	}

	A->Group_element->element_power_int_in_place(Elt1,
		power_of_singer, 0 /* verbose_level */);

	if (f_v) {
		cout << "generator after raising to the "
				"power of " << power_of_singer << ":" << endl;
		A->Group_element->element_print_quick(Elt1, cout);
	}
	nice_gens->copy_in(0, Elt1);


	if (f_v) {
		cout << "strong_generators::generators_for_the_singer_cycle "
				"creating group" << endl;
	}
	if (f_v) {
		cout << "group order factored: ";
		Lint_vec_print(cout, go_factored, 1);
		cout << endl;
		cout << "target_go=" << target_go << endl;
	}
	S = A->create_sims_from_generators_randomized(
		nice_gens,
		true /* f_target_go */,
		target_go, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "strong_generators::generators_for_the_singer_cycle "
				"after creating group" << endl;
	}
	init_from_sims(S, 0);
	if (f_v) {
		cout << "strong_generators::generators_for_the_singer_cycle "
				"after extracting "
				"strong generators" << endl;
	}
	if (f_vv) {
		int f_print_as_permutation = false;
		int f_offset = false;
		int offset = 0;
		int f_do_it_anyway_even_for_big_degree = false;
		int f_print_cycles_of_length_one = false;
		
		cout << "strong generators are:" << endl;
		gens->print(cout,
				f_print_as_permutation,
			f_offset, offset,
			f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0 /* verbose_level */);
	}
	FREE_OBJECT(S);
	//FREE_OBJECT(nice_gens);
	FREE_int(data);
	FREE_lint(go_factored);
	FREE_int(Elt1);
	if (f_v) {
		cout << "strong_generators::generators_for_the_singer_cycle done" << endl;
	}
}

void strong_generators::generators_for_the_singer_cycle_and_the_Frobenius(
		actions::action *A,
		algebra::matrix_group *Mtx, int power_of_singer,
	data_structures_groups::vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Elt1;
	sims *S;
	field_theory::finite_field *F;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object target_go;
	long int *go_factored;
	int n, q;
	//vector_ge *my_gens;
	int *data1;
	int *data2;
	int i;
	number_theory::number_theory_domain NT;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "strong_generators::generators_for_the_singer_cycle_and_the_Frobenius "
				"initializing singer group power_of_singer="
				<< power_of_singer << endl;
	}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (f_v) {
		cout << "n=" << n << " q=" << q << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	go_factored = NEW_lint(2);
	data1 = NEW_int(n * n + 1);
	data2 = NEW_int(n * n + 1);

	// group order
	// = (q^n - 1) / (q - 1) if projective
	// = q^n - 1 if general_linear

	go_factored[0] = Gg.nb_PG_elements(n - 1, q);
	long int g;
	g = NT.gcd_lint(go_factored[0], power_of_singer);
	go_factored[0] = go_factored[0] / g;
	go_factored[1] = n;

	D.multiply_up_lint(
			target_go, go_factored, 2,
			0 /* verbose_level */);
	if (f_v) {
		cout << "group order factored: ";
		Lint_vec_print(cout, go_factored, 2);
		cout << endl;
		cout << "target_go=" << target_go << endl;
	}
	nice_gens = NEW_OBJECT(data_structures_groups::vector_ge);
	nice_gens->init(A, verbose_level - 2);
	nice_gens->allocate(2, verbose_level - 2);



	{
		field_theory::finite_field Fp;

		if (!NT.is_prime(q)) {
			cout << "strong_generators::generators_for_the_singer_cycle_and_the_Frobenius "
					"field order must be a prime" << endl;
			exit(1);
		}

		Fp.finite_field_init_small_order(q,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				0 /*verbose_level*/);
		ring_theory::unipoly_domain FX(&Fp);

		ring_theory::unipoly_object m;
		ring_theory::longinteger_object rk;

		FX.create_object_by_rank(m, 0, verbose_level);

		if (f_v) {
			cout << "search_for_primitive_polynomial_of_given_degree "
					"p=" << q << " degree=" << n << endl;
		}
		FX.get_a_primitive_polynomial(
				m, n, verbose_level - 1);

		Int_vec_zero(data1, n * n);


		// create upper diagonal:
		for (i = 0; i < n - 1; i++) {
			data1[i * n + i + 1] = 1;
		}

		int a, b;

		// create the lower row:
		for (i = 0; i < n; i++) {
			a = FX.s_i(m, i);
			b = F->negate(a);
			data1[(n - 1) * n + i] = b;
		}

		if (Mtx->f_semilinear) {
			data1[n * n] = 0;
		}

		Int_vec_zero(data2, n * n);

		FX.Frobenius_matrix_by_rows(data2, m,
				verbose_level);

		if (Mtx->f_semilinear) {
			data2[n * n] = 0;
		}

	}


	A->Group_element->make_element(Elt1, data1, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "singer cycle 0:" << endl;
		A->Group_element->element_print_quick(Elt1, cout);
	}

	A->Group_element->element_power_int_in_place(
			Elt1,
		power_of_singer, 0 /* verbose_level */);

	if (f_v) {
		cout << "generator after raising to the "
				"power of " << power_of_singer << ":" << endl;
		A->Group_element->element_print_quick(Elt1, cout);
	}
	nice_gens->copy_in(0, Elt1);

	A->Group_element->make_element(Elt1, data2, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "Frob:" << endl;
		A->Group_element->element_print_quick(Elt1, cout);
	}
	nice_gens->copy_in(1, Elt1);



	if (f_v) {
		cout << "strong_generators::generators_for_the_singer_cycle_and_the_Frobenius "
				"creating group" << endl;
	}
	if (f_v) {
		cout << "group order factored: ";
		Lint_vec_print(cout, go_factored, 1);
		cout << endl;
		cout << "target_go=" << target_go << endl;
	}
	S = A->create_sims_from_generators_randomized(
		nice_gens,
		true /* f_target_go */,
		target_go, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "strong_generators::generators_for_the_singer_cycle_and_the_Frobenius "
				"after creating group" << endl;
	}
	init_from_sims(S, 0);
	if (f_v) {
		cout << "strong_generators::generators_for_the_singer_cycle_and_the_Frobenius "
				"after extracting strong generators" << endl;
	}
	if (f_vv) {
		int f_print_as_permutation = false;
		int f_offset = false;
		int offset = 0;
		int f_do_it_anyway_even_for_big_degree = false;
		int f_print_cycles_of_length_one = false;

		cout << "strong generators are:" << endl;
		gens->print(cout,
				f_print_as_permutation,
			f_offset, offset,
			f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0 /* verbose_level */);
	}
	FREE_OBJECT(S);
	//FREE_OBJECT(nice_gens);
	FREE_int(data1);
	FREE_int(data2);
	FREE_lint(go_factored);
	FREE_int(Elt1);
	if (f_v) {
		cout << "strong_generators::generators_for_the_singer_cycle_and_the_Frobenius done" << endl;
	}
}

void strong_generators::generators_for_the_null_polarity_group(
		actions::action *A,
		algebra::matrix_group *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	field_theory::finite_field *F;
	int n, q;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_null_polarity_group" << endl;
	}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (f_v) {
		cout << "n=" << n << " q=" << q << endl;
	}

	algebra::null_polarity_generator *N;

	N = NEW_OBJECT(algebra::null_polarity_generator);


	if (f_v) {
		cout << "strong_generators::generators_for_the_null_polarity_group "
				"before N->init" << endl;
	}
	N->init(F, n, verbose_level - 1);
	if (f_v) {
		cout << "strong_generators::generators_for_the_null_polarity_group "
				"after N->init" << endl;
	}
	
	data_structures_groups::vector_ge *nice_gens;

	init_from_data(
		A, N->Data,
		N->nb_gens, n * n, N->transversal_length, 
		nice_gens,
		verbose_level - 1);


	FREE_OBJECT(N);
	FREE_OBJECT(nice_gens);
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_null_polarity_group done" << endl;
	}
}

void strong_generators::generators_for_symplectic_group(
		actions::action *A,
		algebra::matrix_group *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	field_theory::finite_field *F;
	int n, q;
	
	if (f_v) {
		cout << "strong_generators::generators_for_symplectic_group" << endl;
	}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (f_v) {
		cout << "n=" << n << " q=" << q << endl;
	}

	algebra::generators_symplectic_group *N;

	N = NEW_OBJECT(algebra::generators_symplectic_group);


	if (f_v) {
		cout << "strong_generators::generators_for_symplectic_group before "
				"generators_symplectic_group::init" << endl;
	}
	N->init(F, n, verbose_level - 1);
	if (f_v) {
		cout << "strong_generators::generators_for_symplectic_group after "
				"generators_symplectic_group::init" << endl;
	}
	
		// warning, N->transversal_length[n]
		// but A->base_len = n + 1

	data_structures_groups::vector_ge *nice_gens;
	int *t_len;
	int i;

	t_len = NEW_int(A->base_len());
	for (i = 0; i < A->base_len(); i++) {
		if (i < n) {
			t_len[i] = N->transversal_length[i];
		}
		else {
			t_len[i] = 1;
		}
	}
	if (f_v) {
		cout << "strong_generators::generators_for_symplectic_group t_len=";
		Int_vec_print(cout, t_len, A->base_len());
		cout << endl;
	}

	if (f_v) {
		cout << "strong_generators::generators_for_symplectic_group before "
				"init_from_data" << endl;
	}
	init_from_data(
		A, N->Data,
		N->nb_gens, n * n, t_len,
		nice_gens,
		verbose_level - 1);
	if (f_v) {
		cout << "strong_generators::generators_for_symplectic_group after "
				"init_from_data" << endl;
	}

	ring_theory::longinteger_object target_go;



	target_go.create_product(A->base_len(), tl);
	if (f_v) {
		cout << "strong_generators::generators_for_symplectic_group "
				"target_go = " << target_go << endl;
	}

	if (f_v) {
		cout << "strong_generators::stabilizer_of_cubic_surface_from_catalogue "
				"before init_reduced_generating_set" << endl;
	}
	init_reduced_generating_set(
			gens,
			target_go,
			verbose_level - 1);
	if (f_v) {
		cout << "strong_generators::stabilizer_of_cubic_surface_from_catalogue "
				"after init_reduced_generating_set" << endl;
	}


	FREE_int(t_len);
	FREE_OBJECT(nice_gens);
	FREE_OBJECT(N);
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_symplectic_group done" << endl;
	}
}

void strong_generators::init_centralizer_of_matrix(
		actions::action *A, int *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	sims *S;

	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix" << endl;
	}
	S = A->create_sims_for_centralizer_of_matrix(
			Mtx, verbose_level - 1);
	init_from_sims(S, 0 /* verbose_level */);
	FREE_OBJECT(S);
	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix done" << endl;
	}
}

void strong_generators::init_centralizer_of_matrix_general_linear(
		actions::action *A_projective,
		actions::action *A_general_linear, int *Mtx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	sims *S;
	strong_generators *SG1;
	ring_theory::longinteger_object go1, Q, go;
	ring_theory::longinteger_domain D;
	algebra::matrix_group *M;
	data_structures_groups::vector_ge *new_gens;
	int *data;
	int q, n, i;

	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix_general_linear" << endl;
	}
	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix_general_linear "
				"before A_projective->create_sims_for_centralizer_of_matrix" << endl;
	}
	S = A_projective->create_sims_for_centralizer_of_matrix(
			Mtx, 0/* verbose_level */);
	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix_general_linear "
				"after A_projective->create_sims_for_centralizer_of_matrix" << endl;
	}
	SG1 = NEW_OBJECT(strong_generators);
	SG1->init_from_sims(S, 0 /* verbose_level */);
	FREE_OBJECT(S);

	M = A_projective->G.matrix_grp;
	q = M->GFq->q;
	n = M->n;

	SG1->group_order(go1);
	Q.create(q - 1);
	D.mult(go1, Q, go);

	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix_general_linear "
				"created centralizer "
				"in the projective linear group of "
				"order " << go1 << endl;
	}
	
	new_gens = NEW_OBJECT(data_structures_groups::vector_ge);
	new_gens->init(A_general_linear, verbose_level - 2);
	new_gens->allocate(SG1->gens->len + 1, verbose_level - 2);
	data = NEW_int(n * n + n + 1);
	for (i = 0; i < SG1->gens->len; i++) {

		Int_vec_copy(SG1->gens->ith(i), data, n * n);

		if (M->f_semilinear) {
			data[n * n] = SG1->gens->ith(i)[n * n];
		}
		A_general_linear->Group_element->make_element(
				new_gens->ith(i), data, 0);
	}
	M->GFq->Linear_algebra->diagonal_matrix(
			data, n, M->GFq->primitive_root());
	if (M->f_semilinear) {
		data[n * n] = 0;
	}
	A_general_linear->Group_element->make_element(
			new_gens->ith(SG1->gens->len), data, 0);

	
	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix_general_linear "
				"creating sims for the general "
				"linear centralizer of order " << go << endl;
	}
	S = A_general_linear->create_sims_from_generators_with_target_group_order(
		new_gens, go, 0 /* verbose_level */);
	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix_general_linear "
				"creating sims for the general "
				"linear centralizer of order " << go <<  " done" << endl;
	}
	init_from_sims(S, 0 /* verbose_level */);
	FREE_OBJECT(S);

	
	FREE_int(data);
	FREE_OBJECT(new_gens);
	FREE_OBJECT(SG1);
	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix_general_linear "
				"done" << endl;
	}
}

void strong_generators::field_reduction(
		actions::action *Aq,
		int n, int s,
		field_theory::finite_field *Fq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q, Q, m, t;
	field_theory::finite_field *FQ;
	actions::action *AQ;
	field_theory::subfield_structure *S;
	sims *Sims;
	int *EltQ;
	int *Eltq;
	int *Mtx;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "strong_generators::field_reduction" << endl;
	}
	q = Fq->q;
	Q = NT.i_power_j(q, s);
	m = n / s;
	if (m * s != n) {
		cout << "strong_generators::field_reduction "
				"s must divide n" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "strong_generators::field_reduction "
				"creating subfield structure" << endl;
	}
	if (f_v) {
		cout << "n=" << n << endl;
		cout << "s=" << s << endl;
		cout << "m=" << m << endl;
		cout << "q=" << q << endl;
		cout << "Q=" << Q << endl;
	}
	FQ = NEW_OBJECT(field_theory::finite_field);
	FQ->finite_field_init_small_order(
			Q,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0);

	AQ = NEW_OBJECT(actions::action);
	
	if (f_v) {
		cout << "strong_generators::field_reduction "
				"creating AQ" << endl;
	}

	data_structures_groups::vector_ge *nice_gens;

	if (f_v) {
		cout << "strong_generators::field_reduction "
				"before AQ->Known_groups->init_general_linear_group" << endl;
	}
	AQ->Known_groups->init_general_linear_group(
			m,
			FQ,
			false /* f_semilinear */,
			true /* f_basis */, false /* f_init_sims */,
			nice_gens,
			verbose_level - 2);
	if (f_v) {
		cout << "strong_generators::field_reduction "
				"after AQ->Known_groups->init_general_linear_group" << endl;
	}
	if (f_v) {
		cout << "strong_generators::field_reduction "
				"creating AQ done" << endl;
	}
	FREE_OBJECT(nice_gens);

	ring_theory::longinteger_object order_GLmQ;
	ring_theory::longinteger_object target_go;
	ring_theory::longinteger_domain D;
	int r;

	AQ->group_order(order_GLmQ);
	

	if (f_v) {
		cout << "strong_generators::field_reduction "
				"order of GL(m,Q) = " << order_GLmQ << endl;
	}
	D.integral_division_by_int(order_GLmQ, 
		q - 1, target_go, r);
	if (f_v) {
		cout << "strong_generators::field_reduction "
				"target_go = " << target_go << endl;
	}

	S = NEW_OBJECT(field_theory::subfield_structure);
	S->init(FQ, Fq, verbose_level);

	if (f_v) {
		cout << "strong_generators::field_reduction "
			"creating subfield structure done" << endl;
	}

	data_structures_groups::vector_ge *gens;
	data_structures_groups::vector_ge *gens1;
	int nb_gens;

	gens = AQ->Strong_gens->gens;
	nb_gens = gens->len;

	gens1 = NEW_OBJECT(data_structures_groups::vector_ge);

	Eltq = NEW_int(Aq->elt_size_in_int);
	Mtx = NEW_int(n * n);

	if (f_v) {
		cout << "strong_generators::field_reduction "
				"lifting generators" << endl;
	}
	gens1->init(Aq, verbose_level - 2);
	gens1->allocate(nb_gens, verbose_level - 2);
	for (t = 0; t < nb_gens; t++) {
		cout << "strong_generators::field_reduction " << t
				<< " / " << nb_gens << endl;
		EltQ = gens->ith(t);
		S->lift_matrix(EltQ, m, Mtx, 0 /* verbose_level */);
		if (f_v) {
			cout << "lifted matrix:" << endl;
			Int_matrix_print(Mtx, n, n);
		}
		Aq->Group_element->make_element(
				Eltq, Mtx, verbose_level - 1);
		if (f_v) {
			cout << "after make_element:" << endl;
			Aq->Group_element->element_print_quick(Eltq, cout);
		}
		Aq->Group_element->element_move(Eltq, gens1->ith(t), 0);
		cout << "strong_generators::field_reduction " << t
				<< " / " << nb_gens << " done" << endl;
	}

	if (f_v) {
		cout << "strong_generators::field_reduction "
				"creating lifted group:" << endl;
	}
	Sims = Aq->create_sims_from_generators_with_target_group_order(
		gens1, target_go, 0 /* verbose_level */);

#if 0
	Sims = Aq->create_sims_from_generators_without_target_group_order(
		gens1, MINIMUM(2, verbose_level - 3));
#endif

	if (f_v) {
		cout << "strong_generators::field_reduction "
				"creating lifted group done" << endl;
	}

	ring_theory::longinteger_object go;

	Sims->group_order(go);

	if (f_v) {
		cout << "go=" << go << endl;
	}

	init_from_sims(Sims, 0 /* verbose_level */);
	if (f_v) {
		cout << "strong_generators::field_reduction "
				"strong generators are:" << endl;
		print_generators(cout, verbose_level - 1);
	}

	FREE_OBJECT(gens1);
	FREE_int(Eltq);
	FREE_int(Mtx);
	FREE_OBJECT(Sims);
	FREE_OBJECT(S);
	FREE_OBJECT(AQ);
	FREE_OBJECT(FQ);
	if (f_v) {
		cout << "strong_generators::field_reduction done" << endl;
	}

}

void strong_generators::generators_for_translation_plane_in_andre_model(
		actions::action *A_PGL_n1_q,
		actions::action *A_PGL_n_q,
		algebra::matrix_group *Mtx_n1, algebra::matrix_group *Mtx_n,
	strong_generators *spread_stab_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	field_theory::finite_field *F;
	int n, n1, q;
	data_structures_groups::vector_ge *my_gens;
	int *M, *M1;
	int sz;

	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model" << endl;
	}
	F = Mtx_n->GFq;
	q = F->q;
	n = Mtx_n->n;
	n1 = Mtx_n1->n;

	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model "
				"n=" << n << " n1=" << n1 << endl;
	}
	int f_semilinear;
	int nb_gens, h, cnt, i, j, a, u;


	f_semilinear = Mtx_n1->f_semilinear;

	nb_gens = spread_stab_gens->gens->len + 1 + n * F->e;
	//nb_gens = spread_stab_gens->len + /* 1 + */ n * F->e;

	int alpha;

	alpha = F->primitive_root();

	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model "
				"nb_gens=" << nb_gens << endl;
	}
	sz = n1 * n1 + 1;
	M = NEW_int(sz * nb_gens);
	my_gens = NEW_OBJECT(data_structures_groups::vector_ge);
	my_gens->init(A_PGL_n1_q, verbose_level - 2);
	my_gens->allocate(nb_gens, verbose_level - 2);


	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model "
				"making generators of the first kind:" << endl;
	}
	cnt = 0;
	for (h = 0; h < spread_stab_gens->gens->len; h++, cnt++) {
		if (f_vv) {
			cout << "making generator " << h << ":" << endl;
			//int_matrix_print(spread_stab_gens->ith(h), n, n);
		}

		M1 = M + cnt * sz;
		Int_vec_zero(M1, n1 * n1);
		for (i = 0; i < n1; i++) {
			M1[i * n1 + i] = 1;
		}
		if (f_semilinear) {
			M1[n1 * n1] = 0;
		}
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				a = spread_stab_gens->gens->ith(h)[i * n + j];
				M1[i * n1 + j] = a;
			}
		}
		if (f_semilinear) {
			a = spread_stab_gens->gens->ith(h)[n * n];
			M1[n1 * n1] = a;
		}
	}

#if 1
	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model "
				"making generators of the second kind:" << endl;
	}
	M1 = M + cnt * sz;
	Int_vec_zero(M1, n1 * n1);
	for (i = 0; i < n1; i++) {
		if (i < n1 - 1) {
			M1[i * n1 + i] = alpha;
		}
		else {
			M1[i * n1 + i] = 1;
		}
	}
	if (f_semilinear) {
		M1[n1 * n1] = 0;
	}
	cnt++;
#endif


	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model "
				"making generators of the third kind:" << endl;
	}

	for (h = 0; h < n; h++) {
		for (u = 0; u < F->e; u++, cnt++) {
			M1 = M + cnt * sz;
			Int_vec_zero(M1, n1 * n1);
			for (i = 0; i < n1; i++) {
				M1[i * n1 + i] = 1;
			}
			M1[(n1 - 1) * n1 + h] = F->power(alpha, u);
			// computes alpha^{p^u}
			if (f_semilinear) {
				M1[n1 * n1] = 0;
			}
		}
	}

	if (cnt != nb_gens) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model "
				"cnt != nb_gens" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model "
				"making generators:" << endl;
	}
	for (h = 0; h < nb_gens; h++) {
		M1 = M + h * sz;
		if (f_v) {
			cout << "strong_generators::generators_for_translation_plane_in_andre_model "
					"generator " << h << " / "
					<< nb_gens << endl;
			Int_matrix_print(M1, n1, n1);
			//cout << endl;
		}
		A_PGL_n1_q->Group_element->make_element(
				my_gens->ith(h), M1, 0 /* verbose_level */);
	}

	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object spread_stab_go, target_go, aa, b, bb, c, go;
	

	spread_stab_gens->group_order(spread_stab_go);

	spread_stab_go.assign_to(aa);
	//D.multiply_up(aa, spread_stab_tl, A_PGL_n_q->base_len);

	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model "
				"spread stabilizer has order " << aa << endl;
	}
	b.create_i_power_j(q, n);
	D.mult(aa, b, bb);
	c.create(q - 1);
	D.mult(bb, c, target_go);
	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model "
				"plane stabilizer target_go=" << target_go << endl;
	}

	sims *S;


	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model "
				"creating group" << endl;
	}
	S = A_PGL_n1_q->create_sims_from_generators_with_target_group_order(
		my_gens, target_go, 0 /*verbose_level*/);
	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model "
				"group has been created" << endl;
	}

	S->group_order(go);

	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model "
				"created group of order " << go << endl;
	}

	init_from_sims(S, 0 /* verbose_level */);

	FREE_OBJECT(S);
	FREE_int(M);
	FREE_OBJECT(my_gens);

	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model done" << endl;
	}
}

void strong_generators::generators_for_the_stabilizer_of_two_components(
		actions::action *A_PGL_n_q,
		algebra::matrix_group *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	field_theory::finite_field *F;
	int n, k, q;
	data_structures_groups::vector_ge *my_gens;
	actions::action *A_PGL_k_q;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components" << endl;
	}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	k = n >> 1;
	if (ODD(n)) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components "
				"n must be even" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "n=" << n << " k=" << k << " q=" << q << endl;
	}

	data_structures_groups::vector_ge *nice_gens;


	A_PGL_k_q = NEW_OBJECT(actions::action);
	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components "
				"before A_PGL_k_q->Known_groups->init_projective_group" << endl;
	}
	A_PGL_k_q->Known_groups->init_projective_group(k,
		F, false /*f_semilinear */, true /* f_init_sims */,
		true /* f_basis */,
		nice_gens,
		0 /* verbose_level */);
	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components "
				"after A_PGL_k_q->Known_groups->init_projective_group" << endl;
	}

	FREE_OBJECT(nice_gens);
	my_gens = NEW_OBJECT(data_structures_groups::vector_ge);
	

	actions::action_global AG;

	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components "
				"before AG.make_generators_stabilizer_of_two_components" << endl;
	}
	AG.make_generators_stabilizer_of_two_components(
			A_PGL_n_q, A_PGL_k_q,
		k, my_gens, verbose_level - 1);
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components "
				"after AG.make_generators_stabilizer_of_two_components" << endl;
	}

	ring_theory::longinteger_object go_linear, a, two, target_go;
	ring_theory::longinteger_domain D;

	two.create(1);
	A_PGL_k_q->group_order(go_linear);
	D.mult(go_linear, go_linear, a);
	D.mult(a, two, target_go);
	

	strong_generators *SG;

	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components "
				"before generators_to_strong_generators target_go=" << target_go << endl;
	}

	A_PGL_n_q->generators_to_strong_generators(
		true /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(SG, 0);


	FREE_OBJECT(SG);
	FREE_OBJECT(A_PGL_k_q);
	FREE_OBJECT(my_gens);

	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components "
				"done" << endl;
	}
}

void strong_generators::regulus_stabilizer(
		actions::action *A_PGL_n_q,
		algebra::matrix_group *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	field_theory::finite_field *F;
	int n, k, q;
	data_structures_groups::vector_ge *my_gens;
	actions::action *A_PGL_k_q;
	ring_theory::longinteger_object go, a, b, target_go;
	ring_theory::longinteger_domain D;
	int *P;
	int len1, len;
	int h1, h;
	int Identity[4] = {0,1,1,0};
	int *Q;
	int *Elt1;
	data_structures_groups::vector_ge *gens1;
	
	if (f_v) {
		cout << "strong_generators::regulus_stabilizer" << endl;
	}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (n != 4) {
		cout << "strong_generators::regulus_stabilizer "
				"n must be 4" << endl;
		exit(1);
	}
	k = n >> 1;
	if (ODD(n)) {
		cout << "strong_generators::regulus_stabilizer "
				"n must be even" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "n=" << n << " k=" << k << " q=" << q << endl;
	}

	data_structures_groups::vector_ge *nice_gens;

	A_PGL_k_q = NEW_OBJECT(actions::action);
	if (f_v) {
		cout << "strong_generators::regulus_stabilizer "
				"before A_PGL_k_q->Known_groups->init_projective_group" << endl;
	}
	A_PGL_k_q->Known_groups->init_projective_group(k,
		F, false /*f_semilinear */, true /* f_init_sims */,
		true /* f_basis */,
		nice_gens,
		0 /* verbose_level */);
	if (f_v) {
		cout << "strong_generators::regulus_stabilizer "
				"after A_PGL_k_q->Known_groups->init_projective_group" << endl;
	}
	FREE_OBJECT(nice_gens);
	A_PGL_k_q->group_order(go);
	D.mult(go, go, a);
	if (Mtx->f_semilinear) {
		b.create(F->e);
	}
	else {
		b.create(1);
	}
	D.mult(a, b, target_go);
	if (f_v) {
		cout << "strong_generators::regulus_stabilizer "
				"target_go=" << target_go
			<< " = order of PGL(" << k << "," << q << ")^2 * "
			<< b << " = " << go << "^2 * " << b << endl;
		cout << "action A_PGL_k_q: ";
		A_PGL_k_q->print_info();
	}

	Elt1 = NEW_int(A_PGL_n_q->elt_size_in_int);
	my_gens = NEW_OBJECT(data_structures_groups::vector_ge);
	my_gens->init(A_PGL_n_q, verbose_level - 2);

	gens1 = A_PGL_k_q->Strong_gens->gens;
	len1 = gens1->len;
	if (f_v) {
		cout << "There are " << len1 << " generators in gen1" << endl;
	}
	len = 2 * len1;
	if (Mtx->f_semilinear) {
		len++;
	}
	Q = NEW_int(n * n + 1);
	my_gens->allocate(len, verbose_level - 2);
	

	if (f_vv) {
		cout << "strong_generators::regulus_stabilizer "
				"creating generators for the stabilizer:" << endl;
	}
	for (h = 0; h < len; h++) {
		if (f_vv) {
			cout << "strong_generators::regulus_stabilizer "
					"h=" << h << " / " << len << endl;
		}

		if (h < 2 * len1) {
			h1 = h >> 1;
			P = gens1->ith(h1);
			if (f_vv) {
				cout << "strong_generators::regulus_stabilizer "
						"generator:" << endl;
				A_PGL_k_q->Group_element->print_quick(cout, P);
			}

			if ((h % 2) == 0) {
				F->Linear_algebra->Kronecker_product(P, Identity, 2, Q);
			}
			else {
				F->Linear_algebra->Kronecker_product(Identity, P, 2, Q);
			}
			if (Mtx->f_semilinear) {
				Q[n * n] = P[k * k];
			}
		}
		else {
			F->Linear_algebra->identity_matrix(Q, n);
			Q[n * n] = 1;
		}
		if (f_vv) {
			cout << "strong_generators::regulus_stabilizer "
					"h = " << h << " before make_element:" << endl;
			Int_matrix_print(Q, n, n);
			if (Mtx->f_semilinear) {
				cout << "strong_generators::regulus_stabilizer "
						"semilinear part = " << Q[n * n] << endl;
			}
		}
		A_PGL_n_q->Group_element->make_element(Elt1, Q, 0);
		if (f_vv) {
			cout << "strong_generators::regulus_stabilizer "
					"after make_element:" << endl;
			A_PGL_n_q->Group_element->print_quick(cout, Elt1);
		}
		A_PGL_n_q->Group_element->move(Elt1, my_gens->ith(h));
		
	}
	if (f_vv) {
		for (h = 0; h < len; h++) {
			cout << "strong_generators::regulus_stabilizer "
					"generator " << h << ":" << endl;
			A_PGL_n_q->Group_element->element_print(my_gens->ith(h), cout);
		}
	}

	if (f_v) {
		cout << "strong_generators::regulus_stabilizer "
				"before generators_to_strong_generators "
				"target_go=" << target_go << endl;
	}
	
	strong_generators *SG;

	A_PGL_n_q->generators_to_strong_generators(
		true /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::regulus_stabilizer "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(SG, 0);


	FREE_OBJECT(SG);
	FREE_OBJECT(A_PGL_k_q);
	FREE_OBJECT(my_gens);
	FREE_int(Elt1);
	FREE_int(Q);

	if (f_v) {
		cout << "strong_generators::regulus_stabilizer "
				"done" << endl;
	}
}

void strong_generators::generators_for_the_borel_subgroup_upper(
		actions::action *A_linear,
		algebra::matrix_group *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Elt1;
	data_structures_groups::vector_ge *my_gens;
	field_theory::finite_field *F;
	int *Q;
	int n, i, j, h, alpha, len, q;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_upper" << endl;
	}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	Elt1 = NEW_int(A_linear->elt_size_in_int);
	my_gens = NEW_OBJECT(data_structures_groups::vector_ge);
	my_gens->init(A_linear, verbose_level - 2);

	len = n + ((n * (n - 1)) >> 1);
	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_upper "
				"len=" << len << endl;
	}
	my_gens->allocate(len, verbose_level - 2);
	Q = NEW_int(n * n + 1);
	

	if (f_vv) {
		cout << "strong_generators::generators_for_the_borel_subgroup_upper "
				"creating generators for the stabilizer:" << endl;
	}
	h = 0;
	alpha = F->primitive_root();
	for (i = 0; i < n; i++, h++) {
		F->Linear_algebra->identity_matrix(Q, n);
		Q[i * n + i] = alpha;
		if (Mtx->f_semilinear) {
			Q[n * n] = 0;
		}
		A_linear->Group_element->make_element(Elt1, Q, 0);
		if (f_vv) {
			cout << "strong_generators::generators_for_the_borel_subgroup_upper "
					"after make_element:" << endl;
			A_linear->Group_element->print_quick(cout, Elt1);
		}
		A_linear->Group_element->move(Elt1, my_gens->ith(h));
	}
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			F->Linear_algebra->identity_matrix(Q, n);
			Q[i * n + j] = 1;
			if (Mtx->f_semilinear) {
				Q[n * n] = 0;
			}
			A_linear->Group_element->make_element(Elt1, Q, 0);
			if (f_vv) {
				cout << "strong_generators::generators_for_the_borel_subgroup_upper "
						"after make_element:" << endl;
				A_linear->Group_element->print_quick(cout, Elt1);
			}
			A_linear->Group_element->move(Elt1, my_gens->ith(h));
			h++;
		}
	}
	if (h != len) {
		cout << "strong_generators::generators_for_the_borel_subgroup_upper "
				"n != len" << endl;
		cout << "h=" << h << endl;		
		cout << "len=" << len << endl;		
		exit(1);
	}
	
	
	if (f_vv) {
		for (h = 0; h < len; h++) {
			cout << "strong_generators::generators_for_the_borel_subgroup_upper "
					"generator " << h << " / " << len << endl;
			A_linear->Group_element->element_print(my_gens->ith(h), cout);
		}
	}
	ring_theory::longinteger_object target_go;

	int *factors;
	int nb_factors;
	nb_factors = len;
	factors = NEW_int(nb_factors);
	h = 0;
	for (i = 0; i < n; i++) {
		factors[h++] = q - 1;
	}
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			factors[h++] = q;
		}
	}

	target_go.create_product(nb_factors, factors);
	FREE_int(factors);

	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_upper "
				"before generators_to_strong_generators "
				"target_go=" << target_go << endl;
	}
	

	strong_generators *SG;

	A_linear->generators_to_strong_generators(
		true /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_upper "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(SG, 0);

	FREE_OBJECT(SG);
	FREE_OBJECT(my_gens);
	FREE_int(Elt1);
	FREE_int(Q);
}

void strong_generators::generators_for_the_borel_subgroup_lower(
		actions::action *A_linear,
		algebra::matrix_group *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Elt1;
	data_structures_groups::vector_ge *my_gens;
	field_theory::finite_field *F;
	int *Q;
	int n, i, j, h, alpha, len, q;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_lower" << endl;
	}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	Elt1 = NEW_int(A_linear->elt_size_in_int);
	my_gens = NEW_OBJECT(data_structures_groups::vector_ge);
	my_gens->init(A_linear, verbose_level - 2);

	len = n + ((n * (n - 1)) >> 1);
	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_lower "
				"len=" << len << endl;
	}
	my_gens->allocate(len, verbose_level - 2);
	Q = NEW_int(n * n + 1);
	

	if (f_vv) {
		cout << "strong_generators::generators_for_the_borel_subgroup_lower "
				"creating generators for the stabilizer:" << endl;
	}
	h = 0;
	alpha = F->primitive_root();
	for (i = 0; i < n; i++, h++) {
		F->Linear_algebra->identity_matrix(Q, n);
		Q[i * n + i] = alpha;
		if (Mtx->f_semilinear) {
			Q[n * n] = 0;
		}
		A_linear->Group_element->make_element(Elt1, Q, 0);
		if (f_vv) {
			cout << "strong_generators::generators_for_the_borel_subgroup_lower "
					"after make_element:" << endl;
			A_linear->Group_element->print_quick(cout, Elt1);
		}
		A_linear->Group_element->move(Elt1, my_gens->ith(h));
	}
	for (i = 0; i < n; i++) {
		for (j = 0; j < i; j++) {
			F->Linear_algebra->identity_matrix(Q, n);
			Q[i * n + j] = 1;
			if (Mtx->f_semilinear) {
				Q[n * n] = 0;
			}
			A_linear->Group_element->make_element(Elt1, Q, 0);
			if (f_vv) {
				cout << "strong_generators::generators_for_the_borel_subgroup_lower "
						"after make_element:" << endl;
				A_linear->Group_element->print_quick(cout, Elt1);
			}
			A_linear->Group_element->move(Elt1, my_gens->ith(h));
			h++;
		}
	}
	if (h != len) {
		cout << "strong_generators::generators_for_the_borel_subgroup_lower "
				"n != len" << endl;
		cout << "h=" << h << endl;		
		cout << "len=" << len << endl;		
		exit(1);
	}
	
	
	if (f_vv) {
		for (h = 0; h < len; h++) {
			cout << "strong_generators::generators_for_the_borel_subgroup_lower "
					"generator " << h << " / " << len << endl;
			A_linear->Group_element->element_print(my_gens->ith(h), cout);
		}
	}
	ring_theory::longinteger_object target_go;

	int *factors;
	int nb_factors;
	nb_factors = len;
	factors = NEW_int(nb_factors);
	h = 0;
	for (i = 0; i < n; i++) {
		factors[h++] = q - 1;
	}
	for (i = 0; i < n; i++) {
		for (j = 0; j < i; j++) {
			factors[h++] = q;
		}
	}

	target_go.create_product(nb_factors, factors);
	FREE_int(factors);

	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_lower "
				"before generators_to_strong_generators "
				"target_go=" << target_go << endl;
	}
	

	strong_generators *SG;

	A_linear->generators_to_strong_generators(
		true /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_lower "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(SG, 0);

	FREE_OBJECT(SG);
	FREE_OBJECT(my_gens);
	FREE_int(Elt1);
	FREE_int(Q);
}

void strong_generators::generators_for_the_identity_subgroup(
		actions::action *A_linear,
		algebra::matrix_group *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Elt1;
	data_structures_groups::vector_ge *my_gens;
	field_theory::finite_field *F;
	int *Q;
	int n, i, h, len; //, q;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_identity_subgroup" << endl;
	}
	F = Mtx->GFq;
	//q = F->q;
	n = Mtx->n;
	Elt1 = NEW_int(A_linear->elt_size_in_int);
	my_gens = NEW_OBJECT(data_structures_groups::vector_ge);
	my_gens->init(A_linear, verbose_level - 2);

	len = 1;
	if (f_v) {
		cout << "strong_generators::generators_for_the_identity_subgroup "
				"len=" << len << endl;
	}
	my_gens->allocate(len, verbose_level - 2);
	Q = NEW_int(n * n + 1);
	

	if (f_vv) {
		cout << "strong_generators::generators_for_the_identity_subgroup "
				"creating generators "
				"for the stabilizer:" << endl;
	}
	for (i = 0; i < 1; i++) {
		F->Linear_algebra->identity_matrix(Q, n);
		if (Mtx->f_semilinear) {
			Q[n * n] = 0;
		}
		A_linear->Group_element->make_element(Elt1, Q, 0);
		if (f_vv) {
			cout << "strong_generators::generators_for_the_identity_subgroup "
					"after make_element:" << endl;
			A_linear->Group_element->print_quick(cout, Elt1);
		}
		A_linear->Group_element->move(Elt1, my_gens->ith(i));
	}
	
	
	if (f_vv) {
		for (h = 0; h < len; h++) {
			cout << "strong_generators::generators_for_the_identity_subgroup "
					"generator "
					<< h << " / " << len << endl;
			A_linear->Group_element->element_print(my_gens->ith(h), cout);
		}
	}
	ring_theory::longinteger_object target_go;

	target_go.create(1);

	if (f_v) {
		cout << "strong_generators::generators_for_the_identity_subgroup "
				"before generators_to_strong_generators "
				"target_go=" << target_go << endl;
	}
	

	strong_generators *SG;

	A_linear->generators_to_strong_generators(
		true /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_identity_subgroup "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(SG, 0);

	FREE_OBJECT(SG);
	FREE_OBJECT(my_gens);
	FREE_int(Elt1);
	FREE_int(Q);
}


void strong_generators::generators_for_parabolic_subgroup(
		actions::action *A_PGL_n_q,
		algebra::matrix_group *Mtx, int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	field_theory::finite_field *F;
	int n, q;
	data_structures_groups::vector_ge *my_gens;
	int *data;
	int size;
	int nb_gens;
	
	if (f_v) {
		cout << "strong_generators::generators_for_parabolic_subgroup" << endl;
	}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (f_v) {
		cout << "n=" << n << " k=" << k << " q=" << q << endl;
	}



	algebra::group_generators_domain GGD;

	if (f_v) {
		cout << "strong_generators::generators_for_parabolic_subgroup "
				"before GGD.generators_for_parabolic_subgroup" << endl;
	}
	GGD.generators_for_parabolic_subgroup(
			n, F,
		Mtx->f_semilinear, k, 
		data, size, nb_gens, 
		verbose_level - 2);
	if (f_v) {
		cout << "strong_generators::generators_for_parabolic_subgroup "
				"after GGD.generators_for_parabolic_subgroup" << endl;
	}

	my_gens = NEW_OBJECT(data_structures_groups::vector_ge);
	my_gens->init_from_data(
			A_PGL_n_q, data,
			nb_gens, size,
			0 /*verbose_level*/);

	if (f_v) {
		cout << "strong_generators::generators_for_parabolic_subgroup "
				"after generators_for_parabolic_subgroup" << endl;
	}

	ring_theory::longinteger_object go1, nCk, target_go;
	ring_theory::longinteger_domain D;
	combinatorics::combinatorics_domain C;


	D.group_order_PGL(go1, n, q, Mtx->f_semilinear);

	if (f_v) {
		cout << "strong_generators::generators_for_parabolic_subgroup "
			"go1=" << go1 << endl;
	}

	C.q_binomial_no_table(nCk, n, k, q, 0 /* verbose_level */);

	if (f_v) {
		cout << "strong_generators::generators_for_parabolic_subgroup "
			"nCk=" << nCk << endl;
	}

	D.integral_division_exact(go1, nCk, target_go);

	if (f_v) {
		cout << "strong_generators::generators_for_parabolic_subgroup "
				"before generators_to_strong_generators "
				"target_go=" << target_go << endl;
	}
	
	strong_generators *SG;

	A_PGL_n_q->generators_to_strong_generators(
		true /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::generators_for_parabolic_subgroup "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(SG, 0);


	FREE_OBJECT(SG);
	FREE_OBJECT(my_gens);
	FREE_int(data);

	if (f_v) {
		cout << "strong_generators::generators_for_parabolic_subgroup "
				"done" << endl;
		}
}

void
strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4(
		actions::action *A_PGL_4_q,
		algebra::matrix_group *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	field_theory::finite_field *F;
	int n, q;
	data_structures_groups::vector_ge *my_gens;
	int *data;
	int size;
	int nb_gens;
	number_theory::number_theory_domain NT;
	
	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4" << endl;
	}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (f_v) {
		cout << "n=" << n << " q=" << q << endl;
	}
	if (n != 4) {
		cout << "strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4 "
				"n != 4" << endl;
		exit(1);
	}


	algebra::group_generators_domain GGD;

	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4 "
				"before GGD.generators_for_stabilizer_of_three_collinear_points_in_PGL4" << endl;
	}
	GGD.generators_for_stabilizer_of_three_collinear_points_in_PGL4(
		Mtx->f_semilinear, F,
		data, size, nb_gens, 
		verbose_level - 1);
	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4 "
				"after GGD.generators_for_stabilizer_of_three_collinear_points_in_PGL4" << endl;
	}

	my_gens = NEW_OBJECT(data_structures_groups::vector_ge);

	my_gens->init_from_data(
			A_PGL_4_q, data,
			nb_gens, size,
			0 /*verbose_level*/);


	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4 "
				"after generators_for_stabilizer_of_three_collinear_points_in_PGL4" << endl;
	}

	ring_theory::longinteger_object target_go, a, b, c, d, e, f;
	ring_theory::longinteger_domain D;


	target_go.create(1);
	a.create((q - 1) * 6);
	b.create(q + 1);
	c.create(q);
	d.create(q - 1);
	e.create(NT.i_power_j(q, 4));
	D.mult_in_place(target_go, a);
	D.mult_in_place(target_go, b);
	D.mult_in_place(target_go, c);
	D.mult_in_place(target_go, d);
	D.mult_in_place(target_go, e);
	if (Mtx->f_semilinear) {
		f.create(Mtx->GFq->e);
		D.mult_in_place(target_go, f);
	}
	
	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4 "
				"before generators_to_strong_generators "
				"target_go=" << target_go << endl;
	}
	
	strong_generators *SG;

	A_PGL_4_q->generators_to_strong_generators(
		true /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4 "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(SG, 0);


	FREE_OBJECT(SG);
	FREE_OBJECT(my_gens);
	FREE_int(data);

	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4 "
				"done" << endl;
	}
}

void strong_generators::generators_for_stabilizer_of_triangle_in_PGL4(
		actions::action *A_PGL_4_q,
		algebra::matrix_group *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	field_theory::finite_field *F;
	int n, q;
	data_structures_groups::vector_ge *my_gens;
	int *data;
	int size;
	int nb_gens;
	number_theory::number_theory_domain NT;
	
	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_triangle_in_PGL4" << endl;
	}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (f_v) {
		cout << "n=" << n << " q=" << q << endl;
	}
	if (n != 4) {
		cout << "strong_generators::generators_for_stabilizer_of_triangle_in_PGL4 "
				"n != 4" << endl;
		exit(1);
	}


	algebra::group_generators_domain GGD;

	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_triangle_in_PGL4 "
				"before GGD.generators_for_stabilizer_of_triangle_in_PGL4" << endl;
	}
	GGD.generators_for_stabilizer_of_triangle_in_PGL4(
		Mtx->f_semilinear, F,
		data, size, nb_gens, 
		verbose_level - 1);
	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_triangle_in_PGL4 "
				"after GGD.generators_for_stabilizer_of_triangle_in_PGL4" << endl;
	}

	my_gens = NEW_OBJECT(data_structures_groups::vector_ge);

	my_gens->init_from_data(
			A_PGL_4_q, data,
			nb_gens, size,
			0 /*verbose_level*/);

	

	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_triangle_in_PGL4 "
				"after generators_for_stabilizer_of_triangle_in_PGL4" << endl;
	}

	ring_theory::longinteger_object target_go, a, b, c, f;
	ring_theory::longinteger_domain D;


	target_go.create(1);
	a.create(NT.i_power_j(q, 3));
	b.create(NT.i_power_j(q - 1, 3));
	c.create(6);
	D.mult_in_place(target_go, a);
	D.mult_in_place(target_go, b);
	D.mult_in_place(target_go, c);
	if (Mtx->f_semilinear) {
		f.create(Mtx->GFq->e);
		D.mult_in_place(target_go, f);
	}
	
	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_triangle_in_PGL4 "
				"before generators_to_strong_generators target_go=" << target_go << endl;
	}
	
	strong_generators *SG;

	A_PGL_4_q->generators_to_strong_generators(
		true /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_triangle_in_PGL4 "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(SG, 0);


	FREE_OBJECT(SG);
	FREE_OBJECT(my_gens);
	FREE_int(data);

	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_triangle_in_PGL4 "
				"done" << endl;
	}
}

void strong_generators::generators_for_the_orthogonal_group(
		actions::action *A,
	orthogonal_geometry::orthogonal *O,
	int f_semilinear, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::generators_for_the_orthogonal_group" << endl;
		cout << "f_semilinear=" << f_semilinear << endl;
	}


	actions::action *A2;

	A2 = NEW_OBJECT(actions::action);
	if (f_v) {
		cout << "strong_generators::generators_for_the_orthogonal_group "
				"before A2->init_orthogonal_group_with_O" << endl;
	}

	A2->Known_groups->init_orthogonal_group_with_O(
			O,
		true /* f_on_points */,
		false /* f_on_lines */,
		false /* f_on_points_and_lines */,
		f_semilinear, 
		true /* f_basis */, verbose_level - 1);

	if (f_v) {
		cout << "strong_generators::generators_for_the_orthogonal_group "
				"after A2->init_orthogonal_group_with_O" << endl;
	}

	ring_theory::longinteger_object target_go;
	strong_generators *Strong_gens2;

	A2->Sims->group_order(target_go);

	if (f_v) {
		cout << "strong_generators::generators_for_the_orthogonal_group "
				"before generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go, 
		&A2->Sims->gens, Strong_gens2, 
		0 /* verbose_level */);

	if (f_v) {
		cout << "strong_generators::generators_for_the_orthogonal_group "
				"after generators_to_strong_generators" << endl;
	}
	
	init_copy(Strong_gens2, 0 /* verbose_level */);

	//init_from_sims(A2->Sims, 0 /* verbose_level */);
	FREE_OBJECT(Strong_gens2);
	FREE_OBJECT(A2);

	if (f_v) {
		cout << "strong_generators::generators_for_the_orthogonal_group done" << endl;
	}
}

void strong_generators::stabilizer_of_cubic_surface_from_catalogue(
		actions::action *A,
	field_theory::finite_field *F, int iso,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_cubic_surface_from_catalogue" << endl;
		cout << "q=" << F->q << endl;
		cout << "iso=" << iso << endl;
	}

	strong_generators::A = A;

	int *data;
	int nb_gens;
	int data_size;
	string ascii_target_go;
	ring_theory::longinteger_object target_go;
	knowledge_base::knowledge_base K;
	
	if (f_v) {
		cout << "strong_generators::stabilizer_of_cubic_surface_from_catalogue before "
				"cubic_surface_stab_gens" << endl;
	}
	K.cubic_surface_stab_gens(
			F->q, iso,
			data, nb_gens, data_size, ascii_target_go);

	target_go.create_from_base_10_string(ascii_target_go);


	data_structures_groups::vector_ge *gens;

	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_cubic_surface_from_catalogue before "
				"gens->init_from_data" << endl;
	}
	gens->init_from_data(
			A, data,
			nb_gens, data_size,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "strong_generators::stabilizer_of_cubic_surface_from_catalogue after "
				"gens->init_from_data" << endl;
	}


#if 0
	strong_generators *Strong_gens2;

	if (f_v) {
		cout << "strong_generators::stabilizer_of_cubic_surface_from_catalogue before "
				"generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go, 
		gens, Strong_gens2, 
		0 /* verbose_level */);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_cubic_surface_from_catalogue after "
				"generators_to_strong_generators" << endl;
	}
	
	init_copy(Strong_gens2, 0 /* verbose_level */);

	FREE_OBJECT(Strong_gens2);
#else

	if (f_v) {
		cout << "strong_generators::stabilizer_of_cubic_surface_from_catalogue before "
				"init_reduced_generating_set" << endl;
	}
	init_reduced_generating_set(
			gens,
			target_go,
			verbose_level);
	if (f_v) {
		cout << "strong_generators::stabilizer_of_cubic_surface_from_catalogue after "
				"init_reduced_generating_set" << endl;
	}
#endif

	FREE_OBJECT(gens);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_cubic_surface_from_catalogue done" << endl;
	}
}

void strong_generators::init_reduced_generating_set(
		data_structures_groups::vector_ge *gens,
		ring_theory::longinteger_object &target_go,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::init_reduced_generating_set" << endl;
	}

	strong_generators *Strong_gens2;

	if (f_v) {
		cout << "strong_generators::init_reduced_generating_set "
				"before generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		gens, Strong_gens2,
		0 /* verbose_level */);

	if (f_v) {
		cout << "strong_generators::init_reduced_generating_set "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(Strong_gens2, 0 /* verbose_level */);

	FREE_OBJECT(Strong_gens2);

	if (f_v) {
		cout << "strong_generators::init_reduced_generating_set "
				"done" << endl;
	}
}

void strong_generators::stabilizer_of_quartic_curve_from_catalogue(
		actions::action *A,
	field_theory::finite_field *F, int iso,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_quartic_curve_from_catalogue" << endl;
		cout << "q=" << F->q << endl;
		cout << "iso=" << iso << endl;
	}


	strong_generators::A = A;

	int *data;
	int nb_gens;
	int data_size;
	string ascii_target_go;
	ring_theory::longinteger_object target_go;
	knowledge_base::knowledge_base K;

	if (f_v) {
		cout << "strong_generators::stabilizer_of_quartic_curve_from_catalogue "
				"before K.quartic_curves_stab_gens" << endl;
	}
	K.quartic_curves_stab_gens(
			F->q, iso,
			data, nb_gens, data_size, ascii_target_go);
	if (f_v) {
		cout << "strong_generators::stabilizer_of_quartic_curve_from_catalogue "
				"after K.quartic_curves_stab_gens" << endl;
	}
	if (f_v) {
		cout << "strong_generators::stabilizer_of_quartic_curve_from_catalogue "
				"ascii_target_go = " << ascii_target_go << endl;
	}

	target_go.create_from_base_10_string(ascii_target_go);
	if (f_v) {
		cout << "strong_generators::stabilizer_of_quartic_curve_from_catalogue "
				"target_go = " << target_go << endl;
	}

	if (f_v) {
		cout << "data:" << endl;
		Int_matrix_print(data, nb_gens, data_size);
	}

	data_structures_groups::vector_ge *gens;

	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_quartic_curve_from_catalogue "
				"before gens->init_from_data" << endl;
	}
	gens->init_from_data(
			A, data, nb_gens, data_size,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "strong_generators::stabilizer_of_quartic_curve_from_catalogue "
				"after gens->init_from_data" << endl;
	}



	strong_generators *Strong_gens2;

	if (f_v) {
		cout << "strong_generators::stabilizer_of_quartic_curve_from_catalogue "
				"before generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		gens, Strong_gens2,
		0 /* verbose_level */);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_quartic_curve_from_catalogue "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(Strong_gens2, 0 /* verbose_level */);

	FREE_OBJECT(Strong_gens2);
	FREE_OBJECT(gens);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_quartic_curve_from_catalogue "
				"done" << endl;
	}
}

void
strong_generators::stabilizer_of_Eckardt_surface(
		actions::action *A,
	field_theory::finite_field *F,
	int f_with_normalizer, int f_semilinear,
	data_structures_groups::vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_Eckardt_surface" << endl;
		cout << "q=" << F->q << endl;
		cout << "f_with_normalizer=" << f_with_normalizer << endl;
		cout << "f_semilinear=" << f_semilinear << endl;
	}

	int *data;
	int nb_gens;
	int data_size;
	int group_order;
	ring_theory::longinteger_object target_go;
	algebraic_geometry::algebraic_geometry_global AGG;
	
	if (f_v) {
		cout << "strong_generators::stabilizer_of_Eckardt_surface "
				"before AGG.cubic_surface_family_24_generators" << endl;
	}

	AGG.cubic_surface_family_24_generators(
			F,
			f_with_normalizer,
		f_semilinear, 
		data, nb_gens, data_size, group_order, verbose_level);


	nice_gens = NEW_OBJECT(data_structures_groups::vector_ge);

	nice_gens->init_from_data(
			A, data,
			nb_gens, data_size,
			0 /*verbose_level*/);

	target_go.create(group_order);



	strong_generators *Strong_gens2;

	if (f_v) {
		cout << "strong_generators::stabilizer_of_Eckardt_surface "
				"before generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go, 
		nice_gens, Strong_gens2,
		0 /* verbose_level */);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_Eckardt_surface "
				"after generators_to_strong_generators" << endl;
	}
	
	init_copy(Strong_gens2, 0 /* verbose_level */);

	FREE_int(data);
	FREE_OBJECT(Strong_gens2);
	//FREE_OBJECT(gens);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_Eckardt_surface "
				"done" << endl;
	}
}

void strong_generators::stabilizer_of_G13_surface(
		actions::action *A,
	field_theory::finite_field *F, int a,
	data_structures_groups::vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_G13_surface" << endl;
		cout << "q=" << F->q << endl;
		cout << "a=" << a << endl;
	}

	int *data;
	int nb_gens;
	int data_size;
	int group_order;
	ring_theory::longinteger_object target_go;
	algebraic_geometry::algebraic_geometry_global AGG;

	if (f_v) {
		cout << "strong_generators::stabilizer_of_G13_surface "
				"before AGG.cubic_surface_family_G13_generators" << endl;
	}

	AGG.cubic_surface_family_G13_generators(
			F,
			a,
		data, nb_gens, data_size, group_order, verbose_level);

	nice_gens = NEW_OBJECT(data_structures_groups::vector_ge);

	nice_gens->init_from_data(
			A, data,
			nb_gens, data_size,
			0 /*verbose_level*/);

	target_go.create(group_order);


	strong_generators *Strong_gens2;

	if (f_v) {
		cout << "strong_generators::stabilizer_of_G13_surface "
				"before generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		nice_gens, Strong_gens2,
		verbose_level);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_G13_surface "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(Strong_gens2, 0 /* verbose_level */);

	FREE_int(data);
	FREE_OBJECT(Strong_gens2);
	//FREE_OBJECT(gens);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_G13_surface "
				"done" << endl;
	}
}

void strong_generators::stabilizer_of_F13_surface(
		actions::action *A,
	field_theory::finite_field *F, int a,
	data_structures_groups::vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_F13_surface" << endl;
		cout << "q=" << F->q << endl;
		cout << "a=" << a << endl;
	}

	int *data;
	int nb_gens;
	int data_size;
	int group_order;
	ring_theory::longinteger_object target_go;
	algebraic_geometry::algebraic_geometry_global AGG;

	if (f_v) {
		cout << "strong_generators::stabilizer_of_F13_surface "
				"before AGG.cubic_surface_family_F13_generators" << endl;
	}

	AGG.cubic_surface_family_F13_generators(
			F, a,
		data, nb_gens, data_size, group_order, verbose_level);

	nice_gens = NEW_OBJECT(data_structures_groups::vector_ge);

	target_go.create(group_order);

	nice_gens->init_from_data(
			A, data,
			nb_gens, data_size,
			0 /*verbose_level*/);



	strong_generators *Strong_gens2;

	if (f_v) {
		cout << "strong_generators::stabilizer_of_F13_surface "
				"before generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		nice_gens, Strong_gens2,
		verbose_level);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_F13_surface "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(Strong_gens2, 0 /* verbose_level */);

	FREE_int(data);
	FREE_OBJECT(Strong_gens2);
	//FREE_OBJECT(gens);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_F13_surface "
				"done" << endl;
	}
}


void strong_generators::BLT_set_from_catalogue_stabilizer(
		actions::action *A,
	field_theory::finite_field *F, int iso,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::BLT_set_from_catalogue_stabilizer" << endl;
		cout << "q=" << F->q << endl;
		cout << "iso=" << iso << endl;
	}

	int *data;
	int nb_gens;
	int data_size;
	string ascii_target_go;
	ring_theory::longinteger_object target_go;
	knowledge_base::knowledge_base K;
	
	if (f_v) {
		cout << "strong_generators::BLT_set_from_catalogue_stabilizer "
				"before K.BLT_stab_gens" << endl;
	}
	K.BLT_stab_gens(F->q, iso, data, nb_gens, data_size, ascii_target_go);

	if (f_v) {
		cout << "strong_generators::BLT_set_from_catalogue_stabilizer "
				"data_size=" << data_size << endl;
		cout << "strong_generators::BLT_set_from_catalogue_stabilizer "
				"nb_gens=" << nb_gens << endl;
	}

	data_structures_groups::vector_ge *gens;

	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	gens->init_from_data(
			A, data,
			nb_gens, data_size,
			0 /*verbose_level*/);

	target_go.create_from_base_10_string(ascii_target_go);

	if (f_v) {
		cout << "strong_generators::BLT_set_from_catalogue_stabilizer "
				"generators are:" << endl;
		gens->print_quick(cout);
	}



	strong_generators *Strong_gens2;

	if (f_v) {
		cout << "strong_generators::BLT_set_from_catalogue_stabilizer "
				"before generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go, 
		gens, Strong_gens2, 
		verbose_level - 3);

	if (f_v) {
		cout << "strong_generators::BLT_set_from_catalogue_stabilizer "
				"after generators_to_strong_generators" << endl;
	}
	
	init_copy(Strong_gens2, 0 /* verbose_level */);

	FREE_OBJECT(Strong_gens2);
	FREE_OBJECT(gens);

	if (f_v) {
		cout << "strong_generators::BLT_set_from_catalogue_stabilizer "
				"done" << endl;
	}
}

void strong_generators::stabilizer_of_spread_from_catalogue(
		actions::action *A,
	int q, int k, int iso, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_spread_from_catalogue" << endl;
		cout << "q=" << q << endl;
		cout << "k=" << k << endl;
		cout << "iso=" << iso << endl;
	}

	int *data;
	int nb_gens;
	int data_size;
	string ascii_target_go;
	ring_theory::longinteger_object target_go;
	knowledge_base::knowledge_base K;
	
	if (f_v) {
		cout << "strong_generators::stabilizer_of_spread_from_catalogue "
				"before K.Spread_stab_gens" << endl;
	}
	K.Spread_stab_gens(
			q, k, iso, data, nb_gens, data_size, ascii_target_go);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_spread_from_catalogue "
				"data_size=" << data_size << endl;
		cout << "strong_generators::stabilizer_of_spread_from_catalogue "
				"nb_gens=" << nb_gens << endl;
	}

	data_structures_groups::vector_ge *gens;

	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	gens->init_from_data(
			A, data,
			nb_gens, data_size,
			0 /*verbose_level*/);

	target_go.create_from_base_10_string(ascii_target_go);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_spread_from_catalogue "
				"generators are:" << endl;
		gens->print_quick(cout);
	}



	strong_generators *Strong_gens2;

	if (f_v) {
		cout << "strong_generators::stabilizer_of_spread_from_catalogue "
				"before generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go, 
		gens, Strong_gens2, 
		0 /* verbose_level */);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_spread_from_catalogue "
				"after generators_to_strong_generators" << endl;
	}
	
	init_copy(Strong_gens2, 0 /* verbose_level */);

	FREE_OBJECT(Strong_gens2);
	FREE_OBJECT(gens);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_spread_from_catalogue "
				"done" << endl;
	}
}


void strong_generators::stabilizer_of_pencil_of_conics(
		actions::action *A,
	field_theory::finite_field *F,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_pencil_of_conics" << endl;
		cout << "q=" << F->q << endl;
	}

	ring_theory::longinteger_object target_go;
	number_theory::number_theory_domain NT;
	int i;

	if (f_v) {
		cout << "strong_generators::stabilizer_of_pencil_of_conics "
				"creating generators" << endl;
	}


	data_structures_groups::vector_ge *gens;

	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init(A, verbose_level - 2);
	target_go.create(NT.i_power_j(F->q - 1, 2));

	int *data;
	int nb_gens = 2;
	int data_size = 10; // 9 + 1
	int alpha;

	alpha = F->primitive_root();
	data = NEW_int(data_size);


	gens->allocate(nb_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		Int_vec_zero(data, data_size);
		if (i == 0) {
			// diag(t, 1/t, 1)
			data[0] = alpha;
			data[4] = F->inverse(alpha);
			data[8] = 1;
		}
		else {
			// diag(t, 1, 1)
			data[0] = alpha;
			data[4] = 1;
			data[8] = 1;
		}
		A->Group_element->make_element(gens->ith(i), data, 0);
		}

	if (f_v) {
		cout << "strong_generators::stabilizer_of_pencil_of_conics "
				"generators are:" << endl;
		gens->print_quick(cout);
	}



	strong_generators *Strong_gens2;

	if (f_v) {
		cout << "strong_generators::stabilizer_of_pencil_of_conics "
				"before generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		gens, Strong_gens2,
		0 /* verbose_level */);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_pencil_of_conics "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(Strong_gens2, 0 /* verbose_level */);

	FREE_OBJECT(Strong_gens2);
	FREE_OBJECT(gens);

	if (f_v) {
		cout << "strong_generators::stabilizer_of_pencil_of_conics "
				"done" << endl;
	}
}

void strong_generators::Janko1(
		actions::action *A,
	field_theory::finite_field *F,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::Janko1" << endl;
		cout << "q=" << F->q << endl;
	}

	if (F->q != 11) {
		cout << "strong_generators::Janko1 q != 11" << endl;
		exit(1);
	}
	algebra::matrix_group *M;

	M = A->get_matrix_group();
	if (M->n != 7) {
		cout << "strong_generators::Janko1 "
				"dimension != 7" << endl;
		exit(1);
	}
	ring_theory::longinteger_object target_go;
	number_theory::number_theory_domain NT;
	int i;



	data_structures_groups::vector_ge *gens;

	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init(A, verbose_level - 2);
	target_go.create(
			11 * (11 * 11 * 11 - 1) * (11 + 1));

	int *data;
	int nb_gens = 2;
	int data_size = 50; // 7 * 7 + 1
	//int alpha;
	int j;

	//alpha = F->primitive_root();
	data = NEW_int(data_size);


	gens->allocate(nb_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		Int_vec_zero(data, data_size);
		if (i == 0) {
			for (j = 0; j < 7; j++) {
				if (j < 7 - 1) {
					data[j * 7 + j + 1] = 1;
				}
				else {
					data[j * 7 + 0] = 1;
				}
			}
		}
		else {
			int data2[] = {
					-3, 2, -1, -1, -3, -1, -3,
					-2, 1, 1, 3, 1, 3, 3,
					-1, -1, -3, -1, -3, -3, 2,
					-1, -3, -1, -3, -3, 2, -1,
					-3, -1, -3, -3, 2, -1, -1,
					1, 3, 3, -2, 1, 1, 3,
					3, 3, -2, 1, 1, 3, 1
			};
			for (j = 0; j < 49; j++) {
				data[j] = (data2[j] + 11) % 11;
			}
		}
		A->Group_element->make_element(gens->ith(i), data, 0);
	}

	if (f_v) {
		cout << "strong_generators::Janko1 "
				"generators are:" << endl;
		gens->print_quick(cout);
	}



	strong_generators *Strong_gens2;

	if (f_v) {
		cout << "strong_generators::Janko1 "
				"before generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		gens, Strong_gens2,
		0 /* verbose_level */);

	if (f_v) {
		cout << "strong_generators::Janko1 "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(Strong_gens2, 0 /* verbose_level */);

	FREE_OBJECT(Strong_gens2);
	FREE_OBJECT(gens);

	if (f_v) {
		cout << "strong_generators::Janko1 done" << endl;
	}
}

void strong_generators::Hall_reflection(
	int nb_pairs, int &degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_perms;
	int *perms;
	data_structures_groups::vector_ge *gens;
	algebra::group_generators_domain GG;

	if (f_v) {
		cout << "strong_generators::Hall_reflection" << endl;
	}


	if (f_v) {
		cout << "strong_generators::Hall_reflection "
				"before generators_Hall_reflection" << endl;
	}

	GG.generators_Hall_reflection(
			nb_pairs,
			nb_perms, perms, degree,
			verbose_level);




	if (f_v) {
		cout << "strong_generators::Hall_reflection "
				"after generators_Hall_reflection" << endl;
	}


	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init_from_data(
			A, perms,
			nb_perms, degree, 0 /*verbose_level*/);

#if 0
	gens->init(A, verbose_level - 2);


	int i;

	gens->allocate(nb_perms, verbose_level - 2);
	for (i = 0; i < nb_perms; i++) {
		A->Group_element->make_element(gens->ith(i), perms + i * degree, 0);
	}
#endif

	if (f_v) {
		cout << "strong_generators::Hall_reflection "
				"generators are:" << endl;
		gens->print_quick(cout);
	}



	ring_theory::longinteger_object target_go;


	target_go.create(2);


	if (f_v) {
		cout << "strong_generators::Hall_reflection "
				"target_go=" << target_go << endl;
	}

	if (f_v) {
		cout << "strong_generators::Hall_reflection "
				"before A->init_permutation_group" << endl;
	}
	A = NEW_OBJECT(actions::action);
	int f_no_base = false;

	A->Known_groups->init_permutation_group(
			degree, f_no_base, verbose_level);

	strong_generators *SG;

	if (f_v) {
		cout << "strong_generators::Hall_reflection "
				"before generators_to_strong_generators" << endl;
	}

	A->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		gens, SG, verbose_level - 3);

	if (f_v) {
		cout << "strong_generators::Hall_reflection "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(SG, 0);


	FREE_OBJECT(SG);
	FREE_OBJECT(gens);

	if (f_v) {
		cout << "strong_generators::Hall_reflection done" << endl;
	}
}

void strong_generators::normalizer_of_a_Hall_reflection(
	int nb_pairs, int &degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_perms;
	int *perms;
	data_structures_groups::vector_ge *gens;
	algebra::group_generators_domain GG;

	if (f_v) {
		cout << "strong_generators::normalizer_of_a_Hall_reflection" << endl;
	}


	if (f_v) {
		cout << "strong_generators::normalizer_of_a_Hall_reflection "
				"before generators_Hall_reflection_normalizer_group" << endl;
	}

	GG.generators_Hall_reflection_normalizer_group(
			nb_pairs,
			nb_perms, perms, degree,
			verbose_level);




	if (f_v) {
		cout << "strong_generators::normalizer_of_a_Hall_reflection "
				"after generators_Hall_reflection_normalizer_group" << endl;
	}


	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init_from_data(A, perms,
			nb_perms, degree, 0 /*verbose_level*/);

#if 0
	gens->init(A, verbose_level - 2);

	int i;

	gens->allocate(nb_perms, verbose_level - 2);
	for (i = 0; i < nb_perms; i++) {
		A->Group_element->make_element(gens->ith(i), perms + i * degree, 0);
	}
#endif

	if (f_v) {
		cout << "strong_generators::normalizer_of_a_Hall_reflection "
				"generators are:" << endl;
		gens->print_quick(cout);
	}



	int *factors;
	int nb_factors;
	ring_theory::longinteger_object target_go;


	GG.order_Hall_reflection_normalizer_factorized(
			nb_pairs,
			factors, nb_factors);

	target_go.create_product(nb_factors, factors);
	FREE_int(factors);


	if (f_v) {
		cout << "strong_generators::normalizer_of_a_Hall_reflection "
				"target_go=" << target_go << endl;
	}

	if (f_v) {
		cout << "strong_generators::normalizer_of_a_Hall_reflection "
				"before A->init_permutation_group" << endl;
	}
	A = NEW_OBJECT(actions::action);

	A->Known_groups->init_symmetric_group(
			degree, verbose_level);
	//A->init_permutation_group(degree, verbose_level);

	strong_generators *SG;

	if (f_v) {
		cout << "strong_generators::normalizer_of_a_Hall_reflection "
				"before generators_to_strong_generators" << endl;
	}

	A->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		gens, SG, verbose_level - 3);

	if (f_v) {
		cout << "strong_generators::normalizer_of_a_Hall_reflection "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(SG, 0);


	FREE_OBJECT(SG);
	FREE_OBJECT(gens);

	if (f_v) {
		cout << "strong_generators::normalizer_of_a_Hall_reflection "
				"done" << endl;
	}
}

void strong_generators::hyperplane_lifting_with_two_lines_fixed(
	strong_generators *SG_hyperplane,
	geometry::projective_space *P, int line1, int line2,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::vector_ge *gens;
	int A4[17]; // one more in case of semilinear maps

	if (f_v) {
		cout << "strong_generators::hyperplane_lifting_with_two_lines_fixed" << endl;
	}



	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init(A, verbose_level - 2);

	int i;
	int f_semilinear = false;
	int frobenius = 0;
	algebraic_geometry::algebraic_geometry_global Gg;

	f_semilinear = A->is_semilinear_matrix_group();
	if (f_v) {
		cout << "strong_generators::hyperplane_lifting_with_two_lines_fixed "
				"f_semilinear = " << f_semilinear << endl;
		cout << "generators SG_hyperplane:" << endl;
		SG_hyperplane->print_generators(cout, verbose_level - 1);
	}

	gens->allocate(SG_hyperplane->gens->len, verbose_level - 2);
	for (i = 0; i < SG_hyperplane->gens->len; i++) {
		if (f_v) {
			cout << "strong_generators::hyperplane_lifting_with_two_lines_fixed "
					"lifting generator "
					<< i << " / " << SG_hyperplane->gens->len << endl;
		}
		frobenius = SG_hyperplane->gens->ith(i)[9];
		if (f_v) {
			if (f_semilinear) {
				cout << "strong_generators::hyperplane_lifting_with_two_lines_fixed "
						" frobenius = " << frobenius << endl;
			}
		}

		if (f_v) {
			cout << "strong_generators::hyperplane_lifting_with_two_lines_fixed "
					"lifting generator "
					<< i << " / " << SG_hyperplane->gens->len
					<< " before Gg.hyperplane_lifting_with_two_lines_fixed" << endl;
		}
		Gg.hyperplane_lifting_with_two_lines_fixed(
				P,
				SG_hyperplane->gens->ith(i),
				f_semilinear, frobenius,
				line1, line2,
				A4,
				verbose_level);
		// in case of semilinear maps,
		// A4[16] is set in lifted_action_on_hyperplane_W0_fixing_two_lines

		if (f_v) {
			cout << "strong_generators::hyperplane_lifting_with_two_lines_fixed "
					"lifting generator "
					<< i << " / " << SG_hyperplane->gens->len
					<< " after Gg.hyperplane_lifting_with_two_lines_fixed" << endl;
		}
		A->Group_element->make_element(gens->ith(i), A4, 0);
		if (f_v) {
			cout << "strong_generators::hyperplane_lifting_with_two_lines_fixed "
					"generator "
					<< i << " / " << SG_hyperplane->gens->len
					<< " lifts to " << endl;
			A->Group_element->element_print(gens->ith(i), cout);
		}
	}

	if (f_v) {
		cout << "strong_generators::hyperplane_lifting_with_two_lines_fixed "
				"generators are:" << endl;
		gens->print(cout);
	}



	ring_theory::longinteger_object target_go;


	SG_hyperplane->group_order(target_go);


	if (f_v) {
		cout << "strong_generators::hyperplane_lifting_with_two_lines_fixed "
				"target_go=" << target_go << endl;
	}


	if (f_v) {
		cout << "strong_generators::hyperplane_lifting_with_two_lines_fixed "
				"before generators_to_strong_generators" << endl;
	}

	strong_generators *SG;

	A->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		gens, SG, verbose_level - 3);

	if (f_v) {
		cout << "strong_generators::hyperplane_lifting_with_two_lines_fixed "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(SG, 0);


	FREE_OBJECT(SG);
	FREE_OBJECT(gens);

	if (f_v) {
		cout << "strong_generators::hyperplane_lifting_with_two_lines_fixed done" << endl;
	}
}

void strong_generators::exterior_square(
		actions::action *A_detached,
		strong_generators *SG_original,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::vector_ge *gens;

	if (f_v) {
		cout << "strong_generators::exterior_square" << endl;
	}



	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init(A_detached, verbose_level - 2);

	int i, n, n2;
	int f_semilinear = false;
	int frobenius = 0;
	int *An2;
	field_theory::finite_field *F;

	n = SG_original->A->matrix_group_dimension();
	F = A->matrix_group_finite_field();
	if (f_v) {
		cout << "strong_generators::exterior_square "
				"n = " << n << endl;
	}

	combinatorics::combinatorics_domain Combi;

	n2 = Combi.binomial2(n);

	An2 = NEW_int(n2 * n2 + 1); // in case of semilinear

	f_semilinear = SG_original->A->is_semilinear_matrix_group();
	if (f_v) {
		cout << "strong_generators::exterior_square "
				"f_semilinear = " << f_semilinear << endl;
		cout << "generators SG_original:" << endl;
		SG_original->print_generators(cout, verbose_level - 1);
	}

	gens->allocate(SG_original->gens->len, verbose_level - 2);
	for (i = 0; i < SG_original->gens->len; i++) {
		if (f_v) {
			cout << "strong_generators::exterior_square "
					"lifting generator "
					<< i << " / " << SG_original->gens->len << endl;
		}
		if (f_semilinear) {
			frobenius = SG_original->gens->ith(i)[n * n];
			if (f_v) {
				if (f_semilinear) {
					cout << "strong_generators::exterior_square "
							" frobenius = " << frobenius << endl;
				}
			}
		}
		else {
			frobenius = 0;
		}

		if (f_v) {
			cout << "strong_generators::exterior_square "
					"lifting generator "
					<< i << " / " << SG_original->gens->len
					<< " before P->exterior_square" << endl;
		}

		Int_vec_zero(An2, n2 * n2 + 1);
		F->Linear_algebra->exterior_square(
				SG_original->gens->ith(i), An2, n,
				verbose_level - 2);

		An2[n2 * n2] = frobenius;

		if (f_v) {
			cout << "strong_generators::exterior_square "
					"lifting generator "
					<< i << " / " << SG_original->gens->len
					<< " after P->exterior_square" << endl;
		}
		A_detached->Group_element->make_element(
				gens->ith(i), An2, 0);
		if (f_v) {
			cout << "strong_generators::exterior_square "
					"generator "
					<< i << " / " << SG_original->gens->len
					<< " lifts to " << endl;
			A_detached->Group_element->element_print(gens->ith(i), cout);
		}
	}

	if (f_v) {
		cout << "strong_generators::exterior_square "
				"generators are:" << endl;
		gens->print(cout);
	}

	gens->copy(nice_gens, verbose_level);


	ring_theory::longinteger_object target_go;


	SG_original->group_order(target_go);


	if (f_v) {
		cout << "strong_generators::exterior_square "
				"target_go=" << target_go << endl;
	}


	if (f_v) {
		cout << "strong_generators::exterior_square "
				"before generators_to_strong_generators" << endl;
	}

	strong_generators *SG;

	A->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		gens, SG, verbose_level - 3);

	if (f_v) {
		cout << "strong_generators::exterior_square "
				"after generators_to_strong_generators" << endl;
	}

	init_copy(SG, 0);


	FREE_OBJECT(SG);
	FREE_OBJECT(gens);

	if (f_v) {
		cout << "strong_generators::exterior_square done" << endl;
	}
}


void strong_generators::diagonally_repeat(
		actions::action *An,
		strong_generators *Sn,
		int verbose_level)
// Embeds all generators from Sk in GL(k,q) into GL(n,k)
// by repeating each matrix A twice on the diagonal
// to form
// diag(A,A).
// The new group is isomorphic to the old one,
// but has twice the dimension.
// This function is used in upstep
// to compute the stabilizer of the flag
// from the original generators of the centralizer.
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::vector_ge *new_gens;
	int h, l, i, j, a, n, k;
	int *M;
	int *Elt;
	ring_theory::longinteger_object go;
	sims *Sims;


	if (f_v) {
		cout << "strong_generators::diagonally_repeat" << endl;
	}
	k = A->matrix_group_dimension();
	n = An->matrix_group_dimension();
	M = NEW_int(n * n);
	new_gens = NEW_OBJECT(data_structures_groups::vector_ge);

	new_gens->init(An, verbose_level - 2);
	l = gens->len;
	if (f_v) {
		cout << "strong_generators::diagonally_repeat "
				"l=" << l << endl;
	}
	new_gens->allocate(l, verbose_level - 2);
	for (h = 0; h < l; h++) {
		Elt = gens->ith(h);
		Int_vec_zero(M, n * n);
		for (i = 0; i < k; i++) {
			for (j = 0; j < k; j++) {
				a = Elt[i * k + j];
				M[i * n + j] = a;
				M[(k + i) * n + k + j] = a;
			}
		}
		An->Group_element->make_element(new_gens->ith(h), M, 0);
	}
	group_order(go);

	FREE_int(M);

	if (f_v) {
		cout << "strong_generators::diagonally_repeat "
				"before A->create_sims_from_generators_"
				"with_target_group_order" << endl;
	}

	Sims = An->create_sims_from_generators_with_target_group_order(
			new_gens, go, verbose_level);
	if (f_v) {
		cout << "strong_generators::diagonally_repeat "
				"after A->create_sims_from_generators_"
				"with_target_group_order" << endl;
	}

	if (f_v) {
		cout << "strong_generators::diagonally_repeat "
				"before Sn->init_from_sims" << endl;
	}
	Sn->init_from_sims(Sims, verbose_level - 2);
	if (f_v) {
		cout << "strong_generators::diagonally_repeat "
				"after Sn->init_from_sims" << endl;
	}

	FREE_OBJECT(new_gens);
	FREE_OBJECT(Sims);

	if (f_v) {
		cout << "The old stabilizer has order " << go << endl;
	}
	if (f_v) {
		cout << "strong_generators::diagonally_repeat end" << endl;
	}
}



}}}

