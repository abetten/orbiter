// strong_generators_groups.C
//
// Anton Betten

// started: December 4, 2013
// moved here: Dec 21, 2015


#include "galois.h"
#include "action.h"

void strong_generators::init_linear_group_from_scratch(action *&A, 
	finite_field *F, INT n, 
	INT f_projective, INT f_general, INT f_affine, 
	INT f_semilinear, INT f_special, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "strong_generators::init_linear_group_from_scratch" << endl;
		}


	A = new action;
	strong_generators::A = A;

	INT f_basis = TRUE;
	
	if (f_projective) {
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch projective group" << endl;
			}
		A->init_projective_group(n, F, f_semilinear, 
			f_basis, verbose_level);
		}
	else if (f_general) {
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch general linear group" << endl;
			}
		A->init_general_linear_group(n, F, f_semilinear, 
			f_basis, verbose_level);
		}
	else if (f_affine) {
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch affine group" << endl;
			}
		A->init_affine_group(n, F, f_semilinear, 
			f_basis, verbose_level);
		}
	else {
		cout << "strong_generators::init_linear_group_from_scratch the type of group is not specified" << endl;
		exit(1);
		}


	if (!A->f_has_strong_generators) {
		cout << "strong_generators::init_linear_group_from_scratch fatal: !A->f_has_strong_generators" << endl;
		}

	if (f_special) {


		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch special linear group" << endl;
			}

		special_subgroup(verbose_level);
		
		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch special linear group done" << endl;
			}
		}
	else {

		if (f_v) {
			cout << "strong_generators::init_linear_group_from_scratch creating sims and collecting generators" << endl;
			}
		sims *S;
		S = A->Strong_gens->create_sims(0 /* verbose_level */);
		init_from_sims(S, verbose_level);
		delete S;
		}
	if (f_v) {
		cout << "strong_generators::init_linear_group_from_scratch strong generators have been created" << endl;
		}
	if (f_vv) {
		print_generators();
		print_generators_tex();
		}


	if (f_v) {
		cout << "strong_generators::init_linear_group_from_scratch done" << endl;
		}
}

void strong_generators::special_subgroup(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	action A_on_det;
	longinteger_object go;
		
	if (f_v) {
		cout << "strong_generators::special_subgroup setting up action on determinant" << endl;
		}
	A_on_det.induced_action_on_determinant(A->Sims, verbose_level);
	if (f_v) {
		cout << "strong_generators::special_subgroup induced_action_on_determinant finished" << endl;
		}
	A_on_det.Kernel->group_order(go);
	if (f_v) {
		cout << "strong_generators::special_subgroup kernel has order " << go << endl;
		}


	init_from_sims(A_on_det.Kernel, verbose_level);
	
	if (f_v) {
		cout << "strong_generators::special_subgroup special linear group done" << endl;
		}
}

void strong_generators::even_subgroup(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	action A_on_sign;
	longinteger_object go;
		
	if (f_v) {
		cout << "strong_generators::even_subgroup setting up action on sign" << endl;
		}
	A_on_sign.induced_action_on_sign(A->Sims, verbose_level);
	if (f_v) {
		cout << "strong_generators::even_subgroup induced_action_on_sign finished" << endl;
		}
	A_on_sign.Kernel->group_order(go);
	if (f_v) {
		cout << "strong_generators::even_subgroup kernel has order " << go << endl;
		}


	init_from_sims(A_on_sign.Kernel, verbose_level);
	
	if (f_v) {
		cout << "strong_generators::even_subgroup special linear group done" << endl;
		}
}

void strong_generators::init_single(action *A, INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	sims *S;

	if (f_v) {
		cout << "strong_generators::init_single" << endl;
		}
	S = create_sims_from_single_generator_without_target_group_order(A, 
		Elt, verbose_level);
	init_from_sims(S, verbose_level);
	delete S;

	if (f_v) {
		cout << "strong_generators::init_single done" << endl;
		}
}

void strong_generators::init_trivial_group(action *A, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i;

	if (f_v) {
		cout << "strong_generators::init_trivial_group" << endl;
		}
	strong_generators::A = A;
	tl = NEW_INT(A->base_len);
	for (i = 0; i < A->base_len; i++) {
		tl[i] = 1;
		}
	gens = new vector_ge;
	gens->init(A);
	gens->allocate(0);
	//S->extract_strong_generators_in_order(*gens, tl, 0 /*verbose_level*/);
	if (f_v) {
		cout << "strong_generators::init_trivial_group done" << endl;
		}
}

void strong_generators::generators_for_the_monomial_group(action *A, 
	matrix_group *Mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *Elt1;
	sims *S;
	finite_field *F;
	longinteger_domain D;
	longinteger_object target_go;
	INT *go_factored;
	INT n, q, pos_frobenius;
	vector_ge *my_gens;
	INT *data;
	INT i, h, hh, h1, j, a, b, nb_gens;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_monomial_group initializing monomial group" << endl;
		}
	strong_generators::A = A;
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (f_v) {
		cout << "n=" << n << " q=" << q << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	go_factored = NEW_INT(3 * n + 1);
	data = NEW_INT(n * n + n + 1);

	pos_frobenius = 0;
	if (Mtx->f_projective) {
		cout << "strong_generators::generators_for_the_monomial_group  type is projective" << endl;
		pos_frobenius = n * n;
		}

	if (Mtx->f_affine) {
		cout << "strong_generators::generators_for_the_monomial_group  type is affine" << endl;
		pos_frobenius = n * n + n;
		//exit(1);
		}

	if (Mtx->f_general_linear) {
		cout << "strong_generators::generators_for_the_monomial_group  type is general_linear" << endl;
		pos_frobenius = n * n;
		}


	// group order 
	// = n! * (q - 1)^(n-1) * e if projective
	// = n! * (q - 1)^n * e if general linear
	// = n! * (q - 1)^n * q^n * e if affine
	// where e is the degree of the field if f_semilinear is TRUE
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
	D.multiply_up(target_go, go_factored, 3 * n + 1);
	if (f_v) {
		cout << "group order factored: ";
		INT_vec_print(cout, go_factored, 3 * n + 1);
		cout << endl;
		cout << "target_go=" << target_go << endl;
		}
	my_gens = new vector_ge;
	my_gens->init(A);
	nb_gens = n - 1 + 1 + 1;
	if (Mtx->f_affine) {
		nb_gens += n * F->e;
		}
	my_gens->allocate(nb_gens);
	for (h = 0; h < nb_gens; h++) {

		cout << "strong_generators::generators_for_the_monomial_group generator " << h << " / " << nb_gens << ":" << endl;

		F->identity_matrix(data, n);
		if (Mtx->f_affine) {
			INT_vec_zero(data + n * n, n);
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
			data[n * n + a] = i_power_j(F->p, b); // elements of a field basis of F_q over F_p
			}
		A->make_element(Elt1, data, 0 /*verbose_level - 1*/);
		if (f_vv) {
			cout << "generator " << h << ":" << endl;
			A->element_print_quick(Elt1, cout);
			}
		my_gens->copy_in(h, Elt1);
		}
	if (f_v) {
		cout << "strong_generators::generators_for_the_monomial_group creating group" << endl;
		}
	S = create_sims_from_generators_randomized(A, 
		my_gens, TRUE /* f_target_go */, 
		target_go, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "strong_generators::generators_for_the_monomial_group after creating group" << endl;
		}
	init_from_sims(S, 0);
	if (f_v) {
		cout << "strong_generators::generators_for_the_monomial_group after extracting strong generators" << endl;
		}
	if (f_vv) {
		INT f_print_as_permutation = FALSE;
		INT f_offset = FALSE;
		INT offset = 0;
		INT f_do_it_anyway_even_for_big_degree = FALSE;
		INT f_print_cycles_of_length_one = FALSE;
		
		longinteger_object go;
	
		cout << "computing the group order:" << endl;
		group_order(go);
		cout << "The group order is " << go << endl;
		
		cout << "strong generators are:" << endl;
		gens->print(cout, f_print_as_permutation, 
			f_offset, offset, f_do_it_anyway_even_for_big_degree, 
			f_print_cycles_of_length_one);
		}
	delete S;
	delete my_gens;
	FREE_INT(data);
	FREE_INT(go_factored);
	FREE_INT(Elt1);
	if (f_v) {
		cout << "strong_generators::generators_for_the_monomial_group done" << endl;
		}
}

void strong_generators::generators_for_the_diagonal_group(action *A, 
	matrix_group *Mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *Elt1;
	sims *S;
	finite_field *F;
	longinteger_domain D;
	longinteger_object target_go;
	INT *go_factored;
	INT n, q;
	vector_ge *my_gens;
	INT *data;
	INT i, h;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_diagonal_group initializing diagonal group" << endl;
		}
	strong_generators::A = A;
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (f_v) {
		cout << "n=" << n << " q=" << q << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	go_factored = NEW_INT(n + 1);
	data = NEW_INT(n * n + 1);

	// group order 
	// = q^n * e if not projective
	// = q^(n-1) * e if projective
	// where e is the degree of the field if f_semilinear is TRUE
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
		cout << "strong_generators::generators_for_the_diagonal_group  type is projective" << endl;
		}

	if (Mtx->f_affine) {
		cout << "strong_generators::generators_for_the_diagonal_group  type should not be affine" << endl;
		exit(1);
		}

	if (Mtx->f_general_linear) {
		cout << "strong_generators::generators_for_the_diagonal_group  type is general_linear" << endl;
		}

	if (Mtx->f_semilinear) {
		go_factored[n] = F->e;
		}
	else {
		go_factored[n] = 1;
		}
	D.multiply_up(target_go, go_factored, n + 1);
	if (f_v) {
		cout << "group order factored: ";
		INT_vec_print(cout, go_factored, n + 1);
		cout << endl;
		cout << "target_go=" << target_go << endl;
		}
	my_gens = new vector_ge;
	my_gens->init(A);
	my_gens->allocate(n + 1);
	for (h = 0; h < n + 1; h++) {

		F->identity_matrix(data, n);

		if (h < n) {
			data[h * n + h] = F->alpha_power(1);
			}
		else if (h == n) {
			if (Mtx->f_semilinear) {
				data[n * n] = 1;
				}
			}
		A->make_element(Elt1, data, 0 /*verbose_level - 1*/);
		if (f_vv) {
			cout << "generator " << h << ":" << endl;
			A->element_print_quick(Elt1, cout);
			}
		my_gens->copy_in(h, Elt1);
		}
	if (f_v) {
		cout << "strong_generators::generators_for_the_diagonal_group creating group" << endl;
		}
	S = create_sims_from_generators_randomized(A, 
		my_gens, TRUE /* f_target_go */, 
		target_go, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "strong_generators::generators_for_the_diagonal_group after creating group" << endl;
		}
	init_from_sims(S, 0);
	if (f_v) {
		cout << "strong_generators::generators_for_the_diagonal_group after extracting strong generators" << endl;
		}
	if (f_vv) {
		INT f_print_as_permutation = FALSE;
		INT f_offset = FALSE;
		INT offset = 0;
		INT f_do_it_anyway_even_for_big_degree = FALSE;
		INT f_print_cycles_of_length_one = FALSE;
		
		longinteger_object go;
	
		cout << "computing the group order:" << endl;
		group_order(go);
		cout << "The group order is " << go << endl;
		
		cout << "strong generators are:" << endl;
		gens->print(cout, f_print_as_permutation, 
			f_offset, offset, f_do_it_anyway_even_for_big_degree, 
			f_print_cycles_of_length_one);
		}
	delete S;
	delete my_gens;
	FREE_INT(data);
	FREE_INT(go_factored);
	FREE_INT(Elt1);
	if (f_v) {
		cout << "strong_generators::generators_for_the_diagonal_group done" << endl;
		}
}

void strong_generators::generators_for_the_singer_cycle(action *A, 
	matrix_group *Mtx, INT power_of_singer, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *Elt1;
	sims *S;
	finite_field *F;
	longinteger_domain D;
	longinteger_object target_go;
	INT *go_factored;
	INT n, q;
	vector_ge *my_gens;
	INT *data;
	INT i;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_singer_cycle initializing singer group power_of_singer=" << power_of_singer << endl;
		}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (f_v) {
		cout << "n=" << n << " q=" << q << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	go_factored = NEW_INT(1);
	data = NEW_INT(n * n + 1);

	// group order 
	// = (q^n - 1) / (q - 1) if projective
	// = q^n - 1 if general_linear
	
	go_factored[0] = nb_PG_elements(n - 1, q);
	INT g;
	g = gcd_INT(go_factored[0], power_of_singer);
	go_factored[0] = go_factored[0] / g;

	D.multiply_up(target_go, go_factored, 1);
	if (f_v) {
		cout << "group order factored: ";
		INT_vec_print(cout, go_factored, 1);
		cout << endl;
		cout << "target_go=" << target_go << endl;
		}
	my_gens = new vector_ge;
	my_gens->init(A);
	my_gens->allocate(1);

	

	{
	finite_field Fp;
	
	if (!is_prime(q)) {
		cout << "strong_generators::generators_for_the_singer_cycle field order must be a prime" << endl;
		exit(1);
		}

	Fp.init(q, 0 /*verbose_level*/);
	unipoly_domain FX(&Fp);
	
	unipoly_object m;
	longinteger_object rk;
	
	FX.create_object_by_rank(m, 0);
	
	if (f_v) {
		cout << "search_for_primitive_polynomial_of_given_degree p=" << q << " degree=" << n << endl;
		}
	FX.get_a_primitive_polynomial(m, n, verbose_level - 1);

	INT_vec_zero(data, n * n);

	// create upper diagonal:
	for (i = 0; i < n - 1; i++) {
		data[i * n + i + 1] = 1; 
		}

	INT a, b;
	
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

	
	A->make_element(Elt1, data, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "generator :" << endl;
		A->element_print_quick(Elt1, cout);
		}

	A->element_power_INT_in_place(Elt1, 
		power_of_singer, 0 /* verbose_level */);

	if (f_v) {
		cout << "generator after raising to the power of " << power_of_singer << ":" << endl;
		A->element_print_quick(Elt1, cout);
		}
	my_gens->copy_in(0, Elt1);


	if (f_v) {
		cout << "strong_generators::generators_for_the_singer_cycle creating group" << endl;
		}
	if (f_v) {
		cout << "group order factored: ";
		INT_vec_print(cout, go_factored, 1);
		cout << endl;
		cout << "target_go=" << target_go << endl;
		}
	S = create_sims_from_generators_randomized(A, 
		my_gens, TRUE /* f_target_go */, 
		target_go, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "strong_generators::generators_for_the_singer_cycle after creating group" << endl;
		}
	init_from_sims(S, 0);
	if (f_v) {
		cout << "strong_generators::generators_for_the_singer_cycle after extracting strong generators" << endl;
		}
	if (f_vv) {
		INT f_print_as_permutation = FALSE;
		INT f_offset = FALSE;
		INT offset = 0;
		INT f_do_it_anyway_even_for_big_degree = FALSE;
		INT f_print_cycles_of_length_one = FALSE;
		
		cout << "strong generators are:" << endl;
		gens->print(cout, f_print_as_permutation, 
			f_offset, offset, f_do_it_anyway_even_for_big_degree, 
			f_print_cycles_of_length_one);
		}
	delete S;
	delete my_gens;
	FREE_INT(data);
	FREE_INT(go_factored);
	FREE_INT(Elt1);
	if (f_v) {
		cout << "strong_generators::generators_for_the_singer_cycle done" << endl;
		}
}

void strong_generators::generators_for_the_null_polarity_group(action *A, 
	matrix_group *Mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	finite_field *F;
	INT n, q;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_null_polarity_group" << endl;
		}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (f_v) {
		cout << "n=" << n << " q=" << q << endl;
		}

	null_polarity_generator *N;

	N = new null_polarity_generator;


	if (f_v) {
		cout << "strong_generators::generators_for_the_null_polarity_group calling null_polarity_generator::init" << endl;
		}
	N->init(F, n, verbose_level);
	
	init_from_data(A, N->Data, 
		N->nb_gens, n * n, N->transversal_length, 
		verbose_level);


	delete N;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_null_polarity_group done" << endl;
		}
}

void strong_generators::generators_for_symplectic_group(action *A, 
	matrix_group *Mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	finite_field *F;
	INT n, q;
	
	if (f_v) {
		cout << "strong_generators::generators_for_symplectic_group" << endl;
		}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (f_v) {
		cout << "n=" << n << " q=" << q << endl;
		}

	generators_symplectic_group *N;

	N = new generators_symplectic_group;


	if (f_v) {
		cout << "strong_generators::generators_for_symplectic_group calling generators_symplectic_group::init" << endl;
		}
	N->init(F, n, verbose_level);
	
	init_from_data(A, N->Data, 
		N->nb_gens, n * n, N->transversal_length, 
		verbose_level);


	delete N;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_symplectic_group done" << endl;
		}
}

void strong_generators::init_centralizer_of_matrix(action *A, INT *Mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	sims *S;

	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix" << endl;
		}
	S = create_sims_for_centralizer_of_matrix(A, Mtx, verbose_level - 1);
	init_from_sims(S, 0 /* verbose_level */);
	delete S;
	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix done" << endl;
		}
}

void strong_generators::init_centralizer_of_matrix_general_linear(action *A_projective, action *A_general_linear, INT *Mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	sims *S;
	strong_generators *SG1;
	longinteger_object go1, Q, go;
	longinteger_domain D;
	matrix_group *M;
	vector_ge *new_gens;
	INT *data;
	INT q, n, i;

	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix_general_linear" << endl;
		}
	S = create_sims_for_centralizer_of_matrix(A_projective, Mtx, 0/* verbose_level */);
	SG1 = new strong_generators;
	SG1->init_from_sims(S, 0 /* verbose_level */);
	delete S;

	M = A_projective->G.matrix_grp;
	q = M->GFq->q;
	n = M->n;

	SG1->group_order(go1);
	Q.create(q - 1);
	D.mult(go1, Q, go);

	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix_general_linear created centralizer in the projective linear group of order " << go1 << endl;
		}
	
	new_gens = new vector_ge;
	new_gens->init(A_general_linear);
	new_gens->allocate(SG1->gens->len + 1);
	data = NEW_INT(n * n + n + 1);
	for (i = 0; i < SG1->gens->len; i++) {
		INT_vec_copy(SG1->gens->ith(i), data, n * n);
		if (M->f_semilinear) {
			data[n * n] = SG1->gens->ith(i)[n * n];
			}
		A_general_linear->make_element(new_gens->ith(i), data, 0);
		}
	M->GFq->diagonal_matrix(data, n, M->GFq->primitive_root());
	if (M->f_semilinear) {
		data[n * n] = 0;
		}
	A_general_linear->make_element(new_gens->ith(SG1->gens->len), data, 0);

	
	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix_general_linear creating sims for the general linear centralizer of order " << go << endl;
		}
	S = create_sims_from_generators_with_target_group_order(A_general_linear, 
		new_gens, go, 0 /* verbose_level */);
	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix_general_linear creating sims for the general linear centralizer of order " << go <<  " done" << endl;
		}
	init_from_sims(S, 0 /* verbose_level */);
	delete S;

	
	FREE_INT(data);
	delete new_gens;
	delete SG1;
	if (f_v) {
		cout << "strong_generators::init_centralizer_of_matrix_general_linear done" << endl;
		}
}

void strong_generators::field_reduction(action *Aq, INT n, INT s, finite_field *Fq, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT q, Q, m, t;
	finite_field *FQ;
	action *AQ;
	subfield_structure *S;
	sims *Sims;
	INT *EltQ;
	INT *Eltq;
	INT *Mtx;

	if (f_v) {
		cout << "strong_generators::field_reduction" << endl;
		}
	q = Fq->q;
	Q = i_power_j(q, s);
	m = n / s;
	if (m * s != n) {
		cout << "strong_generators::field_reduction s must divide n" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "strong_generators::field_reduction creating subfield structure" << endl;
		}
	if (f_v) {
		cout << "n=" << n << endl;
		cout << "s=" << s << endl;
		cout << "m=" << m << endl;
		cout << "q=" << q << endl;
		cout << "Q=" << Q << endl;
		}
	FQ = new finite_field;
	FQ->init(Q, 0);

	AQ = new action;
	
	if (f_v) {
		cout << "strong_generators::field_reduction creating AQ" << endl;
		}
	AQ->init_general_linear_group(m, FQ, FALSE /* f_semilinear */, TRUE /* f_basis */, verbose_level - 2);
	if (f_v) {
		cout << "strong_generators::field_reduction creating AQ done" << endl;
		}

	longinteger_object order_GLmQ;
	longinteger_object target_go;
	longinteger_domain D;
	INT r;

	AQ->group_order(order_GLmQ);
	

	cout << "strong_generators::field_reduction order of GL(m,Q) = " << order_GLmQ << endl;
	D.integral_division_by_INT(order_GLmQ, 
		q - 1, target_go, r);
	cout << "strong_generators::field_reduction target_go = " << target_go << endl;

	S = new subfield_structure;
	S->init(FQ, Fq, verbose_level);

	cout << "strong_generators::field_reduction creating subfield structure done" << endl;
		

	vector_ge *gens;
	vector_ge *gens1;
	INT nb_gens;

	gens = AQ->Strong_gens->gens;
	nb_gens = gens->len;

	gens1 = new vector_ge;

	Eltq = NEW_INT(Aq->elt_size_in_INT);
	Mtx = NEW_INT(n * n);

	cout << "strong_generators::field_reduction lifting generators" << endl;
	gens1->init(Aq);
	gens1->allocate(nb_gens);
	for (t = 0; t < nb_gens; t++) {
		cout << "strong_generators::field_reduction " << t << 
" / " << nb_gens << endl;
		EltQ = gens->ith(t);
		S->lift_matrix(EltQ, m, Mtx, 0 /* verbose_level */);
		if (f_v) {
			cout << "lifted matrix:" << endl;
			INT_matrix_print(Mtx, n, n);
			}
		Aq->make_element(Eltq, Mtx, verbose_level - 1);
		if (f_v) {
			cout << "after make_element:" << endl;
			Aq->element_print_quick(Eltq, cout);
			}
		Aq->element_move(Eltq, gens1->ith(t), 0);
		cout << "strong_generators::field_reduction " << t << 
" / " << nb_gens << " done" << endl;
		}

	if (f_v) {
		cout << "strong_generators::field_reduction creating lifted group:" << endl;
		}
	Sims = create_sims_from_generators_with_target_group_order(Aq, 
		gens1, target_go, 0 /* verbose_level */);

#if 0
	Sims = create_sims_from_generators_without_target_group_order(Aq, 
		gens1, MINIMUM(2, verbose_level - 3));
#endif

	if (f_v) {
		cout << "strong_generators::field_reduction creating lifted group done" << endl;
		}

	longinteger_object go;

	Sims->group_order(go);

	if (f_v) {
		cout << "go=" << go << endl;
		}

	init_from_sims(Sims, 0 /* verbose_level */);
	if (f_v) {
		cout << "strong_generators::field_reduction strong generators are:" << endl;
		print_generators();
		}

	delete gens1;
	FREE_INT(Eltq);
	FREE_INT(Mtx);
	delete Sims;
	delete S;
	delete AQ;
	delete FQ;
	if (f_v) {
		cout << "strong_generators::field_reduction done" << endl;
		}

}

void strong_generators::generators_for_translation_plane_in_andre_model(
	action *A_PGL_n1_q, action *A_PGL_n_q, 
	matrix_group *Mtx_n1, matrix_group *Mtx_n, 
	vector_ge *spread_stab_gens, longinteger_object &spread_stab_go, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	finite_field *F;
	INT n, n1, q;
	vector_ge *my_gens;
	INT *M, *M1;
	INT sz;

	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model" << endl;
		}
	F = Mtx_n->GFq;
	q = F->q;
	n = Mtx_n->n;
	n1 = Mtx_n1->n;

	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model n=" << n << " n1=" << n1 << endl;
		}
	INT f_semilinear;
	INT nb_gens, h, cnt, i, j, a, u;


	f_semilinear = Mtx_n1->f_semilinear;
	nb_gens = spread_stab_gens->len + /* 1 + */ n * F->e;

	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model nb_gens=" << nb_gens << endl;
		}
	sz = n1 * n1 + 1;
	M = NEW_INT(sz * nb_gens);
	my_gens = new vector_ge;
	my_gens->init(A_PGL_n1_q);
	my_gens->allocate(nb_gens);


	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model making generators of the first kind:" << endl;
		}
	cnt = 0;
	for (h = 0; h < spread_stab_gens->len; h++, cnt++) {
		if (f_vv) {
			cout << "making generator " << h << ":" << endl;
			//INT_matrix_print(spread_stab_gens->ith(h), n, n);
			}

		M1 = M + cnt * sz;
		INT_vec_zero(M1, n1 * n1);
		for (i = 0; i < n1; i++) {
			M1[i * n1 + i] = 1;
			}
		if (f_semilinear) {
			M1[n1 * n1] = 0;
			}
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				a = spread_stab_gens->ith(h)[i * n + j];
				M1[i * n1 + j] = a;
				}
			}
		if (f_semilinear) {
			a = spread_stab_gens->ith(h)[n * n];
			M1[n1 * n1] = a;
			}
		//An1->make_element(gens->ith(h), M, 0 /*verbose_level*/);
		}

#if 0
	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model making generators of the second kind:" << endl;
		}
	INT alpha;

	alpha = F->primitive_root();
	M1 = M + cnt * sz;
	INT_vec_zero(M1, n1 * n1);
	for (i = 0; i < n1; i++) {
		M1[i * n1 + i] = alpha;
		}
	if (f_semilinear) {
		M1[n1 * n1] = 0;
		}
	//An1->make_element(gens->ith(spread_stab_gens->len), M, 0 /* verbose_level */);
	cnt++;
#endif


	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model making generators of the third kind:" << endl;
		}

	INT alpha;

	alpha = F->primitive_root();
	
	for (h = 0; h < n; h++) {
		for (u = 0; u < F->e; u++, cnt++) {
			M1 = M + cnt * sz;
			INT_vec_zero(M1, n1 * n1);
			for (i = 0; i < n1; i++) {
				M1[i * n1 + i] = 1;
				}
			M1[(n1 - 1) * n1 + h] = F->frobenius_power(alpha, u); // computes alpha^{p^u}
			if (f_semilinear) {
				M1[n1 * n1] = 0;
				}
			//An1->make_element(gens->ith(spread_stab_gens->len + 1 + h * F->e + u), M, 0 /* verbose_level */);
			}
		}

	if (cnt != nb_gens) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model cnt != nb_gens" << endl;
		exit(1);
		}

	for (h = 0; h < nb_gens; h++) {
		M1 = M + h * sz;
		A_PGL_n1_q->make_element(my_gens->ith(h), M1, 0 /* verbose_level */);
		}

	longinteger_domain D;
	longinteger_object target_go, aa, b, go;
	

	spread_stab_go.assign_to(aa);
	//D.multiply_up(aa, spread_stab_tl, A_PGL_n_q->base_len);

	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model spread stabilizer has order " << aa << endl;
		}
	b.create_i_power_j(q, n);
	D.mult(aa, b, target_go);

	sims *S;


	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model creating group" << endl;
		}
	S = create_sims_from_generators_with_target_group_order(A_PGL_n1_q, 
		my_gens, target_go, 0 /*verbose_level*/);
	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model group has been created" << endl;
		}

	S->group_order(go);

	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model created group of order " << go << endl;
		}

	init_from_sims(S, 0 /* verbose_level */);

	delete S;
	FREE_INT(M);
	delete my_gens;

	if (f_v) {
		cout << "strong_generators::generators_for_translation_plane_in_andre_model done" << endl;
		}
}

void strong_generators::generators_for_the_stabilizer_of_two_components(action *A_PGL_n_q, 
	matrix_group *Mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	finite_field *F;
	INT n, k, q;
	vector_ge *my_gens;
	action *A_PGL_k_q;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components" << endl;
		}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	k = n >> 1;
	if (ODD(n)) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components n must be even" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "n=" << n << " k=" << k << " q=" << q << endl;
		}

	A_PGL_k_q = new action;
	A_PGL_k_q->init_projective_group(k, F, FALSE /*f_semilinear */, 
		TRUE /* f_basis */, 0 /* verbose_level */);

	my_gens = new vector_ge;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components before make_generators_stabilizer_of_two_components" << endl;
		}
	make_generators_stabilizer_of_two_components(A_PGL_n_q, A_PGL_k_q, 
		k, my_gens, 0 /*verbose_level */);
		// ACTION/action_global.C
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components after make_generators_stabilizer_of_two_components" << endl;
		}

	longinteger_object go_linear, a, two, target_go;
	longinteger_domain D;

	two.create(1);
	A_PGL_k_q->group_order(go_linear);
	D.mult(go_linear, go_linear, a);
	D.mult(a, two, target_go);
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components before generators_to_strong_generators target_go=" << target_go << endl;
		}
	
	strong_generators *SG;

	generators_to_strong_generators(A_PGL_n_q, 
		TRUE /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components after generators_to_strong_generators" << endl;
		}

	init_copy(SG, 0);


	delete SG;	
	delete A_PGL_k_q;
	delete my_gens;

	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_two_components done" << endl;
		}
}

void strong_generators::regulus_stabilizer(action *A_PGL_n_q, 
	matrix_group *Mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	finite_field *F;
	INT n, k, q;
	vector_ge *my_gens;
	action *A_PGL_k_q;
	longinteger_object go, a, b, target_go;
	longinteger_domain D;
	INT *P;
	INT len1, len;
	INT h1, h;
	INT Identity[4] = {0,1,1,0};
	INT *Q;
	INT *Elt1;
	vector_ge *gens1;
	
	if (f_v) {
		cout << "strong_generators::regulus_stabilizer" << endl;
		}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (n != 4) {
		cout << "strong_generators::regulus_stabilizer n must be 4" << endl;
		exit(1);
		}
	k = n >> 1;
	if (ODD(n)) {
		cout << "strong_generators::regulus_stabilizer n must be even" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "n=" << n << " k=" << k << " q=" << q << endl;
		}

	A_PGL_k_q = new action;
	A_PGL_k_q->init_projective_group(k, F, FALSE /*f_semilinear */, 
		TRUE /* f_basis */, 0 /* verbose_level */);
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
		cout << "strong_generators::regulus_stabilizer target_go=" << target_go 
			<< " = order of PGL(" << k << "," << q << ")^2 * " << b << " = " << go << "^2 * " << b << endl;
		cout << "action A_PGL_k_q: ";
		A_PGL_k_q->print_info();
		}

	Elt1 = NEW_INT(A_PGL_n_q->elt_size_in_INT);
	my_gens = new vector_ge;
	my_gens->init(A_PGL_n_q);

	gens1 = A_PGL_k_q->Strong_gens->gens;
	len1 = gens1->len;
	if (f_v) {
		cout << "There are " << len1 << " generators in gen1" << endl;
		}
	len = 2 * len1;
	if (Mtx->f_semilinear) {
		len++;
		}
	Q = NEW_INT(n * n + 1);
	my_gens->allocate(len);
	

	if (f_vv) {
		cout << "strong_generators::regulus_stabilizer creating generators for the stabilizer:" << endl;
		}
	for (h = 0; h < len; h++) {
		if (f_vv) {
			cout << "strong_generators::regulus_stabilizer h=" << h << " / " << len << endl;
			}

		if (h < 2 * len1) {
			h1 = h >> 1;
			P = gens1->ith(h1);
			if (f_vv) {
				cout << "strong_generators::regulus_stabilizer generator:" << endl;
				A_PGL_k_q->print_quick(cout, P);
				}

			if ((h % 2) == 0) {
				F->Kronecker_product(P, Identity, 2, Q);
				}
			else {
				F->Kronecker_product(Identity, P, 2, Q);
				}
			if (Mtx->f_semilinear) {
				Q[n * n] = P[k * k];
				}
			}
		else {
			F->identity_matrix(Q, n);
			Q[n * n] = 1;
			}
		if (f_vv) {
			cout << "strong_generators::regulus_stabilizer h = " << h << " before make_element:" << endl;
			INT_matrix_print(Q, n, n);
			if (Mtx->f_semilinear) {
				cout << "strong_generators::regulus_stabilizer semilinear part = " << Q[n * n] << endl;
				}
			}
		A_PGL_n_q->make_element(Elt1, Q, 0);
		if (f_vv) {
			cout << "strong_generators::regulus_stabilizer after make_element:" << endl;
			A_PGL_n_q->print_quick(cout, Elt1);
			}
		A_PGL_n_q->move(Elt1, my_gens->ith(h));
		
		}
	if (f_vv) {
		for (h = 0; h < len; h++) {
			cout << "strong_generators::regulus_stabilizer generator " << h << ":" << endl;
			A_PGL_n_q->element_print(my_gens->ith(h), cout);
			}
		}

	if (f_v) {
		cout << "strong_generators::regulus_stabilizer before generators_to_strong_generators target_go=" << target_go << endl;
		}
	
	strong_generators *SG;

	generators_to_strong_generators(A_PGL_n_q, 
		TRUE /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::regulus_stabilizer after generators_to_strong_generators" << endl;
		}

	init_copy(SG, 0);


	delete SG;	
	delete A_PGL_k_q;
	delete my_gens;
	FREE_INT(Elt1);
	FREE_INT(Q);

	if (f_v) {
		cout << "strong_generators::regulus_stabilizer done" << endl;
		}
}

void strong_generators::generators_for_the_borel_subgroup_upper(action *A_linear, 
	matrix_group *Mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *Elt1;
	vector_ge *my_gens;
	finite_field *F;
	INT *Q;
	INT n, i, j, h, alpha, len, q;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_upper" << endl;
		}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	Elt1 = NEW_INT(A_linear->elt_size_in_INT);
	my_gens = new vector_ge;
	my_gens->init(A_linear);

	len = n + ((n * (n - 1)) >> 1);
	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_upper len=" << len << endl;
		}
	my_gens->allocate(len);
	Q = NEW_INT(n * n + 1);
	

	if (f_vv) {
		cout << "strong_generators::generators_for_the_borel_subgroup_upper creating generators for the stabilizer:" << endl;
		}
	h = 0;
	alpha = F->primitive_root();
	for (i = 0; i < n; i++, h++) {
		F->identity_matrix(Q, n);
		Q[i * n + i] = alpha;
		if (Mtx->f_semilinear) {
			Q[n * n] = 0;
			}
		A_linear->make_element(Elt1, Q, 0);
		if (f_vv) {
			cout << "strong_generators::generators_for_the_borel_subgroup_upper after make_element:" << endl;
			A_linear->print_quick(cout, Elt1);
			}
		A_linear->move(Elt1, my_gens->ith(h));
		}
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			F->identity_matrix(Q, n);
			Q[i * n + j] = 1;
			if (Mtx->f_semilinear) {
				Q[n * n] = 0;
				}
			A_linear->make_element(Elt1, Q, 0);
			if (f_vv) {
				cout << "strong_generators::generators_for_the_borel_subgroup_upper after make_element:" << endl;
				A_linear->print_quick(cout, Elt1);
				}
			A_linear->move(Elt1, my_gens->ith(h));
			h++;
			}
		}
	if (h != len) {
		cout << "strong_generators::generators_for_the_borel_subgroup_upper n != len" << endl;
		cout << "h=" << h << endl;		
		cout << "len=" << len << endl;		
		exit(1);
		}
	
	
	if (f_vv) {
		for (h = 0; h < len; h++) {
			cout << "strong_generators::generators_for_the_borel_subgroup_upper generator " << h << " / " << len << endl;
			A_linear->element_print(my_gens->ith(h), cout);
			}
		}
	longinteger_object target_go;

	INT *factors;
	INT nb_factors;
	nb_factors = len;
	factors = NEW_INT(nb_factors);
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
	FREE_INT(factors);

	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_upper before generators_to_strong_generators target_go=" << target_go << endl;
		}
	

	strong_generators *SG;

	generators_to_strong_generators(A_linear, 
		TRUE /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_upper after generators_to_strong_generators" << endl;
		}

	init_copy(SG, 0);

	delete SG;
	delete my_gens;
	FREE_INT(Elt1);
	FREE_INT(Q);
}

void strong_generators::generators_for_the_borel_subgroup_lower(action *A_linear, 
	matrix_group *Mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *Elt1;
	vector_ge *my_gens;
	finite_field *F;
	INT *Q;
	INT n, i, j, h, alpha, len, q;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_lower" << endl;
		}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	Elt1 = NEW_INT(A_linear->elt_size_in_INT);
	my_gens = new vector_ge;
	my_gens->init(A_linear);

	len = n + ((n * (n - 1)) >> 1);
	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_lower len=" << len << endl;
		}
	my_gens->allocate(len);
	Q = NEW_INT(n * n + 1);
	

	if (f_vv) {
		cout << "strong_generators::generators_for_the_borel_subgroup_lower creating generators for the stabilizer:" << endl;
		}
	h = 0;
	alpha = F->primitive_root();
	for (i = 0; i < n; i++, h++) {
		F->identity_matrix(Q, n);
		Q[i * n + i] = alpha;
		if (Mtx->f_semilinear) {
			Q[n * n] = 0;
			}
		A_linear->make_element(Elt1, Q, 0);
		if (f_vv) {
			cout << "strong_generators::generators_for_the_borel_subgroup_lower after make_element:" << endl;
			A_linear->print_quick(cout, Elt1);
			}
		A_linear->move(Elt1, my_gens->ith(h));
		}
	for (i = 0; i < n; i++) {
		for (j = 0; j < i; j++) {
			F->identity_matrix(Q, n);
			Q[i * n + j] = 1;
			if (Mtx->f_semilinear) {
				Q[n * n] = 0;
				}
			A_linear->make_element(Elt1, Q, 0);
			if (f_vv) {
				cout << "strong_generators::generators_for_the_borel_subgroup_lower after make_element:" << endl;
				A_linear->print_quick(cout, Elt1);
				}
			A_linear->move(Elt1, my_gens->ith(h));
			h++;
			}
		}
	if (h != len) {
		cout << "strong_generators::generators_for_the_borel_subgroup_lower n != len" << endl;
		cout << "h=" << h << endl;		
		cout << "len=" << len << endl;		
		exit(1);
		}
	
	
	if (f_vv) {
		for (h = 0; h < len; h++) {
			cout << "strong_generators::generators_for_the_borel_subgroup_lower generator " << h << " / " << len << endl;
			A_linear->element_print(my_gens->ith(h), cout);
			}
		}
	longinteger_object target_go;

	INT *factors;
	INT nb_factors;
	nb_factors = len;
	factors = NEW_INT(nb_factors);
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
	FREE_INT(factors);

	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_lower before generators_to_strong_generators target_go=" << target_go << endl;
		}
	

	strong_generators *SG;

	generators_to_strong_generators(A_linear, 
		TRUE /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_borel_subgroup_lower after generators_to_strong_generators" << endl;
		}

	init_copy(SG, 0);

	delete SG;
	delete my_gens;
	FREE_INT(Elt1);
	FREE_INT(Q);
}

void strong_generators::generators_for_the_identity_subgroup(action *A_linear, 
	matrix_group *Mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *Elt1;
	vector_ge *my_gens;
	finite_field *F;
	INT *Q;
	INT n, i, h, len; //, q;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_identity_subgroup" << endl;
		}
	F = Mtx->GFq;
	//q = F->q;
	n = Mtx->n;
	Elt1 = NEW_INT(A_linear->elt_size_in_INT);
	my_gens = new vector_ge;
	my_gens->init(A_linear);

	len = 1;
	if (f_v) {
		cout << "strong_generators::generators_for_the_identity_subgroup len=" << len << endl;
		}
	my_gens->allocate(len);
	Q = NEW_INT(n * n + 1);
	

	if (f_vv) {
		cout << "strong_generators::generators_for_the_identity_subgroup creating generators for the stabilizer:" << endl;
		}
	for (i = 0; i < 1; i++) {
		F->identity_matrix(Q, n);
		if (Mtx->f_semilinear) {
			Q[n * n] = 0;
			}
		A_linear->make_element(Elt1, Q, 0);
		if (f_vv) {
			cout << "strong_generators::generators_for_the_identity_subgroup after make_element:" << endl;
			A_linear->print_quick(cout, Elt1);
			}
		A_linear->move(Elt1, my_gens->ith(i));
		}
	
	
	if (f_vv) {
		for (h = 0; h < len; h++) {
			cout << "strong_generators::generators_for_the_identity_subgroup generator " << h << " / " << len << endl;
			A_linear->element_print(my_gens->ith(h), cout);
			}
		}
	longinteger_object target_go;

	target_go.create(1);

	if (f_v) {
		cout << "strong_generators::generators_for_the_identity_subgroup before generators_to_strong_generators target_go=" << target_go << endl;
		}
	

	strong_generators *SG;

	generators_to_strong_generators(A_linear, 
		TRUE /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_identity_subgroup after generators_to_strong_generators" << endl;
		}

	init_copy(SG, 0);

	delete SG;
	delete my_gens;
	FREE_INT(Elt1);
	FREE_INT(Q);
}


void strong_generators::generators_for_parabolic_subgroup(action *A_PGL_n_q, 
	matrix_group *Mtx, INT k, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	finite_field *F;
	INT n, q, i;
	vector_ge *my_gens;
	INT *data;
	INT size;
	INT nb_gens;
	
	if (f_v) {
		cout << "strong_generators::generators_for_parabolic_subgroup" << endl;
		}
	F = Mtx->GFq;
	q = F->q;
	n = Mtx->n;
	if (f_v) {
		cout << "n=" << n << " k=" << k << " q=" << q << endl;
		}

	if (f_v) {
		cout << "strong_generators::generators_for_parabolic_subgroup before generators_for_parabolic_subgroup" << endl;
		}

	::generators_for_parabolic_subgroup(n, F, 
		Mtx->f_semilinear, k, 
		data, size, nb_gens, 
		verbose_level);
		// GALOIS/group_generators.C

	my_gens = new vector_ge;
	my_gens->init(A_PGL_n_q);
	my_gens->allocate(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		A_PGL_n_q->make_element(my_gens->ith(i), data + i * size, 0);
		}
	

	if (f_v) {
		cout << "strong_generators::generators_for_parabolic_subgroup after generators_for_parabolic_subgroup" << endl;
		}

	longinteger_object go1, nCk, target_go;
	longinteger_domain D;


	D.group_order_PGL(go1, n, q, Mtx->f_semilinear);

	cout << "strong_generators::generators_for_parabolic_subgroup go1=" << go1 << endl;

	D.q_binomial_no_table(nCk, n, k, q, 0 /* verbose_level */);

	cout << "strong_generators::generators_for_parabolic_subgroup nCk=" << nCk << endl;

	D.integral_division_exact(go1, nCk, target_go);

	if (f_v) {
		cout << "strong_generators::generators_for_parabolic_subgroup before generators_to_strong_generators target_go=" << target_go << endl;
		}
	
	strong_generators *SG;

	generators_to_strong_generators(A_PGL_n_q, 
		TRUE /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::generators_for_parabolic_subgroup after generators_to_strong_generators" << endl;
		}

	init_copy(SG, 0);


	delete SG;	
	delete my_gens;
	FREE_INT(data);

	if (f_v) {
		cout << "strong_generators::generators_for_parabolic_subgroup done" << endl;
		}
}

void strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4(action *A_PGL_4_q, 
	matrix_group *Mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	finite_field *F;
	INT n, q, i;
	vector_ge *my_gens;
	INT *data;
	INT size;
	INT nb_gens;
	
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
		cout << "strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4 n != 4" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4 before generators_for_stabilizer_of_three_collinear_points_in_PGL4" << endl;
		}

	::generators_for_stabilizer_of_three_collinear_points_in_PGL4(F, 
		Mtx->f_semilinear, 
		data, size, nb_gens, 
		verbose_level);
		// GALOIS/group_generators.C

	my_gens = new vector_ge;
	my_gens->init(A_PGL_4_q);
	my_gens->allocate(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		A_PGL_4_q->make_element(my_gens->ith(i), data + i * size, 0);
		}
	

	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4 after generators_for_stabilizer_of_three_collinear_points_in_PGL4" << endl;
		}

	longinteger_object target_go, a, b, c, d, e, f;
	longinteger_domain D;


	target_go.create(1);
	a.create((q - 1) * 6);
	b.create(q + 1);
	c.create(q);
	d.create(q - 1);
	e.create(i_power_j(q, 4));
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
		cout << "strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4 before generators_to_strong_generators target_go=" << target_go << endl;
		}
	
	strong_generators *SG;

	generators_to_strong_generators(A_PGL_4_q, 
		TRUE /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4 after generators_to_strong_generators" << endl;
		}

	init_copy(SG, 0);


	delete SG;	
	delete my_gens;
	FREE_INT(data);

	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_three_collinear_points_in_PGL4 done" << endl;
		}
}

void strong_generators::generators_for_stabilizer_of_triangle_in_PGL4(action *A_PGL_4_q, 
	matrix_group *Mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	finite_field *F;
	INT n, q, i;
	vector_ge *my_gens;
	INT *data;
	INT size;
	INT nb_gens;
	
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
		cout << "strong_generators::generators_for_stabilizer_of_triangle_in_PGL4 n != 4" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_triangle_in_PGL4 before generators_for_stabilizer_of_triangle_in_PGL4" << endl;
		}

	::generators_for_stabilizer_of_triangle_in_PGL4(F, 
		Mtx->f_semilinear, 
		data, size, nb_gens, 
		verbose_level);
		// GALOIS/group_generators.C

	my_gens = new vector_ge;
	my_gens->init(A_PGL_4_q);
	my_gens->allocate(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		A_PGL_4_q->make_element(my_gens->ith(i), data + i * size, 0);
		}
	

	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_triangle_in_PGL4 after generators_for_stabilizer_of_triangle_in_PGL4" << endl;
		}

	longinteger_object target_go, a, b, c, f;
	longinteger_domain D;


	target_go.create(1);
	a.create(i_power_j(q, 3));
	b.create(i_power_j(q - 1, 3));
	c.create(6);
	D.mult_in_place(target_go, a);
	D.mult_in_place(target_go, b);
	D.mult_in_place(target_go, c);
	if (Mtx->f_semilinear) {
		f.create(Mtx->GFq->e);
		D.mult_in_place(target_go, f);
		}
	
	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_triangle_in_PGL4 before generators_to_strong_generators target_go=" << target_go << endl;
		}
	
	strong_generators *SG;

	generators_to_strong_generators(A_PGL_4_q, 
		TRUE /* f_target_go */, target_go, 
		my_gens, SG, verbose_level - 3);
	
	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_triangle_in_PGL4 after generators_to_strong_generators" << endl;
		}

	init_copy(SG, 0);


	delete SG;	
	delete my_gens;
	FREE_INT(data);

	if (f_v) {
		cout << "strong_generators::generators_for_stabilizer_of_triangle_in_PGL4 done" << endl;
		}
}

void strong_generators::generators_for_the_orthogonal_group(action *A, 
	finite_field *F, INT n, 
	INT epsilon, 
	INT f_semilinear, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::generators_for_the_orthogonal_group" << endl;
		cout << "n=" << n << endl;
		cout << "epsilon=" << epsilon << endl;
		cout << "q=" << F->q << endl;
		cout << "f_semilinear=" << f_semilinear << endl;
		}

	action *A2;

	A2 = new action;
	if (f_v) {
		cout << "strong_generators::generators_for_the_orthogonal_group before A2->init_orthogonal_group" << endl;
		}

	A2->init_orthogonal_group(epsilon, 
		n, F, 
		TRUE /* f_on_points */, FALSE /* f_on_lines */, FALSE /* f_on_points_and_lines */, 
		f_semilinear, 
		TRUE /* f_basis */, verbose_level);

	if (f_v) {
		cout << "strong_generators::generators_for_the_orthogonal_group after A2->init_orthogonal_group" << endl;
		}

	longinteger_object target_go;
	strong_generators *Strong_gens2;

	A2->Sims->group_order(target_go);

	if (f_v) {
		cout << "strong_generators::generators_for_the_orthogonal_group before generators_to_strong_generators" << endl;
		}
	generators_to_strong_generators(A, 
		TRUE /* f_target_go */, target_go, 
		&A2->Sims->gens, Strong_gens2, 
		0 /* verbose_level */);

	if (f_v) {
		cout << "strong_generators::generators_for_the_orthogonal_group after generators_to_strong_generators" << endl;
		}
	
	init_copy(Strong_gens2, 0 /* verbose_level */);

	//init_from_sims(A2->Sims, 0 /* verbose_level */);
	delete Strong_gens2;
	delete A2;

	if (f_v) {
		cout << "strong_generators::generators_for_the_orthogonal_group done" << endl;
		}
}

void strong_generators::generators_for_the_stabilizer_of_the_cubic_surface(action *A, 
	finite_field *F, INT iso, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_the_cubic_surface" << endl;
		cout << "q=" << F->q << endl;
		cout << "iso=" << iso << endl;
		}

	INT *data;
	INT nb_gens;
	INT data_size;
	const BYTE *ascii_target_go;
	longinteger_object target_go;
	INT i;
	
	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_the_cubic_surface before cubic_surface_stab_gens" << endl;
		}
	cubic_surface_stab_gens(F->q, iso, data, nb_gens, data_size, ascii_target_go);


	vector_ge *gens;

	gens = new vector_ge;
	gens->init(A);
	target_go.create_from_base_10_string(ascii_target_go);


	gens->allocate(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		A->make_element(gens->ith(i), data + i * data_size, 0);
		}



	strong_generators *Strong_gens2;

	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_the_cubic_surface before generators_to_strong_generators" << endl;
		}
	generators_to_strong_generators(A, 
		TRUE /* f_target_go */, target_go, 
		gens, Strong_gens2, 
		0 /* verbose_level */);

	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_the_cubic_surface after generators_to_strong_generators" << endl;
		}
	
	init_copy(Strong_gens2, 0 /* verbose_level */);

	delete Strong_gens2;
	delete gens;

	if (f_v) {
		cout << "strong_generators::generators_for_the_stabilizer_of_the_cubic_surface done" << endl;
		}
}




