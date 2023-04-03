// action_init.cpp
//
// Anton Betten
// 1/1/2009

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {


void action::init_group_from_generators(
	int *group_generator_data, int group_generator_size,
	int f_group_order_target,
	const char *group_order_target,
	data_structures_groups::vector_ge *gens,
	groups::strong_generators *&Strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object go, cur_go;
	groups::sims S;
	int *Elt;
	int nb_gens, i;
	int nb_times = 200;

	if (f_v) {
		cout << "action::init_group_from_generators" << endl;
		cout << "group_generator_size=" << group_generator_size << endl;
	}
	if (f_group_order_target) {
		cout << "group_order_target=" << group_order_target << endl;
	}
	go.create_from_base_10_string(group_order_target, 0);
	if (f_group_order_target) {
		cout << "group_order_target=" << go << endl;
	}
	S.init(this, verbose_level - 2);
	Elt = NEW_int(elt_size_in_int);
	nb_gens = group_generator_size / make_element_size;
	if (nb_gens * make_element_size != group_generator_size) {
		cout << "action::init_group_from_generators fatal: "
				"group_generator_size is not "
				"divisible by make_element_size"
				<< endl;
		cout << "make_element_size=" << make_element_size << endl;
		cout << "group_generator_size=" << group_generator_size << endl;
		exit(1);
	}
	gens->init(this, verbose_level - 2);
	gens->allocate(nb_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		if (f_v) {
			cout << "parsing generator " << i << ":" << endl;
		}
		Int_vec_print(cout, group_generator_data +
			i * make_element_size, make_element_size);
		cout << endl;
		Group_element->make_element(Elt,
			group_generator_data + i * make_element_size, verbose_level - 2);
		Group_element->element_move(Elt, gens->ith(i), 0);
	}
	if (f_v) {
		cout << "done parsing generators" << endl;
	}
	S.init_trivial_group(verbose_level);
	S.init_generators(*gens, verbose_level);
	S.compute_base_orbits(verbose_level);
	while (true) {
		S.closure_group(nb_times, 0/*verbose_level*/);
		S.group_order(cur_go);
		cout << "cur_go=" << cur_go << endl;
		if (!f_group_order_target) {
			break;
		}
		if (D.compare(cur_go, go) == 0) {
			cout << "reached target group order" << endl;
			break;
		}
		cout << "did not reach target group order, continuing" << endl;
	}

	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->init_from_sims(&S, verbose_level - 1);

	FREE_int(Elt);
}

void action::init_group_from_generators_by_base_images(
		groups::sims *parent_group_S,
	int *group_generator_data, int group_generator_size,
	int f_group_order_target,
	const char *group_order_target,
	data_structures_groups::vector_ge *gens,
	groups::strong_generators *&Strong_gens_out,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object go, cur_go;
	groups::sims S;
	int *Elt;
	int nb_gens, i;
	int nb_times = 200;

	if (f_v) {
		cout << "action::init_group_from_generators_by_base_images" << endl;
	}
	if (f_v) {
		cout << "group_generator_size=" << group_generator_size << endl;
	}
	if (f_group_order_target) {
		cout << "group_order_target=" << group_order_target << endl;
		go.create_from_base_10_string(group_order_target, 0);
	}
	if (f_group_order_target) {
		cout << "group_order_target=" << go << endl;
	}
	S.init(this, verbose_level - 2);
	Elt = NEW_int(elt_size_in_int);
	nb_gens = group_generator_size / base_len();
	if (f_v) {
		cout << "nb_gens=" << nb_gens << endl;
		cout << "base_len=" << base_len() << endl;
	}
	if (nb_gens * base_len() != group_generator_size) {
		cout << "action::init_group_from_generators_by_base_images fatal: "
				"group_generator_size is not divisible by base_len" << endl;
		cout << "base_len=" << base_len() << endl;
		cout << "group_generator_size=" << group_generator_size << endl;
		exit(1);
	}
	gens->init(this, verbose_level - 2);
	gens->allocate(nb_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		if (f_v) {
			cout << "parsing generator " << i << ":" << endl;
		}
		Int_vec_print(cout, group_generator_data +
			i * base_len(), base_len());
		cout << endl;
		Group_element->make_element_from_base_image(Elt, parent_group_S,
			group_generator_data + i * base_len(),
			verbose_level - 2);
		Group_element->element_move(Elt, gens->ith(i), 0);
	}
	if (f_v) {
		cout << "done parsing generators" << endl;
	}
	S.init_trivial_group(verbose_level);
	S.init_generators(*gens, verbose_level);
	S.compute_base_orbits(verbose_level);
	while (true) {
		S.closure_group(nb_times, 0/*verbose_level*/);
		S.group_order(cur_go);
		cout << "cur_go=" << cur_go << endl;
		if (!f_group_order_target) {
			break;
		}
		if (D.compare(cur_go, go) == 0) {
			cout << "reached target group order" << endl;
			break;
		}
		cout << "did not reach target group order, continuing" << endl;
	}

	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->init_from_sims(&S, verbose_level - 1);
	f_has_strong_generators = true;

	FREE_int(Elt);
	if (f_v) {
		cout << "action::init_group_from_generators_by_base_images done" << endl;
	}
}

void action::build_up_automorphism_group_from_aut_data(
	int nb_auts, int *aut_data,
	groups::sims &S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, i, coset;
	int *Elt1, *Elt2;
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "action::build_up_automorphism_group_from_aut_data "
				"action=" << label << " nb_auts=" << nb_auts << endl;
	}
	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);
	S.init(this, verbose_level - 2);
	S.init_trivial_group(verbose_level - 1);
	for (h = 0; h < nb_auts; h++) {
		if (f_v) {
			cout << "aut_data[" << h << "]=";
			Int_vec_print(cout, aut_data + h * base_len(), base_len());
			cout << endl;
		}
		for (i = 0; i < base_len(); i++) {
			coset = aut_data[h * base_len() + i];
			//image_point = Sims->orbit[i][coset];
			Sims->path[i] = coset;
				//Sims->orbit_inv[i][aut_data[h * base_len + i]];
		}
		if (f_v) {
			cout << "path=";
			Int_vec_print(cout, Sims->path, base_len());
			cout << endl;
		}
		Sims->element_from_path_inv(Elt1);
		if (S.strip_and_add(Elt1, Elt2, 0/*verbose_level*/)) {
			S.group_order(go);
			if (f_v) {
				cout << "generator " << h
						<< " added, group order has been updated to " << go << endl;
				S.print_transversal_lengths();
				S.print_transversals_short();
			}
		}
		else {
			if (f_v) {
				cout << "generator " << h << " strips through" << endl;
			}
		}
	}
	FREE_int(Elt1);
	FREE_int(Elt2);
}



groups::sims *action::create_sims_from_generators_with_target_group_order_factorized(
		data_structures_groups::vector_ge *gens,
		int *tl, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object go;
	ring_theory::longinteger_domain D;
	groups::sims *S;

	if (f_v) {
		cout << "action::create_sims_from_generators_with_target_group_order_factorized" << endl;
	}
	D.multiply_up(go, tl, len, 0 /* verbose_level */);
	if (f_v) {
		cout << "action::create_sims_from_generators_with_target_group_order_factorized go=" << go << endl;
	}
	if (f_v) {
		cout << "action::create_sims_from_generators_with_target_group_order_factorized "
				"before create_sims_from_generators_randomized" << endl;
	}
	S = create_sims_from_generators_randomized(
		gens, true /* f_target_go */, go, verbose_level - 3);
	if (f_v) {
		cout << "action::create_sims_from_generators_with_target_group_order_factorized done" << endl;
	}
	return S;
}

groups::sims *action::create_sims_from_generators_with_target_group_order_lint(
		data_structures_groups::vector_ge *gens,
		long int target_go, int verbose_level)
{
	ring_theory::longinteger_object tgo;

	tgo.create(target_go, __FILE__, __LINE__);
	return create_sims_from_generators_with_target_group_order(
			gens, tgo, verbose_level - 3);
}

groups::sims *action::create_sims_from_generators_with_target_group_order(
		data_structures_groups::vector_ge *gens,
	ring_theory::longinteger_object &target_go,
	int verbose_level)
{
	return create_sims_from_generators_randomized(
		gens, true /* f_target_go */, target_go, verbose_level - 3);
}

groups::sims *action::create_sims_from_generators_without_target_group_order(
		data_structures_groups::vector_ge *gens, int verbose_level)
{
	ring_theory::longinteger_object dummy;

	return create_sims_from_generators_randomized(
		gens, false /* f_target_go */, dummy, verbose_level - 3);
}

groups::sims *action::create_sims_from_single_generator_without_target_group_order(
	int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::sims *S;
	data_structures_groups::vector_ge *gens;
	ring_theory::longinteger_object dummy;

	if (f_v) {
		cout << "action::create_sims_from_single_generator_"
				"without_target_group_order" << endl;
	}
	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init_single(this, Elt, verbose_level - 2);

	S = create_sims_from_generators_randomized(
		gens, false /* f_target_go */, dummy, verbose_level - 3);

	FREE_OBJECT(gens);
	if (f_v) {
		cout << "action::create_sims_from_single_generator_"
				"without_target_group_order done" << endl;
	}
	return S;
}

groups::sims *action::create_sims_from_generators_randomized(
		data_structures_groups::vector_ge *gens,
		int f_target_go, ring_theory::longinteger_object &target_go,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::sims *S;

	if (f_v) {
		cout << "action::create_sims_from_generators_randomized" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		if (f_target_go) {
			cout << "creating a group of order " << target_go << endl;
			if (target_go.is_zero()) {
				cout << "action::create_sims_from_generators_randomized target_go is zero" << endl;
				exit(1);
			}
		}
		else {
			cout << "action::create_sims_from_generators_randomized no target group order given" << endl;
		}
	}

	groups::schreier_sims *ss;

	ss = NEW_OBJECT(groups::schreier_sims);

	if (f_v) {
		cout << "action::create_sims_from_generators_randomized "
				"before ss->init" << endl;
	}
	ss->init(this, verbose_level - 1);
	if (f_v) {
		cout << "action::create_sims_from_generators_randomized "
				"after ss->init" << endl;
	}

	//ss->interested_in_kernel(A_subaction, verbose_level - 1);

	if (f_target_go) {
		ss->init_target_group_order(target_go, 0 /*verbose_level - 1*/);
	}

	if (f_v) {
		cout << "action::create_sims_from_generators_randomized "
				"before ss->init_generators" << endl;
	}
	ss->init_generators(gens, verbose_level - 2);
	if (f_v) {
		cout << "action::create_sims_from_generators_randomized "
				"after ss->init_generators" << endl;
	}

	if (f_v) {
		cout << "action::create_sims_from_generators_randomized "
				"before ss->create_group" << endl;
	}
	ss->create_group(0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "action::create_sims_from_generators_randomized "
				"after ss->create_group" << endl;
	}

	S = ss->G;
	ss->G = NULL;
	//*this = *ss->G;

	//ss->G->null();

	//cout << "create_sims_from_generators_randomized
	// before FREE_OBJECT ss" << endl;
	FREE_OBJECT(ss);
	//cout << "create_sims_from_generators_randomized
	// after FREE_OBJECT ss" << endl;

	if (f_v) {
		cout << "action::create_sims_from_generators_randomized done" << endl;
	}
	return S;
}

groups::sims *action::create_sims_for_centralizer_of_matrix(
		int *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	algebra::matrix_group *M;
	field_theory::finite_field *F;
	int d, q, i;
	algebra::gl_classes *C;

	if (f_v) {
		cout << "action::create_sims_for_centralizer_of_matrix" << endl;
	}

	if (type_G != matrix_group_t) {
		cout << "action::create_sims_for_centralizer_of_matrix "
				"action not of type matrix_group" << endl;
		exit(1);
	}

	M = G.matrix_grp;
	F = M->GFq;
	q = F->q;
	d = M->n;


	if (M->C == NULL) {
		if (f_v) {
			cout << "action::create_sims_for_centralizer_of_matrix "
					"before M->init_gl_classes" << endl;
		}
		M->init_gl_classes(verbose_level - 2);
	}

	C = M->C;

	if (f_v) {
		cout << "action::create_sims_for_centralizer_of_matrix "
				"d = " << d << " q = " << q << endl;
		cout << "Mtx=" << endl;
		Int_matrix_print(Mtx, d, d);
	}

	//gl_classes C;
	//gl_class_rep *Reps;
	//int nb_classes;

	//C.init(d, F, 0 /*verbose_level - 2*/);


#if 0
	C.make_classes(Reps, nb_classes, 0 /*verbose_level - 2*/);

	if (f_v) {
		cout << "create_sims_for_centralizer_of_matrix "
				"There are " << nb_classes << " conjugacy classes" << endl;
	}
	if (f_vv) {
		cout << "create_sims_for_centralizer_of_matrix "
				"The conjugacy classes are:" << endl;
		for (i = 0; i < nb_classes; i++) {
			cout << "Class " << i << ":" << endl;
			int_matrix_print(Reps[i].type_coding.M,
					Reps[i].type_coding.m, Reps[i].type_coding.n);
			cout << "Centralizer order = "
					<< Reps[i].centralizer_order << endl;
		}
	}
#endif


	//int class_rep;

	int *Elt;

	Elt = NEW_int(elt_size_in_int);

	algebra::gl_class_rep *R1;

	R1 = NEW_OBJECT(algebra::gl_class_rep);

	int *Basis;
	int **Gens;
	int nb_gens;
	int nb_alloc = 20;

	Gens = NEW_pint(nb_alloc);
	nb_gens = 0;

	Basis = NEW_int(d * d);
	if (f_v) {
		cout << "action::create_sims_for_centralizer_of_matrix "
				"before generators_for_centralizer" << endl;
	}
	C->generators_for_centralizer(Mtx, R1, Basis, Gens,
			nb_gens, nb_alloc, verbose_level - 2);

	if (f_v) {
		cout << "action::create_sims_for_centralizer_of_matrix "
				"Basis=" << endl;
		Int_matrix_print(Basis, d, d);
		cout << "create_sims_for_centralizer_of_matrix "
				"We found " << nb_gens << " centralizing matrices" << endl;
	}

	if (f_vv) {
		cout << "action::create_sims_for_centralizer_of_matrix "
				"Gens=" << endl;
		for (i = 0; i < nb_gens; i++) {
			cout << "Gen " << i << " / " << nb_gens << " is:" << endl;
			Int_matrix_print(Gens[i], d, d);
		}
	}

	for (i = 0; i < nb_gens; i++) {
		if (!F->Linear_algebra->test_if_commute(Mtx, Gens[i], d,
				0/*verbose_level*/)) {
			cout << "The matrices do not commute" << endl;
			cout << "Mtx=" << endl;
			Int_matrix_print(Mtx, d, d);
			cout << "Gens[i]=" << endl;
			Int_matrix_print(Gens[i], d, d);
			exit(1);
		}
	}

	//C.identify_matrix(Elt, R1, verbose_level);

	if (f_v) {
		cout << "The type of the matrix under "
				"consideration is:" << endl;
		Int_matrix_print(R1->type_coding->M,
				R1->type_coding->m, R1->type_coding->n);
	}


#if 0
	class_rep = C.find_class_rep(Reps, nb_classes,
			R1, 0 /* verbose_level */);

	if (f_v) {
		cout << "The index of the class of the "
				"matrix is = " << class_rep << endl;
	}
#endif


	data_structures_groups::vector_ge *gens;
	data_structures_groups::vector_ge *SG;
	int *tl;
	ring_theory::longinteger_object centralizer_order, cent_go;
	int *Elt1;

	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	SG = NEW_OBJECT(data_structures_groups::vector_ge);
	tl = NEW_int(base_len());
	gens->init(this, verbose_level - 2);
	gens->allocate(nb_gens, verbose_level - 2);
	Elt1 = NEW_int(elt_size_in_int);

	for (i = 0; i < nb_gens; i++) {
		Group_element->make_element(Elt1, Gens[i], 0);
		Group_element->element_move(Elt1, gens->ith(i), 0);
	}
	groups::sims *Cent;


	if (f_v) {
		cout << "before centralizer_order_Kung" << endl;
	}
	R1->centralizer_order_Kung(C, centralizer_order, verbose_level);
	if (f_v) {
		cout << "after centralizer_order_Kung, "
				"centralizer_order=" << centralizer_order << endl;
	}

	Cent = create_sims_from_generators_with_target_group_order(
			gens,
		centralizer_order /*Reps[class_rep].centralizer_order*/,
		0 /* verbose_level */);
	//Cent = create_sims_from_generators_without_target_group_order(
	// A, gens, 0 /* verbose_level */);
	Cent->group_order(cent_go);

	if (f_v) {
		cout << "action::create_sims_for_centralizer_of_matrix "
				"The order of the centralizer is " << cent_go << endl;
	}




	for (i = 0; i < nb_gens; i++) {
		FREE_int(Gens[i]);
	}
	FREE_pint(Gens);

	FREE_OBJECT(R1);
	FREE_OBJECT(gens);
	FREE_OBJECT(SG);
	FREE_int(tl);
	FREE_int(Elt1);
	FREE_int(Elt);
	FREE_int(Basis);

	if (f_v) {
		cout << "action::create_sims_for_centralizer_of_matrix done" << endl;
	}
	return Cent;
}


void action::init_automorphism_group_from_group_table(
	std::string &fname_base,
	int *Table, int group_order, int *gens, int nb_gens,
	groups::strong_generators *&Aut_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *N_gens;
	int N_nb_gens;
	int N_go;
	ring_theory::longinteger_object go;
	//int i;
	interfaces::magma_interface Magma;

	if (f_v) {
		cout << "action::init_automorphism_group_from_group_table" << endl;
	}

	Magma.normalizer_in_Sym_n(fname_base,
		group_order, Table, gens, nb_gens,
		N_gens, N_nb_gens, N_go, verbose_level);

	if (f_v) {
		cout << "action::init_automorphism_group_from_group_table "
				"The holomorph has order " << N_go
				<< " and is generated by " << N_nb_gens << " elements" << endl;
	}
	go.create(N_go, __FILE__, __LINE__);

#if 0
	for (i = 0; i < N_nb_gens; i++) {
		cout << "holomorph generator " << i << " / "
				<< N_nb_gens << ":" << endl;

		ord = perm_order(N_gens + i * H->group_order, H->group_order);
		cout << "an element of order " << ord << endl;
		for (j = 0; j < nb_gens; j++) {
			a = gens[j];
			b = N_gens[i * H->group_order + a];
			cout << a << " -> " << b << " : ";
			H->unrank_element(H->Elt1, a);
			H->unrank_element(H->Elt2, b);
			int_vec_print(cout, H->Elt1, H->len);
			cout << " -> ";
			int_vec_print(cout, H->Elt2, H->len);
			cout << endl;
			}
		}
	given_base_length = H->len;
	given_base = NEW_int(given_base_length);
	for (i = 0; i < given_base_length; i++) {
		given_base[i] = i_power_j(q, i);
		}
	cout << "given base: ";
	int_vec_print(cout, given_base, given_base_length);
	cout << endl;
#endif



	if (f_v) {
		cout << "action::init_automorphism_group_from_group_table "
				"creating holomorph" << endl;
	}

	long int *gens1;
	int i;
	gens1 = NEW_lint(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		gens1[i] = gens[i];
	}

	Known_groups->init_permutation_group_from_generators(
		group_order /* degree */,
		true, go,
		N_nb_gens, N_gens,
		nb_gens /* given_base_length */, gens1 /* given_base */,
		false /* f_no_base */,
		verbose_level);
	{
		ring_theory::longinteger_object go;
		action::group_order(go);
		if (f_v) {
			cout << "action::init_automorphism_group_from_group_table "
					"The order of the holomorph is " << go << endl;
		}
	}

	ring_theory::longinteger_object Aut_order;
	if (f_v) {
		cout << "action::init_automorphism_group_from_group_table "
				"creating automorphism group" << endl;
	}
	Aut_gens = Strong_gens->point_stabilizer(0 /* pt */, verbose_level);
	Aut_gens->group_order(Aut_order);
	if (f_v) {
		cout << "action::init_automorphism_group_from_group_table "
				"The automorphism group has order " << Aut_order << endl;
	}

}






}}}



