/*
 * group_theory_global.cpp
 *
 *  Created on: Sep 27, 2024
 *      Author: betten
 */






#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {


group_theory_global::group_theory_global()
{
	Record_birth();

}


group_theory_global::~group_theory_global()
{
	Record_death();

}


void group_theory_global::strong_generators_conjugate_avGa(
		strong_generators *SG_in,
		int *Elt_a,
		strong_generators *&SG_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//actions::action *A;
	data_structures_groups::vector_ge *gens;
	algebra::ring_theory::longinteger_object go;

	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_avGa" << endl;
	}

	//A = SG_in->A;

	SG_in->group_order(go);
	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_avGa "
				"go=" << go << endl;
	}
	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_avGa "
				"before gens->init_conjugate_svas_of" << endl;
	}
	gens->init_conjugate_svas_of(
			SG_in->gens, Elt_a, verbose_level);
	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_avGa "
				"after gens->init_conjugate_svas_of" << endl;
	}

	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_avGa "
				"before generators_to_strong_generators" << endl;
	}
	SG_in->A->generators_to_strong_generators(
		true /* f_target_go */, go,
		gens, SG_out,
		0 /*verbose_level*/);

	FREE_OBJECT(gens);

	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_avGa done" << endl;
	}
}


void group_theory_global::strong_generators_conjugate_aGav(
		strong_generators *SG_in,
		int *Elt_a,
		strong_generators *&SG_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures_groups::vector_ge *gens;
	algebra::ring_theory::longinteger_object go;

	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_aGav" << endl;
	}

	SG_in->group_order(go);
	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_aGav "
				"go=" << go << endl;
	}
	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	gens->init_conjugate_sasv_of(
			SG_in->gens, Elt_a, 0 /* verbose_level */);



	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_aGav "
				"before generators_to_strong_generators" << endl;
	}
	SG_in->A->generators_to_strong_generators(
		true /* f_target_go */, go,
		gens, SG_out,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_aGav "
				"after generators_to_strong_generators" << endl;
	}

	FREE_OBJECT(gens);

	if (f_v) {
		cout << "group_theory_global::strong_generators_conjugate_aGav done" << endl;
	}
}

void group_theory_global::set_of_coset_representatives(
		groups::strong_generators *Subgroup_gens_H,
		groups::strong_generators *Subgroup_gens_G,
		data_structures_groups::vector_ge *&coset_reps,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_global::set_of_coset_representatives" << endl;
	}

	//actions::action *A1;
	//actions::action *A2;

	//A1 = A;
	//A2 = AG_secondary->A;


	groups::sims *S;

	S = Subgroup_gens_G->create_sims(verbose_level);
	// the large group

	if (f_v) {

		algebra::ring_theory::longinteger_object go_G;

		S->group_order(go_G);

		cout << "group_theory_global::set_of_coset_representatives the large group has order " << go_G << endl;

	}

	if (f_v) {
		cout << "group_theory_global::set_of_coset_representatives "
				"before Subgroup_gens_H->set_of_coset_representatives" << endl;
	}
	Subgroup_gens_H->set_of_coset_representatives(
			S,
			coset_reps,
			verbose_level);
	if (f_v) {
		cout << "group_theory_global::set_of_coset_representatives "
				"after Subgroup_gens_H->set_of_coset_representatives" << endl;
		cout << "group_theory_global::set_of_coset_representatives "
				"number of coset reps = " << coset_reps->len << endl;
	}



	FREE_OBJECT(S);

	if (f_v) {
		cout << "group_theory_global::set_of_coset_representatives done" << endl;
	}
}


void group_theory_global::conjugacy_classes_based_on_normal_forms(
		actions::action *A,
		groups::sims *override_Sims,
		std::string &label,
		std::string &label_tex,
		int verbose_level)
// called from group_theoretic_activity by means of any_group::classes_based_on_normal_form
{
	int f_v = (verbose_level >= 1);
	string prefix;
	string fname_output;
	other::orbiter_kernel_system::file_io Fio;
	int d;
	algebra::field_theory::finite_field *F;


	if (f_v) {
		cout << "group_theory_global::conjugacy_classes_based_on_normal_forms" << endl;
	}

	prefix.assign(label);
	fname_output.assign(label);


	d = A->matrix_group_dimension();
	F = A->matrix_group_finite_field();

	if (f_v) {
		cout << "group_theory_global::conjugacy_classes_based_on_normal_forms "
				"d=" << d << endl;
		cout << "group_theory_global::conjugacy_classes_based_on_normal_forms "
				"q=" << F->q << endl;
	}

	algebra::linear_algebra::gl_classes C;
	algebra::linear_algebra::gl_class_rep *R;
	int nb_classes;
	int *Mtx;
	int *Elt;
	int i, order;
	long int a;


	fname_output += "_classes_based_on_normal_forms_"
			+ std::to_string(d) + "_" + std::to_string(F->q) + ".tex";

	C.init(d, F, verbose_level);

	if (f_v) {
		cout << "before C.make_classes" << endl;
	}
	C.make_classes(
			R, nb_classes, false /*f_no_eigenvalue_one*/,
			verbose_level);
	if (f_v) {
		cout << "after C.make_classes" << endl;
	}

	Mtx = NEW_int(d * d + 1);
	Elt = NEW_int(A->elt_size_in_int);

	int *Order;

	Order = NEW_int(nb_classes);

	for (i = 0; i < nb_classes; i++) {

		if (f_v) {
			cout << "class " << i << " / " << nb_classes << ":" << endl;
		}

		Int_vec_zero(Mtx, d * d + 1);
		C.make_matrix_from_class_rep(
				Mtx, R + i, verbose_level - 1);

		A->Group_element->make_element(Elt, Mtx, 0);

		if (f_v) {
			cout << "before override_Sims->element_rank_lint" << endl;
		}
		a = override_Sims->element_rank_lint(Elt);
		if (f_v) {
			cout << "after override_Sims->element_rank_lint" << endl;
		}

		cout << "Representative of class " << i << " / "
				<< nb_classes << " has rank " << a << "\\\\" << endl;
		Int_matrix_print(Elt, d, d);

		if (f_v) {
			cout << "before C.print_matrix_and_centralizer_order_latex" << endl;
		}
		C.print_matrix_and_centralizer_order_latex(
				cout, R + i);
		if (f_v) {
			cout << "after C.print_matrix_and_centralizer_order_latex" << endl;
		}

		if (f_v) {
			cout << "before A->element_order" << endl;
		}
		order = A->Group_element->element_order(Elt);
		if (f_v) {
			cout << "after A->element_order" << endl;
		}

		cout << "The element order is : " << order << "\\\\" << endl;

		Order[i] = order;

	}

	other::data_structures::tally T_order;

	T_order.init(Order, nb_classes, false, 0);


	{
		ofstream ost(fname_output);
		other::l1_interfaces::latex_interface L;

		L.head_easy(ost);
		//C.report(fp, verbose_level);


		ost << "The distribution of element orders is:" << endl;
#if 0
		ost << "$$" << endl;
		T_order.print_file_tex_we_are_in_math_mode(ost, false /* f_backwards */);
		ost << "$$" << endl;
#endif

		//ost << "$" << endl;
		T_order.print_file_tex(ost, false /* f_backwards */);
		ost << "\\\\" << endl;

		ost << "$$" << endl;
		T_order.print_array_tex(ost, false /* f_backwards */);
		ost << "$$" << endl;



		int t, f, l, a, h, c;

		for (t = 0; t < T_order.nb_types; t++) {
			f = T_order.type_first[t];
			l = T_order.type_len[t];
			a = T_order.data_sorted[f];

			if (f_v) {
				cout << "class type " << t << " / " << T_order.nb_types << ":" << endl;
			}

			ost << "\\section{The Classes of Elements of Order $" << a << "$}" << endl;


			ost << "There are " << l << " classes of elements of order "
					<< a << "\\\\" << endl;

			for (h = 0; h < l; h++) {

				c = f + h;

				i = T_order.sorting_perm_inv[c];

				if (f_v) {
					cout << "class " << h << " / " << l
							<< " of elements of order " << a << ":" << endl;
				}

				Int_vec_zero(Mtx, d * d + 1);
				C.make_matrix_from_class_rep(Mtx, R + i, verbose_level - 1);

				A->Group_element->make_element(Elt, Mtx, 0);

				if (f_v) {
					cout << "before override_Sims->element_rank_lint" << endl;
				}
				a = override_Sims->element_rank_lint(Elt);
				if (f_v) {
					cout << "after override_Sims->element_rank_lint" << endl;
				}

				ost << "Representative of class " << i << " / "
						<< nb_classes << " has rank " << a << "\\\\" << endl;
				Int_matrix_print(Elt, d, d);

				if (f_v) {
					cout << "before C.print_matrix_and_centralizer_order_latex" << endl;
				}
				C.print_matrix_and_centralizer_order_latex(ost, R + i);
				if (f_v) {
					cout << "after C.print_matrix_and_centralizer_order_latex" << endl;
				}

				if (f_v) {
					cout << "before A->element_order" << endl;
				}
				order = A->Group_element->element_order(Elt);
				if (f_v) {
					cout << "after A->element_order" << endl;
				}

				ost << "The element order is : " << order << "\\\\" << endl;


			}

		}
		L.foot(ost);
	}
	cout << "Written file " << fname_output << " of size "
			<< Fio.file_size(fname_output) << endl;

	FREE_int(Mtx);
	FREE_int(Elt);
	FREE_OBJECTS(R);

	if (f_v) {
		cout << "group_theory_global::conjugacy_classes_based_on_normal_forms done" << endl;
	}
}


void group_theory_global::find_singer_cycle(
		groups::any_group *Any_group,
		actions::action *A1, actions::action *A2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_global::find_singer_cycle" << endl;
	}
	groups::sims *H;
	groups::strong_generators *SG;

	SG = Any_group->get_strong_generators();

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = SG->create_sims(verbose_level);

	if (f_v) {
		//cout << "group order G = " << G->group_order_int() << endl;
		cout << "group order H = " << H->group_order_lint() << endl;
	}

	int *Elt;
	algebra::ring_theory::longinteger_object go;
	int i, d, q, cnt, ord, order;
	algebra::number_theory::number_theory_domain NT;

	if (!A1->is_matrix_group()) {
		cout << "group_theory_global::find_singer_cycle "
				"needs matrix group" << endl;
		exit(1);
	}
	algebra::basic_algebra::matrix_group *M;

	M = A1->get_matrix_group();
	q = M->GFq->q;
	d = A1->matrix_group_dimension();

	if (A1->is_projective()) {
		order = (NT.i_power_j(q, d) - 1) / (q - 1);
	}
	else {
		order = NT.i_power_j(q, d) - 1;
	}
	if (f_v) {
		cout << "group_theory_global::find_singer_cycle "
				"looking for an "
				"element of order " << order << endl;
	}

	Elt = NEW_int(A1->elt_size_in_int);
	H->group_order(go);

	cnt = 0;
	for (i = 0; i < go.as_int(); i++) {
		H->element_unrank_lint(i, Elt);


		ord = A2->Group_element->element_order(Elt);

	#if 0
		cout << "Element " << setw(5) << i << " / "
				<< go.as_int() << ":" << endl;
		A->element_print(Elt, cout);
		cout << endl;
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
	#endif

		if (ord != order) {
			continue;
		}
		if (!M->Element->has_shape_of_singer_cycle(Elt)) {
			continue;
		}
		if (f_v) {
			cout << "Element " << setw(5) << i << " / "
						<< go.as_int() << " = " << cnt << ":" << endl;
			A2->Group_element->element_print(Elt, cout);
			cout << endl;
			A2->Group_element->element_print_as_permutation(Elt, cout);
			cout << endl;
		}
		cnt++;
	}
	if (f_v) {
		cout << "we found " << cnt
				<< " group elements of order " << order << endl;
	}

	FREE_int(Elt);
	if (f_v) {
		cout << "group_theory_global::find_singer_cycle done" << endl;
	}
}

void group_theory_global::relative_order_vector_of_cosets(
		actions::action *A, groups::strong_generators *SG,
		data_structures_groups::vector_ge *cosets,
		int *&relative_order_table, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt1;
	int *Elt2;
	//int *Elt3;
	groups::sims *S;
	int i, drop_out_level, image, order;

	if (f_v) {
		cout << "group_theory_global::relative_order_vector_of_cosets" << endl;
	}

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	//Elt3 = NEW_int(A->elt_size_in_int);

	relative_order_table = NEW_int(cosets->len);

	S = SG->create_sims(0 /*verbose_level */);
	for (i = 0; i < cosets->len; i++) {
		A->Group_element->element_move(cosets->ith(i), Elt1, 0);
		order = 1;
		while (true) {
			if (S->strip(Elt1, Elt2, drop_out_level, image, 0 /*verbose_level*/)) {
				break;
			}
			A->Group_element->element_mult(cosets->ith(i), Elt1, Elt2, 0);
			A->Group_element->element_move(Elt2, Elt1, 0);
			order++;
		}
		relative_order_table[i] = order;
	}


	FREE_int(Elt1);
	FREE_int(Elt2);

	if (f_v) {
		cout << "group_theory_global::relative_order_vector_of_cosets done" << endl;
	}
}


void group_theory_global::order_of_all_elements(
		actions::action *A, groups::strong_generators *SG,
		int *&order_table, int &go, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	groups::sims *S;
	long int rk;
	int order;

	if (f_v) {
		cout << "group_theory_global::order_of_all_elements" << endl;
	}

	go = SG->group_order_as_lint();

	Elt = NEW_int(A->elt_size_in_int);

	order_table = NEW_int(go);

	S = SG->create_sims(0 /*verbose_level */);
	for (rk = 0; rk < go; rk++) {

		S->element_unrank_lint(
				rk, Elt, 0 /* verbose_level */);


		order = A->Group_element->element_order(Elt);
		order_table[rk] = order;
	}

	FREE_OBJECT(S);
	FREE_int(Elt);

	if (f_v) {
		cout << "group_theory_global::order_of_all_elements done" << endl;
	}
}


std::string group_theory_global::order_invariant(
		actions::action *A, groups::strong_generators *SG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_global::order_invariant" << endl;
	}

	int *order_table;
	int go;

	if (f_v) {
		cout << "group_theory_global::order_invariant "
				"before order_of_all_elements" << endl;
	}
	order_of_all_elements(
			A, SG,
			order_table, go, verbose_level - 1);
	if (f_v) {
		cout << "group_theory_global::order_invariant "
				"after order_of_all_elements" << endl;
	}

	other::data_structures::tally *C;
	int f_second = false;

	C = NEW_OBJECT(other::data_structures::tally);
	C->init(order_table, go, f_second, 0);
	if (f_v) {
		cout << "group_theory_global::order_invariant: "
				"order invariant: ";
		C->print(false /*f_backwards*/);
		C->print_bare_tex(cout, false /*f_backwards*/);
		cout << endl;
	}

	string s;

	s = C->stringify_bare_tex(false /*f_backwards*/);

	FREE_OBJECT(C);

	if (f_v) {
		cout << "group_theory_global::order_invariant done" << endl;
	}
	return s;
}

void group_theory_global::search_element_of_order(
		groups::sims *H,
		actions::action *A1, actions::action *A2,
		int order, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::search_element_of_order" << endl;
	}
#if 0
	groups::sims *H;
	groups::strong_generators *SG;

	SG = Any_group->get_strong_generators();

	//G = LG->initial_strong_gens->create_sims(verbose_level);
	H = SG->create_sims(verbose_level);
#endif

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;
	algebra::ring_theory::longinteger_object go;
	int i, cnt, ord;

	Elt = NEW_int(A1->elt_size_in_int);
	H->group_order(go);

	cnt = 0;
	for (i = 0; i < go.as_int(); i++) {
		H->element_unrank_lint(i, Elt);


		ord = A2->Group_element->element_order(Elt);

	#if 0
		cout << "Element " << setw(5) << i << " / "
				<< go.as_int() << ":" << endl;
		A->element_print(Elt, cout);
		cout << endl;
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
	#endif

		if (ord != order) {
			continue;
		}
		if (f_v) {
			cout << "Element " << setw(5) << i << " / "
						<< go.as_int() << " = " << cnt << ":" << endl;
			A2->Group_element->element_print(Elt, cout);
			cout << endl;
			A2->Group_element->element_print_as_permutation(Elt, cout);
			cout << endl;
		}
		cnt++;
	}
	if (f_v) {
		cout << "we found " << cnt << " group elements of order " << order << endl;
	}

	FREE_int(Elt);
	if (f_v) {
		cout << "group_theory_global::search_element_of_order done" << endl;
	}
}

void group_theory_global::find_standard_generators(
		groups::sims *H,
		actions::action *A1, actions::action *A2,
		int order_a, int order_b, int order_ab, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_global::find_standard_generators" << endl;
	}
#if 0
	groups::sims *H;
	groups::strong_generators *SG;

	SG = Any_group->get_strong_generators();

	H = SG->create_sims(verbose_level);
#endif

	if (f_v) {
		cout << "group_theory_global::find_standard_generators "
				"group order H = " << H->group_order_lint() << endl;
	}

	int *Elt_a;
	int *Elt_b;
	int *Elt_ab;
	algebra::ring_theory::longinteger_object go;
	long int i, j, cnt, ord;
	long int goi;

	Elt_a = NEW_int(A1->elt_size_in_int);
	Elt_b = NEW_int(A1->elt_size_in_int);
	Elt_ab = NEW_int(A1->elt_size_in_int);
	H->group_order(go);

	goi = go.as_lint();
	cnt = 0;
	for (i = 0; i < goi; i++) {
		H->element_unrank_lint(i, Elt_a);


		ord = A2->Group_element->element_order(Elt_a);

	#if 0
		cout << "Element " << setw(5) << i << " / "
				<< go.as_int() << ":" << endl;
		A->element_print(Elt, cout);
		cout << endl;
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
	#endif

		if (ord != order_a) {
			continue;
		}

		for (j = 0; j < goi; j++) {
			H->element_unrank_lint(j, Elt_b);


			ord = A2->Group_element->element_order(Elt_b);

			if (ord != order_b) {
				continue;
			}

			A2->Group_element->element_mult(Elt_a, Elt_b, Elt_ab, 0);

			ord = A2->Group_element->element_order(Elt_ab);

			if (ord != order_ab) {
				continue;
			}

			if (f_v) {
				cout << "group_theory_global::find_standard_generators "
						"a = " << setw(5) << i << ", b=" << setw(5) << j
						<< " : " << cnt << ":" << endl;
				cout << "a=" << endl;
				A2->Group_element->element_print(Elt_a, cout);
				cout << endl;
				A2->Group_element->element_print_as_permutation(Elt_a, cout);
				cout << endl;
				cout << "b=" << endl;
				A2->Group_element->element_print(Elt_b, cout);
				cout << endl;
				A2->Group_element->element_print_as_permutation(Elt_b, cout);
				cout << endl;
				cout << "ab=" << endl;
				A2->Group_element->element_print(Elt_ab, cout);
				cout << endl;
				A2->Group_element->element_print_as_permutation(Elt_ab, cout);
				cout << endl;
			}
			cnt++;
		}
	}
	if (f_v) {
		cout << "group_theory_global::find_standard_generators "
				"we found " << cnt << " group elements with "
				"ord_a = " << order_a << " ord_b  = " << order_b
				<< " and ord_ab = " << order_ab << endl;
	}

	FREE_int(Elt_a);
	FREE_int(Elt_b);
	FREE_int(Elt_ab);
	if (f_v) {
		cout << "group_theory_global::find_standard_generators done" << endl;
	}
}


void group_theory_global::find_standard_generators_M24(
		groups::sims *H,
		actions::action *A1, actions::action *A2,
		int *Elt_a, int *Elt_b,
		int verbose_level)
// creates a sims object from the set of strong generators in Any_group
// the order of elements will be computed using the action A2
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_global::find_standard_generators_M24" << endl;
	}
#if 0
	groups::sims *H;
	groups::strong_generators *SG;

	SG = Any_group->get_strong_generators();

	H = SG->create_sims(verbose_level);
#endif

	algebra::ring_theory::longinteger_object go;
	long int goi;

	H->group_order(go);

	goi = go.as_lint();

	if (f_v) {
		cout << "group_theory_global::find_standard_generators_M24 "
				"group order H = " << goi << endl;
	}


	int *Elt_1;
	//int *Elt_a;
	//int *Elt_b;
	int *Elt_ab;
	int *Elt_x0;
	int *Elt_y0;
	int *Elt_x;
	int *Elt_y;
	int *Elt_xy;
	int *Elt_s;
	int *Elt_t;
	int *Elt_sv;
	int *Elt_tv;
	long int i, j, s, t, ord;
	int cnt_i, cnt_j, cnt_k;

	Elt_1 = NEW_int(A1->elt_size_in_int);
	//Elt_a = NEW_int(A1->elt_size_in_int);
	//Elt_b = NEW_int(A1->elt_size_in_int);
	Elt_ab = NEW_int(A1->elt_size_in_int);
	Elt_x0 = NEW_int(A1->elt_size_in_int);
	Elt_y0 = NEW_int(A1->elt_size_in_int);
	Elt_x = NEW_int(A1->elt_size_in_int);
	Elt_y = NEW_int(A1->elt_size_in_int);
	Elt_xy = NEW_int(A1->elt_size_in_int);
	Elt_s = NEW_int(A1->elt_size_in_int);
	Elt_t = NEW_int(A1->elt_size_in_int);
	Elt_sv = NEW_int(A1->elt_size_in_int);
	Elt_tv = NEW_int(A1->elt_size_in_int);



	other::orbiter_kernel_system::os_interface Os;



	cnt_i = 0;

	while (true) {

		i = Os.random_integer(goi);
		cnt_i++;


		H->element_unrank_lint(i, Elt_x0);


		ord = A2->Group_element->element_order(Elt_x0);

		if (ord == 10) {
			break;
		}
	}


	if (f_v) {
		cout << "group_theory_global::find_standard_generators_M24 "
				"Element x0 = " << setw(5) << i << " / "
				<< goi << " has order 10, found after " << cnt_i << " iterations" << endl;
		A1->Group_element->element_print(Elt_x0, cout);
		cout << endl;
	}


	A1->Group_element->element_power_int(
			Elt_x0, Elt_x,
			5, 0 /* verbose_level*/);


	ord = A2->Group_element->element_order(Elt_x);

	if (ord != 2) {
		cout << "group_theory_global::find_standard_generators_M24 "
				"Elt_x must have order 2" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "group_theory_global::find_standard_generators_M24 "
				"Element x has order 2" << endl;
		A1->Group_element->element_print(Elt_x, cout);
		cout << endl;
	}



	cnt_j = 0;

	while (true) {

		j = Os.random_integer(goi);
		cnt_j++;

		H->element_unrank_lint(j, Elt_y0);


		ord = A2->Group_element->element_order(Elt_y0);

		if (ord == 15) {
			break;
		}
	}

	if (f_v) {
		cout << "group_theory_global::find_standard_generators_M24 "
				"Element y0 = " << setw(5) << i << " / "
				<< goi << " has order 15, found after " << cnt_j << " iterations" << endl;
		A1->Group_element->element_print(Elt_y0, cout);
		cout << endl;
		//A->element_print_as_permutation(Elt_y0, cout);
		//cout << endl;
	}


	A1->Group_element->element_power_int(
			Elt_y0, Elt_y,
			5, 0 /* verbose_level*/);


	ord = A2->Group_element->element_order(Elt_y);

	if (ord != 3) {
		cout << "group_theory_global::find_standard_generators_M24 "
				"Elt_y must have order 3" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "group_theory_global::find_standard_generators_M24 "
				"Element y has order 3" << endl;
		A1->Group_element->element_print(Elt_y, cout);
		cout << endl;
	}



	cnt_k = 0;
	while (true) {

		s = Os.random_integer(goi);
		t = Os.random_integer(goi);
		cnt_k++;

		H->element_unrank_lint(s, Elt_s);
		H->element_unrank_lint(t, Elt_t);

		A1->Group_element->element_invert(
				Elt_s, Elt_sv, 0 /* verbose_level*/);
		A1->Group_element->element_invert(
				Elt_t, Elt_tv, 0 /* verbose_level*/);

		A1->Group_element->mult_abc(
				Elt_sv,
				Elt_x,
				Elt_s,
				Elt_a,
				0 /* verbose_level*/);

		A1->Group_element->mult_abc(
				Elt_tv,
				Elt_y,
				Elt_t,
				Elt_b,
				0 /* verbose_level*/);

		A1->Group_element->element_mult(
				Elt_a, Elt_b, Elt_ab, 0 /* verbose_level*/);

		ord = A2->Group_element->element_order(Elt_ab);

		if (ord == 23) {
			break;
		}
	}

	std::string word;

	word = "abababbababbabb"; // ab(ababb)^2abb

	A1->Group_element->evaluate_word_in_ab(
			Elt_a, Elt_b, Elt_1,
			word, 0 /* verbose_level*/);

	ord = A2->Group_element->element_order(Elt_1);


	if (ord == 5) {
		if (f_v) {
			cout << "group_theory_global::find_standard_generators_M24 "
					"Element " << word << " has order 5" << endl;
			A1->Group_element->element_print(Elt_y, cout);
			cout << endl;
			cout << "inverting b" << endl;
		}
		A1->Group_element->invert_in_place(Elt_b);
	}


	if (f_v) {
		cout << "group_theory_global::find_standard_generators_M24 "
				"Element a*b has order 23, after "
				"(" << cnt_i << "," << cnt_j << "," << cnt_k << ") "
						"iterations" << endl;

		cout << "group_theory_global::find_standard_generators_M24 "
				"a = " << endl;
		A1->Group_element->element_print(Elt_a, cout);
		cout << endl;

		cout << "group_theory_global::find_standard_generators_M24 "
				"b = " << endl;
		A1->Group_element->element_print(Elt_b, cout);
		cout << endl;

	}

#if 0
	if (f_v) {
		cout << "group_theory_global::find_standard_generators_M24 "
				"a = " << setw(5) << i << ", b=" << setw(5) << j
				<< " : " << cnt << ":" << endl;
		cout << "a=" << endl;
		A2->Group_element->element_print(Elt_a, cout);
		cout << endl;
		A2->Group_element->element_print_as_permutation(Elt_a, cout);
		cout << endl;
		cout << "b=" << endl;
		A2->Group_element->element_print(Elt_b, cout);
		cout << endl;
		A2->Group_element->element_print_as_permutation(Elt_b, cout);
		cout << endl;
		cout << "ab=" << endl;
		A2->Group_element->element_print(Elt_ab, cout);
		cout << endl;
		A2->Group_element->element_print_as_permutation(Elt_ab, cout);
		cout << endl;
	}
#endif


#if 0
	if (f_v) {
		cout << "group_theory_global::find_standard_generators_M24 "
				"we found " << cnt << " group elements with "
				"ord_a = " << order_a << " ord_b  = " << order_b
				<< " and ord_ab = " << order_ab << endl;
	}
#endif

	FREE_int(Elt_1);
	//FREE_int(Elt_a);
	//FREE_int(Elt_b);
	FREE_int(Elt_ab);
	FREE_int(Elt_x0);
	FREE_int(Elt_y0);
	FREE_int(Elt_x);
	FREE_int(Elt_y);
	FREE_int(Elt_xy);
	FREE_int(Elt_s);
	FREE_int(Elt_t);
	FREE_int(Elt_sv);
	FREE_int(Elt_tv);
	if (f_v) {
		cout << "group_theory_global::find_standard_generators_M24 done" << endl;
	}
}

void group_theory_global::compute_regular_representation(
		actions::action *A,
		groups::sims *S,
		data_structures_groups::vector_ge *SG, int *&perm,
		int verbose_level)
// this functions is not called from anywhere
// allocates perm[SG->len * goi]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_global::compute_regular_representation" << endl;
	}
	algebra::ring_theory::longinteger_object go;
	int goi, i;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	S->group_order(go);
	goi = go.as_int();

	if (f_v) {
	cout << "computing the regular representation of degree "
			<< go << ":" << endl;
	}

	perm = NEW_int(SG->len * goi);

	for (i = 0; i < SG->len; i++) {
		S->regular_representation(SG->ith(i),
				perm + i * goi, verbose_level);
	}

	if (f_v) {
		cout << endl;
		for (i = 0; i < SG->len; i++) {
			Combi.Permutations->perm_print_offset(
					cout,
				perm + i * goi, goi, 1 /* offset */,
				false /* f_print_cycles_of_length_one */,
				false /* f_cycle_length */, false, 0,
				true /* f_orbit_structure */,
				NULL, NULL);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "group_theory_global::compute_regular_representation done" << endl;
	}
}

#if 0
void group_theory_global::presentation(
		actions::action *A, groups::sims *S, int goi,
		data_structures_groups::vector_ge *gens, int *primes,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_global::presentation" << endl;
	}
	int *Elt1, *Elt2, *Elt3, *Elt4;
	int i, j, jj, k, l, a, b;
	int word[100];
	int *word_list;
	int *inverse_word_list;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);

	word_list = NEW_int(goi);
	inverse_word_list = NEW_int(goi);

	l = gens->len;

	if (f_v) {
		cout << "presentation of length " << l << endl;
		cout << "primes: ";
		Int_vec_print(cout, primes, l);
		cout << endl;
	}

#if 0
	// replace g5 by  g5 * g3:
	A->mult(gens->ith(5), gens->ith(3), Elt1);
	A->move(Elt1, gens->ith(5));

	// replace g7 by  g7 * g4:
	A->mult(gens->ith(7), gens->ith(4), Elt1);
	A->move(Elt1, gens->ith(7));
#endif



	for (i = 0; i < goi; i++) {
		inverse_word_list[i] = -1;
	}
	for (i = 0; i < goi; i++) {
		A->Group_element->one(Elt1);
		j = i;
		for (k = 0; k < l; k++) {
			b = j % primes[k];
			word[k] = b;
			j = j - b;
			j = j / primes[k];
		}
		for (k = 0; k < l; k++) {
			b = word[k];
			while (b) {
				A->Group_element->mult(Elt1, gens->ith(k), Elt2);
				A->Group_element->move(Elt2, Elt1);
				b--;
			}
		}

		A->Group_element->move(Elt1, Elt2);
		a = S->element_rank_lint(Elt2);
		word_list[i] = a;
		inverse_word_list[a] = i;

		if (f_v) {
			cout << "word " << i << " = ";
			Int_vec_print(cout, word, 9);
			cout << " gives " << endl;
			A->Group_element->print(cout, Elt1);
			cout << "which is element " << word_list[i] << endl;
			cout << endl;
		}
	}
	if (f_v) {
		cout << "i : word_list[i] : inverse_word_list[i]" << endl;
		for (i = 0; i < goi; i++) {
			cout << setw(5) << i << " : " << setw(5) << word_list[i]
				<< " : " << setw(5) << inverse_word_list[i] << endl;
		}
	}



	if (f_v) {
		for (i = 0; i < l; i++) {
			cout << "generator " << i << ":" << endl;
			A->Group_element->print(cout, gens->ith(i));
			cout << endl;
		}
	}
	for (i = 0; i < l; i++) {
		A->Group_element->move(gens->ith(i), Elt1);
		A->Group_element->element_power_int_in_place(Elt1, primes[i], 0);

		a = S->element_rank_lint(Elt1);

		j = inverse_word_list[a];
		for (k = 0; k < l; k++) {
			b = j % primes[k];
			word[k] = b;
			j = j - b;
			j = j / primes[k];
		}

		if (f_v) {
			cout << "generator " << i << " to the power " << primes[i]
				<< " is elt " << a << " which is word "
				<< inverse_word_list[a];
			Int_vec_print(cout, word, l);
			cout << " :" << endl;
			A->Group_element->print(cout, Elt1);
			cout << endl;
		}
	}


	for (i = 0; i < l; i++) {
		A->Group_element->move(gens->ith(i), Elt1);
		A->Group_element->invert(Elt1, Elt2);
		for (j = 0; j < i; j++) {
			A->Group_element->mult(Elt2, gens->ith(j), Elt3);
			A->Group_element->mult(Elt3, Elt1, Elt4);

			a = S->element_rank_lint(Elt4);
			jj = inverse_word_list[a];
			for (k = 0; k < l; k++) {
				b = jj % primes[k];
				word[k] = b;
				jj = jj - b;
				jj = jj / primes[k];
			}

			if (f_v) {
				cout << "g_" << j << "^{g_" << i << "} =" << endl;
				cout << "which is element " << a << " which is word "
					<< inverse_word_list[a] << " = ";
				Int_vec_print(cout, word, l);
				cout << endl;
				A->Group_element->print(cout, Elt4);
				cout << endl;
			}
		}
		if (f_v) {
			cout << endl;
		}
	}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);

	FREE_int(word_list);
	FREE_int(inverse_word_list);
	if (f_v) {
		cout << "group_theory_global::presentation done" << endl;
	}
}
#endif



void group_theory_global::permutation_representation_of_element(
		actions::action *A,
		std::string &element_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	string prefix;

	if (f_v) {
		cout << "group_theory_global::permutation_representation_of_element "
				"element_description=" << element_description << endl;
	}

	prefix = A->label + "_elt";

	Elt = NEW_int(A->elt_size_in_int);

	int *data;
	int data_len;


	Int_vec_scan(element_description, data, data_len);


	if (data_len != A->make_element_size) {
		cout << "data_len != A->make_element_size" << endl;
		exit(1);
	}

	A->Group_element->make_element(Elt, data, 0 /* verbose_level */);




	{
		string fname, title, author, extra_praeamble;

		fname = prefix + "_permutation.tex";
		title = "Permutation representation of element";


		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "group_theory_global::permutation_representation_of_element "
						"before report" << endl;
			}

			ost << "$$" << endl;
			A->Group_element->element_print_latex(Elt, ost);
			ost << "$$" << endl;

			ost << "$$" << endl;
			A->Group_element->element_print_as_permutation(Elt, ost);
			ost << "$$" << endl;

			if (f_v) {
				cout << "group_theory_global::permutation_representation_of_element "
						"after report" << endl;
			}


			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
	}


	FREE_int(data);




	if (f_v) {
		cout << "group_theory_global::permutation_representation_of_element done" << endl;
	}
}






void group_theory_global::representation_on_polynomials(
		group_constructions::linear_group *LG,
		algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level)
// creates an action object for the induced action on polynomials
// assumes a linear group
{
	int f_v = (verbose_level >= 1);
	//int f_stabilizer = true;
	//int f_draw_tree = true;


	if (f_v) {
		cout << "group_theory_global::representation_on_polynomials" << endl;
	}


	//field_theory::finite_field *F;
	actions::action *A;
	//matrix_group *M;
	int n;
	//int degree;
	algebra::ring_theory::longinteger_object go;

	A = LG->A_linear;
	//F = A->matrix_group_finite_field();
	A->group_order(go);

	n = A->matrix_group_dimension();
	int degree_of_poly;

	degree_of_poly = HPD->degree;

	if (f_v) {
		cout << "n = " << n << endl;
		cout << "degree_of_poly = " << degree_of_poly << endl;
	}

	if (f_v) {
		cout << "strong generators:" << endl;
		//A->Strong_gens->print_generators();
		A->Strong_gens->print_generators_tex();
	}

#if 0
	//ring_theory::homogeneous_polynomial_domain *HPD;

	HPD = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);


	monomial_ordering_type Monomial_ordering_type = t_PART;


	HPD->init(F, n /* nb_var */, degree_of_poly,
			Monomial_ordering_type,
			verbose_level);
#endif

	actions::action *A2;

	//A2 = NEW_OBJECT(actions::action);
	if (f_v) {
		cout << "group_theory_global::representation_on_polynomials "
				"before A->Induced_action->induced_action_on_homogeneous_polynomials" << endl;
	}
	A2 = A->Induced_action->induced_action_on_homogeneous_polynomials(
		HPD,
		false /* f_induce_action */, NULL,
		verbose_level);
	if (f_v) {
		cout << "group_theory_global::representation_on_polynomials "
				"after A->Induced_action->induced_action_on_homogeneous_polynomials" << endl;
	}

	if (f_v) {
		cout << "created action A2" << endl;
		A2->print_info();
	}


	induced_actions::action_on_homogeneous_polynomials *A_on_HPD;
	int *M;
	int nb_gens;
	int i;

	A_on_HPD = A2->G.OnHP;

	if (LG->f_has_nice_gens) {
		if (f_v) {
			cout << "group_theory_global::representation_on_polynomials "
					"using nice generators" << endl;
		}
		LG->nice_gens->matrix_representation(
				A_on_HPD, M, nb_gens, verbose_level);
	}
	else {
		if (f_v) {
			cout << "group_theory_global::representation_on_polynomials "
					"using strong generators" << endl;
		}
		LG->Strong_gens->gens->matrix_representation(
				A_on_HPD, M, nb_gens, verbose_level);
	}

	for (i = 0; i < nb_gens; i++) {
		cout << "matrix " << i << " / " << nb_gens << ":" << endl;
		Int_matrix_print(
				M + i * A_on_HPD->dimension * A_on_HPD->dimension,
				A_on_HPD->dimension, A_on_HPD->dimension);
	}

	for (i = 0; i < nb_gens; i++) {
		string fname;
		other::orbiter_kernel_system::file_io Fio;

		fname = LG->label + "_rep_" + std::to_string(degree_of_poly) + "_" + std::to_string(i) + ".csv";

		Fio.Csv_file_support->int_matrix_write_csv(
				fname, M + i * A_on_HPD->dimension * A_on_HPD->dimension,
				A_on_HPD->dimension, A_on_HPD->dimension);

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		fname = LG->label + "_rep_" + std::to_string(degree_of_poly) + "_" + std::to_string(i) + ".gap";

		Fio.int_matrix_write_cas_friendly(
				fname, M + i * A_on_HPD->dimension * A_on_HPD->dimension,
				A_on_HPD->dimension, A_on_HPD->dimension);

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;



	}
	if (f_v) {
		cout << "group_theory_global::representation_on_polynomials done" << endl;
	}
}



sims *group_theory_global::create_sims_for_subgroup_given_by_generator_ranks(
		actions::action *A,
		groups::sims *Big_group,
		long int *generators_by_rank, int nb_gens, long int subgroup_order,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_global::create_sims_for_subgroup_given_by_generator_ranks" << endl;
	}

	data_structures_groups::vector_ge *gen_vec;

	gen_vec = NEW_OBJECT(data_structures_groups::vector_ge);

	gen_vec->init(A, verbose_level);
	gen_vec->allocate(nb_gens, 0 /* verbose_level */);

	int i;

	for (i = 0; i < nb_gens; i++) {
		Big_group->element_unrank_lint(generators_by_rank[i], gen_vec->ith(i));
	}

	long int target_go;
	groups::sims *subgroup_sims;

	target_go = subgroup_order;

	if (f_v) {
		cout << "group_theory_global::create_sims_for_subgroup_given_by_generator_ranks "
				"before Subgroup_lattice->A->create_sims_from_generators_with_target_group_order_lint" << endl;
	}
	subgroup_sims = A->create_sims_from_generators_with_target_group_order_lint(
			gen_vec,
			target_go,
			verbose_level - 2);
	if (f_v) {
		cout << "group_theory_global::create_sims_for_subgroup_given_by_generator_ranks "
				"after Subgroup_lattice->A->create_sims_from_generators_with_target_group_order_lint" << endl;
	}


	FREE_OBJECT(gen_vec);

	if (f_v) {
		cout << "group_theory_global::create_sims_for_subgroup_given_by_generator_ranks done" << endl;
	}

	return subgroup_sims;
}


groups::strong_generators *group_theory_global::conjugate_strong_generators(
		groups::strong_generators *Strong_gens_in,
		int *Elt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_global::conjugate_strong_generators" << endl;
	}

	// apply the transformation to the set of generators:

	groups::strong_generators *Strong_gens_out;

	Strong_gens_out = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "group_theory_global::conjugate_strong_generators "
				"before Strong_gens_out->init_generators_for_the_conjugate_group_avGa" << endl;
	}
	Strong_gens_out->init_generators_for_the_conjugate_group_avGa(
			Strong_gens_in, Elt, verbose_level - 2);

	if (f_v) {
		cout << "group_theory_global::conjugate_strong_generators "
				"after Strong_gens_out->init_generators_for_the_conjugate_group_avGa" << endl;
	}

	if (f_v) {
		cout << "group_theory_global::conjugate_strong_generators done" << endl;
	}
	return Strong_gens_out;
}

geometry::algebraic_geometry::variety_object *group_theory_global::variety_apply_single_transformation(
		geometry::algebraic_geometry::variety_object *Variety_object_in,
		actions::action *A,
		actions::action *A_on_lines,
		int f_inverse,
		int *transformation_coeffs,
		int f_has_group, groups::strong_generators *Strong_gens_in,
		groups::strong_generators *&Strong_gens_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "group_theory_global::apply_single_transformation" << endl;
	}

	//actions::action *A;
	int *Elt1;
	int *Elt2;
	int *Elt3;

	//A = PA->A;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	if (f_v) {
		cout << "group_theory_global::apply_single_transformation "
				"before making elements" << endl;
	}

	A->Group_element->make_element(
			Elt1, transformation_coeffs, 0 /*verbose_level*/);

	if (f_inverse) {
		A->Group_element->element_invert(
				Elt1, Elt2, 0 /*verbose_level*/);
	}
	else {
		A->Group_element->element_move(
				Elt1, Elt2, 0 /*verbose_level*/);
	}

	//A->element_transpose(Elt2, Elt3, 0 /*verbose_level*/);

	A->Group_element->element_invert(
			Elt2, Elt3, 0 /*verbose_level*/);

	if (f_v) {
		cout << "group_theory_global::apply_single_transformation "
				"after making elements" << endl;
	}


	if (f_v) {
		cout << "group_theory_global::apply_transformations "
				"applying the transformation given by:" << endl;
		cout << "$$" << endl;
		A->Group_element->print_quick(cout, Elt2);
		cout << endl;
		cout << "$$" << endl;
		cout << "group_theory_global::apply_transformations "
				"The inverse is:" << endl;
		cout << "$$" << endl;
		A->Group_element->print_quick(cout, Elt3);
		cout << endl;
		cout << "$$" << endl;
	}


	geometry::projective_geometry::projective_space *P;


	P = Variety_object_in->Projective_space;

	int *eqn_out;
	int nb_monomials;
	int f_semilinear;
	int d, d2;

	nb_monomials = Variety_object_in->Ring->get_nb_monomials();

	d = A->matrix_group_dimension();

	f_semilinear = A->is_semilinear_matrix_group();

	if (f_v) {
		cout << "group_theory_global::apply_transformations "
				"f_semilinear = " << f_semilinear << endl;
		cout << "group_theory_global::apply_transformations "
				"d = " << d << endl;
	}

	d2 = d * d;

	eqn_out = NEW_int(nb_monomials);

	if (f_v) {
		cout << "group_theory_global::apply_transformations "
				"before Variety_object_in->Ring->substitute_semilinear" << endl;
	}
	Variety_object_in->Ring->substitute_semilinear(
			Variety_object_in->eqn /*coeff_in */,
			eqn_out /*coeff_out*/,
			f_semilinear,
			Elt3[d2] /*frob*/,
			Elt3 /* Mtx_inv*/,
			verbose_level);
	if (f_v) {
		cout << "group_theory_global::apply_transformations "
				"after Variety_object_in->Ring->substitute_semilinear" << endl;
	}


	Variety_object_in->Ring->get_F()->Projective_space_basic->PG_element_normalize_from_front(
			eqn_out, 1, nb_monomials);

	//Int_vec_copy(eqn15, QO->eqn15, 15);
	if (f_v) {
		cout << "group_theory_global::apply_transformations "
				"The equation of the transformed surface is:" << endl;
		cout << "$$" << endl;

		Variety_object_in->Ring->print_equation_with_line_breaks_tex(
				cout, eqn_out, 8 /* nb_terms_per_line*/,
				"\\\\\n" /* const char *new_line_text*/);

		//QCDA->Dom->print_equation_with_line_breaks_tex(cout, nb_monomials);
		cout << endl;
		cout << "$$" << endl;
	}




	if (f_has_group) {

		if (f_v) {
			cout << "group_theory_global::apply_transformations "
					"f_has_group is true" << endl;
		}

		// apply the transformation to the set of generators:

		groups::group_theory_global Group_theory_global;

		if (f_v) {
			cout << "group_theory_global::apply_transformations "
					"before Group_theory_global.conjugate_strong_generators" << endl;
		}
		Strong_gens_out = Group_theory_global.conjugate_strong_generators(
				Strong_gens_in,
				Elt2,
				verbose_level - 2);
		if (f_v) {
			cout << "group_theory_global::apply_transformations "
					"after Group_theory_global.conjugate_strong_generators" << endl;
		}

		//FREE_OBJECT(Sg);
		//Sg = SG2;

#if 0
		groups::strong_generators *SG2;

		SG2 = NEW_OBJECT(groups::strong_generators);
		if (f_v) {
			cout << "group_theory_global::apply_transformations "
					"before SG2->init_generators_for_the_conjugate_group_avGa" << endl;
		}
		SG2->init_generators_for_the_conjugate_group_avGa(
				Sg, Elt2, verbose_level);

		if (f_v) {
			cout << "group_theory_global::apply_transformations "
					"after SG2->init_generators_for_the_conjugate_group_avGa" << endl;
		}

		FREE_OBJECT(Sg);
		Sg = SG2;

		if (f_v) {
			cout << "group_theory_global::apply_transformations "
					"A->print_info()" << endl;
		}
		Sg->A->print_info();
#endif

		//QOG->Aut_gens = Sg;


		//f_has_nice_gens = false;
		// ToDo: need to conjugate nice_gens
	}
	else {

		if (f_v) {
			cout << "group_theory_global::apply_transformations "
					"f_has_group is false" << endl;
		}


	}

	long int *Lines_in;
	long int *Lines_out;
	int nb_lines;


	if (Variety_object_in->Line_sets) {
		if (f_v) {
			cout << "group_theory_global::apply_transformations "
					"applying transformation to lines" << endl;
		}
		if (false) {
			cout << "group_theory_global::apply_transformations "
					"lines = ";
			Lint_vec_print(cout, Variety_object_in->Line_sets->Sets[0], Variety_object_in->Line_sets->Set_size[0]);
			cout << endl;
		}


		nb_lines = Variety_object_in->Line_sets->Set_size[0];

		Lines_in = Variety_object_in->Line_sets->Sets[0];

		Lines_out = NEW_lint(nb_lines);


		int i;

		// apply the transformation to the set of lines:


		for (i = 0; i < nb_lines; i++) {
			if (false) {
				cout << "line " << i << ":" << endl;
				P->Subspaces->Grass_lines->print_single_generator_matrix_tex(
						cout, Lines_in[i]);
			}
			Lines_out[i] = A_on_lines->Group_element->element_image_of(
					Lines_in[i], Elt2, 0 /*verbose_level*/);
			if (false) {
				cout << "maps to " << endl;
				P->Subspaces->Grass_lines->print_single_generator_matrix_tex(
						cout, Lines_out[i]);
			}
		}
		if (f_v) {
			cout << "group_theory_global::apply_transformations "
					"applying transformation to lines done" << endl;
		}
	}
	else {
		nb_lines = 0;
		Lines_in = NULL;
		Lines_out = NULL;
	}

	// apply the transformation to the set of points:

	long int *Points_in;
	long int *Points_out;
	int nb_points;

	if (Variety_object_in->Point_sets) {

		if (f_v) {
			cout << "group_theory_global::apply_transformations "
					"applying transformation to points" << endl;
		}

		nb_points = Variety_object_in->Point_sets->Set_size[0];

		if (f_v) {
			cout << "group_theory_global::apply_transformations "
					"nb_points = " << nb_points << endl;
		}


		Points_in = Variety_object_in->Point_sets->Sets[0];

		if (f_v) {
			cout << "group_theory_global::apply_transformations "
					"before NEW_lint" << endl;
		}

 		Points_out = NEW_lint(nb_points);

		if (f_v) {
			cout << "group_theory_global::apply_transformations "
					"after NEW_lint" << endl;
		}


		int i;

		for (i = 0; i < nb_points; i++) {
			if (f_v) {
				cout << "group_theory_global::apply_transformations "
						"point" << i << endl;
			}
			if (f_v) {
				cout << "group_theory_global::apply_transformations "
						"point" << i << " = " << Points_in[i] << endl;
			}
			Points_out[i] = A->Group_element->element_image_of(
					Points_in[i], Elt2, 0 /*verbose_level*/);
			if (f_v) {
				cout << "group_theory_global::apply_transformations "
						"maps to " << Points_out[i] << endl;
			}
	#if 0
			int a;

			a = Surf->Poly3_4->evaluate_at_a_point_by_rank(
					coeffs_out, QO->Pts[i]);
			if (a) {
				cout << "group_theory_global::apply_transformations something is wrong, "
						"the image point does not lie on the transformed surface" << endl;
				exit(1);
			}
	#endif

		}

		if (f_v) {
			cout << "group_theory_global::apply_transformations "
					"before sorting points" << endl;
		}

		other::data_structures::sorting Sorting;

		Sorting.lint_vec_heapsort(Points_out, nb_points);

		if (f_v) {
			cout << "group_theory_global::apply_transformations "
					"applying transformation to points done" << endl;
		}

	}
	else {
		Points_in = NULL;
		Points_out = NULL;
		nb_points = 0;
	}


	geometry::algebraic_geometry::variety_object *Variety_object_out;


	Variety_object_out = NEW_OBJECT(geometry::algebraic_geometry::variety_object);


	std::string label_txt;
	std::string label_tex;

	label_txt = Variety_object_in->label_txt + "_t";
	label_tex = Variety_object_in->label_tex + "{\\rm \\_t}";

	if (f_v) {
		cout << "group_theory_global::apply_transformations "
				"before Variety_object_out->init_equation_and_points_and_lines_and_labels" << endl;
	}

	Variety_object_out->init_equation_and_points_and_lines_and_labels(
			P,
			Variety_object_in->Ring,
			eqn_out,
			Points_out, nb_points,
			Lines_out, nb_lines,
			label_txt,
			label_tex,
			verbose_level);

	if (f_v) {
		cout << "group_theory_global::apply_transformations "
				"after Variety_object_out->init_equation_and_points_and_lines_and_labels" << endl;
	}

	FREE_int(eqn_out);

	if (nb_points) {
		FREE_lint(Points_out);
	}
	if (nb_lines) {
		FREE_lint(Lines_out);
	}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	if (f_v) {
		cout << "group_theory_global::apply_single_transformation done" << endl;
	}
	return Variety_object_out;
}



}}}


