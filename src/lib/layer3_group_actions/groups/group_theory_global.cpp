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

}


group_theory_global::~group_theory_global()
{

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
	ring_theory::longinteger_object go;

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
	ring_theory::longinteger_object go;

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
	orbiter_kernel_system::file_io Fio;
	int d;
	field_theory::finite_field *F;


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

	linear_algebra::gl_classes C;
	linear_algebra::gl_class_rep *R;
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

	data_structures::tally T_order;

	T_order.init(Order, nb_classes, false, 0);


	{
		ofstream ost(fname_output);
		l1_interfaces::latex_interface L;

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
	ring_theory::longinteger_object go;
	int i, d, q, cnt, ord, order;
	number_theory::number_theory_domain NT;

	if (!A1->is_matrix_group()) {
		cout << "group_theory_global::find_singer_cycle "
				"needs matrix group" << endl;
		exit(1);
	}
	algebra::matrix_group *M;

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





}}}


