/*
 * cayley_graph_search.cpp
 *
 *  Created on: Oct 28, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_graph_theory {


void cayley_graph_search::init(
		int level,
		int group, int subgroup, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cayley_graph_search::init" << endl;
		}
	cayley_graph_search::level = level;
	cayley_graph_search::group = group;
	cayley_graph_search::subgroup = subgroup;

	Strong_gens_subgroup = NULL;

	degree = 0;
	data_size = 0;

	if (f_v) {
		cout << "cayley_graph_search::init "
				"before init_group" << endl;
		}
	init_group(verbose_level);
	if (f_v) {
		cout << "cayley_graph_search::init "
				"after init_group" << endl;
		}


	if (f_v) {
		cout << "cayley_graph_search::init "
				"before init_group2" << endl;
		}
	init_group2(verbose_level);
	if (f_v) {
		cout << "cayley_graph_search::init "
				"after init_group2" << endl;
		}

	int i, j;
	cout << "The elements of the subgroup are:" << endl;
	for (i = 0; i < go; i++) {
		if (f_subgroup[i]) {
			cout << i << " ";
			}
		}
	cout << endl;
	list_of_elements = NEW_int(go);
	list_of_elements_inverse = NEW_int(go);
	for (i = 0; i < go_subgroup; i++) {
		S_subgroup->element_unrank_lint(i, Elt1);
		cout << "Element " << setw(5) << i << " / "
				<< go_subgroup << ":" << endl;
		A->Group_element->element_print(Elt1, cout);
		j = S->element_rank_lint(Elt1);
		cout << "is element " << j << endl;
		list_of_elements[i] = j;
		}

#if 0
	cout << "generators:" << endl;
	for (i = 0; i < nb_generators; i++) {
		cout << "generator " << i << " / " << nb_generators
				<< " is " << generators[i] << endl;
		A->Group_element->element_print_quick(Strong_gens->gens->ith(i), cout);
		cout << endl;
		}
#endif

	for (i = 0; i < go_subgroup; i++) {
		S->element_unrank_lint(list_of_elements[i], Elt1);
		A->Group_element->element_mult(Elt1, Strong_gens->gens->ith(0), Elt2, 0);
		j = S->element_rank_lint(Elt2);
		list_of_elements[go_subgroup + i] = j;
		cout << "Element " << setw(5) << i << " / "
				<< go_subgroup << " * b = " << endl;
		A->Group_element->element_print(Elt2, cout);
		j = S->element_rank_lint(Elt1);
		cout << "is element " << j << endl;
		}

	for (i = 0; i < go; i++) {
		j = list_of_elements[i];
		list_of_elements_inverse[j] = i;
		}


	cout << "List of elements and inverse:" << endl;
	for (i = 0; i < go; i++) {
		cout << i << " : " << list_of_elements[i] << " : "
				<< list_of_elements_inverse[i] << endl;
		}



	if (f_v) {
		cout << "cayley_graph_search::init done" << endl;
		}


}

void cayley_graph_search::init_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cayley_graph_search::init_group" << endl;
		}
	if (level == 3) {
		init_group_level_3(verbose_level);
		}
	else if (level == 4) {
		init_group_level_4(verbose_level);
		}
	else if (level == 5) {
		init_group_level_5(verbose_level);
		}
	else {
		cout << "cayley_graph_search::init illegal level" << endl;
		cout << "level = " << level << endl;
		exit(1);
		}
	if (f_v) {
		cout << "cayley_graph_search::init_group done" << endl;
		}

}

void cayley_graph_search::init_group2(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "cayley_graph_search::init_group2" << endl;
		}
	S = Strong_gens->create_sims(0 /* verbose_level */);
	S->print_all_group_elements();

	if (Strong_gens_subgroup == NULL) {
		cout << "Strong_gens_subgroup = NULL" << endl;
		exit(1);
		}

	S_subgroup = Strong_gens_subgroup->create_sims(
			0 /* verbose_level */);


	f_has_order2 = NEW_int(go);
	f_subgroup = NEW_int(go);

	Int_vec_zero(f_subgroup, go);

	if (level == 4) {
		if (group == 2 || group == 3 || group == 5) {
			for (i = 0; i < go_subgroup; i++) {
				S_subgroup->element_unrank_lint(i, Elt1);
				cout << "Element " << setw(5) << i << " / "
						<< go_subgroup << ":" << endl;
				A->Group_element->element_print(Elt1, cout);
				j = S->element_rank_lint(Elt1);
				f_subgroup[j] = true;
				}
			}
		else if (group == 4) {
			for (i = 0; i < go; i++) {
				S->element_unrank_lint(i, Elt1);
				cout << "Element " << setw(5) << i << " / "
						<< go << ":" << endl;
				A->Group_element->element_print(Elt1, cout);
				cout << endl;
				if (F->Linear_algebra->is_identity_matrix(Elt1, 4)) {
					f_subgroup[i] = true;
					}
				else {
					f_subgroup[i] = false;
					}
				}
			}
		}
	else if (level == 5) {
		for (i = 0; i < go_subgroup; i++) {
			S_subgroup->element_unrank_lint(i, Elt1);
			cout << "Element " << setw(5) << i << " / "
					<< go_subgroup << ":" << endl;
			A->Group_element->element_print(Elt1, cout);
			j = S->element_rank_lint(Elt1);
			f_subgroup[j] = true;
			}
		}


	nb_involutions = 0;
	for (i = 0; i < go; i++) {
		S->element_unrank_lint(i, Elt1);
		cout << "Element " << setw(5) << i << " / "
				<< go << ":" << endl;
		A->Group_element->element_print(Elt1, cout);
		cout << endl;
		ord = A->Group_element->element_order(Elt1);
		if (ord == 2) {
			f_has_order2[i] = true;
			nb_involutions++;
			}
		else {
			f_has_order2[i] = false;
			}
		}

	cout << "We found " << nb_involutions << " involutions" << endl;




#if 1

	Table = NEW_OBJECT(data_structures_groups::group_table_and_generators);


	if (f_v) {
		cout << "modified_group_create::create_automorphism_group "
				"before Table->init" << endl;
	}

	Table->init(
			S,
			Strong_gens->gens,
			verbose_level);

	if (f_v) {
		cout << "modified_group_create::create_automorphism_group "
				"after Table->init" << endl;
	}

#if 0
	nb_generators = Strong_gens->gens->len;
	generators = NEW_int(nb_generators);
	for (i = 0; i < nb_generators; i++) {
		generators[i] = S->element_rank_lint(Strong_gens->gens->ith(i));
	}

	S->create_group_table(Table, go, verbose_level);
#endif

	fname_base = "Ferdinand" + std::to_string(level) + "_" + std::to_string(group);

	//Aut = NEW_OBJECT(actions::action);
	interfaces::magma_interface Magma;

	Magma.compute_automorphism_group_from_group_table(
			fname_base,
			Table,
		Aut,
		Aut_gens,
		verbose_level);

	Aut_gens->group_order(Aut_order);
#endif

	//FREE_OBJECT(Table);


#if 1

	//A2 = NEW_OBJECT(actions::action);
	A2 = S->A->Induced_action->induced_action_by_right_multiplication(
		false /* f_basis */, S,
		S /* Base_group */, false /* f_ownership */,
		verbose_level);
#endif


	if (f_v) {
		cout << "cayley_graph_search::init_group2 done" << endl;
		}
}

void cayley_graph_search::init_group_level_3(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "cayley_graph_search::init_group_level_3" << endl;
		}
	target_depth = 5;
	go = 16;
	degree = 6;
	data_size = degree;


	A = NEW_OBJECT(actions::action);

	int f_no_base = false;

	A->Known_groups->init_permutation_group(degree, f_no_base, verbose_level);


	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);


	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init(A, verbose_level - 2);

	if (group == 1) {
		int data[] = {
			1,2,3,0,4,5, // (0,1,2,3)
			2,1,0,3,4,5, // (0,2)
			0,1,2,3,5,4, // (4,5)
			};

		gens->allocate(3, verbose_level - 2);

		for (i = 0; i < 3; i++) {
			A->Group_element->make_element(Elt1,
					data + i * data_size,
					0 /*verbose_level*/);
			A->Group_element->element_move(Elt1, gens->ith(i), 0);
			}
		}
	else {
		cout << "illegal group" << endl;
		exit(1);
		}
	go_subgroup = go / 2;
	target_go.create(go);
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		gens, Strong_gens,
		verbose_level);
	Strong_gens->print_generators(cout, verbose_level - 1);
	if (f_v) {
		cout << "cayley_graph_search::init_group_level_3 done" << endl;
		}
}

void cayley_graph_search::init_group_level_4(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cayley_graph_search::init_group_level_4" << endl;
		}
	target_depth = 32;
	go = 32;

	int i, j;

	A = NEW_OBJECT(actions::action);


	if (group == 2) {
		degree = 10;
		data_size = degree;
		}
	else if (group == 3) {
		degree = 8;
		data_size = degree;
		}
	else if (group == 4) {
		degree = 0;
		data_size = 20;
		}
	else if (group == 5) {
		degree = 12;
		data_size = degree;
		}


	if (degree) {
		int f_no_base = false;
		A->Known_groups->init_permutation_group(degree, f_no_base, verbose_level);
		}
	else if (group == 4) {
		int q = 2;
		data_structures_groups::vector_ge *nice_gens;

		F = NEW_OBJECT(field_theory::finite_field);
		F->finite_field_init_small_order(q,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				0);
		A->Known_groups->init_affine_group(4, F,
			false /* f_semilinear */,
			true /* f_basis */, true /* f_init_sims */,
			nice_gens,
			verbose_level);
		FREE_OBJECT(nice_gens);
		}
	else {
		cout << "group " << group << " not yet implemented" << endl;
		exit(1);
		}

	go_subgroup = go / 2;



	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens_subgroup = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init(A, verbose_level - 2);
	gens_subgroup->init(A, verbose_level - 2);

	if (group == 2) {
		int data[] = { // C_4 x C_2 x C_2 x C_2
			1,2,3,0,4,5,6,7,8,9, // (0,1,2,3)
			0,1,2,3,5,4,6,7,8,9, // (4,5)
			0,1,2,3,4,5,7,6,8,9, // (6,7)
			0,1,2,3,4,5,6,7,9,8, // (8,9)
			};
		int data_subgroup[] = {
			2,3,0,1,4,5,6,7,8,9, // (0,2)(1,3)
			0,1,2,3,5,4,6,7,8,9, // (4,5)
			0,1,2,3,4,5,7,6,8,9, // (6,7)
			0,1,2,3,4,5,6,7,9,8, // (8,9)
			};

		gens->allocate(4, verbose_level - 2);

		for (i = 0; i < 4; i++) {
			A->Group_element->make_element(Elt1,
					data + i * data_size,
					0 /*verbose_level*/);
			A->Group_element->element_move(Elt1, gens->ith(i), 0);
			}
		gens_subgroup->allocate(4, verbose_level - 2);

		for (i = 0; i < 4; i++) {
			A->Group_element->make_element(Elt1,
					data_subgroup + i * data_size,
					0 /*verbose_level*/);
			A->Group_element->element_move(Elt1, gens_subgroup->ith(i), 0);
			}
		}
	else if (group == 3) {
		int data[] = { // D_8 x C_2 x C_2
			1,2,3,0,4,5,6,7, // (0,1,2,3)
			2,1,0,3,4,5,6,7, // (0,2)
			0,1,2,3,5,4,6,7, // (4,5)
			0,1,2,3,4,5,7,6, // (6,7)
			};
		int data_subgroup1[] = {
			2,1,0,3,4,5,6,7, // (0,2)
			0,3,2,1,4,5,6,7, // (1,3)
			0,1,2,3,5,4,6,7, // (4,5)
			0,1,2,3,4,5,7,6, // (6,7)
			};
		int data_subgroup2[] = {
			1,0,3,2,4,5,6,7, // (0,1)(2,3)
			3,2,1,0,4,5,6,7, // (0,3)(1,2)
			0,1,2,3,5,4,6,7, // (4,5)
			0,1,2,3,4,5,7,6, // (6,7)
			};

		gens->allocate(4, verbose_level - 2);

		for (i = 0; i < 4; i++) {
			A->Group_element->make_element(Elt1,
					data + i * data_size,
					0 /*verbose_level*/);
			A->Group_element->element_move(Elt1, gens->ith(i), 0);
			}
		gens_subgroup->allocate(4, verbose_level - 2);


		if (subgroup == 1) {
			for (i = 0; i < 4; i++) {
				A->Group_element->make_element(Elt1,
						data_subgroup1 + i * data_size,
						0 /*verbose_level*/);
				A->Group_element->element_move(Elt1, gens_subgroup->ith(i), 0);
				}
			}
		else if (subgroup == 2) {
			for (i = 0; i < 4; i++) {
				A->Group_element->make_element(Elt1,
						data_subgroup2 + i * data_size,
						0 /*verbose_level*/);
				A->Group_element->element_move(Elt1, gens_subgroup->ith(i), 0);
				}
			}
		else {
			cout << "unkown subgroup" << endl;
			exit(1);
			}
		}
	else if (group == 4) {
		int data[] = {
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 1,0,0,0,
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 0,1,0,0,
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 0,0,1,0,
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 0,0,0,1,
			1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1, 0,0,0,0
			};

		int data_subgroup[] = {
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 1,0,0,0,
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 0,1,0,0,
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 0,0,1,0,
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 0,0,0,1,
			};

		gens->allocate(5, verbose_level - 2);
		for (i = 0; i < 5; i++) {
			A->Group_element->make_element(Elt1, data + i * data_size, 0 /*verbose_level*/);
			A->Group_element->element_move(Elt1, gens->ith(i), 0);
			}

		gens_subgroup->allocate(4, verbose_level - 2);
		for (i = 0; i < 4; i++) {
			A->Group_element->make_element(Elt1, data_subgroup + i * data_size, 0 /*verbose_level*/);
			A->Group_element->element_move(Elt1, gens_subgroup->ith(i), 0);
			}


		}
	else if (group == 5) {
		const char *data_str[] = {
			"(1,2)(5,8,6,7)(9,12,10,11)",
			"(1,4)(2,3)(5,12)(6,11)(7,10)(8,9)",
			"(5,10)(6,9)(7,12)(8,11)",
			"(1,2)(3,4)(5,6)(7,8)(9,10)(11,12)",
			"(5,6)(7,8)(9,10)(11,12)"
			};
		data_structures::string_tools ST;

		gens->allocate(5, verbose_level - 2);

		for (i = 0; i < 5; i++) {
			int *perm;
			int degree;
			string s;

			s.assign(data_str[i]);

			ST.scan_permutation_from_string(
					s, perm, degree,
					0 /* verbose_level */);
			cout << "degree=" << degree << endl;
			for (j = 0; j < degree; j++) {
				cout << perm[j] << " ";
				}
			cout << " : ";

			for (j = 1; j < degree; j++) {
				perm[j - 1] = perm[j] - 1;
				}
			for (j = 0; j < degree - 1; j++) {
				cout << perm[j] << " ";
				}
			cout << endl;

			A->Group_element->make_element(Elt1, perm, 0 /*verbose_level*/);
			A->Group_element->element_move(Elt1, gens->ith(i), 0);
			}
		const char *data_subgroup_str[] = {
			"(1,4)(2,3)(5,8)(6,7)(9,12)(10,11)",
			"(1,4)(2,3)(5,12)(6,11)(7,10)(8,9)",
			"(1,2)(3,4)(12)",
			"(5,6)(7,8)(9,10)(11,12)"
			};

		gens_subgroup->allocate(4, verbose_level - 2);

		for (i = 0; i < 4; i++) {
			int *perm;
			int degree;
			string s;

			s.assign(data_subgroup_str[i]);

			ST.scan_permutation_from_string(
					s, perm, degree,
					0 /* verbose_level */);
			for (j = 0; j < degree; j++) {
				cout << perm[j] << " ";
				}
			cout << " : ";

			for (j = 1; j < degree; j++) {
				perm[j - 1] = perm[j] - 1;
				}
			for (j = 0; j < degree - 1; j++) {
				cout << perm[j] << " ";
				}
			cout << endl;


			A->Group_element->make_element(Elt1, perm, 0 /*verbose_level*/);
			A->Group_element->element_move(Elt1, gens_subgroup->ith(i), 0);
			}
		}
	else {
		cout << "illegal group" << endl;
		exit(1);
		}
	target_go.create(go);
	target_go_subgroup.create(go_subgroup);

	cout << "creating generators for the group:" << endl;
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		gens, Strong_gens,
		verbose_level);

	cout << "creating generators for the subgroup:" << endl;
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go_subgroup,
		gens_subgroup, Strong_gens_subgroup,
		verbose_level);

	if (f_v) {
		cout << "cayley_graph_search::init_group_level_4 done" << endl;
		}

}

void cayley_graph_search::init_group_level_5(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cayley_graph_search::init_group_level_5" << endl;
		}

	target_depth = 15;
	go = 64;

	int i;

	A = NEW_OBJECT(actions::action);


	if (group == 1) {
		degree = 0;
		data_size = 30;
		}
	else {
		cout << "unknown type of group" << endl;
		}


	if (group == 1) {
		int q = 2;
		data_structures_groups::vector_ge *nice_gens;

		F = NEW_OBJECT(field_theory::finite_field);
		F->finite_field_init_small_order(q,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				0);
		A->Known_groups->init_affine_group(5, F,
			false /* f_semilinear */,
			true /* f_basis */, true /* f_init_sims */,
			nice_gens,
			verbose_level);
		FREE_OBJECT(nice_gens);
		}
	else {
		cout << "group " << group << " not yet implemented" << endl;
		exit(1);
		}

	go_subgroup = go / 2;



	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens_subgroup = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init(A, verbose_level - 2);
	gens_subgroup->init(A, verbose_level - 2);


	if (group == 1) {
		int data[] = {
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 1,0,0,0,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,1,0,0,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,0,1,0,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,0,0,1,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,0,0,0,1,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1, 0,0,0,0,0,
			};

		int data_subgroup[] = {
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 1,0,0,0,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,1,0,0,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,0,1,0,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,0,0,1,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,0,0,0,1,
			};

		gens->allocate(6, verbose_level - 2);
		for (i = 0; i < 6; i++) {
			A->Group_element->make_element(Elt1, data + i * data_size, 0 /*verbose_level*/);
			A->Group_element->element_move(Elt1, gens->ith(i), 0);
			}

		gens_subgroup->allocate(5, verbose_level - 2);
		for (i = 0; i < 5; i++) {
			A->Group_element->make_element(Elt1, data_subgroup + i * data_size, 0 /*verbose_level*/);
			A->Group_element->element_move(Elt1, gens_subgroup->ith(i), 0);
			}


		}
	else {
		cout << "illegal group" << endl;
		exit(1);
		}
	target_go.create(go);
	target_go_subgroup.create(go_subgroup);

	cout << "creating generators for the group:" << endl;
	A->generators_to_strong_generators(
		false /* f_target_go */, target_go,
		gens, Strong_gens,
		verbose_level);

	ring_theory::longinteger_object go1;
	Strong_gens->group_order(go1);
	cout << "go1=" << go1 << endl;
	//exit(1);


	cout << "creating generators for the subgroup:" << endl;
	A->generators_to_strong_generators(
		false /* f_target_go */, target_go_subgroup,
		gens_subgroup, Strong_gens_subgroup,
		verbose_level);

	if (f_v) {
		cout << "cayley_graph_search::init_group_level_5 done" << endl;
		}

}

int cayley_graph_search::incremental_check_func(
		int len, long int *S, int verbose_level)
{
	int f_OK = true;
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int a;

	if (f_v) {
		cout << "checking set ";
		Lint_vec_print(cout, S, len);
		cout << " (incrementally)";
		}
	if (len) {
		a = S[len - 1];
		if (a == 0) {
			f_OK = false;
			}
		if (f_has_order2[a] == false) {
			f_OK = false;
			}
		}
	if (f_OK) {
		if (f_v) {
			cout << "OK" << endl;
			}
		return true;
		}
	else {
		return false;
		}
}



void cayley_graph_search::classify_subsets(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cayley_graph_search::classify_subsets" << endl;
		}





	prefix = "Ferdinand" + std::to_string(group) + "_" + std::to_string(group);

	cout << "classifying subsets:" << endl;

	Control = NEW_OBJECT(poset_classification::poset_classification_control);
	Control->f_W = true;
	Control->f_w = true;

	Control->problem_label = prefix;
	Control->f_problem_label = true;

	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	Poset->init_subset_lattice(Aut, Aut,
			Aut_gens,
			verbose_level);

	gen = NEW_OBJECT(poset_classification::poset_classification);

	gen->compute_orbits_on_subsets(
		target_depth,
		//prefix,
		//f_W, f_w,
		Control,
		Poset,
		// ToDo
		//NULL /* ferdinand3_early_test_func */,
		//NULL /* void *early_test_func_data */,
		//ferdinand_incremental_check_func /* int (*candidate_incremental_check_func)(int len, int *S, void *data, int verbose_level)*/,
		//this /* void *candidate_incremental_check_data */,
		verbose_level);



	if (f_v) {
		cout << "cayley_graph_search::classify_subsets "
				"done" << endl;
		}

}



void cayley_graph_search::write_file(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cayley_graph_search::write_file" << endl;
		}

	int nb_orbits;
	int sz;
	int i, j;
	int cnt = 0;

	for (sz = 1; sz <= target_depth; sz++) {

		fname_graphs = "ferdinand" + std::to_string(level) + "_" + std::to_string(group)
				+ "_subgroup_" + std::to_string(subgroup) + "_graphs_sz_" + std::to_string(sz) + ".txt";


		{
		ofstream fp(fname_graphs);

		for (i = 0; i < go; i++) {
			S->element_unrank_lint(i, Elt1);
			cout << "Element " << setw(5) << i << " / "
					<< go << ":" << endl;
			A->Group_element->element_print(Elt1, cout);
			cout << endl;

			fp << "[";
			A2->Group_element->element_print_as_permutation(Elt1, fp);
			fp << "],";
			}
		fp << endl;
		for (i = 0; i < go; i++) {
			S->element_unrank_lint(i, Elt1);
			cout << "Element " << setw(5) << i << " / "
					<< go << ":" << endl;
			A->Group_element->element_print(Elt1, cout);
			cout << endl;

			if (f_subgroup[i]) {
				fp << "[";
				A2->Group_element->element_print_as_permutation(Elt1, fp);
				fp << "],";
				}
			}
		fp << endl;

		int n;
		long int *Adj;

		Adj = NEW_lint(go * sz);

		nb_orbits = gen->nb_orbits_at_level(sz);
		cout << "We found " << nb_orbits << " orbits on "
				<< sz << "-subsets" << endl;
		//fp << "[";
#if 0
		for (n = 0; n < nb_orbits; n++) {

			set_and_stabilizer *SaS;

			SaS = gen->get_set_and_stabilizer(sz, n, 0 /*verbose_level*/);

			cout << n << " / " << nb_orbits << " : ";
			SaS->print_set_tex(cout);
			cout << "  # " << cnt << endl;
			cout << endl;
			}
#endif
		for (n = 0; n < nb_orbits; n++) {

			data_structures_groups::set_and_stabilizer *SaS;

			SaS = gen->get_set_and_stabilizer(sz, n, 0 /*verbose_level*/);

			if ((n % 1000) == 0) {
				cout << n << " / " << nb_orbits << " : ";
				SaS->print_set_tex(cout);
				cout << "  # " << cnt << endl;
				cout << endl;
				}


			create_Adjacency_list(Adj,
				SaS->data, sz,
				verbose_level);
			//cout << "The adjacency sets are:" << endl;
			//print_integer_matrix_with_standard_labels(cout,
			//Adj, go, sz, false /* f_tex */);
			fp << "{";
			for (i = 0; i < go; i++) {
				fp << i << ":[";
				for (j = 0; j < sz; j++) {
					fp << Adj[i * sz + j];
					if (j < sz - 1) {
						fp << ",";
						}
					}
				fp << "]";
#if 1
				if (i < go - 1) {
						fp << ",";
					}
#endif
				}
			fp << "}";

			//action *create_automorphism_group_of_graph(
			//int *Adj, int n, int verbose_level);
#if 0
			if (n < nb_orbits - 1) {
				fp << ", ";
				}
#endif
			fp << "  # " << cnt << endl;
			cnt++;

			delete SaS;
			}
		//fp << "];" << endl;

		FREE_lint(Adj);
		} // end of fp

		orbiter_kernel_system::file_io Fio;

	cout << "written file " << fname_graphs << " of size "
			<< Fio.file_size(fname_graphs) << endl;
	} // next sz


#if 0
	int set[5] = {6,7,8,10,15};
	int canonical_set[5];
	int *Elt;
	int orb;

	Elt = NEW_int(A->elt_size_in_int);

	orb = gen->trace_set(set, 5, 5,
		canonical_set, Elt,
		0 /*verbose_level */);
	cout << "canonical set : ";
	int_vec_print(cout, canonical_set, 5);
	cout << endl;
	cout << "orb=" << orb << endl;
	cout << "transporter : ";
	Aut->element_print(Elt, cout);
	//A->element_print(Elt, cout);
	cout << endl;
#endif

	if (f_v) {
		cout << "cayley_graph_search::write_file done" << endl;
		}
}

void cayley_graph_search::create_Adjacency_list(
		long int *Adj,
	long int *connection_set, int connection_set_sz,
	int verbose_level)
// Adj[go * connection_set_sz]
{
	int f_v = false;//(verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "cayley_graph_search::create_Adjacency_list" << endl;
		}
	for (i = 0; i < go; i++) {
		S->element_unrank_lint(i, Elt1);

#if 0
		cout << "Element " << setw(5) << i << " / "
				<< go << ":" << endl;

		cout << "set: ";
		int_vec_print(cout, connection_set, connection_set_sz);
		cout << endl;
#endif

		A2->Group_element->map_a_set_and_reorder(connection_set,
				Adj + i * connection_set_sz,
				connection_set_sz, Elt1,
				0 /* verbose_level */);

#if 0
		cout << "image_set: ";
		int_vec_print(cout,
				Adj + i * connection_set_sz, connection_set_sz);
		cout << endl;
#endif

		}
#if 0
	//cout << "The adjacency sets are:" << endl;
	//print_integer_matrix_with_standard_labels(cout,
	//Adj, go, sz, false /* f_tex */);
	cout << "{";
	for (i = 0; i < go; i++) {
		cout << i << ":[";
		for (j = 0; j < sz; j++) {
			cout << Adj[i * connection_set_sz + j];
			if (j < connection_set_sz - 1) {
				cout << ",";
				}
			}
		cout << "]";
#if 1
		if (i < go - 1) {
				cout << ",";
			}
#endif
		}
	cout << "}";
#endif

	if (f_v) {
		cout << "cayley_graph_search::create_Adjacency_list "
				"done" << endl;
		}
}

void cayley_graph_search::create_additional_edges(
	long int *Additional_neighbor,
	int *Additional_neighbor_sz,
	long int connection_element,
	int verbose_level)
// Additional_neighbor[go], Additional_neighbor_sz[go]
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "cayley_graph_search::create_Adjacency_list" << endl;
	}
	for (i = 0; i < go; i++) {
		Additional_neighbor_sz[i] = 0;
	}
	//S->element_unrank_int(connection_element, Elt2);
	for (i = 0; i < go; i++) {
		if (!f_subgroup[i]) {
			continue;
		}
		S->element_unrank_lint(i, Elt1);
		Additional_neighbor[i] = A2->Group_element->element_image_of(
				connection_element, Elt1, 0 /* verbose_level */);
		Additional_neighbor_sz[i]++;
	}
}



}}}


