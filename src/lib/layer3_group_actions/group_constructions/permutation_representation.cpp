/*
 * permutation_representation.cpp
 *
 *  Created on: Aug 22, 2019
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;



namespace orbiter {
namespace layer3_group_actions {
namespace group_constructions {



permutation_representation::permutation_representation()
{
	Record_birth();
	A_original = NULL;
	f_stay_in_the_old_action = false;
	nb_gens = 0;
	gens = NULL;
	Perms = NULL;
	degree = 0;
	//longinteger_object target_go;
	P = NULL;
	perm_offset = 0;
	elt_size_int = 0;
	make_element_size = 0;
	char_per_elt = 0;
	elt1 = NULL;
	label[0] = 0;
	label_tex[0] = 0;
	Page_storage = NULL;
	Elts = NULL;
	//null();
}

permutation_representation::~permutation_representation()
{
	Record_death();
	if (P) {
		FREE_OBJECT(P);
	}
	if (elt1) {
		FREE_char((char *) elt1);
	}
	if (Page_storage) {
		FREE_OBJECT(Page_storage);
	}
	if (Elts) {
		FREE_int(Elts);
	}
	//free();
}

void permutation_representation::init(
		actions::action *A_original,
		int f_stay_in_the_old_action,
		data_structures_groups::vector_ge *gens,
		int *Perms, int degree,
		int verbose_level)
// Perms is degree x nb_gens
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "permutation_representation::init "
				"A_original=" << A_original->label << endl;
		cout << "permutation_representation::init "
				"f_stay_in_the_old_action=" << f_stay_in_the_old_action << endl;
	}
	permutation_representation::A_original = A_original;
	permutation_representation::f_stay_in_the_old_action = f_stay_in_the_old_action;
	permutation_representation::gens = gens;
	nb_gens = gens->len;
	permutation_representation::Perms = Perms;
	permutation_representation::degree = degree;
	P = NEW_OBJECT(permutation_representation_domain);
	P->init(degree, 10 /* page_length_log */, verbose_level - 2);
	perm_offset = A_original->elt_size_in_int;
	elt_size_int = perm_offset + P->elt_size_int;
	make_element_size = A_original->make_element_size + degree;

	label = A_original->label + "_perm_rep_deg" + std::to_string(degree);

	label_tex = A_original->label_tex + " degree " + std::to_string(degree);

	char_per_elt = A_original->coded_elt_size_in_char + char_per_elt;
	elt1 = (uchar *) NEW_char(char_per_elt);

	Page_storage = NEW_OBJECT(other::data_structures::page_storage);
	Page_storage->init(char_per_elt /* entry_size */,
			10 /* page_length_log */, verbose_level);

	int i, j;

	Elts = NEW_int(nb_gens * elt_size_int);
	for (i = 0; i < nb_gens; i++) {
		Int_vec_copy(gens->ith(i), Elts + i * elt_size_int, A_original->elt_size_in_int);
		for (j = 0; j < degree; j++) {
			Elts[i * elt_size_int + perm_offset + j] = Perms[j * nb_gens + i];
		}
	}

	if (f_v) {
		cout << "permutation_representation::init done" << endl;
	}
}

long int permutation_representation::element_image_of(
		int *Elt,
		long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int b;

	if (f_v) {
		cout << "permutation_representation::element_image_of" << endl;
	}
	if (f_stay_in_the_old_action) {
		if (f_v) {
			cout << "permutation_representation::element_image_of "
					"using the old action" << endl;
		}
		b = A_original->Group_element->element_image_of(a, Elt, verbose_level);
	}
	else {
		if (f_v) {
			cout << "permutation_representation::element_image_of "
					"using the permutation representation (new action)" << endl;
		}
		b = Elt[perm_offset + a];
		if (f_v) {
			cout << "permutation_representation::element_image_of " << a
					<< " maps to " << b << endl;
		}
	}
	return b;
}

void permutation_representation::element_one(
		int *Elt)
{
	int verbose_level = 0;

	A_original->Group_element->element_one(Elt, verbose_level);
	P->one(Elt + perm_offset);
}

int permutation_representation::element_is_one(
		int *Elt)
{
		if (!P->is_one(Elt)) {
			return false;
		}
		return true;
}

void permutation_representation::element_mult(
		int *A, int *B, int *AB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "permutation_representation::element_mult" << endl;
	}
	A_original->Group_element->element_mult(A, B, AB, verbose_level);
	Combi.Permutations->perm_mult(A + perm_offset, B + perm_offset, AB + perm_offset, degree);
	if (f_v) {
		cout << "permutation_representation::element_mult done" << endl;
	}
}

void permutation_representation::element_move(
		int *A, int *B, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "permutation_representation::element_move" << endl;
	}
	Int_vec_copy(A, B, elt_size_int);
	if (f_v) {
		cout << "permutation_representation::element_move done" << endl;
	}
}

void permutation_representation::element_invert(
		int *A, int *Av, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "permutation_representation::element_invert" << endl;
	}
	A_original->Group_element->element_invert(A, Av, verbose_level);
	Combi.Permutations->perm_inverse(A + perm_offset, Av + perm_offset, degree);
	if (f_v) {
		cout << "permutation_representation::element_invert done" << endl;
	}
}

void permutation_representation::element_pack(
		int *Elt, uchar *elt)
{
	cout << "permutation_representation::element_pack not yet implemented" << endl;
}

void permutation_representation::element_unpack(
		uchar *elt, int *Elt)
{
	cout << "permutation_representation::element_unpack not yet implemented" << endl;
}

void permutation_representation::element_print_for_make_element(
		int *Elt, std::ostream &ost)
{
	A_original->Group_element->element_print_for_make_element(Elt, ost);
}

void permutation_representation::element_print_easy(
		int *Elt, std::ostream &ost)
{
	A_original->Group_element->element_print_quick(Elt, ost);
}

void permutation_representation::element_print_latex(
		int *Elt, std::ostream &ost)
{
	A_original->Group_element->element_print_latex(Elt, ost);
}




}}}

