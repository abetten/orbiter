// strong_generators.cpp
//
// Anton Betten
// December 4, 2013

#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {

strong_generators::strong_generators()
{
	null();
}

strong_generators::~strong_generators()
{
	freeself();
}

void strong_generators::null()
{
	A = NULL;
	tl = NULL;
	gens = NULL;
}

void strong_generators::freeself()
{
	if (tl) {
		FREE_int(tl);
	}
	if (gens) {
		FREE_OBJECT(gens);
	}
	null();
}

void strong_generators::swap_with(strong_generators *SG)
{
	action *my_A;
	int *my_tl;
	vector_ge *my_gens;

	my_A = A;
	A = SG->A;
	SG->A = my_A;

	my_tl = tl;
	tl = SG->tl;
	SG->tl = my_tl;

	my_gens = gens;
	gens = SG->gens;
	SG->gens = my_gens;

}

void strong_generators::init(action *A)
{
	init(A, 0);
}

void strong_generators::init(action *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::init" << endl;
	}
	strong_generators::A = A;
	if (f_v) {
		cout << "strong_generators::init done" << endl;
	}
}

void strong_generators::init_from_sims(sims *S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "strong_generators::init_from_sims" << endl;
		cout << "action = " << S->A->label << endl;
		cout << "base_len = " << S->A->base_len() << endl;
	}
	A = S->A;
	tl = NEW_int(A->base_len());
	gens = NEW_OBJECT(vector_ge);
	if (f_v) {
		cout << "strong_generators::init_from_sims before "
				"S->extract_strong_generators_in_order" << endl;
	}
	S->extract_strong_generators_in_order(*gens, tl,
			verbose_level - 5);
	if (f_v) {
		cout << "strong_generators::init_from_sims after "
				"S->extract_strong_generators_in_order" << endl;
		cout << "strong_generators::init_from_sims tl=";
		Orbiter->Int_vec.print(cout, tl, A->base_len());
		cout << endl;
	}
	if (f_v) {
		cout << "strong_generators::init_from_sims done" << endl;
	}
}

void strong_generators::init_from_ascii_coding(action *A,
		char *ascii_coding, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object go;
	group_container *G;

	if (f_v) {
		cout << "strong_generators::init_from_ascii_coding" << endl;
	}
	G = NEW_OBJECT(group_container);
	G->init(A, verbose_level - 2);
	if (f_vv) {
		cout << "strong_generators::init_from_ascii_coding "
				"before G->init_ascii_coding_to_sims" << endl;
	}
	G->init_ascii_coding_to_sims(ascii_coding, verbose_level - 2);
	if (f_vv) {
		cout << "strong_generators::init_from_ascii_coding "
				"after G->init_ascii_coding_to_sims" << endl;
	}
		

	G->S->group_order(go);

	if (f_vv) {
		cout << "strong_generators::init_from_ascii_coding "
				"Group order=" << go << endl;
	}

	init_from_sims(G->S, 0 /* verbose_level */);

	FREE_OBJECT(G);
	if (f_v) {
		cout << "strong_generators::init_from_ascii_coding "
				"done" << endl;
	}
}


strong_generators *strong_generators::create_copy()
{
	strong_generators *S;

	S = NEW_OBJECT(strong_generators);
	S->init_copy(this, 0);
	return S;
}

void strong_generators::init_copy(strong_generators *S,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "strong_generators::init_copy" << endl;
	}
	A = S->A;
	tl = NEW_int(A->base_len());
	//cout << "strong_generators::init_copy before int_vec_copy" << endl;
	Orbiter->Int_vec.copy(S->tl, tl, A->base_len());
	gens = NEW_OBJECT(vector_ge);
	gens->init(A, verbose_level - 2);
	gens->allocate(S->gens->len, verbose_level - 2);
	for (i = 0; i < S->gens->len; i++) {
		//cout << "strong_generators::init_copy before
		// element_move i=" << i << endl;
		A->element_move(S->gens->ith(i), gens->ith(i), 0);
	}
	if (f_v) {
		cout << "strong_generators::init_copy done" << endl;
	}
}

void strong_generators::init_by_hdl_and_with_tl(action *A,
		std::vector<int> &gen_handle,
		std::vector<int> &tl,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "strong_generators::init_by_hdl_and_with_tl" << endl;
	}


	init(A, 0);
	strong_generators::tl = NEW_int(A->base_len());
	for (i = 0; i < A->base_len(); i++) {
		strong_generators::tl[i] = tl[i];
	}
	gens = NEW_OBJECT(vector_ge);
	gens->init(A, verbose_level - 2);
	gens->allocate(gen_handle.size(), verbose_level - 2);
	for (i = 0; i < gen_handle.size(); i++) {
		A->element_retrieve(gen_handle[i], gens->ith(i), 0);
	}


	if (f_v) {
		cout << "strong_generators::init_by_hdl_and_with_tl done" << endl;
	}
}


void strong_generators::init_by_hdl(action *A,
		int *gen_hdl, int nb_gen, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "strong_generators::init_by_hdl" << endl;
	}
	init(A, 0);
	tl = NEW_int(A->base_len());
	for (i = 0; i < A->base_len(); i++) {
		tl[i] = 1;
	}
	gens = NEW_OBJECT(vector_ge);
	gens->init(A, verbose_level - 2);
	gens->allocate(nb_gen, verbose_level - 2);
	for (i = 0; i < nb_gen; i++) {
		A->element_retrieve(gen_hdl[i], gens->ith(i), 0);
	}
	if (f_v) {
		cout << "strong_generators::init_by_hdl done" << endl;
	}
}

void strong_generators::init_from_permutation_representation(
	action *A, sims *parent_group_S, int *data,
	int nb_elements, long int group_order, vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::init_from_permutation_representation" << endl;
	}
	if (group_order == 0) {
		cout << "strong_generators::init_from_permutation_representation "
				"group_order == 0" << endl;
		exit(1);
	}
	if (group_order < 0) {
		cout << "strong_generators::init_from_permutation_representation "
				"group_order < 0" << endl;
		cout << "group_order=" << group_order << endl;
		exit(1);
	}
	init(A, verbose_level - 2);

	//vector_ge *nice_gens;
	nice_gens = NEW_OBJECT(vector_ge);

	if (f_v) {
		cout << "strong_generators::init_from_permutation_representation "
				"before nice_gens->init_from_permutation_representation" << endl;
	}
	nice_gens->init_from_permutation_representation(A, parent_group_S, data,
		nb_elements, verbose_level - 3);
	if (f_v) {
		cout << "strong_generators::init_from_permutation_representation "
				"after nice_gens->init_from_permutation_representation" << endl;
	}
	
	sims *S;

	if (f_v) {
		cout << "strong_generators::init_from_permutation_representation before "
				"A->create_sims_from_generators_with_target_group_order_int" << endl;
	}
	S = A->create_sims_from_generators_with_target_group_order_lint(
			nice_gens, group_order, verbose_level - 3);
	if (f_v) {
		cout << "strong_generators::init_from_permutation_representation after "
				"A->create_sims_from_generators_with_target_group_order_int" << endl;
	}
	
	if (f_v) {
		cout << "strong_generators::init_from_permutation_representation "
				"before init_from_sims" << endl;
	}
	init_from_sims(S, verbose_level - 3);
	if (f_v) {
		cout << "strong_generators::init_from_permutation_representation "
				"after init_from_sims" << endl;
	}

	//tl = NEW_int(A->base_len);
	//int_vec_copy(transversal_length, tl, A->base_len);

	//FREE_OBJECT(my_gens);
	if (f_v) {
		cout << "strong_generators::init_from_permutation_representation "
				"done, found a group of order " << group_order << endl;
	}
}

void strong_generators::init_from_data(action *A, int *data, 
	int nb_elements, int elt_size, int *transversal_length, 
	vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::init_from_data" << endl;
	}
	init(A, verbose_level - 2);
	gens = NEW_OBJECT(vector_ge);
	nice_gens = NEW_OBJECT(vector_ge);

	gens->init_from_data(A, data, 
		nb_elements, elt_size, verbose_level);
	
	nice_gens->init_from_data(A, data,
		nb_elements, elt_size, verbose_level);

	tl = NEW_int(A->base_len());
	Orbiter->Int_vec.copy(transversal_length, tl, A->base_len());

	if (f_v) {
		cout << "strong_generators::init_from_data done" << endl;
	}
}

void strong_generators::init_from_data_with_target_go_ascii(
	action *A, int *data,
	int nb_elements, int elt_size, const char *ascii_target_go,
	vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object target_go;

	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go_ascii" << endl;
	}
	strong_generators::A = A;
	target_go.create_from_base_10_string(ascii_target_go);
	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go_ascii before init_from_data_with_target_go" << endl;
	}
	init_from_data_with_target_go(A, data, 
		elt_size, nb_elements, target_go,
		nice_gens,
		verbose_level);
	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go_ascii after init_from_data_with_target_go" << endl;
	}
	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go_ascii done" << endl;
	}
}

void strong_generators::init_from_data_with_target_go(
	action *A, int *data_gens,
	int data_gens_size, int nb_gens,
	longinteger_object &target_go,
	vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go" << endl;
	}

	strong_generators::A = A;

	//vector_ge *my_gens;

	nice_gens = NEW_OBJECT(vector_ge);
	nice_gens->init(A, verbose_level - 2);
	nice_gens->allocate(nb_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		if (f_v) {
			cout << "strong_generators::init_from_data_with_target_go "
					<< i << " / " << nb_gens << endl;
		}

		A->make_element(nice_gens->ith(i),
				data_gens + i * data_gens_size,
				verbose_level);
	}
	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go "
				"generators are:" << endl;
		nice_gens->print_quick(cout);
	}

	strong_generators *SG;

	SG = NEW_OBJECT(strong_generators);
	
	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go before A->generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		TRUE /* f_target_go */, target_go, 
		nice_gens, SG,
		verbose_level - 1);
	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go after A->generators_to_strong_generators" << endl;
	}

	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go "
				"strong generators are:" << endl;
		SG->print_generators(cout);
	}

	//FREE_OBJECT(my_gens);

	if (gens) {
		FREE_OBJECT(gens);
	}
	gens = SG->gens;
	SG->gens = NULL;
	if (tl) {
		FREE_int(tl);
	}
	tl = SG->tl;
	SG->tl = NULL;
	
	FREE_OBJECT(SG);
	
	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go done" << endl;
	}
}

void strong_generators::init_from_data_with_go(
	action *A, std::string &generators_data,
	std::string &go_text,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i;

	if (f_v) {
		cout << "strong_generators::init_from_data_with_go" << endl;
	}


	int *gens_data;
	int gens_data_sz;
	int nb_elements;

	Orbiter->Int_vec.scan(generators_data, gens_data, gens_data_sz);
	cout << "gens_data = ";
	Orbiter->Int_vec.print(cout, gens_data, gens_data_sz);
	cout << endl;
	cout << "go_text = " << go_text << endl;


	init(A);


	nb_elements = gens_data_sz / A->make_element_size;

	//strong_generators *Gens;
	vector_ge *nice_gens;
	//int orbit_length;

	//Gens = NEW_OBJECT(strong_generators);

	cout << "before SG->init_from_data_with_target_go_ascii" << endl;
	init_from_data_with_target_go_ascii(A,
			gens_data,
			nb_elements, A->make_element_size,
			go_text.c_str(),
			nice_gens,
			verbose_level);


	FREE_OBJECT(nice_gens);

	if (f_v) {
		cout << "strong_generators::init_from_data_with_go done" << endl;
	}

}
void
strong_generators::init_point_stabilizer_of_arbitrary_point_through_schreier(
	schreier *Sch,
	int pt, int &orbit_idx, longinteger_object &full_group_order,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	strong_generators *SG0;
	
	if (f_v) {
		cout << "strong_generators::init_point_stabilizer_of_arbitrary_"
				"point_through_schreier" << endl;
	}
	Elt = NEW_int(A->elt_size_in_int);
	Sch->transporter_from_point_to_orbit_rep(pt, orbit_idx, Elt,
			0 /* verbose_level */);

	SG0 = NEW_OBJECT(strong_generators);
	SG0->init(A);

	SG0->init_point_stabilizer_orbit_rep_schreier(Sch, orbit_idx,
			full_group_order, verbose_level);
	init_generators_for_the_conjugate_group_aGav(SG0, Elt, 0 /* verbose_level */);
	
	FREE_OBJECT(SG0);
	FREE_int(Elt);
	if (f_v) {
		cout << "strong_generators::init_point_stabilizer_of_arbitrary_"
				"point_through_schreier done" << endl;
	}
}

void strong_generators::init_point_stabilizer_orbit_rep_schreier(
	schreier *Sch,
	int orbit_idx, longinteger_object &full_group_order,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sims *Stab;
	
	if (f_v) {
		cout << "strong_generators::init_point_stabilizer_orbit_rep_schreier" << endl;
	}
	Sch->point_stabilizer(A, full_group_order,
			Stab, orbit_idx, verbose_level);
	init_from_sims(Stab, 0 /* verbose_level */);
	FREE_OBJECT(Stab);
	if (f_v) {
		cout << "strong_generators::init_point_stabilizer_orbit_rep_schreier done" << endl;
	}
}

void strong_generators::init_generators_for_the_conjugate_group_avGa(
		strong_generators *SG, int *Elt_a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	vector_ge *gens;
	longinteger_object go;
	//int i;	
	
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_avGa" << endl;
	}
	
	SG->group_order(go);
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_avGa go=" << go << endl;
	}
	gens = NEW_OBJECT(vector_ge);

#if 0
	gens->init(SG->A);
	gens->allocate(SG->gens->len);
	for (i = 0; i < SG->gens->len; i++) {
		A->element_conjugate_bvab(SG->gens->ith(i), Elt_a,
				gens->ith(i), 0 /* verbose_level */);
	}
#else
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_avGa before gens->init_conjugate_svas_of" << endl;
	}
	gens->init_conjugate_svas_of(SG->gens, Elt_a, verbose_level);
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_avGa after gens->init_conjugate_svas_of" << endl;
	}
#endif

	strong_generators *SG1;
	
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_avGa "
				"before generators_to_strong_generators" << endl;
	}
	SG->A->generators_to_strong_generators(
		TRUE /* f_target_go */, go, 
		gens, SG1, 
		0 /*verbose_level*/);

	swap_with(SG1);
	FREE_OBJECT(gens);
	FREE_OBJECT(SG1);
	
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_avGa done" << endl;
	}
}

void strong_generators::init_generators_for_the_conjugate_group_aGav(
		strong_generators *SG, int *Elt_a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	vector_ge *gens;
	longinteger_object go;
	//int i;	
	
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_aGav" << endl;
	}

	SG->group_order(go);
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_aGav go=" << go << endl;
	}
	gens = NEW_OBJECT(vector_ge);

#if 0
	gens->init(SG->A);
	gens->allocate(SG->gens->len);
	for (i = 0; i < SG->gens->len; i++) {
		if (f_v) {
			cout << i << " / " << SG->gens->len << ":" << endl;
			}
		SG->A->element_conjugate_babv(SG->gens->ith(i),
				Elt_a, gens->ith(i), verbose_level);
		}
#else
	gens->init_conjugate_sasv_of(SG->gens, Elt_a, 0 /* verbose_level */);
#endif

	strong_generators *SG1;
	
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_aGav "
				"before generators_to_strong_generators" << endl;
	}
	SG->A->generators_to_strong_generators(
		TRUE /* f_target_go */, go, 
		gens, SG1, 
		0 /*verbose_level*/);
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_aGav "
				"after generators_to_strong_generators" << endl;
	}

	swap_with(SG1);
	FREE_OBJECT(gens);
	FREE_OBJECT(SG1);
	
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_aGav done" << endl;
	}
}

void strong_generators::init_transposed_group(
		strong_generators *SG, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	vector_ge *gens;
	longinteger_object go;
	int i;
	
	if (f_v) {
		cout << "strong_generators::init_transposed_group" << endl;
	}

	SG->group_order(go);
	gens = NEW_OBJECT(vector_ge);

	gens->init(A, verbose_level - 2);
	gens->allocate(SG->gens->len, verbose_level - 2);
	for (i = 0; i < SG->gens->len; i++) {
		if (f_v) {
			cout << "before element_transpose " << i << " / "
					<< SG->gens->len << ":" << endl;
			A->element_print_quick(SG->gens->ith(i), cout);
		}
		A->element_transpose(SG->gens->ith(i), gens->ith(i),
				0 /* verbose_level*/);
		if (f_v) {
			cout << "after element_transpose " << i << " / "
					<< SG->gens->len << ":" << endl;
			A->element_print_quick(gens->ith(i), cout);
		}
	}

	strong_generators *SG1;
	
	if (f_v) {
		cout << "strong_generators::init_transposed_group "
				"before A->generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		TRUE /* f_target_go */, go, 
		gens, SG1, 
		verbose_level);
	if (f_v) {
		cout << "strong_generators::init_transposed_group "
				"after A->generators_to_strong_generators" << endl;
	}

	swap_with(SG1);
	FREE_OBJECT(gens);
	FREE_OBJECT(SG1);
	
	if (f_v) {
		cout << "strong_generators::init_transposed_group done" << endl;
	}
}

void strong_generators::init_group_extension(
	strong_generators *subgroup, int *data, int index,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	longinteger_object target_go;
	longinteger_domain D;

	if (f_v) {
		cout << "strong_generators::init_group_extension" << endl;
	}

	A = subgroup->A;

	vector_ge *my_gens;
	int nb_gens;

	my_gens = NEW_OBJECT(vector_ge);
	my_gens->init(A, verbose_level - 2);
	nb_gens = subgroup->gens->len;
	my_gens->allocate(nb_gens + 1, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		A->element_move(subgroup->gens->ith(i), my_gens->ith(i), 0);
	}
	A->make_element(my_gens->ith(nb_gens), data, 0);

	subgroup->group_order(target_go);
	D.mult_integer_in_place(target_go, index);
	
	strong_generators *SG;

	SG = NEW_OBJECT(strong_generators);

	if (f_v) {
		cout << "strong_generators::init_group_extension "
				"before generators_to_strong_generators, "
				"target_go=" << target_go << endl;
	}
	
	A->generators_to_strong_generators(
		TRUE /* f_target_go */, target_go, 
		my_gens, SG, 
		0 /*verbose_level*/);

	if (FALSE) {
		cout << "strong_generators::init_group_extension "
				"strong generators are:" << endl;
		SG->print_generators(cout);
	}

	FREE_OBJECT(my_gens);

	if (gens) {
		FREE_OBJECT(gens);
	}
	gens = SG->gens;
	SG->gens = NULL;
	if (tl) {
		FREE_int(tl);
	}
	tl = SG->tl;
	SG->tl = NULL;
	
	FREE_OBJECT(SG);
	
	if (f_v) {
		cout << "strong_generators::init_group_extension done" << endl;
	}
}

void strong_generators::init_group_extension(
	strong_generators *subgroup,
	vector_ge *new_gens, int index,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	longinteger_object target_go;
	longinteger_domain D;

	if (f_v) {
		cout << "strong_generators::init_group_extension" << endl;
	}

	A = subgroup->A;

	vector_ge *my_gens;
	int nb_gens, nb_new_gens;

	my_gens = NEW_OBJECT(vector_ge);
	my_gens->init(A, verbose_level - 2);
	nb_gens = subgroup->gens->len;
	nb_new_gens = new_gens->len;
	my_gens->allocate(nb_gens + nb_new_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		A->element_move(subgroup->gens->ith(i), my_gens->ith(i), 0);
	}
	for (i = 0; i < nb_new_gens; i++) {
		A->element_move(new_gens->ith(i), my_gens->ith(nb_gens + i), 0);
	}

	if (f_v) {
		cout << "strong_generators::init_group_extension "
				"my_gens=" << endl;
		my_gens->print_quick(cout);
	}


	subgroup->group_order(target_go);
	D.mult_integer_in_place(target_go, index);
	
	if (f_v) {
		cout << "strong_generators::init_group_extension "
				"target_go=" << target_go << endl;
		cout << "A=" << endl;
		A->print_info();
	}
	
	strong_generators *SG;

	SG = NEW_OBJECT(strong_generators);
	
	if (f_v) {
		cout << "strong_generators::init_group_extension "
				"before generators_to_strong_generators, "
				"target_go=" << target_go << endl;
	}

	A->generators_to_strong_generators(
		TRUE /* f_target_go */, target_go, 
		my_gens, SG, 
		0 /*verbose_level - 2*/);

	if (FALSE) {
		cout << "strong_generators::init_group_extension "
				"strong generators are:" << endl;
		SG->print_generators(cout);
	}

	FREE_OBJECT(my_gens);

	if (gens) {
		FREE_OBJECT(gens);
	}
	gens = SG->gens;
	SG->gens = NULL;
	if (tl) {
		FREE_int(tl);
	}
	tl = SG->tl;
	SG->tl = NULL;
	
	FREE_OBJECT(SG);
	
	if (f_v) {
		cout << "strong_generators::init_group_extension "
				"done" << endl;
	}
}

void strong_generators::switch_to_subgroup(
	const char *rank_vector_text,
	const char *subgroup_order_text, sims *S,
	int *&subgroup_gens_idx, int &nb_subgroup_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object target_go;

	if (f_v) {
		cout << "strong_generators::switch_to_subgroup" << endl;
	}


	//sims *S;

	//S = create_sims(0 /* verbose_level */);
	

	Orbiter->Int_vec.scan(rank_vector_text, subgroup_gens_idx, nb_subgroup_gens);
	if (f_v) {
		cout << "strong_generators::switch_to_subgroup "
				"after scanning: ";
		Orbiter->Int_vec.print(cout, subgroup_gens_idx, nb_subgroup_gens);
		cout << endl;
	}


	vector_ge *my_gens;

	my_gens = NEW_OBJECT(vector_ge);
	my_gens->init(A, verbose_level - 2);
	my_gens->extract_subset_of_elements_by_rank_text_vector(
			rank_vector_text, S, verbose_level);


	if (f_v) {
		cout << "strong_generators::switch_to_subgroup "
				"chosen generators:" << endl;
		my_gens->print_quick(cout);
	}

	target_go.create_from_base_10_string(subgroup_order_text);

	
	strong_generators *SG;

	SG = NEW_OBJECT(strong_generators);
	
	A->generators_to_strong_generators(
		TRUE /* f_target_go */, target_go, 
		my_gens, SG, 
		0 /*verbose_level*/);

	if (FALSE) {
		cout << "strong_generators::switch_to_subgroup "
				"strong generators are:" << endl;
		SG->print_generators(cout);
	}

	FREE_OBJECT(my_gens);

	if (gens) {
		FREE_OBJECT(gens);
	}
	gens = SG->gens;
	SG->gens = NULL;
	if (tl) {
		FREE_int(tl);
	}
	tl = SG->tl;
	SG->tl = NULL;
	
	FREE_OBJECT(SG);
	
	if (f_v) {
		cout << "strong_generators::switch_to_subgroup done" << endl;
	}
}

void strong_generators::init_subgroup(action *A,
	int *subgroup_gens_idx, int nb_subgroup_gens,
	const char *subgroup_order_text, 
	sims *S, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object target_go;

	if (f_v) {
		cout << "strong_generators::init_subgroup" << endl;
	}


	strong_generators::A = A;
	
	vector_ge *my_gens;

	my_gens = NEW_OBJECT(vector_ge);
	my_gens->init(A, verbose_level - 2);
	my_gens->extract_subset_of_elements_by_rank(
		subgroup_gens_idx, nb_subgroup_gens, S, verbose_level);


	if (f_v) {
		cout << "strong_generators::init_subgroup "
				"chosen generators:" << endl;
		my_gens->print_quick(cout);
	}

	target_go.create_from_base_10_string(subgroup_order_text);

	
	strong_generators *SG;

	SG = NEW_OBJECT(strong_generators);
	
	A->generators_to_strong_generators(
		TRUE /* f_target_go */, target_go, 
		my_gens, SG, 
		0 /*verbose_level*/);

	if (FALSE) {
		cout << "strong_generators::init_subgroup "
				"strong generators are:" << endl;
		SG->print_generators(cout);
	}

	FREE_OBJECT(my_gens);

	if (gens) {
		FREE_OBJECT(gens);
	}
	gens = SG->gens;
	SG->gens = NULL;
	if (tl) {
		FREE_int(tl);
	}
	tl = SG->tl;
	SG->tl = NULL;
	
	FREE_OBJECT(SG);
	
	if (f_v) {
		cout << "strong_generators::init_subgroup done" << endl;
	}
}

void strong_generators::init_subgroup_by_generators(action *A,
	int nb_subgroup_gens,
	int *subgroup_gens,
	std::string &subgroup_order_text,
	vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object target_go;

	if (f_v) {
		cout << "strong_generators::init_subgroup_by_generators" << endl;
	}


	strong_generators::A = A;

	//vector_ge *my_gens;

	nice_gens = NEW_OBJECT(vector_ge);
	nice_gens->init(A, verbose_level - 2);
	nice_gens->allocate(nb_subgroup_gens, verbose_level - 2);
	for (int h = 0; h < nb_subgroup_gens; h++) {
		if (f_v) {
			cout << "strong_generators::init_subgroup_by_generators "
					"generator " << h << " / " << nb_subgroup_gens << endl;
		}
		A->make_element(nice_gens->ith(h), subgroup_gens + h * A->make_element_size, verbose_level);
	}


	if (f_v) {
		cout << "strong_generators::init_subgroup_by_generators "
				"chosen generators:" << endl;
		nice_gens->print_quick(cout);
	}

	target_go.create_from_base_10_string(subgroup_order_text);


	strong_generators *SG;

	SG = NEW_OBJECT(strong_generators);

	A->generators_to_strong_generators(
		TRUE /* f_target_go */, target_go,
		nice_gens, SG,
		0 /*verbose_level*/);

	if (FALSE) {
		cout << "strong_generators::init_subgroup_by_generators "
				"strong generators are:" << endl;
		SG->print_generators(cout);
	}

	//FREE_OBJECT(my_gens);

	if (gens) {
		FREE_OBJECT(gens);
	}
	gens = SG->gens;
	SG->gens = NULL;
	if (tl) {
		FREE_int(tl);
	}
	tl = SG->tl;
	SG->tl = NULL;

	FREE_OBJECT(SG);

	if (f_v) {
		cout << "strong_generators::init_subgroup_by_generators done" << endl;
	}
}


sims *strong_generators::create_sims(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sims *S;


	if (f_v) {
		cout << "strong_generators::create_sims verbose_level=" << verbose_level << endl;
	}
	
	if (gens == NULL) {
		cout << "strong_generators::create_sims "
				"gens == NULL" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "strong_generators::create_sims before "
				"create_sims_from_generators_with_target_group_order_factorized" << endl;
	}
	S = A->create_sims_from_generators_with_target_group_order_factorized(
		gens, tl, A->base_len(), verbose_level - 2);

	if (f_v) {
		cout << "strong_generators::create_sims after "
				"create_sims_from_generators_with_target_group_order_factorized" << endl;
	}

	if (f_v) {
		cout << "strong_generators::create_sims done" << endl;
	}
	return S;
}

sims *strong_generators::create_sims_in_different_action(
		action *A_given, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sims *S;


	if (f_v) {
		cout << "strong_generators::create_sims_in_different_action" << endl;
	}
	
	if (gens == NULL) {
		cout << "strong_generators::create_sims_in_different_action "
				"gens == NULL" << endl;
		exit(1);
	}
	S = A_given->create_sims_from_generators_with_target_group_order_factorized(
		gens, tl, A->base_len(),
		0 /* verbose_level */);

	if (f_v) {
		cout << "strong_generators::create_sims_in_different_action "
				"done" << endl;
	}
	return S;
}

void strong_generators::add_generators(
		vector_ge *coset_reps, int group_index,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	sims *S;
	vector_ge *gens1; 
	int *tl1;
	int *coset_reps_vec;
	int i;

	if (f_v) {
		cout << "strong_generators::add_generators" << endl;
	}
	if (f_vv) {
		cout << "group_index=" << group_index << endl;
		cout << "action=";
		A->print_info();
	}

	coset_reps_vec = NEW_int(group_index * A->elt_size_in_int);
	for (i = 0; i < group_index; i++) {
		A->element_move(coset_reps->ith(i),
				coset_reps_vec + i * A->elt_size_in_int, 0);
	}

	gens1 = NEW_OBJECT(vector_ge);
	tl1 = NEW_int(A->base_len());
	
	S = create_sims(verbose_level - 1);

	S->transitive_extension_using_coset_representatives_extract_generators(
		coset_reps_vec, group_index, 
		*gens1, tl1, 
		verbose_level - 2);

	FREE_int(coset_reps_vec);

	if (gens) {
		FREE_OBJECT(gens);
	}
	if (tl) {
		FREE_int(tl);
	}
	gens = gens1;
	tl = tl1;

	FREE_OBJECT(S);
	if (f_v) {
		cout << "strong_generators::add_generators done" << endl;
	}
}

void strong_generators::add_single_generator(
		int *Elt, int group_index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sims *S;
	vector_ge *gens1; 
	int *tl1;

	if (f_v) {
		cout << "strong_generators::add_single_generator" << endl;
		cout << "action=";
		A->print_info();
	}

	gens1 = NEW_OBJECT(vector_ge);
	tl1 = NEW_int(A->base_len());
	
	S = create_sims(verbose_level - 1);

	S->transitive_extension_using_generators(
		Elt, 1, group_index, 
		*gens1, tl1, 
		verbose_level);

	if (gens) {
		FREE_OBJECT(gens);
	}
	if (tl) {
		FREE_int(tl);
	}
	gens = gens1;
	tl = tl1;

	FREE_OBJECT(S);
	if (f_v) {
		cout << "strong_generators::add_single_generator "
				"done" << endl;
	}
}

void strong_generators::group_order(longinteger_object &go)
{
	longinteger_domain D;

	D.multiply_up(go, tl, A->base_len(), 0 /* verbose_level */);
}

long int strong_generators::group_order_as_lint()
{
	longinteger_domain D;
	longinteger_object go;

	D.multiply_up(go, tl, A->base_len(), 0 /* verbose_level */);
	return go.as_lint();
}

void strong_generators::print_group_order(std::ostream &ost)
{
	longinteger_object go;

	group_order(go);
	ost << go;
}

void strong_generators::print_generators_gap(std::ostream &ost)
{
	int i;

	ost << "Generators in GAP format are:" << endl;
	if (A->degree < 200) {
		ost << "G := Group([";
		for (i = 0; i < gens->len; i++) {
			A->element_print_as_permutation_with_offset(
					gens->ith(i), ost,
					1 /*offset*/,
					TRUE /* f_do_it_anyway_even_for_big_degree */,
					FALSE /* f_print_cycles_of_length_one */,
					0 /* verbose_level*/);
			if (i < gens->len - 1) {
				ost << ", " << endl;
			}
		}
		ost << "]);" << endl;
	}
	else {
		ost << "too big to print" << endl;
	}
}


void strong_generators::print_generators_gap_in_different_action(std::ostream &ost, action *A2)
{
	int i;

	ost << "Generators in GAP format are:" << endl;
	if (A->degree < 200) {
		ost << "G := Group([";
		for (i = 0; i < gens->len; i++) {
			A2->element_print_as_permutation_with_offset(
					gens->ith(i), ost,
					1 /*offset*/,
					TRUE /* f_do_it_anyway_even_for_big_degree */,
					FALSE /* f_print_cycles_of_length_one */,
					0 /* verbose_level*/);
			if (i < gens->len - 1) {
				ost << ", " << endl;
			}
		}
		ost << "]);" << endl;
	}
	else {
		ost << "too big to print" << endl;
	}
}


void strong_generators::print_generators_compact(std::ostream &ost)
{
	int i, j, a;

	ost << "Generators in compact permutation form are:" << endl;
	if (A->degree < 200) {
		ost << gens->len << " " << A->degree << endl;
		for (i = 0; i < gens->len; i++) {
			for (j = 0; j < A->degree; j++) {
				a = A->element_image_of(j,
						gens->ith(i), 0 /* verbose_level */);
				ost << a << " ";
				}
			ost << endl;
			}
		ost << "-1" << endl;
	}
	else {
		ost << "too big to print" << endl;
	}
}

void strong_generators::print_generators(std::ostream &ost)
{
	int i;
	longinteger_object go;

	cout << "strong_generators::print_generators computing group order" << endl;
	group_order(go);
	ost << "Strong generators for a group of order "
			<< go << " tl=";
	Orbiter->Int_vec.print(cout, tl, A->base_len());
	ost << endl;

	for (i = 0; i < gens->len; i++) {
		ost << "generator " << i << " / "
				<< gens->len << " is: " << endl;
		A->element_print(gens->ith(i), ost);
		ost << "as permutation: " << endl;
		if (A->degree < 400) {
			A->element_print_as_permutation_with_offset(
					gens->ith(i), ost,
					0 /* offset*/,
					TRUE /* f_do_it_anyway_even_for_big_degree*/,
					TRUE /* f_print_cycles_of_length_one*/,
					0 /* verbose_level*/);
			//A->element_print_as_permutation(SG->gens->ith(i), cout);
			ost << endl;
		}
		else {
			ost << "too big to print" << endl;
		}
	}

	ost << "Generators as permutations are:" << endl;



	if (A->degree < 400) {
		for (i = 0; i < gens->len; i++) {
			A->element_print_as_permutation(gens->ith(i), ost);
			ost << endl;
		}
	}
	else {
		ost << "too big to print" << endl;
	}
}

void strong_generators::print_generators_in_latex_individually(std::ostream &ost)
{
	int i;
	longinteger_object go;

	group_order(go);

	ost << "The stabilizer of order $" << go
			<< "$ is generated by:\\\\" << endl;

	for (i = 0; i < gens->len; i++) {

		string label;
		char str[1000];

		sprintf(str, "g_{%d} = ", i + 1);
		label.assign(str);

		//A->element_print_latex_with_extras(gens->ith(i), label, ost);

		ost << "$" << str << "$ ";

		//A->element_print_latex_not_in_math_mode(gens->ith(i), ost);

		if (A->f_is_linear) {
			ost << "$";
			A->element_print_latex(gens->ith(i), ost);
			ost << "$";
		}
		else {
			A->element_print_latex(gens->ith(i), ost);
		}

		//ost << "\\\\" << endl;


		int n, ord;

		ord = A->element_order(gens->ith(i));

		ost << " of order " << ord;

		n = A->count_fixed_points(gens->ith(i), 0 /* verbose_level */);
		ost << " and with " << n << " fixed points.\\\\" << endl;

		}
	ost << endl << "\\bigskip" << endl;
}

void strong_generators::print_generators_in_source_code()
{
	int i;
	longinteger_object go;

	group_order(go);
	cout << "Strong generators for a group of "
			"order " << go << " tl=";
	Orbiter->Int_vec.print(cout, tl, A->base_len());
	cout << endl;
	A->print_base();
	for (i = 0; i < gens->len; i++) {
		//cout << "Generator " << i << " / "
		// << gens->len << " is:" << endl;
		A->print_for_make_element(cout, gens->ith(i));
		cout << endl;
	}
}

void strong_generators::print_generators_in_source_code_to_file(
		const char *fname)
{
	int i;
	longinteger_object go;
	file_io Fio;

	{
		ofstream f(fname);
		group_order(go);
		f << gens->len << " " << go << endl;
		for (i = 0; i < gens->len; i++) {
			//cout << "Generator " << i << " / "
			//<< gens->len << " is:" << endl;
			A->print_for_make_element_no_commas(f, gens->ith(i));
			f << endl;
		}
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
}

void strong_generators::print_generators_even_odd()
{
	int i, sgn;
	longinteger_object go;

	group_order(go);
	cout << "Strong generators for a group of order " << go << " tl=";
	Orbiter->Int_vec.print(cout, tl, A->base_len());
	cout << endl;
	for (i = 0; i < gens->len; i++) {
		cout << "Generator " << i << " / " << gens->len << " is:" << endl;
		A->element_print(gens->ith(i), cout);

		sgn = A->element_signum_of_permutation(gens->ith(i));
		cout << " sgn=" << sgn;
		cout << endl;
	}
}

void strong_generators::print_generators_MAGMA(action *A, std::ostream &ost)
{
	int i;

	for (i = 0; i < gens->len; i++) {
		//cout << "Generator " << i << " / "
		// << gens->len << " is:" << endl;
		A->element_print_as_permutation_with_offset(
			gens->ith(i), ost,
			1 /* offset */,
			TRUE /* f_do_it_anyway_even_for_big_degree */,
			FALSE /* f_print_cycles_of_length_one */,
			0 /* verbose_level */);
		if (i < gens->len - 1) {
			ost << ", " << endl;
		}
	}
}

void strong_generators::export_magma(action *A, std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::export_magma" << endl;
		A->print_info();
	}
	if (A->type_G == matrix_group_t) {
		matrix_group *M;
		int *Elt;
		int h, i, j;

		M = A->get_matrix_group();
		if (M->f_semilinear) {
			cout << "cannot export to magma if semilinear" << endl;
			return;
		}
		finite_field *F;

		F = M->GFq;
		if (F->e > 1) {
			int a;

			if (f_v) {
				cout << "strong_generators::export_magma extension field" << endl;
			}
			ost << "F<w>:=GF(" << F->q << ");" << endl;
			ost << "G := GeneralLinearGroup(" << M->n << ", F);" << endl;
			ost << "H := sub< G | ";
			for (h = 0; h < gens->len; h++) {
				Elt = gens->ith(h);
				ost << "[";
				for (i = 0; i < M->n; i++) {
					for (j = 0; j < M->n; j++) {
						a = Elt[i * M->n + j];
						if (a < F->p) {
							ost << a;
						}
						else {
							ost << "w^" << F->log_alpha(a);
						}
						if (j < M->n - 1) {
							ost << ",";
						}
					}
					if (i < M->n - 1) {
						ost << ", ";
					}
				}
				ost << "]";
				if (h < gens->len - 1) {
					ost << ", " << endl;
				}
			}
			ost << " >;" << endl;

		}
		else {
			ost << "G := GeneralLinearGroup(" << M->n << ", GF(" << F->q << "));" << endl;
			ost << "H := sub< G | ";
			for (h = 0; h < gens->len; h++) {
				Elt = gens->ith(h);
				ost << "[";
				for (i = 0; i < M->n; i++) {
					for (j = 0; j < M->n; j++) {
						ost << Elt[i * M->n + j];
						if (j < M->n - 1) {
							ost << ",";
						}
					}
					if (i < M->n - 1) {
						ost << ", ";
					}
				}
				ost << "]";
				if (h < gens->len - 1) {
					ost << ", " << endl;
				}
			}
			ost << " >;" << endl;
		}
	}
	if (f_v) {
		cout << "strong_generators::export_magma done" << endl;
	}
}


//GL42 := GeneralLinearGroup(4, GF(2));
//> Ominus42 := sub< GL42 | [1,0,0,0, 1,1,0,1, 1,0,1,0, 0,0,0,1 ],
//>                               [0,1,0,0, 1,0,0,0, 0,0,1,0, 0,0,0,1 ],
//>                               [0,1,0,0, 1,0,0,0, 0,0,1,0, 0,0,1,1 ] >;


void strong_generators::canonical_image_GAP(std::string &input_set_text, std::ostream &ost)
{
	int i;

	//ost << "Generators in GAP format are:" << endl;
	ost << "G := Group([";
	for (i = 0; i < gens->len; i++) {
		A->element_print_as_permutation_with_offset(
				gens->ith(i), ost,
				1 /*offset*/,
				TRUE /* f_do_it_anyway_even_for_big_degree */,
				FALSE /* f_print_cycles_of_length_one */,
				0 /* verbose_level*/);
		if (i < gens->len - 1) {
			ost << ", " << endl;
		}
	}
	ost << "]);" << endl;

	long int *set;
	int sz;
	string_tools ST;
	std::string output;


	Orbiter->Lint_vec.scan(input_set_text, set, sz);

	// add one because GAP is 1-based:
	for (i = 0; i < sz; i++) {
		set[i]++;
	}

	ST.create_comma_separated_list(output, set, sz);

	ost << "LoadPackage(\"images\");" << endl;
	ost << "MinimalImage(G, [" << output << "], OnSets);" << endl;
}


void strong_generators::print_generators_tex()
{
	print_generators_tex(cout);
}

void strong_generators::print_generators_tex(std::ostream &ost)
{
	int i;
	longinteger_object go;

	group_order(go);
	ost << "Strong generators for a group of order " << go << ":" << endl;
	ost << "$$" << endl;
	for (i = 0; i < gens->len; i++) {
		//cout << "Generator " << i << " / " << gens->len
		// << " is:" << endl;
		A->element_print_latex(gens->ith(i), ost);
		if (i < gens->len - 1) {
			ost << ", " << endl;
		}
		if (((i + 1) % 1) == 0 && i < gens->len - 1) {
			ost << "$$" << endl;
			ost << "$$" << endl;
		}
	}
	ost << "$$" << endl;
	for (i = 0; i < gens->len; i++) {
		//cout << "Generator " << i << " / " << gens->len
		// << " is:" << endl;
		A->element_print_for_make_element(gens->ith(i), ost);
		ost << "\\\\" << endl;
	}
}

void strong_generators::print_generators_in_different_action_tex(std::ostream &ost, action *A2)
{
	int i;
	longinteger_object go;

	group_order(go);
	ost << "Strong generators for a group of order " << go << ":" << endl;
	ost << "$$" << endl;
	for (i = 0; i < gens->len; i++) {
		//cout << "Generator " << i << " / " << gens->len
		// << " is:" << endl;
		A2->element_print_as_permutation(gens->ith(i), ost);
		if (i < gens->len - 1) {
			ost << ", " << endl;
		}
		if (((i + 1) % 1) == 0 && i < gens->len - 1) {
			ost << "$$" << endl;
			ost << "$$" << endl;
		}
	}
	ost << "$$" << endl;
	for (i = 0; i < gens->len; i++) {
		//cout << "Generator " << i << " / " << gens->len
		// << " is:" << endl;
		A->element_print_for_make_element(gens->ith(i), ost);
		ost << "\\\\" << endl;
	}
}


void strong_generators::print_generators_tex_with_print_point_function(
		action *A_given,
		std::ostream &ost,
		void (*point_label)(stringstream &sstr, long int pt, void *data),
		void *point_label_data)
{
	int i;
	longinteger_object go;

	group_order(go);
	ost << "Strong generators for a group of order " << go << ":" << endl;
	//ost << "$$" << endl;
	for (i = 0; i < gens->len; i++) {
		cout << "Generator " << i << " / " << gens->len << " is:" << endl;
		ost << "$$" << endl;
		A->element_print_latex(gens->ith(i), ost);
		ost << "$$" << endl;
		ost << "$$" << endl;
		A_given->element_print_latex_with_print_point_function(
				gens->ith(i), ost,
				point_label, point_label_data);
		ost << "$$" << endl;
	}
	//ost << "$$" << endl;
	for (i = 0; i < gens->len; i++) {
		//cout << "Generator " << i << " / " << gens->len
		// << " is:" << endl;
		A->element_print_for_make_element(gens->ith(i), ost);
		ost << "\\\\" << endl;
	}

	for (i = 0; i < gens->len; i++) {
		ost << "$";
		A_given->element_print_latex(gens->ith(i), ost);
		ost << "$\\\\" << endl;
	}

}

void strong_generators::print_generators_for_make_element(std::ostream &ost)
{
	int i;
	longinteger_object go;

	group_order(go);
	ost << "Strong generators for a group of order " << go << ":\\\\" << endl;
	for (i = 0; i < gens->len; i++) {
		//ost << "";
		A->element_print_for_make_element(gens->ith(i), ost);
		ost << "\\\\" << endl;
	}
}


void strong_generators::print_generators_as_permutations()
{
	int i;
	longinteger_object go;

	group_order(go);
	cout << "Strong generators for a group of order "
			<< go << ":" << endl;
	for (i = 0; i < gens->len; i++) {
		cout << "Generator " << i << " / "
				<< gens->len << " is:" << endl;
		A->element_print(gens->ith(i), cout);
		if (A->degree < 1000) {
			A->element_print_as_permutation(gens->ith(i), cout);
			cout << endl;
		}
		else {
			cout << "strong_generators::print_generators_as_permutations "
					"the degree is too large, we won't print "
					"the permutation representation" << endl;
		}
	}
}

void strong_generators::print_generators_as_permutations_tex(std::ostream &ost, action *A2)
{
	int i;
	longinteger_object go;

	group_order(go);
	ost << "Strong generators for a group of order " << go << ":" << endl;
	ost << "\\\\" << endl;
	for (i = 0; i < gens->len; i++) {
		ost << "Generator " << i << " / "
				<< gens->len << " is: $" << endl;
		//A->element_print(gens->ith(i), cout);
		if (A->degree < 1000) {
			A2->element_print_as_permutation(gens->ith(i), ost);
		}
		else {
			cout << "strong_generators::print_generators_as_permutations_tex "
					"the degree is too large, we won't print "
					"the permutation representation" << endl;
		}
		ost << "$\\\\" << endl;
	}
}

void strong_generators::print_with_given_action(
		std::ostream &ost, action *A2)
{
	int i;
	
	for (i = 0; i < gens->len; i++) {
		ost << "Generator " << i << " / "
				<< gens->len << " is:" << endl;
		ost << "$$" << endl;
		A2->element_print_latex(gens->ith(i), ost);
		//ost << endl;
		ost << "$$" << endl;
		ost << "as permutation:" << endl;
		//ost << "$$" << endl;
		if (A2->degree < 1000) {
			A2->element_print_as_permutation(gens->ith(i), ost);
		}
		else {
			cout << "strong_generators::print_with_given_action "
					"the degree is too large, we won't print "
					"the permutation representation" << endl;
			ost << "too big to print";
		}
		//ost << endl;
		//ost << "$$" << endl;
		ost << "\\\\" << endl;
	}
}

void strong_generators::print_elements_ost(std::ostream &ost)
{
	long int i;
	longinteger_object go;
	sims *S;
	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);
	S = create_sims(0 /*verbose_level */);
	ost << "Group elements for a group of order " << go << " tl=";
	Orbiter->Int_vec.print(ost, tl, A->base_len());
	ost << "\\\\" << endl;
	for (i = 0; i < go.as_lint(); i++) {
		S->element_unrank_lint(i, Elt, 0 /* verbose_level */);
		ost << "Element " << i << " / " << go << " is:" << endl;
		ost << "$$" << endl;
		A->element_print_latex(Elt, ost);
		ost << "$$" << endl;
	}
	FREE_OBJECT(S);
	FREE_int(Elt);
}

void strong_generators::print_elements_with_special_orthogonal_action_ost(std::ostream &ost)
{
	long int i;
	longinteger_object go;
	sims *S;
	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);
	S = create_sims(0 /*verbose_level */);
	ost << "Group elements for a group of order " << go << " tl=";
	Orbiter->Int_vec.print(ost, tl, A->base_len());
	ost << "\\\\" << endl;
	for (i = 0; i < go.as_lint(); i++) {
		S->element_unrank_lint(i, Elt, 0 /* verbose_level */);

		ost << "Element " << i << " / " << go << " is:" << endl;
		ost << "$$" << endl;
		A->element_print_latex(Elt, ost);
		if (A->matrix_group_dimension() == 4) {
			int A6[36];
			finite_field *F;

			F = A->matrix_group_finite_field();
			F->isomorphism_to_special_orthogonal(Elt, A6, 0 /* verbose_level*/);
			ost << "=" << endl;
			F->print_matrix_latex(ost, A6, 6, 6);
		}
		ost << "$$" << endl;
	}
	FREE_OBJECT(S);
	FREE_int(Elt);
}


void strong_generators::print_elements_with_given_action(std::ostream &ost, action *A2)
{
	long int i;
	longinteger_object go;
	sims *S;
	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);
	S = create_sims(0 /*verbose_level */);
	ost << "Group elements for a group of order " << go << " tl=";
	Orbiter->Int_vec.print(ost, tl, A->base_len());
	ost << "\\\\" << endl;
	for (i = 0; i < go.as_lint(); i++) {
		S->element_unrank_lint(i, Elt, 0 /* verbose_level */);
		ost << "Element " << i << " / " << go << " is:" << endl;
		ost << "$$" << endl;
		if (A2->degree < 1000) {
			A2->element_print_as_permutation(Elt, ost);
		}
		else {
			cout << "strong_generators::print_with_given_action "
					"the degree is too large, we won't print "
					"the permutation representation" << endl;
		}
		ost << endl;
		ost << "$$" << endl;
	}
	FREE_OBJECT(S);
	FREE_int(Elt);
}

void strong_generators::print_elements_latex_ost(std::ostream &ost)
{
	long int i, order, m;
	longinteger_object go;
	sims *S;
	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);
	S = create_sims(0 /*verbose_level */);
	ost << "Group elements for a group of order " << go << " tl=";
	Orbiter->Int_vec.print(ost, tl, A->base_len());
	ost << "\\\\" << endl;
	m = MINIMUM(go.as_int(), 100);
	if (m < go.as_int()) {
		ost << "We will only list the first " << m
				<< " elements:\\\\" << endl;
	}
	for (i = 0; i < m; i++) {
		S->element_unrank_lint(i, Elt, 0 /* verbose_level */);
		order = A->element_order(Elt);
		ost << "Element " << i << " / " << go << " is:" << endl;
		ost << "$$" << endl;
		A->element_print_latex(Elt, ost);
		ost << "$$" << endl;
		ost << "The element has order " << order << ".\\\\" << endl;
	}
	FREE_OBJECT(S);
	FREE_int(Elt);
}

void strong_generators::print_elements_latex_ost_with_print_point_function(
		action *A_given,
		std::ostream &ost,
		void (*point_label)(std::stringstream &sstr, long int pt, void *data),
		void *point_label_data)
{
	long int i, order, m;
	longinteger_object go;
	sims *S;
	int *Elt;
	int *power_elt;
	int *nb_fix_points;
	int *cycle_type;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);
	power_elt = NEW_int(go.as_int());
	nb_fix_points = NEW_int(go.as_int());
	cycle_type = NEW_int(A_given->degree);
	S = create_sims(0 /*verbose_level */);
	ost << "Group elements for a group of order " << go << " tl=";
	Orbiter->Int_vec.print(ost, tl, A->base_len());
	ost << "\\\\" << endl;
	m = MINIMUM(go.as_int(), 500);
	if (m < go.as_int()) {
		ost << "We will only list the first " << m
				<< " elements:\\\\" << endl;
	}
	for (i = 0; i < m; i++) {
		S->element_unrank_lint(i, Elt, 0 /* verbose_level */);
		//cout << "element " << i << " / " << m << " before A->element_order" << endl;
		order = A->element_order(Elt);
		//cout << "element " << i << " / " << m << " before A->element_order_and_cycle_type" << endl;
		A_given->element_order_and_cycle_type(Elt, cycle_type);
		ost << "Element " << i << " / " << go << " is:" << endl;
		ost << "$$" << endl;
		A->element_print_latex(Elt, ost);
		ost << "$$" << endl;
		ost << "$$" << endl;
		A_given->element_print_latex_with_print_point_function(Elt, ost,
				point_label, point_label_data);
		ost << "$$" << endl;
		ost << "The element has order " << order << ".\\\\" << endl;
		S->compute_all_powers(i, order, power_elt, 0 /*verbose_level*/);
		ost << "The powers are: ";
		Orbiter->Int_vec.print(ost, power_elt, order);
		ost << ".\\\\" << endl;
		nb_fix_points[i] = cycle_type[0];
		ost << "The element has " << nb_fix_points[i] << " fix points.\\\\" << endl;
	}
	tally C;

	C.init(nb_fix_points, m, FALSE, 0);
	ost << "The distribution of the number of fix points is $";
	C.print_file_tex_we_are_in_math_mode(ost, TRUE);
	ost << "$\\\\" << endl;
	FREE_OBJECT(S);
	FREE_int(Elt);
	FREE_int(power_elt);
	FREE_int(nb_fix_points);
	FREE_int(cycle_type);
}

void strong_generators::create_group_table(
		int *&Table, long int &go, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sims *S;

	if (f_v) {
		cout << "strong_generators::create_group_table" << endl;
	}
	S = create_sims(0 /*verbose_level */);
	S->create_group_table(Table, go, verbose_level - 1);
	FREE_OBJECT(S);
	if (f_v) {
		cout << "strong_generators::create_group_table done" << endl;
	}
}

void strong_generators::list_of_elements_of_subgroup(
	strong_generators *gens_subgroup,
	long int *&Subgroup_elements_by_index, long int &sz_subgroup, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, a;
	longinteger_object go;
	sims *S;
	sims *U;
	int *Elt;

	if (f_v) {
		cout << "strong_generators::list_of_elements_of_subgroup" << endl;
	}
	Elt = NEW_int(A->elt_size_in_int);
	S = create_sims(0 /*verbose_level */);
	U = gens_subgroup->create_sims(0 /*verbose_level */);
	U->group_order(go);
	sz_subgroup = go.as_lint();
	Subgroup_elements_by_index = NEW_lint(go.as_int());
	for (i = 0; i < sz_subgroup; i++) {
		U->element_unrank_lint(i, Elt, 0 /* verbose_level */);
		a = S->element_rank_lint(Elt);
		Subgroup_elements_by_index[i] = a;
	}
	FREE_OBJECT(S);
	FREE_OBJECT(U);
	FREE_int(Elt);
	if (f_v) {
		cout << "strong_generators::list_of_elements_of_subgroup done" << endl;
	}
}

void strong_generators::compute_schreier_with_given_action(
		action *A_given, schreier *&Sch, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::compute_schreier_with_given_action" << endl;
		cout << "action=";
		A->print_info();
	}
	Sch = NEW_OBJECT(schreier);

	Sch->init(A_given, verbose_level - 2);
	Sch->initialize_tables();
	Sch->init_generators(*gens, verbose_level - 2);
	Sch->compute_all_point_orbits(verbose_level - 2);


	if (f_v) {
		cout << "strong_generators::compute_schreier_with_given_action done, we found "
				<< Sch->nb_orbits << " orbits" << endl;
	}
}

void strong_generators::compute_schreier_with_given_action_on_a_given_set(
		action *A_given, schreier *&Sch, long int *set, int len,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::compute_schreier_with_given_action_on_a_given_set" << endl;
		cout << "action=";
		A->print_info();
	}
	Sch = NEW_OBJECT(schreier);

	Sch->init(A_given, verbose_level - 2);
	Sch->initialize_tables();
	Sch->init_generators(*gens, verbose_level - 2);
	Sch->compute_all_orbits_on_invariant_subset_lint(len, set,
			0 /* verbose_level */);
	//Sch->compute_all_point_orbits(verbose_level);


	if (f_v) {
		cout << "strong_generators::compute_schreier_with_given_action_on_a_given_set "
				"done, we found "
				<< Sch->nb_orbits << " orbits" << endl;
	}
}

void strong_generators::orbits_on_points(int &nb_orbits,
		int *&orbit_reps, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	schreier *Sch;
	int i, f, a;

	if (f_v) {
		cout << "strong_generators::orbits_on_points" << endl;
		cout << "action=";
		A->print_info();
	}

	compute_schreier_with_given_action(A, Sch, verbose_level - 1);


	nb_orbits = Sch->nb_orbits;
	orbit_reps = NEW_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		f = Sch->orbit_first[i];
		a = Sch->orbit[f];
		orbit_reps[i] = a;
	}

	FREE_OBJECT(Sch);

	if (f_v) {
		cout << "strong_generators::orbits_on_points done, "
				"we found " << nb_orbits << " orbits" << endl;
	}
}

void strong_generators::orbits_on_points_with_given_action(
		action *A_given, int &nb_orbits, int *&orbit_reps,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	schreier *Sch;
	int i, f, a;

	if (f_v) {
		cout << "strong_generators::orbits_on_points_with_given_action" << endl;
		cout << "action=";
		A->print_info();
	}
	compute_schreier_with_given_action(A_given, Sch, verbose_level - 1);

	nb_orbits = Sch->nb_orbits;
	orbit_reps = NEW_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		f = Sch->orbit_first[i];
		a = Sch->orbit[f];
		orbit_reps[i] = a;
	}

	FREE_OBJECT(Sch);

	if (f_v) {
		cout << "strong_generators::orbits_on_points_with_given_action "
				"done, we found "
				<< nb_orbits << " orbits" << endl;
	}
}

schreier *strong_generators::orbits_on_points_schreier(
		action *A_given, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	schreier *Sch;

	if (f_v) {
		cout << "strong_generators::orbits_on_points_schreier "
				"degree = " << A_given->degree << endl;
		cout << "A_given=";
		A_given->print_info();
	}

	longinteger_object go;
	group_order(go);

	if (f_v) {
		cout << "strong_generators::orbits_on_points_schreier "
				"go = " << go << endl;
	}


	if (f_v) {
		cout << "strong_generators::orbits_on_points_schreier "
				"generators:" << endl;
		print_generators_tex();
	}
	Sch = NEW_OBJECT(schreier);

	Sch->init(A_given, verbose_level - 2);
	Sch->initialize_tables();
	if (f_v) {
		cout << "strong_generators::orbits_on_points_schreier "
				"before Sch->init_generators" << endl;
	}
	Sch->init_generators(*gens, verbose_level - 2);
	if (f_v) {
		cout << "strong_generators::orbits_on_points_schreier "
				"before Sch->compute_all_point_orbits" << endl;
	}
	Sch->compute_all_point_orbits(0 /*verbose_level*/);
	if (f_v) {
		cout << "strong_generators::orbits_on_points_schreier "
				"after Sch->compute_all_point_orbits" << endl;
	}

	if (f_v) {
		cout << "strong_generators::orbits_on_points_schreier "
				"done, we found " << Sch->nb_orbits << " orbits" << endl;
	}
	return Sch;
}

schreier *strong_generators::orbit_of_one_point_schreier(
		action *A_given, int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	schreier *Sch;

	if (f_v) {
		cout << "strong_generators::orbit_of_one_point_schreier "
				"degree = " << A_given->degree << " point = "
				<< pt << endl;
	}
	Sch = NEW_OBJECT(schreier);

	Sch->init(A_given, verbose_level - 2);
	Sch->initialize_tables();
	Sch->init_generators(*gens, verbose_level - 2);
	Sch->compute_point_orbit(pt, verbose_level);

	if (f_v) {
		cout << "strong_generators::orbit_of_one_point_schreier "
				"done, we found one orbit of length "
				<< Sch->orbit_len[0] << endl;
	}
	return Sch;
}

void strong_generators::orbits_light(action *A_given, 
	int *&Orbit_reps, int *&Orbit_lengths, int &nb_orbits, 
	int **&Pts_per_generator, int *&Nb_per_generator, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);	
	int f_vv = FALSE; //(verbose_level >= 2);	
	bitvector *Has_been_reached;
	//uchar *reached;
	int Orbit_allocated;
	int Orbit_len;
	int *Orbit;
	int *Q;
	int Q_allocated;
	int Q_len;
	int pt, i = 0, h, nb_gens, a, b, idx;
	int Orbit_reps_allocated;
	int nb_reached;
	int *Generator_idx;
	sorting Sorting;

	if (f_v) {
		cout << "strong_generators::orbits_light "
				"degree = " << A_given->degree << endl;
	}

	Orbit_reps_allocated = 1024;
	Orbit_reps = NEW_int(Orbit_reps_allocated);
	Orbit_lengths = NEW_int(Orbit_reps_allocated);
	nb_orbits = 0;

	if (f_v) {
		cout << "strong_generators::orbits_light "
				"allocating array Generator_idx" << endl;
	}
	Generator_idx = NEW_int(A_given->degree);
	if (f_v) {
		cout << "strong_generators::orbits_light "
				"allocating array Generator_idx done" << endl;
	}
	for (pt = 0; pt < A_given->degree; pt++) {
		Generator_idx[pt] = -1;
	}
	Has_been_reached = NEW_OBJECT(bitvector);
	Has_been_reached->allocate(A_given->degree);

	nb_reached = 0;

	Orbit_allocated = 1024;
	Orbit = NEW_int(Orbit_allocated);

	Q_allocated = 1024;
	Q = NEW_int(Q_allocated);
	
	nb_gens = gens->len;

	if (A_given->degree > ONE_MILLION) {
		f_v = TRUE;
	}

	Nb_per_generator = NEW_int(nb_gens);
	Orbiter->Int_vec.zero(Nb_per_generator, nb_gens);
	Pts_per_generator = NEW_pint(nb_gens);

	for (pt = 0; pt < A_given->degree; pt++) {
		if (Has_been_reached->s_i(pt)) {
			continue;
		}
		if (f_vv) {
			cout << "strong_generators::orbits_light "
					"computing orbit of point " << pt << endl;
		}
		Q[0] = pt;
		Q_len = 1;
		
		Orbit[0] = pt;
		Orbit_len = 1;

		while (Q_len) {
			if (f_vv) {
				cout << "strong_generators::orbits_light "
						"considering the next element in the queue" << endl;
			}
			a = Q[0];
			for (i = 1; i < Q_len; i++) {
				Q[i - 1] = Q[i];
			}
			Q_len--;
			if (f_vv) {
				cout << "strong_generators::orbits_light "
						"looking at element " << a << endl;
			}
			for (h = 0; h < nb_gens; h++) {
				if (f_vv) {
					cout << "strong_generators::orbits_light "
							"applying generator " << h << endl;
				}
				b = A_given->element_image_of(a, gens->ith(h), FALSE);
				if (f_vv) {
					cout << "strong_generators::orbits_light "
							"under generator " << h
							<< " it maps to " << b << endl;
				}
				if (!Sorting.int_vec_search(Orbit, Orbit_len, b, idx)) {
					if (Orbit_len == Orbit_allocated) {
						int new_oa;
						int *O;

						new_oa = 2 * Orbit_allocated;
						O = NEW_int(new_oa);
						for (i = 0; i < Orbit_len; i++) {
							O[i] = Orbit[i];
						}
						FREE_int(Orbit);
						Orbit = O;
						Orbit_allocated = new_oa;
					}
					for (i = Orbit_len; i > idx; i--) {
						Orbit[i] = Orbit[i - 1];
					}
					Orbit[idx] = b;
					Orbit_len++;
					Generator_idx[b] = h;
					Nb_per_generator[h]++;

					if (f_vv) {
						cout << "current orbit: ";
						Orbiter->Int_vec.print(cout, Orbit, Orbit_len);
						cout << endl;
					}

					Has_been_reached->m_i(b, 1);
					nb_reached++;
					if (f_v && ((nb_reached & ((1 << 18) - 1)) == 0)) {
						cout << "strong_generators::orbits_light "
								"nb_reached =  " << nb_reached << " / "
								<< A_given->degree << endl;
					}

					if (Q_len == Q_allocated) {
						int new_qa;
						int *new_Q;

						new_qa = 2 * Q_allocated;
						new_Q = NEW_int(new_qa);
						for (i = 0; i < Q_len; i++) {
							new_Q[i] = Q[i];
						}
						FREE_int(Q);
						Q = new_Q;
						Q_allocated = new_qa;
					}

					Q[Q_len++] = b;

					if (f_vv) {
						cout << "current Queue: ";
						Orbiter->Int_vec.print(cout, Q, Q_len);
						cout << endl;
					}

				}
			} // next h
		} // while (Q_len)

		if (f_vv) {
			cout << "Orbit of point " << pt << " has length "
					<< Orbit_len << endl;
		}
		if (nb_orbits == Orbit_reps_allocated) {
			int an;
			int *R;
			int *L;

			an = 2 * Orbit_reps_allocated;
			R = NEW_int(an);
			L = NEW_int(an);
			for (i = 0; i < nb_orbits; i++) {
				R[i] = Orbit_reps[i];
				L[i] = Orbit_lengths[i];
			}
			FREE_int(Orbit_reps);
			FREE_int(Orbit_lengths);
			Orbit_reps = R;
			Orbit_lengths = L;
			Orbit_reps_allocated = an;
		}
		Orbit_reps[nb_orbits] = pt;
		Orbit_lengths[nb_orbits] = Orbit_len;
		nb_orbits++;
	} // for pt
	if (f_v) {
		cout << "strong_generators::orbits_light degree = "
				<< A_given->degree << " we found " << nb_orbits
				<< " orbits" << endl;
		cout << i << " : " << Nb_per_generator[i] << endl;
		for (i = 0; i < nb_gens; i++) {
			cout << i << " : " << Nb_per_generator[i] << endl;
		}
	}


	if (f_v) {
		cout << "strong_generators::orbits_light computing the arrays "
				"Pts_per_generator" << endl;
	}
	for (i = 0; i < nb_gens; i++) { 
		int *v;
		int j;

		v = NEW_int(Nb_per_generator[i]);
		j = 0;
		for (pt = 0; pt < A_given->degree; pt++) {
			if (Generator_idx[pt] == i) {
				v[j] = pt;
				j++;
			}
		}
		if (j != Nb_per_generator[i]) {
			cout << "strong_generators::orbits_light j != Nb_per_generator[i]" << endl;
			exit(1);
		}
		Pts_per_generator[i] = v;
	}

	FREE_int(Orbit);
	FREE_int(Q);
	//FREE_uchar(reached);
	FREE_OBJECT(Has_been_reached);
	FREE_int(Generator_idx);
	//FREE_int(Nb_per_generator);
	if (f_v) {
		cout << "strong_generators::orbits_light degree = "
				<< A_given->degree << " we found "
				<< nb_orbits << " orbits" << endl;
	}
}


void strong_generators::write_to_file_binary(
		ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "strong_generators::write_to_file_binary" << endl;
	}

	if (!A->f_has_base()) {
		cout << "strong_generators::write_to_file_binary "
				"!A->f_has_base" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "strong_generators::write_to_file_binary "
				"A->base_len=" << A->base_len() << endl;
	}
	int bl;

	bl = A->base_len();
	fp.write((char *) &bl, sizeof(int));

	if (tl == NULL) {
		cout << "strong_generators::write_to_file_binary tl == NULL" << endl;
		exit(1);
	}
	for (i = 0; i < A->base_len(); i++) {
		if (f_v) {
			cout << "strong_generators::write_to_file_binary "
					"before writing tl[" << i << "]" << endl;
		}
		fp.write((char *) &tl[i], sizeof(int));
	}
	if (f_v) {
		cout << "strong_generators::write_to_file_binary "
				"before gens->write_to_file_binary" << endl;
	}
	gens->write_to_file_binary(fp, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "strong_generators::write_to_file_binary "
				"after gens->write_to_file_binary" << endl;
	}
	if (f_v) {
		cout << "strong_generators::write_to_file_binary done" << endl;
	}
}

void strong_generators::read_from_file_binary(
		action *A, ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l;

	if (f_v) {
		cout << "strong_generators::read_from_file_binary" << endl;
	}
	init(A, 0);
	if (f_v) {
		cout << "strong_generators::read_from_file_binary "
				"action A=" << A->label << endl;
	}
	fp.read((char *) &l, sizeof(int));
	if (l != A->base_len()) {
		cout << "strong_generators::read_from_file_binary "
				"l != A->base_len()" << endl;
		cout << "l=" << l << endl;
		cout << "A->base_len()=" << A->base_len() << endl;
		exit(1);
	}
	if (f_v) {
		cout << "strong_generators::read_from_file_binary "
				"A->base_len()=" << A->base_len() << endl;
	}
	tl = NEW_int(A->base_len());
	for (i = 0; i < A->base_len(); i++) {
		fp.read((char *) &tl[i], sizeof(int));
	}
	gens = NEW_OBJECT(vector_ge);
	gens->init(A, verbose_level - 2);
	if (f_v) {
		cout << "strong_generators::read_from_file_binary "
				"before gens->read_from_file_binary" << endl;
	}
	gens->read_from_file_binary(fp, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "strong_generators::read_from_file_binary done" << endl;
	}
}

void strong_generators::write_file(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;
	
	if (f_v) {
		cout << "strong_generators::write_file" << endl;
	}
	{
		ofstream fp(fname, ios::binary);

		write_to_file_binary(fp, verbose_level - 1);
	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "strong_generators::write_file done" << endl;
	}
}

void strong_generators::read_file(action *A,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "strong_generators::read_file reading "
				"file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	if (Fio.file_size(fname) <= 0) {
		cout << "strong_generators::read_file "
				"file " << fname << " does not exist" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "strong_generators::read_file reading file " << fname << endl;
	}

	{
		ifstream fp(fname, ios::binary);

		read_from_file_binary(A, fp, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "strong_generators::read_file "
			"Read file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "strong_generators::read_file done" << endl;
	}
}


#if 0
void strong_generators::generators_for_shallow_schreier_tree(
		char *label, vector_ge *chosen_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *AR;
	sims *S;
	int go;
	double avg;
	double log_go;
	int cnt = 0;
	int i;
	
	go = group_order_as_int();
	log_go = log(go);
	if (f_v) {
		cout << "strong_generators::generators_for_shallow_"
				"schreier_tree group of order " << go << endl;
		cout << "log_go = " << log_go << endl;
		}
	S = create_sims(verbose_level - 2);
	if (f_v) {
		cout << "strong_generators::generators_for_shallow_"
				"schreier_tree created sims" << endl;
		}
	AR = new_action_by_right_multiplication(S,
			TRUE /* f_transfer_ownership */, verbose_level - 2);
	if (f_v) {
		cout << "strong_generators::generators_for_shallow_"
				"schreier_tree created action by right "
				"multiplication" << endl;
		}

	chosen_gens->init(A);
	chosen_gens->allocate(gens->len);
	for (i = 0; i < gens->len; i++) {
		A->element_move(gens->ith(i), chosen_gens->ith(i), 0);
		}

	while (TRUE) {

		schreier *Sch;

		Sch = NEW_OBJECT(schreier);
		Sch->init(AR);
		Sch->initialize_tables();
		Sch->init_generators(*chosen_gens);
		if (f_v) {
			cout << "strong_generators::generators_for_shallow_"
					"schreier_tree before computing all orbits" << endl;
			}
		Sch->compute_all_point_orbits(verbose_level - 2);
		if (f_v) {
			cout << "strong_generators::generators_for_shallow_"
					"schreier_tree after computing all orbits" << endl;
			}
		if (Sch->nb_orbits > 1) {
			cout << "strong_generators::generators_for_shallow_"
					"schreier_tree Sch->nb_orbits > 1" << endl;
			exit(1);
			}
		char label1[1000];
		int xmax = 1000000;
		int ymax = 1000000;
		int f_circletext = TRUE;
		int rad = 3000;

		sprintf(label1, "%s_%d", label, cnt);
		Sch->draw_tree(label1, 0 /* orbit_no */,
			xmax, ymax, f_circletext, rad,
			TRUE /* f_embedded */, FALSE /* f_sideways */, 
			0.3 /* scale */, 1. /* line_width */, 
			FALSE, NULL, 
			0 /* verbose_level */);

		
		int *Depth;
		int avgi, f, /*l,*/ idx;

		Depth = NEW_int(Sch->A->degree);
		for (i = 0; i < Sch->A->degree; i++) {
			Depth[i] = Sch->depth_in_tree(i);
			}
		tally Cl;

		Cl.init(Depth, Sch->A->degree, FALSE, 0);
		if (f_v) {
			cout << "distribution of depth in tree is: ";
			Cl.print(TRUE);
			cout << endl;
			}
		avg = Cl.average();
		if (f_v) {
			cout << "average = " << avg << endl;
			cout << "log_go = " << log_go << endl;
			}

		if (avg < log_go) {
			if (f_v) {
				cout << "strong_generators::generators_for_shallow_"
					"schreier_tree average < log_go, we are done" << endl;
				}
			break;
			}

		avgi = (int) avg;
		if (f_v) {
			cout << "average as int = " << avgi << endl;
			}
		f = 0;
		for (i = 0; i < Cl.nb_types; i++) {
			f = Cl.type_first[i];
			//l = Cl.type_len[i];
			if (Cl.data_sorted[f] == avgi) {
				break;
				}
			}
		if (i == Cl.nb_types) {
			cout << "strong_generators::generators_for_shallow_"
				"schreier_tree cannot find element of depth "
				<< avgi << endl;
			exit(1);
			}
		idx = Cl.sorting_perm_inv[f];
		if (f_v) {
			cout << "strong_generators::generators_for_shallow_"
					"schreier_tree idx = " << idx << endl;
			}
		Sch->coset_rep(idx);
		chosen_gens->append(Sch->cosetrep);
		

		FREE_int(Depth);
		FREE_OBJECT(Sch);
		cnt++;
		}

		
	if (f_v) {
		cout << "strong_generators::generators_for_shallow_"
				"schreier_tree done" << endl;
		}
}
#endif

void strong_generators::compute_ascii_coding(
		char *&ascii_coding, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int sz, i, j;
	char *p;
	os_interface Os;

	if (f_v) {
		cout << "strong_generators::compute_ascii_coding" << endl;
	}
	sz = 2 * ((2 + A->base_len() + A->base_len()) * sizeof(int_4) +
			A->coded_elt_size_in_char * gens->len) + 1;
	ascii_coding = NEW_char(sz);
	p = ascii_coding;
	Os.code_int4(p, (int_4) A->base_len());

	Os.code_int4(p, (int_4) gens->len);
	for (i = 0; i < A->base_len(); i++) {
		Os.code_int4(p, (int_4) A->base_i(i));
	}
	for (i = 0; i < A->base_len(); i++) {
		Os.code_int4(p, (int_4) tl[i]);
	}
	for (i = 0; i < gens->len; i++) {
		A->element_pack(gens->ith(i), A->elt1, FALSE);
		for (j = 0; j < A->coded_elt_size_in_char; j++) {
			Os.code_uchar(p, A->elt1[j]);
		}
	}
	*p++ = 0;
	if (p - ascii_coding != sz) {
		cout << "strong_generators::compute_ascii_coding "
				"p - ascii_coding != sz" << endl;
		exit(1);
	}
	

	if (f_v) {
		cout << "strong_generators::compute_ascii_coding "
				"done" << endl;
	}
}

void strong_generators::decode_ascii_coding(
		char *ascii_coding, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int str_len, len, nbsg, i, j;
	char *p, *p0;
	action *A_save;
	int *base1;
	os_interface Os;

	if (f_v) {
		cout << "strong_generators::decode_ascii_coding" << endl;
	}

	// clean up before we go:
	A_save = A;
	freeself();
	A = A_save;

	p = ascii_coding;
	p0 = p;
	str_len = strlen(ascii_coding);
	len = Os.decode_int4(p);
	nbsg = Os.decode_int4(p);
	if (len != A->base_len()) {
		cout << "strong_generators::decode_ascii_coding "
				"len != A->base_len" << endl;
		cout << "len=" << len << " (from file)" << endl;
		cout << "A->base_len=" << A->base_len() << endl;
		cout << "action A is " << A->label << endl;
		exit(1);
	}
	gens = NEW_OBJECT(vector_ge);
	gens->init(A, verbose_level - 2);
	gens->allocate(nbsg, verbose_level - 2);
	base1 = NEW_int(A->base_len());
	tl = NEW_int(A->base_len());
	for (i = 0; i < A->base_len(); i++) {
		base1[i] = Os.decode_int4(p);
		}
	for (i = 0; i < A->base_len(); i++) {
		if (base1[i] != A->base_i(i)) {
			cout << "strong_generators::decode_ascii_coding "
					"base element " << i << " does not match "
					"current base" << endl;
			exit(1);
		}
	}
	for (i = 0; i < A->base_len(); i++) {
		tl[i] = Os.decode_int4(p);
	}
	for (i = 0; i < nbsg; i++) {
		for (j = 0; j < A->coded_elt_size_in_char; j++) {
			Os.decode_uchar(p, A->elt1[j]);
		}
		A->element_unpack(A->elt1, gens->ith(i), FALSE);
	}
	FREE_int(base1);
	if (p - p0 != str_len) {
		cout << "strong_generators::decode_ascii_coding "
				"p - p0 != str_len" << endl;
		cout << "p - p0 = " << p - p0 << endl;
		cout << "str_len = " << str_len << endl;
		exit(1);
	}
	if (f_v) {
		cout << "strong_generators::decode_ascii_coding done" << endl;
	}
}

void strong_generators::export_permutation_group_to_magma(
		std::string &fname, action *A2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	file_io Fio;

	if (f_v) {
		cout << "strong_generators::export_permutation_group_to_magma" << endl;
	}
	{
		ofstream fp(fname);

		fp << "G := sub< Sym(" << A2->degree << ") |" << endl;
		for (i = 0; i < gens->len; i++) {
			A2->element_print_as_permutation_with_offset(
				gens->ith(i), fp,
				1 /* offset */,
				TRUE /* f_do_it_anyway_even_for_big_degree */,
				FALSE /* f_print_cycles_of_length_one */,
				0 /* verbose_level */);
			if (i < gens->len - 1) {
				fp << ", " << endl;
			}
		}
		fp << ">;" << endl;

	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "strong_generators::export_permutation_group_to_magma done" << endl;
	}
}

void strong_generators::export_permutation_group_to_GAP(
		std::string &fname, action *A2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	file_io Fio;

	if (f_v) {
		cout << "strong_generators::export_permutation_group_to_GAP" << endl;
	}
	{
		ofstream fp(fname);

		fp << "G := Group([" << endl;
		for (i = 0; i < gens->len; i++) {
			A2->element_print_as_permutation_with_offset(
				gens->ith(i), fp,
				1 /* offset */,
				TRUE /* f_do_it_anyway_even_for_big_degree */,
				FALSE /* f_print_cycles_of_length_one */,
				0 /* verbose_level */);
			if (i < gens->len - 1) {
				fp << ", " << endl;
			}
		}
		fp << "]);" << endl;

	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "strong_generators::export_permutation_group_to_GAP done" << endl;
	}
}






void strong_generators::compute_and_print_orbits_on_a_given_set(
		action *A_given, long int *set, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	schreier *Sch;
	int i, j, f, l, a;

	if (f_v) {
		cout << "strong_generators::compute_and_print_orbits_on_a_given_set" << endl;
	}
	compute_schreier_with_given_action_on_a_given_set(
			A_given, Sch, set, len, verbose_level - 2);

	cout << "orbits on the set: " << endl;
	for (i = 0; i < Sch->nb_orbits; i++) {
		f = Sch->orbit_first[i];
		for (j = 0; j < Sch->orbit_len[i]; j++) {
			a = Sch->orbit[f + j];
			cout << a << " ";
		}
		if (i < Sch->nb_orbits - 1) {
			cout << "| ";
		}
	}
	cout << endl;
	cout << "partition: " << len << " = ";
	for (i = 0; i < Sch->nb_orbits; i++) {
		l = Sch->orbit_len[i];
		cout << l << " ";
		if (i < Sch->nb_orbits - 1) {
			cout << "+ ";
		}
	}
	cout << endl;
	cout << "representatives for each of the "
			<< Sch->nb_orbits << " orbits:" << endl;
	for (i = 0; i < Sch->nb_orbits; i++) {
		f = Sch->orbit_first[i];
		l = Sch->orbit_len[i];
		a = Sch->orbit[f + 0];
		cout << setw(5) << a << " : " << setw(5) << l << " : ";
		A_given->print_point(a, cout);
		cout << endl;
	}
	FREE_OBJECT(Sch);

}

void strong_generators::compute_and_print_orbits(
		action *A_given, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	schreier *Sch;
	int i, j, f, l, a;

	if (f_v) {
		cout << "strong_generators::compute_and_print_orbits" << endl;
	}
	compute_schreier_with_given_action(A_given, Sch, verbose_level - 2);

	cout << "orbits on the set: " << endl;
	for (i = 0; i < Sch->nb_orbits; i++) {
		f = Sch->orbit_first[i];
		l = Sch->orbit_len[i];
		if (l >= 10) {
			cout << "too long to list ";
		}
		else {
			for (j = 0; j < Sch->orbit_len[i]; j++) {
				a = Sch->orbit[f + j];
				cout << a << " ";
			}
		}
		if (i < Sch->nb_orbits - 1) {
			cout << "| ";
		}
	}
	cout << endl;
	cout << "partition: " << A_given->degree << " = ";
	for (i = 0; i < Sch->nb_orbits; i++) {
		l = Sch->orbit_len[i];
		cout << l << " ";
		if (i < Sch->nb_orbits - 1) {
			cout << "+ ";
		}
	}
	cout << endl;
	cout << "representatives for each of the "
			<< Sch->nb_orbits << " orbits:" << endl;
	for (i = 0; i < Sch->nb_orbits; i++) {
		f = Sch->orbit_first[i];
		l = Sch->orbit_len[i];
		a = Sch->orbit[f + 0];
		cout << setw(5) << a << " : " << setw(5) << l << " : ";
		A_given->print_point(a, cout);
		cout << endl;
	}

#if 0
	set_of_sets *S;

	Sch->orbits_as_set_of_sets(S, 0 /* verbose_level */);

	const char *fname = "orbits.csv";
	
	cout << "writing orbits to file " << fname << endl;
	S->save_csv(fname, 1 /* verbose_level */);
	
	FREE_OBJECT(S);
#endif
	FREE_OBJECT(Sch);

}

int strong_generators::test_if_normalizing(sims *S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "strong_generators::test_if_normalizing" << endl;
	}
	for (i = 0; i < gens->len; i++) {
		if (f_v) {
			cout << "strong_generators::test_if_normalizing "
					"testing generator " << i << " / "
					<< gens->len << endl;
		}
		if (!S->is_normalizing(gens->ith(i), verbose_level)) {
			if (f_v) {
				cout << "strong_generators::test_if_normalizing "
						"generator " << i << " / " << gens->len
						<< " does not normalize the given group" << endl;
			}
			return FALSE;
		}
	}
	if (f_v) {
		cout << "strong_generators::test_if_normalizing done, "
				"the given generators normalize the given group" << endl;
	}
	return TRUE;
}


void strong_generators::test_if_set_is_invariant_under_given_action(
		action *A_given, long int *set, int set_sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "strong_generators::test_if_set_is_invariant_under_given_action" << endl;
	}
	for (i = 0; i < gens->len; i++) {

		if (!A_given->test_if_set_stabilizes(gens->ith(i),
				set_sz, set, 0 /* verbose_level */)) {
			cout << "strong_generators::test_if_set_is_invariant_under_given_action "
					"the generator does not fix the set" << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "strong_generators::test_if_set_is_invariant_under_given_action done" << endl;
	}
}

strong_generators *strong_generators::point_stabilizer(
		int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::point_stabilizer" << endl;
	}

	schreier *Sch;
	sims *Stab;
	strong_generators *Stab_gens;
	longinteger_object G_order, stab_go;

	if (f_v) {
		cout << "computing orbit of point " << pt << ":" << endl;
	}
	group_order(G_order);
	Sch = orbit_of_one_point_schreier(A, pt, verbose_level);
	if (f_v) {
		cout << "orbit of point " << pt << " has length "
				<< Sch->orbit_len[0] << endl;
	}
	Sch->point_stabilizer(A, G_order, 
		Stab, 0 /* orbit_no */, verbose_level);
	Stab->group_order(stab_go);
	if (f_v) {
		cout << "stabilizer of point " << pt << " has order "
				<< stab_go << endl;
	}
	Stab_gens = NEW_OBJECT(strong_generators);
	Stab_gens->init_from_sims(Stab, verbose_level);
	if (f_v) {
		cout << "generators for the stabilizer "
				"have been computed" << endl;
	}
	
	if (f_v) {
		cout << "strong_generators::point_stabilizer done" << endl;
	}
	FREE_OBJECT(Sch);
	FREE_OBJECT(Stab);
	return Stab_gens;
}

strong_generators *strong_generators::find_cyclic_subgroup_with_exactly_n_fixpoints(
		int nb_fixpoints, action *A_given, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::find_cyclic_subgroup_with_exactly_n_fixpoints" << endl;
	}

	sims *H;
	int *Elt;
	int order;
	strong_generators *Sub_gens;

	if (f_v) {
		cout << "finding element with n fixpoints, where n = " << nb_fixpoints << ":" << endl;
	}

	H = create_sims(verbose_level - 2);

	Elt = NEW_int(A->elt_size_in_int);

	order = H->find_element_with_exactly_n_fixpoints_in_given_action(
			Elt, nb_fixpoints, A_given, verbose_level);

	Sub_gens = NEW_OBJECT(strong_generators);
	if (f_v) {
		cout << "strong_generators::find_cyclic_subgroup_with_exactly_n_fixpoints "
				"before init_single_with_target_go" << endl;
		cout << "Elt=" << endl;
		A->element_print(Elt, cout);
	}
	Sub_gens->init_single_with_target_go(A, Elt, order, verbose_level);
	if (f_v) {
		cout << "strong_generators::find_cyclic_subgroup_with_exactly_n_fixpoints "
				"after init_single_with_target_go" << endl;
	}

	if (f_v) {
		cout << "strong_generators::find_cyclic_subgroup_with_exactly_n_fixpoints done" << endl;
	}
	FREE_int(Elt);
	FREE_OBJECT(H);
	return Sub_gens;
}


void strong_generators::make_element_which_moves_a_point_from_A_to_B(
	action *A_given,
	int pt_A, int pt_B, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::make_element_which_moves_a_point_from_A_to_B" << endl;
	}

	schreier *Orb;
	int orbit_idx;
	int len;


	Orb = orbit_of_one_point_schreier(A_given, pt_A,
			0 /*verbose_level */);
	len = Orb->orbit_len[0];
	if (Orb->orbit_inv[pt_B] >= len) {
		cout << "strong_generators::make_element_which_moves_"
				"a_point_from_A_to_B the two points are not "
				"in the same orbit" << endl;
		exit(1);
	}
	Orb->transporter_from_orbit_rep_to_point(pt_B, orbit_idx, Elt,
			0 /* verbose_level */);

	if (A_given->element_image_of(pt_A, Elt, 0 /* verbose_level*/)
			!= pt_B) {
		cout << "strong_generators::make_element_which_moves_a_point_from_A_to_B "
				"the image of A is not B" << endl;
		exit(1);
	}

	FREE_OBJECT(Orb);
	if (f_v) {
		cout << "strong_generators::make_element_which_moves_a_point_from_A_to_B done" << endl;
	}
}

void strong_generators::export_group_to_magma_and_copy_to_latex(
		std::string &label_txt,
		ostream &ost,
		action *A2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;


	if (f_v) {
		cout << "strong_generators::export_group_to_magma_and_copy_to_latex" << endl;
	}
	string export_fname;

	export_fname.assign(label_txt);
	export_fname.append("_group.magma");

	export_permutation_group_to_magma(
			export_fname, A2, verbose_level - 2);
	if (f_v) {
		cout << "written file " << export_fname << " of size "
				<< Fio.file_size(export_fname) << endl;
	}

	ost << "\\subsection*{Magma Export}" << endl;
	ost << "To export the group to Magma, "
			"use the following file\\\\" << endl;
	ost << "\\begin{verbatim}" << endl;

	{
		ifstream fp1(export_fname);
		char line[100000];

		while (TRUE) {
			if (fp1.eof()) {
				break;
			}

			//cout << "count_number_of_orbits_in_file reading
			//line, nb_sol = " << nb_sol << endl;
			fp1.getline(line, 100000, '\n');
			ost << line << endl;
		}

	}
	ost << "\\end{verbatim}" << endl;

	if (f_v) {
		cout << "strong_generators::export_group_to_magma_and_copy_to_latex done" << endl;
	}
}

void strong_generators::export_group_to_GAP_and_copy_to_latex(
		std::string &label_txt,
		ostream &ost,
		action *A2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;


	if (f_v) {
		cout << "strong_generators::export_group_to_GAP_and_copy_to_latex" << endl;
	}
	string export_fname;

	export_fname.assign(label_txt);
	export_fname.append("_group.gap");

	export_permutation_group_to_GAP(
			export_fname, A2, verbose_level - 2);
	if (f_v) {
		cout << "written file " << export_fname << " of size "
				<< Fio.file_size(export_fname) << endl;
	}

	ost << "\\subsection*{GAP Export}" << endl;
	ost << "To export the group to GAP, "
			"use the following file\\\\" << endl;
	ost << "\\begin{verbatim}" << endl;

	{
		ifstream fp1(export_fname);
		char line[100000];

		while (TRUE) {
			if (fp1.eof()) {
				break;
			}

			//cout << "count_number_of_orbits_in_file reading
			//line, nb_sol = " << nb_sol << endl;
			fp1.getline(line, 100000, '\n');
			ost << line << endl;
		}

	}
	ost << "\\end{verbatim}" << endl;

	if (f_v) {
		cout << "strong_generators::export_group_to_GAP_and_copy_to_latex done" << endl;
	}
}

void strong_generators::export_group_and_copy_to_latex(
		std::string &label_txt,
		ostream &ost,
		action *A2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "strong_generators::export_group_and_copy_to_latex" << endl;
	}
	export_group_to_magma_and_copy_to_latex(label_txt, ost, A2, verbose_level);
	export_group_to_GAP_and_copy_to_latex(label_txt, ost, A2, verbose_level);
	if (f_v) {
		cout << "strong_generators::export_group_and_copy_to_latex done" << endl;
	}

}


void strong_generators::report_fixed_objects_in_P3(
		ostream &ost,
		projective_space *P3,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "strong_generators::report_fixed_objects_in_P3" << endl;
	}

	ost << "\\begin{enumerate}[(1)]" << endl;
	for (i = 0; i < gens->len; i++) {

		ost << "\\item" << endl;
		A->report_fixed_objects_in_P3(ost,
				P3,
				gens->ith(i),
				verbose_level);
	}
	ost << "\\end{enumerate}" << endl;
	if (f_v) {
		cout << "strong_generators::report_fixed_objects_in_P3" << endl;
	}
}

void strong_generators::reverse_isomorphism_exterior_square(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::reverse_isomorphism_exterior_square" << endl;
	}

	if (f_v) {
		cout << "strong_generators::reverse_isomorphism_exterior_square "
				"before gens->reverse_isomorphism_exterior_square" << endl;
	}

	gens->reverse_isomorphism_exterior_square(verbose_level);

	if (f_v) {
		cout << "strong_generators::reverse_isomorphism_exterior_square "
				"after gens->reverse_isomorphism_exterior_square" << endl;
	}

	if (f_v) {
		cout << "strong_generators::reverse_isomorphism_exterior_square" << endl;
	}
}

void strong_generators::get_gens_data(int *&data, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::get_gens_data" << endl;
	}
	int i;

	sz = gens->len * A->make_element_size;
	data = NEW_int(sz);
	for (i = 0; i < gens->len; i++) {
		Orbiter->Int_vec.copy(gens->ith(i), data + i * A->make_element_size, A->make_element_size);
	}
	if (f_v) {
		cout << "strong_generators::get_gens_data done" << endl;
	}
}

void strong_generators::get_gens_data_as_string_with_quotes(std::string &str, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::get_gens_data_as_string_with_quotes" << endl;
	}
	int *data;
	int sz;

	get_gens_data(data, sz, verbose_level);


	Orbiter->Int_vec.create_string_with_quotes(str, data, sz);

	if (f_v) {
		cout << "strong_generators::get_gens_data_as_string_with_quotes done" << endl;
	}
}

void strong_generators::export_to_orbiter_as_bsgs(
		action *A2,
		std::string &fname, std::string &label, std::string &label_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int a;
	file_io Fio;
	longinteger_object go;

	if (f_v) {
		cout << "strong_generators::export_to_orbiter_as_bsgs" << endl;
	}

	group_order(go);
	if (f_v) {
		cout << "strong_generators::export_to_orbiter_as_bsgs go = " << go << endl;
		cout << "strong_generators::export_to_orbiter_as_bsgs number of generators = " << gens->len << endl;
		cout << "strong_generators::export_to_orbiter_as_bsgs degree = " << A2->degree << endl;
	}
	{
		ofstream fp(fname);

		string fname_generators;

		fname_generators.assign(label);
		fname_generators.append("_gens.csv");


#if 0
		for (i = 0; i < gens->len; i++) {
			fp << "GENERATOR_" << label << "_" << i << " = \\" << endl;
			fp << "\t\"";
			for (j = 0; j < A2->degree; j++) {
				if (FALSE) {
					cout << "strong_generators::export_to_orbiter_as_bsgs computing image of " << j << " under generator " << i << endl;
				}
				a = A2->element_image_of(j, gens->ith(i), 0 /* verbose_level*/);
				fp << a;
				if (j < A2->degree - 1) {
					fp << ",";
				}
			}
			fp << "\"";
			fp << endl;
		}
#else
		{
			long int *Data;

			Data = NEW_lint(gens->len * A2->degree);
			for (i = 0; i < gens->len; i++) {
				for (j = 0; j < A2->degree; j++) {
					a = A2->element_image_of(j, gens->ith(i), 0 /* verbose_level*/);
					Data[i * A2->degree + j] = a;
				}
			}


			Fio.lint_matrix_write_csv(fname_generators, Data, gens->len, A2->degree);


			FREE_lint(Data);
		}
#endif

		fp << endl;
		fp << label << ":" << endl;
		fp << "\t$(ORBITER_PATH)orbiter.out -v 2 \\" << endl;
		fp << "\t\t-define gens -vector -file " << fname_generators << " -end \\" << endl;
		fp << "\t\t-define G -permutation_group \\" << endl;
		fp << "\t\t-bsgs " << label << " \"" << label_tex << "\" "
				<< A2->degree << " " << go << " ";
		fp << "\"";
		A->print_bare_base(fp);
		fp << "\"";
		fp << " ";
		fp << gens->len;
		fp << " gens -end \\" << endl;
#if 0
		for (i = 0; i < gens->len; i++) {
			fp << "\t\t\t" << "$(GENERATOR_" << label << "_" << i << ") \\" << endl;
		}
		fp << "\t\t-end" << endl;
#endif

		//$(ORBITER_PATH)orbiter.out -v 10 \
		//	-define G -permutation_group \
		//		-bsgs C13 C_{13} 13 13 0 1 \
		//			$(GEN_C13) \
		//		-end \

		// with backslashes at the end of the line

	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "strong_generators::export_to_orbiter_as_bsgs" << endl;
	}
}







}}


