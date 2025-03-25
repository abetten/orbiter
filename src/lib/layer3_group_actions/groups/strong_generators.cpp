// strong_generators.cpp
//
// Anton Betten
// December 4, 2013

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;

namespace orbiter {
namespace layer3_group_actions {
namespace groups {



strong_generators::strong_generators()
{
	Record_birth();
	A = NULL;
	tl = NULL;
	gens = NULL;
}

strong_generators::~strong_generators()
{
	Record_death();
	if (tl) {
		FREE_int(tl);
	}
	if (gens) {
		FREE_OBJECT(gens);
	}
}

void strong_generators::swap_with(
		strong_generators *SG)
{
	actions::action *my_A;
	int *my_tl;
	data_structures_groups::vector_ge *my_gens;

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

void strong_generators::init(
		actions::action *A)
{
	init(A, 0);
}

void strong_generators::init(
		actions::action *A, int verbose_level)
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

void strong_generators::init_from_sims(
		groups::sims *S, int verbose_level)
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
	gens = NEW_OBJECT(data_structures_groups::vector_ge);
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
		Int_vec_print(cout, tl, A->base_len());
		cout << endl;
	}
	if (f_v) {
		cout << "strong_generators::init_from_sims done" << endl;
	}
}

void strong_generators::init_from_ascii_coding(
		actions::action *A,
		std::string &ascii_coding, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	algebra::ring_theory::longinteger_object go;
	data_structures_groups::group_container *G;

	if (f_v) {
		cout << "strong_generators::init_from_ascii_coding" << endl;
	}
	G = NEW_OBJECT(data_structures_groups::group_container);
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


strong_generators *strong_generators::create_copy(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::create_copy" << endl;
	}
	strong_generators *S;

	S = NEW_OBJECT(strong_generators);
	if (f_v) {
		cout << "strong_generators::create_copy before S->init_copy" << endl;
	}
	S->init_copy(this, verbose_level - 1);
	if (f_v) {
		cout << "strong_generators::create_copy after S->init_copy" << endl;
	}
	if (f_v) {
		cout << "strong_generators::create_copy done" << endl;
	}
	return S;
}

void strong_generators::init_copy(
		strong_generators *S,
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
	Int_vec_copy(S->tl, tl, A->base_len());

	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init(A, verbose_level - 2);
	gens->allocate(S->gens->len, verbose_level - 2);
	for (i = 0; i < S->gens->len; i++) {
		//cout << "strong_generators::init_copy before
		// element_move i=" << i << endl;
		A->Group_element->element_move(S->gens->ith(i), gens->ith(i), 0);
	}
	if (f_v) {
		cout << "strong_generators::init_copy done" << endl;
	}
}

void strong_generators::init_by_hdl_and_with_tl(
		actions::action *A,
		std::vector<int> &gen_handle,
		std::vector<int> &tl,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "strong_generators::init_by_hdl_and_with_tl" << endl;
	}

	if (f_v) {
		cout << "strong_generators::init_by_hdl_and_with_tl A->base_len() = " << A->base_len() << endl;
	}

	init(A, 0);
	strong_generators::tl = NEW_int(A->base_len());
	for (i = 0; i < A->base_len(); i++) {
		strong_generators::tl[i] = tl[i];
	}
	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	if (f_v) {
		cout << "strong_generators::init_by_hdl_and_with_tl "
				"before gens->init_by_hdl" << endl;
	}
	gens->init_by_hdl(A, gen_handle, verbose_level);
	if (f_v) {
		cout << "strong_generators::init_by_hdl_and_with_tl "
				"after gens->init_by_hdl" << endl;
	}

	if (f_v) {
		cout << "strong_generators::init_by_hdl_and_with_tl done" << endl;
	}
}


void strong_generators::init_by_hdl(
		actions::action *A,
		int *gen_hdl, int nb_gen, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "strong_generators::init_by_hdl" << endl;
	}

	//int i;

	init(A, 0);
	tl = NEW_int(A->base_len());
	Int_vec_one(tl, A->base_len());
	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init_by_hdl(A, gen_hdl, nb_gen, 0 /*verbose_level */);
	if (f_v) {
		cout << "strong_generators::init_by_hdl done" << endl;
	}
}

void strong_generators::init_from_permutation_representation(
		actions::action *A,
		sims *parent_group_S, int *data,
	int nb_elements, long int group_order,
	data_structures_groups::vector_ge *&nice_gens,
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

	nice_gens = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "strong_generators::init_from_permutation_representation "
				"before nice_gens->init_from_permutation_representation" << endl;
	}
	nice_gens->init_from_permutation_representation(
			A, parent_group_S, data,
		nb_elements, verbose_level - 1);
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

	if (f_v) {
		cout << "strong_generators::init_from_permutation_representation "
				"done, found a group of order " << group_order << endl;
	}
}

void strong_generators::init_from_data(
		actions::action *A, int *data,
	int nb_elements, int elt_size, int *transversal_length, 
	data_structures_groups::vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::init_from_data" << endl;
		cout << "strong_generators::init_from_data A = " << A->label << endl;
	}
	init(A, verbose_level - 2);
	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	nice_gens = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "strong_generators::init_from_data "
				"before gens->init_from_data" << endl;
	}
	gens->init_from_data(
			A, data,
		nb_elements, elt_size,
		verbose_level - 1);
	if (f_v) {
		cout << "strong_generators::init_from_data "
				"after gens->init_from_data" << endl;
	}
	
	if (f_v) {
		cout << "strong_generators::init_from_data "
				"before nice_gens->init_from_data" << endl;
	}
	nice_gens->init_from_data(
			A, data,
		nb_elements, elt_size,
		verbose_level - 2);
	if (f_v) {
		cout << "strong_generators::init_from_data "
				"after nice_gens->init_from_data" << endl;
	}

	tl = NEW_int(A->base_len());
	Int_vec_copy(transversal_length, tl, A->base_len());

	if (f_v) {
		cout << "strong_generators::init_from_data done" << endl;
	}
}

void strong_generators::init_from_data_with_target_go_ascii(
		actions::action *A, int *data,
	int nb_elements, int elt_size, std::string &ascii_target_go,
	data_structures_groups::vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::ring_theory::longinteger_object target_go;

	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go_ascii" << endl;
	}
	strong_generators::A = A;
	target_go.create_from_base_10_string(ascii_target_go);
	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go_ascii "
				"before init_from_data_with_target_go" << endl;
	}
	init_from_data_with_target_go(
			A, data,
		elt_size, nb_elements, target_go,
		nice_gens,
		verbose_level);
	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go_ascii "
				"after init_from_data_with_target_go" << endl;
	}
	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go_ascii done" << endl;
	}
}

void strong_generators::init_from_data_with_target_go(
		actions::action *A, int *data_gens,
	int data_gens_size, int nb_gens,
	algebra::ring_theory::longinteger_object &target_go,
	data_structures_groups::vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i;

	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go" << endl;
	}

	strong_generators::A = A;

	nice_gens = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go "
				"before nice_gens->init_from_data" << endl;
	}
	nice_gens->init_from_data(
			A, data_gens,
			nb_gens, data_gens_size, verbose_level - 2);
	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go "
				"after nice_gens->init_from_data" << endl;
	}


	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go "
				"generators are:" << endl;
		nice_gens->print_quick(cout);
	}

	strong_generators *SG;

	SG = NEW_OBJECT(strong_generators);
	
	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go "
				"before A->generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		true /* f_target_go */, target_go, 
		nice_gens, SG,
		verbose_level - 1);
	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go "
				"after A->generators_to_strong_generators" << endl;
	}

	if (f_v) {
		cout << "strong_generators::init_from_data_with_target_go "
				"strong generators are:" << endl;
		SG->print_generators(cout, verbose_level - 1);
	}

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
		actions::action *A, std::string &generators_data,
	std::string &go_text,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::init_from_data_with_go" << endl;
	}


	int *gens_data;
	int gens_data_sz;
	int nb_elements;

	Int_vec_scan(generators_data, gens_data, gens_data_sz);
	if (f_v) {
		cout << "gens_data = ";
		Int_vec_print(cout, gens_data, gens_data_sz);
		cout << endl;
		cout << "go_text = " << go_text << endl;
	}


	init(A);


	nb_elements = gens_data_sz / A->make_element_size;

	data_structures_groups::vector_ge *nice_gens;

	if (f_v) {
		cout << "strong_generators::init_from_data_with_go "
				"before SG->init_from_data_with_target_go_ascii" << endl;
	}
	init_from_data_with_target_go_ascii(
			A,
			gens_data,
			nb_elements, A->make_element_size,
			go_text,
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
	int pt, int &orbit_idx, algebra::ring_theory::longinteger_object &full_group_order,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	strong_generators *SG0;
	
	if (f_v) {
		cout << "strong_generators::init_point_stabilizer_of_arbitrary_point_through_schreier" << endl;
	}

	Elt = NEW_int(A->elt_size_in_int);

	if (f_v) {
		cout << "strong_generators::init_point_stabilizer_of_arbitrary_point_through_schreier "
				"before Sch->transporter_from_point_to_orbit_rep" << endl;
	}
	Sch->transporter_from_point_to_orbit_rep(
			pt, orbit_idx, Elt,
			0 /* verbose_level */);
	if (f_v) {
		cout << "strong_generators::init_point_stabilizer_of_arbitrary_point_through_schreier "
				"after Sch->transporter_from_point_to_orbit_rep" << endl;
	}

	SG0 = NEW_OBJECT(strong_generators);
	SG0->init(A);

	if (f_v) {
		cout << "strong_generators::init_point_stabilizer_of_arbitrary_point_through_schreier "
				"before SG0->init_point_stabilizer_orbit_rep_schreier" << endl;
	}
	SG0->init_point_stabilizer_orbit_rep_schreier(
			Sch, orbit_idx,
			full_group_order, verbose_level);
	if (f_v) {
		cout << "strong_generators::init_point_stabilizer_of_arbitrary_point_through_schreier "
				"after SG0->init_point_stabilizer_orbit_rep_schreier" << endl;
	}

	if (f_v) {
		cout << "strong_generators::init_point_stabilizer_of_arbitrary_point_through_schreier "
				"before init_generators_for_the_conjugate_group_aGav" << endl;
	}
	init_generators_for_the_conjugate_group_aGav(
			SG0, Elt, 0 /* verbose_level */);
	if (f_v) {
		cout << "strong_generators::init_point_stabilizer_of_arbitrary_point_through_schreier "
				"after init_generators_for_the_conjugate_group_aGav" << endl;
	}
	
	FREE_OBJECT(SG0);
	FREE_int(Elt);

	if (f_v) {
		cout << "strong_generators::init_point_stabilizer_of_arbitrary_point_through_schreier "
				"done" << endl;
	}
}

void strong_generators::init_point_stabilizer_orbit_rep_schreier(
	schreier *Sch,
	int orbit_idx,
	algebra::ring_theory::longinteger_object &full_group_order,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sims *Stab;
	
	if (f_v) {
		cout << "strong_generators::init_point_stabilizer_orbit_rep_schreier" << endl;
	}
	Sch->point_stabilizer(
			A, full_group_order,
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
	data_structures_groups::vector_ge *gens;
	algebra::ring_theory::longinteger_object go;

	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_avGa" << endl;
	}
	
	A = SG->A;

	SG->group_order(go);
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_avGa "
				"go=" << go << endl;
	}
	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_avGa "
				"before gens->init_conjugate_svas_of" << endl;
	}
	gens->init_conjugate_svas_of(
			SG->gens, Elt_a, verbose_level);
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_avGa "
				"after gens->init_conjugate_svas_of" << endl;
	}

	strong_generators *SG1;
	
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_avGa "
				"before generators_to_strong_generators" << endl;
	}
	SG->A->generators_to_strong_generators(
		true /* f_target_go */, go, 
		gens, SG1, 
		0 /*verbose_level*/);
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_avGa "
				"after generators_to_strong_generators" << endl;
	}

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
	data_structures_groups::vector_ge *gens;
	algebra::ring_theory::longinteger_object go;
	//int i;	
	
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_aGav" << endl;
	}

	SG->group_order(go);
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_aGav "
				"go=" << go << endl;
	}
	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	gens->init_conjugate_sasv_of(
			SG->gens, Elt_a, 0 /* verbose_level */);


	strong_generators *SG1;
	
	if (f_v) {
		cout << "strong_generators::init_generators_for_the_conjugate_group_aGav "
				"before generators_to_strong_generators" << endl;
	}
	SG->A->generators_to_strong_generators(
		true /* f_target_go */, go, 
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
	data_structures_groups::vector_ge *gens;
	algebra::ring_theory::longinteger_object go;
	//int i;
	
	if (f_v) {
		cout << "strong_generators::init_transposed_group" << endl;
	}

	SG->group_order(go);
	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "strong_generators::init_transposed_group "
				"before gens->init_transposed" << endl;
	}
	gens->init_transposed(
			SG->gens,
			0 /* verbose_level*/);
	if (f_v) {
		cout << "strong_generators::init_transposed_group "
				"after gens->init_transposed" << endl;
	}


	strong_generators *SG1;
	
	if (f_v) {
		cout << "strong_generators::init_transposed_group "
				"before A->generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		true /* f_target_go */, go, 
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
	algebra::ring_theory::longinteger_object target_go;
	algebra::ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "strong_generators::init_group_extension" << endl;
	}

	A = subgroup->A;

	data_structures_groups::vector_ge *my_gens;
	int nb_gens;

	my_gens = NEW_OBJECT(data_structures_groups::vector_ge);
	my_gens->init(A, verbose_level - 2);
	nb_gens = subgroup->gens->len;
	my_gens->allocate(nb_gens + 1, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		A->Group_element->element_move(subgroup->gens->ith(i), my_gens->ith(i), 0);
	}
	A->Group_element->make_element(my_gens->ith(nb_gens), data, 0);

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
		true /* f_target_go */, target_go, 
		my_gens, SG, 
		0 /*verbose_level*/);

	if (false) {
		cout << "strong_generators::init_group_extension "
				"strong generators are:" << endl;
		SG->print_generators(cout, verbose_level - 1);
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
	data_structures_groups::vector_ge *new_gens, int index,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	algebra::ring_theory::longinteger_object target_go;
	algebra::ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "strong_generators::init_group_extension" << endl;
	}

	A = subgroup->A;

	data_structures_groups::vector_ge *my_gens;
	int nb_gens, nb_new_gens;

	my_gens = NEW_OBJECT(data_structures_groups::vector_ge);
	my_gens->init(A, verbose_level - 2);
	nb_gens = subgroup->gens->len;
	nb_new_gens = new_gens->len;
	my_gens->allocate(nb_gens + nb_new_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		A->Group_element->element_move(subgroup->gens->ith(i), my_gens->ith(i), 0);
	}
	for (i = 0; i < nb_new_gens; i++) {
		A->Group_element->element_move(new_gens->ith(i), my_gens->ith(nb_gens + i), 0);
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
		true /* f_target_go */, target_go, 
		my_gens, SG, 
		0 /*verbose_level - 2*/);

	if (false) {
		cout << "strong_generators::init_group_extension "
				"strong generators are:" << endl;
		SG->print_generators(cout, verbose_level - 1);
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
	std::string &rank_vector_text,
	std::string &subgroup_order_text, sims *S,
	int *&subgroup_gens_idx, int &nb_subgroup_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::ring_theory::longinteger_object target_go;

	if (f_v) {
		cout << "strong_generators::switch_to_subgroup" << endl;
	}

	

	Int_vec_scan(rank_vector_text, subgroup_gens_idx, nb_subgroup_gens);
	if (f_v) {
		cout << "strong_generators::switch_to_subgroup "
				"after scanning: ";
		Int_vec_print(cout, subgroup_gens_idx, nb_subgroup_gens);
		cout << endl;
	}


	data_structures_groups::vector_ge *my_gens;

	my_gens = NEW_OBJECT(data_structures_groups::vector_ge);
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
		true /* f_target_go */, target_go, 
		my_gens, SG, 
		0 /*verbose_level*/);

	if (false) {
		cout << "strong_generators::switch_to_subgroup "
				"strong generators are:" << endl;
		SG->print_generators(cout, verbose_level - 1);
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

void strong_generators::init_subgroup(
		actions::action *A,
	int *subgroup_gens_idx, int nb_subgroup_gens,
	const char *subgroup_order_text, 
	sims *S, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::ring_theory::longinteger_object target_go;

	if (f_v) {
		cout << "strong_generators::init_subgroup" << endl;
	}


	strong_generators::A = A;
	
	data_structures_groups::vector_ge *my_gens;

	my_gens = NEW_OBJECT(data_structures_groups::vector_ge);
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
		true /* f_target_go */, target_go, 
		my_gens, SG, 
		0 /*verbose_level*/);

	if (false) {
		cout << "strong_generators::init_subgroup "
				"strong generators are:" << endl;
		SG->print_generators(cout, verbose_level - 1);
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

void strong_generators::init_subgroup_by_generators(
		actions::action *A,
	int nb_subgroup_gens,
	int *subgroup_gens,
	std::string &subgroup_order_text,
	data_structures_groups::vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::ring_theory::longinteger_object target_go;

	if (f_v) {
		cout << "strong_generators::init_subgroup_by_generators" << endl;
	}


	strong_generators::A = A;


	nice_gens = NEW_OBJECT(data_structures_groups::vector_ge);

	nice_gens->init_from_data(
			A, subgroup_gens,
			nb_subgroup_gens, A->make_element_size,
			verbose_level - 2);



	if (f_v) {
		cout << "strong_generators::init_subgroup_by_generators "
				"chosen generators:" << endl;
		nice_gens->print_quick(cout);
	}

	target_go.create_from_base_10_string(subgroup_order_text);


	strong_generators *SG;

	SG = NEW_OBJECT(strong_generators);

	A->generators_to_strong_generators(
		true /* f_target_go */, target_go,
		nice_gens, SG,
		0 /*verbose_level*/);

	if (false) {
		cout << "strong_generators::init_subgroup_by_generators "
				"strong generators are:" << endl;
		SG->print_generators(cout, verbose_level - 1);
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
		cout << "strong_generators::create_sims "
				"verbose_level=" << verbose_level << endl;
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
		actions::action *A_given, int verbose_level)
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
		data_structures_groups::vector_ge *coset_reps, int group_index,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	sims *S;
	data_structures_groups::vector_ge *gens1;
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
		A->Group_element->element_move(coset_reps->ith(i),
				coset_reps_vec + i * A->elt_size_in_int, 0);
	}

	gens1 = NEW_OBJECT(data_structures_groups::vector_ge);
	tl1 = NEW_int(A->base_len());
	
	S = create_sims(verbose_level - 1);

	if (f_v) {
		cout << "strong_generators::add_generators "
				"before S->transitive_extension_using_coset_representatives_extract_generators" << endl;
	}
	S->transitive_extension_using_coset_representatives_extract_generators(
		coset_reps_vec, group_index, 
		*gens1, tl1, 
		verbose_level - 2);
	if (f_v) {
		cout << "strong_generators::add_generators "
				"after S->transitive_extension_using_coset_representatives_extract_generators" << endl;
	}

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
	data_structures_groups::vector_ge *gens1;
	int *tl1;

	if (f_v) {
		cout << "strong_generators::add_single_generator" << endl;
		cout << "action=";
		A->print_info();
	}

	gens1 = NEW_OBJECT(data_structures_groups::vector_ge);
	tl1 = NEW_int(A->base_len());
	
	S = create_sims(verbose_level - 1);

	if (f_v) {
		cout << "strong_generators::add_single_generator "
				"before S->transitive_extension_using_generators" << endl;
	}
	S->transitive_extension_using_generators(
		Elt, 1, group_index, 
		*gens1, tl1, 
		verbose_level);
	if (f_v) {
		cout << "strong_generators::add_single_generator "
				"after S->transitive_extension_using_generators" << endl;
	}

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

void strong_generators::group_order(
		algebra::ring_theory::longinteger_object &go)
{
	algebra::ring_theory::longinteger_domain D;

	D.multiply_up(go, tl, A->base_len(), 0 /* verbose_level */);
}

std::string strong_generators::group_order_stringify()
{
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object go;
	string s;

	D.multiply_up(go, tl, A->base_len(), 0 /* verbose_level */);
	s = go.stringify();
	return s;
}

long int strong_generators::group_order_as_lint()
{
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object go;

	D.multiply_up(go, tl, A->base_len(), 0 /* verbose_level */);
	return go.as_lint();
}

void strong_generators::print_group_order(
		std::ostream &ost)
{
	algebra::ring_theory::longinteger_object go;

	group_order(go);
	ost << go;
}

void strong_generators::print_generators_gap(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::print_generators_gap" << endl;
	}

	if (f_v) {
		cout << "strong_generators::print_generators_gap "
				"before gens->print_generators_gap" << endl;
	}
	gens->print_generators_gap(ost, verbose_level);
	if (f_v) {
		cout << "strong_generators::print_generators_gap "
				"after gens->print_generators_gap" << endl;
	}

	if (f_v) {
		cout << "strong_generators::print_generators_gap done" << endl;
	}
}


void strong_generators::print_generators_gap_in_different_action(
		std::ostream &ost, actions::action *A2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::print_generators_gap_in_different_action" << endl;
	}

	if (f_v) {
		cout << "strong_generators::print_generators_gap_in_different_action "
				"before gens->print_generators_gap_in_different_action" << endl;
	}
	gens->print_generators_gap_in_different_action(ost, A2, verbose_level);
	if (f_v) {
		cout << "strong_generators::print_generators_gap_in_different_action "
				"after gens->print_generators_gap_in_different_action" << endl;
	}

	if (f_v) {
		cout << "strong_generators::print_generators_gap_in_different_action done" << endl;
	}
}


void strong_generators::print_generators_compact(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::print_generators_compact" << endl;
	}

	if (f_v) {
		cout << "strong_generators::print_generators_compact "
				"before gens->print_generators_compact" << endl;
	}
	gens->print_generators_compact(ost, verbose_level);
	if (f_v) {
		cout << "strong_generators::print_generators_compact "
				"after gens->print_generators_compact" << endl;
	}

	if (f_v) {
		cout << "strong_generators::print_generators_compact done" << endl;
	}
}

void strong_generators::print_generators(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::print_generators" << endl;
	}
	int i;
	algebra::ring_theory::longinteger_object go;

	if (f_v) {
		cout << "strong_generators::print_generators before group order" << endl;
	}
	group_order(go);
	if (f_v) {
		ost << "strong_generators::print_generators "
				"Strong generators for a group of order "
				<< go << " tl=";
		Int_vec_print(cout, tl, A->base_len());
		ost << endl;
	}

	for (i = 0; i < gens->len; i++) {
		ost << "generator " << i << " / "
				<< gens->len << " is: " << endl;
		A->Group_element->element_print(gens->ith(i), ost);

#if 0
		string s;

		s = A->Group_element->stringify_base_images(
				gens->ith(i), 0 /* verbose_level */);
		ost << "base images: " << s << endl;
#endif

		ost << "as permutation: " << endl;
		if (A->degree < 400) {
			A->Group_element->element_print_as_permutation_with_offset(
					gens->ith(i), ost,
					0 /* offset*/,
					true /* f_do_it_anyway_even_for_big_degree*/,
					true /* f_print_cycles_of_length_one*/,
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
			A->Group_element->element_print_as_permutation(
					gens->ith(i), ost);
			ost << endl;
		}
	}
	else {
		ost << "too big to print" << endl;
	}
}

void strong_generators::print_generators_in_latex_individually(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::print_generators_in_latex_individually" << endl;
	}

	int i;
	algebra::ring_theory::longinteger_object go;


	if (f_v) {
		cout << "strong_generators::print_generators_in_latex_individually "
				"before group order" << endl;
	}
	group_order(go);
	if (f_v) {
		cout << "strong_generators::print_generators_in_latex_individually "
				"after group order" << endl;
	}

	ost << "The stabilizer of order $" << go
			<< "$ is generated by:\\\\" << endl;

	if (gens->len > 20) {
		ost << "Too many generators to print. There are " << gens->len <<  " generators.\\\\" << endl;
		ost << endl << "\\noindent" << endl;
	}
	else {
		for (i = 0; i < gens->len; i++) {

			if (f_v) {
				cout << "strong_generators::print_generators_in_latex_individually "
						"generator " << i << " / " << gens->len << endl;
			}
			string label;

			label = "g_{" + std::to_string(i + 1) + "} = ";

			//A->element_print_latex_with_extras(gens->ith(i), label, ost);

			ost << "$" << label << "$ ";

			//A->element_print_latex_not_in_math_mode(gens->ith(i), ost);

			if (true /* A->f_is_linear */) {
				ost << "$";
				A->Group_element->element_print_latex(gens->ith(i), ost);
				ost << "$";
			}
			else {
				A->Group_element->element_print_latex(gens->ith(i), ost);
			}

			ost << "\\\\" << endl;


	#if 0
			int n, ord;

			ord = A->Group_element->element_order(gens->ith(i));

			ost << " of order " << ord;

			n = A->Group_element->count_fixed_points(gens->ith(i), 0 /* verbose_level */);
			ost << " and with " << n << " fixed points.\\\\" << endl;
	#endif
			}
		//ost << "\\\\" << endl;
		ost << endl << "\\noindent" << endl;

		print_for_make_element(ost);
	}

	if (f_v) {
		cout << "strong_generators::print_generators_in_latex_individually done" << endl;
	}
}

void strong_generators::print_generators_in_source_code(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::print_generators_in_source_code" << endl;
	}
	int i;
	algebra::ring_theory::longinteger_object go;

	group_order(go);
	cout << "Strong generators for a group of "
			"order " << go << " tl=";
	Int_vec_print(cout, tl, A->base_len());
	cout << endl;
	A->print_base();
	if (gens->len > 20) {
		cout << "Too many generators to print. "
				"There are " << gens->len <<  " generators.\\\\" << endl;
		cout << endl << "\\noindent" << endl;
	}
	else {
		for (i = 0; i < gens->len; i++) {
			//cout << "Generator " << i << " / "
			// << gens->len << " is:" << endl;
			A->Group_element->print_for_make_element(cout, gens->ith(i));
			cout << endl;
		}
	}
	if (f_v) {
		cout << "strong_generators::print_generators_in_source_code done" << endl;
	}
}

void strong_generators::print_generators_in_source_code_to_file(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::print_generators_in_source_code_to_file" << endl;
	}

	int i;
	algebra::ring_theory::longinteger_object go;
	other::orbiter_kernel_system::file_io Fio;

	{
		ofstream f(fname);
		group_order(go);
		f << gens->len << " " << go << endl;
		for (i = 0; i < gens->len; i++) {
			//cout << "Generator " << i << " / "
			//<< gens->len << " is:" << endl;
			A->Group_element->print_for_make_element_no_commas(f, gens->ith(i));
			f << endl;
		}
	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "strong_generators::print_generators_in_source_code_to_file done" << endl;
	}
}

void strong_generators::print_generators_even_odd(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::print_generators_even_odd" << endl;
	}

	int i, sgn;
	algebra::ring_theory::longinteger_object go;

	group_order(go);
	cout << "Strong generators for a group of order " << go << " tl=";
	Int_vec_print(cout, tl, A->base_len());
	cout << endl;
	for (i = 0; i < gens->len; i++) {
		cout << "Generator " << i << " / " << gens->len << " is:" << endl;
		A->Group_element->element_print(gens->ith(i), cout);

		sgn = A->Group_element->element_signum_of_permutation(gens->ith(i));
		cout << " sgn=" << sgn;
		cout << endl;
	}
	if (f_v) {
		cout << "strong_generators::print_generators_even_odd done" << endl;
	}
}

void strong_generators::print_generators_MAGMA(
		actions::action *A, std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::print_generators_MAGMA" << endl;
	}

	interfaces::magma_interface M;

	M.print_generators_MAGMA(A, this, ost);
	if (f_v) {
		cout << "strong_generators::print_generators_MAGMA done" << endl;
	}
}

void strong_generators::export_magma(
		actions::action *A, std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::export_magma" << endl;
	}

	interfaces::magma_interface M;

	if (f_v) {
		cout << "strong_generators::export_magma "
				"before M.export_group" << endl;
	}
	M.export_group(A, this, ost, verbose_level);
	if (f_v) {
		cout << "strong_generators::export_magma "
				"after M.export_group" << endl;
	}


	if (f_v) {
		cout << "strong_generators::export_magma done" << endl;
	}
}

void strong_generators::export_fining(
		actions::action *A, std::ostream &ost,
		int verbose_level)
// at the moment, A is not used
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::export_fining" << endl;
	}

	interfaces::l3_interface_gap Interface;

	if (A->is_matrix_group()) {
		if (f_v) {
			cout << "strong_generators::export_fining "
					"before M.export_group" << endl;
		}
		Interface.export_collineation_group_to_fining(
				ost,
				this,
				verbose_level);
		if (f_v) {
			cout << "strong_generators::export_fining "
					"after M.export_group" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "strong_generators::export_fining "
					"not a matrix group, skipping" << endl;
		}

	}


	if (f_v) {
		cout << "strong_generators::export_fining done" << endl;
	}
}


void strong_generators::canonical_image_GAP(
		std::string &input_set_text, std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::canonical_image_GAP" << endl;
	}
	long int *set;
	int sz;

	Get_lint_vector_from_label(input_set_text, set, sz, 0 /* verbose_level */);

	if (f_v) {
		cout << "strong_generators::canonical_image_GAP "
				"found a set of size " << sz << endl;
	}

	interfaces::l3_interface_gap Interface;

	if (f_v) {
		cout << "strong_generators::canonical_image_GAP "
				"before Interface.canonical_image_GAP" << endl;
	}
	Interface.canonical_image_GAP(
			this,
			set, sz,
			ost, verbose_level);
	if (f_v) {
		cout << "strong_generators::canonical_image_GAP "
				"after Interface.canonical_image_GAP" << endl;
	}

	if (f_v) {
		cout << "strong_generators::canonical_image_GAP done" << endl;
	}
}


void strong_generators::canonical_image_orbiter(
		std::string &input_set_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::canonical_image_orbiter" << endl;
	}
	long int *set;
	int sz;

	Get_lint_vector_from_label(input_set_text, set, sz, 0 /* verbose_level */);

	if (f_v) {
		cout << "strong_generators::canonical_image_orbiter "
				"found a set of size " << sz << endl;
	}


	long int *canonical_set;
	int *transporter;
	long int total_backtrack_nodes;
	int f_get_automorphism_group = true;
	groups::sims Aut;

	transporter = NEW_int(A->elt_size_in_int);
	canonical_set = NEW_lint(sz);


	actions::action_global Action_global;
	groups::sims *Sims;


	Sims = create_sims(verbose_level);


	if (f_v) {
		cout << "strong_generators::canonical_image_orbiter "
				"before Action_global.make_canonical" << endl;
	}
	Action_global.make_canonical(
			A, Sims,
			sz, set,
		canonical_set, transporter,
		total_backtrack_nodes,
		f_get_automorphism_group, &Aut,
		verbose_level);
	if (f_v) {
		cout << "strong_generators::canonical_image_orbiter "
				"after Action_global.make_canonical" << endl;
	}


	FREE_lint(canonical_set);
	FREE_OBJECT(Sims);


	if (f_v) {
		cout << "strong_generators::canonical_image_orbiter done" << endl;
	}
}



void strong_generators::print_generators_tex()
{
	print_generators_tex(cout);
}

void strong_generators::print_generators_tex(
		std::ostream &ost)
{
	int i;
	algebra::ring_theory::longinteger_object go;

	group_order(go);
	ost << "Strong generators for a group of order " << go << ":\\\\" << endl;
	for (i = 0; i < gens->len; i++) {
		//cout << "Generator " << i << " / " << gens->len
		// << " is:" << endl;
		ost << "$" << endl;
		A->Group_element->element_print_latex(gens->ith(i), ost);
		ost << "$" << endl;


#if 0
		string s;

		s = A->Group_element->stringify_base_images(
				gens->ith(i), 0 /* verbose_level */);
		ost << "base images: " << s << " ";
#endif

		if (i < gens->len - 1) {
			ost << ", " << endl;
		}
#if 0
		if (((i + 1) % 1) == 0 && i < gens->len - 1) {
			ost << "$$" << endl;
			ost << "$$" << endl;
		}
#endif
	}
	//ost << "$$" << endl;
	ost << "\\\\" << endl;
	print_for_make_element(ost);
}

void strong_generators::print_for_make_element(
		std::ostream &ost)
{
	int i;

	for (i = 0; i < gens->len; i++) {
		//cout << "Generator " << i << " / " << gens->len
		// << " is:" << endl;
		A->Group_element->element_print_for_make_element(gens->ith(i), ost);
		ost << "\\\\" << endl;
	}
}

void strong_generators::print_generators_in_different_action_tex(
		std::ostream &ost, actions::action *A2)
{
	int i;
	algebra::ring_theory::longinteger_object go;

	group_order(go);
	ost << "Strong generators for a group of order " << go << ":" << endl;
	ost << "$$" << endl;
	for (i = 0; i < gens->len; i++) {
		//cout << "Generator " << i << " / " << gens->len
		// << " is:" << endl;
		A2->Group_element->element_print_as_permutation(gens->ith(i), ost);
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
		A->Group_element->element_print_for_make_element(gens->ith(i), ost);
		ost << "\\\\" << endl;
	}
}


void strong_generators::print_generators_tex_with_point_labels(
		actions::action *A_given,
		std::ostream &ost,
		std::string *Point_labels, void *data)
{
	int i;
	algebra::ring_theory::longinteger_object go;

	group_order(go);
	ost << "Strong generators for a group of order " << go << ":" << endl;
	//ost << "$$" << endl;
	for (i = 0; i < gens->len; i++) {
		cout << "Generator " << i << " / " << gens->len << " is:" << endl;
		ost << "$$" << endl;
		A->Group_element->element_print_latex(gens->ith(i), ost);
		ost << "$$" << endl;
		ost << "$$" << endl;
		A_given->Group_element->element_print_latex_with_point_labels(
				gens->ith(i), ost,
				Point_labels, data);
		ost << "$$" << endl;
	}
	//ost << "$$" << endl;
	for (i = 0; i < gens->len; i++) {
		//cout << "Generator " << i << " / " << gens->len
		// << " is:" << endl;
		A->Group_element->element_print_for_make_element(gens->ith(i), ost);
		ost << "\\\\" << endl;
	}

	for (i = 0; i < gens->len; i++) {
		ost << "$";
		A_given->Group_element->element_print_latex(gens->ith(i), ost);
		ost << "$\\\\" << endl;
	}

}

void strong_generators::print_generators_for_make_element(
		std::ostream &ost)
{
	int i;
	algebra::ring_theory::longinteger_object go;

	group_order(go);
	ost << "Strong generators for a group of order " << go << ":\\\\" << endl;
	for (i = 0; i < gens->len; i++) {
		//ost << "";
		A->Group_element->element_print_for_make_element(gens->ith(i), ost);
		ost << "\\\\" << endl;
	}
}


void strong_generators::print_generators_as_permutations()
{
	int i;
	algebra::ring_theory::longinteger_object go;

	group_order(go);
	cout << "Strong generators for a group of order "
			<< go << ":" << endl;
	for (i = 0; i < gens->len; i++) {
		cout << "Generator " << i << " / "
				<< gens->len << " is:" << endl;
		A->Group_element->element_print(gens->ith(i), cout);
		if (A->degree < 1000) {
			A->Group_element->element_print_as_permutation(gens->ith(i), cout);
			cout << endl;
		}
		else {
			cout << "strong_generators::print_generators_as_permutations "
					"the degree is too large, we won't print "
					"the permutation representation" << endl;
		}
	}
}

void strong_generators::print_generators_as_permutations_tex(
		std::ostream &ost, actions::action *A2)
{
	int i;
	algebra::ring_theory::longinteger_object go;

	group_order(go);
	ost << "Strong generators for a group of order " << go << ":" << endl;
	ost << "\\\\" << endl;
	for (i = 0; i < gens->len; i++) {
		ost << "Generator " << i << " / "
				<< gens->len << " is: $" << endl;
		//A->element_print(gens->ith(i), cout);
		if (A->degree < 1000) {
			A2->Group_element->element_print_as_permutation(gens->ith(i), ost);
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
		std::ostream &ost, actions::action *A2)
{
	int i;
	
	for (i = 0; i < gens->len; i++) {
		ost << "Generator " << i << " / "
				<< gens->len << " is:" << endl;
		ost << "$$" << endl;
		A2->Group_element->element_print_latex(gens->ith(i), ost);
		//ost << endl;
		ost << "$$" << endl;
		ost << "as permutation:" << endl;
		//ost << "$$" << endl;
		if (A2->degree < 1000) {
			A2->Group_element->element_print_as_permutation(gens->ith(i), ost);
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

void strong_generators::print_elements_ost(
		std::ostream &ost)
{
	long int i;
	algebra::ring_theory::longinteger_object go;
	sims *S;
	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);
	S = create_sims(0 /*verbose_level */);
	ost << "Group elements for a group of order " << go << " tl=";
	Int_vec_print(ost, tl, A->base_len());
	ost << "\\\\" << endl;
	for (i = 0; i < go.as_lint(); i++) {
		S->element_unrank_lint(i, Elt, 0 /* verbose_level */);
		ost << "Element " << i << " / " << go << " is:" << endl;
		ost << "$$" << endl;
		A->Group_element->element_print_latex(Elt, ost);
		ost << "$$" << endl;
	}
	FREE_OBJECT(S);
	FREE_int(Elt);
}

void strong_generators::print_elements_with_special_orthogonal_action_ost(
		std::ostream &ost)
{
	long int i;
	algebra::ring_theory::longinteger_object go;
	sims *S;
	int *Elt;
	geometry::other_geometry::geometry_global Geo;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);
	S = create_sims(0 /*verbose_level */);
	ost << "Group elements for a group of order " << go << " tl=";
	Int_vec_print(ost, tl, A->base_len());
	ost << "\\\\" << endl;
	for (i = 0; i < go.as_lint(); i++) {
		S->element_unrank_lint(i, Elt, 0 /* verbose_level */);

		ost << "Element " << i << " / " << go << " is:" << endl;
		ost << "$$" << endl;
		A->Group_element->element_print_latex(Elt, ost);
		if (A->matrix_group_dimension() == 4) {
			int A6[36];
			algebra::field_theory::finite_field *F;

			F = A->matrix_group_finite_field();
			Geo.isomorphism_to_special_orthogonal(F, Elt, A6, 0 /* verbose_level*/);
			ost << "=" << endl;
			F->Io->print_matrix_latex(ost, A6, 6, 6);
		}
		ost << "$$" << endl;
	}
	FREE_OBJECT(S);
	FREE_int(Elt);
}


void strong_generators::print_elements_with_given_action(
		std::ostream &ost, actions::action *A2)
{
	long int i;
	algebra::ring_theory::longinteger_object go;
	sims *S;
	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);
	S = create_sims(0 /*verbose_level */);
	ost << "Group elements for a group of order " << go << " tl=";
	Int_vec_print(ost, tl, A->base_len());
	ost << "\\\\" << endl;
	for (i = 0; i < go.as_lint(); i++) {
		S->element_unrank_lint(i, Elt, 0 /* verbose_level */);
		ost << "Element " << i << " / " << go << " is:" << endl;
		ost << "$$" << endl;
		if (A2->degree < 1000) {
			A2->Group_element->element_print_as_permutation(Elt, ost);
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

void strong_generators::print_elements_latex_ost(
		std::ostream &ost)
{
	long int i, order, m;
	algebra::ring_theory::longinteger_object go;
	sims *S;
	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);
	group_order(go);
	S = create_sims(0 /*verbose_level */);
	ost << "Group elements for a group of order " << go << " tl=";
	Int_vec_print(ost, tl, A->base_len());
	ost << "\\\\" << endl;
	m = MINIMUM(go.as_int(), 100);
	if (m < go.as_int()) {
		ost << "We will only list the first " << m
				<< " elements:\\\\" << endl;
	}
	for (i = 0; i < m; i++) {
		S->element_unrank_lint(i, Elt, 0 /* verbose_level */);
		order = A->Group_element->element_order(Elt);
		ost << "Element " << i << " / " << go << " is:" << endl;
		ost << "$$" << endl;
		A->Group_element->element_print_latex(Elt, ost);
		ost << "$$" << endl;
		ost << "The element has order " << order << ".\\\\" << endl;
	}
	FREE_OBJECT(S);
	FREE_int(Elt);
}

void strong_generators::print_elements_latex_ost_with_point_labels(
		actions::action *A_given,
		std::ostream &ost,
		std::string *Point_labels, void *data)
{
	long int i, order, m;
	algebra::ring_theory::longinteger_object go;
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
	Int_vec_print(ost, tl, A->base_len());
	ost << "\\\\" << endl;
	m = MINIMUM(go.as_int(), 1500);
	if (m < go.as_int()) {
		ost << "We will only list the first " << m
				<< " elements:\\\\" << endl;
	}
	for (i = 0; i < m; i++) {
		S->element_unrank_lint(i, Elt, 0 /* verbose_level */);
		//cout << "element " << i << " / " << m << " before A->element_order" << endl;
		order = A->Group_element->element_order(Elt);
		//cout << "element " << i << " / " << m << " before A->element_order_and_cycle_type" << endl;
		A_given->Group_element->element_order_and_cycle_type(Elt, cycle_type);
		ost << "Element " << i << " / " << go << " is:" << endl;
		ost << "$$" << endl;
		A->Group_element->element_print_latex(Elt, ost);
		ost << "$$" << endl;
		ost << "$$" << endl;
		A_given->Group_element->element_print_latex_with_point_labels(
				Elt, ost,
				Point_labels, data);
		ost << "$$" << endl;
		ost << "The element has order " << order << ".\\\\" << endl;
		S->compute_all_powers(i, order, power_elt, 0 /*verbose_level*/);
		ost << "The powers are: ";
		Int_vec_print(ost, power_elt, order);
		ost << ".\\\\" << endl;
		nb_fix_points[i] = cycle_type[0];
		ost << "The element has " << nb_fix_points[i] << " fix points.\\\\" << endl;
	}
	other::data_structures::tally C;

	C.init(nb_fix_points, m, false, 0);
	ost << "The distribution of the number of fix points is $";
	C.print_file_tex_we_are_in_math_mode(ost, true);
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
	algebra::ring_theory::longinteger_object go;
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
		actions::action *A_given, schreier *&Sch, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::compute_schreier_with_given_action" << endl;
		cout << "action=";
		A->print_info();
	}
	Sch = NEW_OBJECT(schreier);

	int print_interval = 10000;

	Sch->init(A_given, verbose_level - 2);
	//Sch->initialize_tables();
	Sch->init_generators(*gens, verbose_level - 2);
	Sch->compute_all_point_orbits(print_interval, verbose_level - 2);


	if (f_v) {
		cout << "strong_generators::compute_schreier_with_given_action done, we found "
				<< Sch->Forest->nb_orbits << " orbits" << endl;
	}
}

void strong_generators::compute_schreier_with_given_action_on_a_given_set(
		actions::action *A_given, schreier *&Sch, long int *set, int len,
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
	//Sch->initialize_tables();
	Sch->init_generators(*gens, verbose_level - 2);
	Sch->compute_all_orbits_on_invariant_subset_lint(
			len, set,
			0 /* verbose_level */);
	//Sch->compute_all_point_orbits(verbose_level);


	if (f_v) {
		cout << "strong_generators::compute_schreier_with_given_action_on_a_given_set "
				"done, we found "
				<< Sch->Forest->nb_orbits << " orbits" << endl;
	}
}

void strong_generators::orbits_on_points(
		int &nb_orbits,
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

	if (f_v) {
		cout << "strong_generators::orbits_on_points "
				"before compute_schreier_with_given_action" << endl;
	}
	compute_schreier_with_given_action(A, Sch, verbose_level - 1);
	if (f_v) {
		cout << "strong_generators::orbits_on_points "
				"after compute_schreier_with_given_action" << endl;
	}


	nb_orbits = Sch->Forest->nb_orbits;
	orbit_reps = NEW_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		f = Sch->Forest->orbit_first[i];
		a = Sch->Forest->orbit[f];
		orbit_reps[i] = a;
	}

	FREE_OBJECT(Sch);

	if (f_v) {
		cout << "strong_generators::orbits_on_points done, "
				"we found " << nb_orbits << " orbits" << endl;
	}
}

void strong_generators::orbits_on_set_with_given_action_after_restriction(
		actions::action *A_given,
		long int *Set, int set_sz,
		std::stringstream &orbit_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::orbits_on_set_with_given_action_after_restriction" << endl;
		cout << "action=";
		A->print_info();
		cout << "set=";
		Lint_vec_print(cout, Set, set_sz);
	}

	actions::action *Ar;
	orbits_on_something *Orb;
	std::string prefix;

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_on_set");
	label_of_set_tex.assign("\\_on\\_set");

	if (f_v) {
		cout << "strong_generators::orbits_on_set_with_given_action_after_restriction "
				"before A_given->Induced_action->restricted_action" << endl;
	}
	Ar = A_given->Induced_action->restricted_action(
			Set, set_sz, label_of_set, label_of_set_tex,
			verbose_level);
	if (f_v) {
		cout << "strong_generators::orbits_on_set_with_given_action_after_restriction "
				"after A_given->Induced_action->restricted_action" << endl;
	}

	Orb = NEW_OBJECT(orbits_on_something);

	prefix.assign(Ar->label);

	int print_interval = 10000;

	if (f_v) {
		cout << "strong_generators::orbits_on_set_with_given_action_after_restriction "
				"before Orb->init" << endl;
	}
	Orb->init(Ar,
			this,
			false /* f_load_save */,
			prefix,
			print_interval,
			verbose_level - 1);
	if (f_v) {
		cout << "strong_generators::orbits_on_set_with_given_action_after_restriction "
				"after Orb->init" << endl;
	}

	other::data_structures::tally *Classify_orbits_by_length;

	Classify_orbits_by_length = NEW_OBJECT(other::data_structures::tally);
	Classify_orbits_by_length->init(Orb->Sch->Forest->orbit_len, Orb->Sch->Forest->nb_orbits, false, 0);
	Classify_orbits_by_length->print_bare_stringstream(orbit_type, true /* f_backwards */);


	FREE_OBJECT(Classify_orbits_by_length);
	FREE_OBJECT(Orb);
	FREE_OBJECT(Ar);

	if (f_v) {
		cout << "strong_generators::orbits_on_set_with_given_action_after_restriction done" << endl;
	}
}


void strong_generators::extract_orbit_on_set_with_given_action_after_restriction_by_length(
		actions::action *A_given,
		long int *Set, int set_sz,
		int desired_orbit_length,
		long int *&extracted_set,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::extract_orbit_on_set_with_given_action_after_restriction" << endl;
		cout << "action=";
		A->print_info();
		cout << "set=";
		Lint_vec_print(cout, Set, set_sz);
	}

	actions::action *Ar;
	orbits_on_something *Orb;
	std::string prefix;
	std::string label_of_set;
	std::string label_of_set_tex;

	label_of_set.assign("_on_set");
	label_of_set_tex.assign("\\_on\\_set");

	if (f_v) {
		cout << "strong_generators::extract_orbit_on_set_with_given_action_after_restriction "
				"before A_given->Induced_action->restricted_action" << endl;
	}
	Ar = A_given->Induced_action->restricted_action(
			Set, set_sz,
			label_of_set, label_of_set_tex,
			verbose_level);
	if (f_v) {
		cout << "strong_generators::extract_orbit_on_set_with_given_action_after_restriction "
				"after A_given->Induced_action->restricted_action" << endl;
	}

	Orb = NEW_OBJECT(orbits_on_something);

	prefix.assign(Ar->label);

	int print_interval = 10000;

	if (f_v) {
		cout << "strong_generators::extract_orbit_on_set_with_given_action_after_restriction "
				"before Orb->init" << endl;
	}
	Orb->init(Ar,
			this,
			false /* f_load_save */,
			prefix,
			print_interval,
			verbose_level);
	if (f_v) {
		cout << "strong_generators::extract_orbit_on_set_with_given_action_after_restriction "
				"after Orb->init" << endl;
	}

	other::data_structures::tally *Classify_orbits_by_length;
	other::data_structures::set_of_sets *SoS;
	int *types;
	int nb_types;
	int idx, orb_idx, len;

	Classify_orbits_by_length = NEW_OBJECT(other::data_structures::tally);
	Classify_orbits_by_length->init(Orb->Sch->Forest->orbit_len, Orb->Sch->Forest->nb_orbits, false, 0);

	SoS = Classify_orbits_by_length->get_set_partition_and_types(types,
			nb_types, verbose_level);

	if (f_v) {
		cout << "strong_generators::extract_orbit_on_set_with_given_action_after_restriction" << endl;
		cout << "types=";
		Int_vec_print(cout, types, nb_types);
		cout << endl;
	}

	for (idx = 0; idx < nb_types; idx++) {
		if (types[idx] == desired_orbit_length) {
			break;
		}
	}
	if (idx == nb_types) {
		cout << "could not find orbit of length " << desired_orbit_length << endl;
		exit(1);
	}
	if (SoS->Set_size[idx] != 1) {
		cout << "Orbit of length " << desired_orbit_length << " is not unique" << endl;
		exit(1);

	}

	orb_idx = SoS->Sets[idx][0];
	extracted_set = NEW_lint(desired_orbit_length);

	Orb->Sch->Forest->get_orbit(orb_idx, extracted_set, len, verbose_level);
	if (len != desired_orbit_length) {
		cout << "len != desired_orbit_length" << endl;
		exit(1);
	}

	//Classify_orbits_by_length->print_bare_stringstream(orbit_type, true /* f_backwards */);


	FREE_OBJECT(SoS);
	FREE_OBJECT(Classify_orbits_by_length);
	FREE_OBJECT(Orb);
	FREE_OBJECT(Ar);

	if (f_v) {
		cout << "strong_generators::extract_orbit_on_set_with_given_action_after_restriction done" << endl;
	}
}

void strong_generators::extract_specific_orbit_on_set_with_given_action_after_restriction_by_length(
		actions::action *A_given,
		long int *Set, int set_sz,
		int desired_orbit_length,
		int desired_orbit_idx,
		long int *&extracted_set,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::extract_specific_orbit_on_set_with_given_action_after_restriction_by_length" << endl;
		cout << "action=";
		A->print_info();
		cout << "set=";
		Lint_vec_print(cout, Set, set_sz);
	}

	actions::action *Ar;
	orbits_on_something *Orb;
	std::string prefix;
	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_on_set");
	label_of_set_tex.assign("\\_on\\_set");


	if (f_v) {
		cout << "strong_generators::extract_specific_orbit_on_set_with_given_action_after_restriction_by_length "
				"before A_given->Induced_action->restricted_action" << endl;
	}
	Ar = A_given->Induced_action->restricted_action(
			Set, set_sz,
			label_of_set, label_of_set_tex,
			verbose_level);
	if (f_v) {
		cout << "strong_generators::extract_specific_orbit_on_set_with_given_action_after_restriction_by_length "
				"after A_given->Induced_action->restricted_action" << endl;
	}

	Orb = NEW_OBJECT(orbits_on_something);

	prefix.assign(Ar->label);

	int print_interval = 10000;

	if (f_v) {
		cout << "strong_generators::extract_specific_orbit_on_set_with_given_action_after_restriction_by_length "
				"before Orb->init" << endl;
	}
	Orb->init(Ar,
			this,
			false /* f_load_save */,
			prefix,
			print_interval,
			verbose_level);
	if (f_v) {
		cout << "strong_generators::extract_specific_orbit_on_set_with_given_action_after_restriction_by_length "
				"after Orb->init" << endl;
	}

	other::data_structures::tally *Classify_orbits_by_length;
	other::data_structures::set_of_sets *SoS;
	int *types;
	int nb_types;
	int idx, orb_idx, len;

	Classify_orbits_by_length = NEW_OBJECT(other::data_structures::tally);
	Classify_orbits_by_length->init(
			Orb->Sch->Forest->orbit_len,
			Orb->Sch->Forest->nb_orbits,
			false, 0);

	SoS = Classify_orbits_by_length->get_set_partition_and_types(types,
			nb_types, verbose_level);

	if (f_v) {
		cout << "strong_generators::extract_specific_orbit_on_set_with_given_action_after_restriction_by_length" << endl;
		cout << "types=";
		Int_vec_print(cout, types, nb_types);
		cout << endl;
	}

	for (idx = 0; idx < nb_types; idx++) {
		if (types[idx] == desired_orbit_length) {
			break;
		}
	}
	if (idx == nb_types) {
		cout << "could not find orbit of length " << desired_orbit_length << endl;
		exit(1);
	}
	if (desired_orbit_idx >= SoS->Set_size[idx]) {
		cout << "Orbit of length " << desired_orbit_length << ": desired index is out of range" << endl;
		exit(1);
	}

	orb_idx = SoS->Sets[idx][desired_orbit_idx];
	extracted_set = NEW_lint(desired_orbit_length);

	Orb->Sch->Forest->get_orbit(orb_idx, extracted_set, len, verbose_level);
	if (len != desired_orbit_length) {
		cout << "len != desired_orbit_length" << endl;
		exit(1);
	}

	//Classify_orbits_by_length->print_bare_stringstream(orbit_type, true /* f_backwards */);


	FREE_OBJECT(SoS);
	FREE_OBJECT(Classify_orbits_by_length);
	FREE_OBJECT(Orb);
	FREE_OBJECT(Ar);

	if (f_v) {
		cout << "strong_generators::extract_specific_orbit_on_set_with_given_action_after_restriction_by_length done" << endl;
	}
}




void strong_generators::orbits_on_points_with_given_action(
		actions::action *A_given, int &nb_orbits, int *&orbit_reps,
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
	if (f_v) {
		cout << "strong_generators::orbits_on_points_with_given_action "
				"before compute_schreier_with_given_action" << endl;
	}
	compute_schreier_with_given_action(A_given, Sch, verbose_level - 1);
	if (f_v) {
		cout << "strong_generators::orbits_on_points_with_given_action "
				"after compute_schreier_with_given_action" << endl;
	}

	nb_orbits = Sch->Forest->nb_orbits;
	orbit_reps = NEW_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		f = Sch->Forest->orbit_first[i];
		a = Sch->Forest->orbit[f];
		orbit_reps[i] = a;
	}

	FREE_OBJECT(Sch);

	if (f_v) {
		cout << "strong_generators::orbits_on_points_with_given_action "
				"done, we found "
				<< nb_orbits << " orbits" << endl;
	}
}

schreier *strong_generators::compute_all_point_orbits_schreier(
		actions::action *A_given,
		int print_interval,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	schreier *Sch;

	if (f_v) {
		cout << "strong_generators::compute_all_point_orbits_schreier " << endl;
		cout << "strong_generators::compute_all_point_orbits_schreier print_interval = " << print_interval << endl;
	}
	if (f_v) {
		cout << "strong_generators::compute_all_point_orbits_schreier "
				"degree = " << A_given->degree << endl;
		cout << "A_given=";
		A_given->print_info();
	}

	algebra::ring_theory::longinteger_object go;
	group_order(go);

	if (f_v) {
		cout << "strong_generators::compute_all_point_orbits_schreier "
				"go = " << go << endl;
	}


	if (f_v) {
		cout << "strong_generators::compute_all_point_orbits_schreier "
				"generators:" << endl;
		//print_generators_tex();
	}
	Sch = NEW_OBJECT(schreier);

	Sch->init(A_given, verbose_level - 2);
	//Sch->initialize_tables();
	if (f_v) {
		cout << "strong_generators::compute_all_point_orbits_schreier "
				"before Sch->init_generators" << endl;
	}
	Sch->init_generators(*gens, verbose_level - 2);
	if (f_v) {
		cout << "strong_generators::compute_all_point_orbits_schreier "
				"before Sch->compute_all_point_orbits" << endl;
	}
	Sch->compute_all_point_orbits(print_interval, verbose_level - 1);
	if (f_v) {
		cout << "strong_generators::compute_all_point_orbits_schreier "
				"after Sch->compute_all_point_orbits" << endl;
	}

	if (f_v) {
		cout << "strong_generators::compute_all_point_orbits_schreier "
				"done, we found " << Sch->Forest->nb_orbits << " orbits" << endl;
	}
	return Sch;
}

schreier *strong_generators::orbit_of_one_point_schreier(
		actions::action *A_given, int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	schreier *Sch;

	if (f_v) {
		cout << "strong_generators::orbit_of_one_point_schreier "
				"degree = " << A_given->degree << " point = "
				<< pt << endl;
	}
	Sch = NEW_OBJECT(schreier);

	int print_interval = 10000;

	Sch->init(A_given, verbose_level - 2);
	//Sch->initialize_tables();
	Sch->init_generators(*gens, verbose_level - 2);
	if (f_v) {
		cout << "strong_generators::orbit_of_one_point_schreier "
				"before Sch->compute_point_orbit" << endl;
	}
	Sch->compute_point_orbit(pt, print_interval, 0 /* verbose_level */);
	if (f_v) {
		cout << "strong_generators::orbit_of_one_point_schreier "
				"after Sch->compute_point_orbit" << endl;
	}

	if (f_v) {
		cout << "strong_generators::orbit_of_one_point_schreier "
				"done, we found one orbit of length "
				<< Sch->Forest->orbit_len[0] << endl;
	}
	return Sch;
}

void strong_generators::orbits_light(
		actions::action *A_given,
	int *&Orbit_reps, int *&Orbit_lengths, int &nb_orbits, 
	int **&Pts_per_generator, int *&Nb_per_generator, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);	
	int f_vv = false; //(verbose_level >= 2);	
	other::data_structures::bitvector *Has_been_reached;
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
	other::data_structures::sorting Sorting;

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
	Has_been_reached = NEW_OBJECT(other::data_structures::bitvector);
	Has_been_reached->allocate(A_given->degree);

	nb_reached = 0;

	Orbit_allocated = 1024;
	Orbit = NEW_int(Orbit_allocated);

	Q_allocated = 1024;
	Q = NEW_int(Q_allocated);
	
	nb_gens = gens->len;

	if (A_given->degree > ONE_MILLION) {
		f_v = true;
	}

	Nb_per_generator = NEW_int(nb_gens);
	Int_vec_zero(Nb_per_generator, nb_gens);
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
				b = A_given->Group_element->element_image_of(a, gens->ith(h), false);
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
						cout << "strong_generators::orbits_light current orbit: ";
						Int_vec_print(cout, Orbit, Orbit_len);
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
						cout << "strong_generators::orbits_light current Queue: ";
						Int_vec_print(cout, Q, Q_len);
						cout << endl;
					}

				}
			} // next h
		} // while (Q_len)

		if (f_vv) {
			cout << "strong_generators::orbits_light Orbit of point " << pt << " has length "
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
		std::ofstream &fp, int verbose_level)
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
		actions::action *A, std::ifstream &fp, int verbose_level)
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
	gens = NEW_OBJECT(data_structures_groups::vector_ge);
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

void strong_generators::write_file(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;
	
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

void strong_generators::read_file(
		actions::action *A,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;

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
		cout << "strong_generators::read_file "
				"reading file " << fname << endl;
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



void strong_generators::compute_ascii_coding(
		std::string &ascii_coding, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int sz, i, j;
	char *p, *p0;
	other::orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "strong_generators::compute_ascii_coding" << endl;
	}
	sz = 2 * ((2 + A->base_len() + A->base_len()) * sizeof(int_4) +
			A->coded_elt_size_in_char * gens->len) + 1;
	p = NEW_char(sz);
	p0 = p;
	Os.code_int4(p, (int_4) A->base_len());

	Os.code_int4(p, (int_4) gens->len);
	for (i = 0; i < A->base_len(); i++) {
		Os.code_int4(p, (int_4) A->base_i(i));
	}
	for (i = 0; i < A->base_len(); i++) {
		Os.code_int4(p, (int_4) tl[i]);
	}
	for (i = 0; i < gens->len; i++) {
		A->Group_element->element_pack(
				gens->ith(i), A->Group_element->elt1,
				0 /* verbose_level */);
		for (j = 0; j < A->coded_elt_size_in_char; j++) {
			Os.code_uchar(p, A->Group_element->elt1[j]);
		}
	}
	*p++ = 0;
	if (p - p0 != sz) {
		cout << "strong_generators::compute_ascii_coding "
				"p - ascii_coding != sz" << endl;
		exit(1);
	}
	
	ascii_coding.assign(p0);

	if (f_v) {
		cout << "strong_generators::compute_ascii_coding "
				"done" << endl;
	}
}

void strong_generators::decode_ascii_coding(
		std::string &ascii_coding, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int str_len, len, nbsg, i, j;
	const char *p, *p0;
	actions::action *A_save;
	int *base1;
	other::orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "strong_generators::decode_ascii_coding" << endl;
	}

	// clean up before we go:
	A_save = A;
	//freeself(); // ToDo
	A = A_save;

	p = ascii_coding.c_str();
	p0 = p;
	str_len = ascii_coding.length();
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
	gens = NEW_OBJECT(data_structures_groups::vector_ge);
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
			Os.decode_uchar(p, A->Group_element->elt1[j]);
		}
		A->Group_element->element_unpack(
				A->Group_element->elt1, gens->ith(i),
				0 /* verbose_level */);
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





void strong_generators::compute_and_print_orbits_on_a_given_set(
		actions::action *A_given,
		long int *set, int len, int verbose_level)
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
	for (i = 0; i < Sch->Forest->nb_orbits; i++) {
		f = Sch->Forest->orbit_first[i];
		for (j = 0; j < Sch->Forest->orbit_len[i]; j++) {
			a = Sch->Forest->orbit[f + j];
			cout << a << " ";
		}
		if (i < Sch->Forest->nb_orbits - 1) {
			cout << "| ";
		}
	}
	cout << endl;
	cout << "partition: " << len << " = ";
	for (i = 0; i < Sch->Forest->nb_orbits; i++) {
		l = Sch->Forest->orbit_len[i];
		cout << l << " ";
		if (i < Sch->Forest->nb_orbits - 1) {
			cout << "+ ";
		}
	}
	cout << endl;
	cout << "representatives for each of the "
			<< Sch->Forest->nb_orbits << " orbits:" << endl;
	for (i = 0; i < Sch->Forest->nb_orbits; i++) {
		f = Sch->Forest->orbit_first[i];
		l = Sch->Forest->orbit_len[i];
		a = Sch->Forest->orbit[f + 0];
		cout << setw(5) << a << " : " << setw(5) << l << " : ";
		A_given->Group_element->print_point(a, cout);
		cout << endl;
	}
	FREE_OBJECT(Sch);

}

void strong_generators::compute_and_print_orbits(
		actions::action *A_given, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	schreier *Sch;
	int i, j, f, l, a;

	if (f_v) {
		cout << "strong_generators::compute_and_print_orbits" << endl;
	}
	compute_schreier_with_given_action(A_given, Sch, verbose_level - 2);

	cout << "orbits on the set: " << endl;
	for (i = 0; i < Sch->Forest->nb_orbits; i++) {
		f = Sch->Forest->orbit_first[i];
		l = Sch->Forest->orbit_len[i];
		if (l >= 10) {
			cout << "too long to list ";
		}
		else {
			for (j = 0; j < Sch->Forest->orbit_len[i]; j++) {
				a = Sch->Forest->orbit[f + j];
				cout << a << " ";
			}
		}
		if (i < Sch->Forest->nb_orbits - 1) {
			cout << "| ";
		}
	}
	cout << endl;
	cout << "partition: " << A_given->degree << " = ";
	for (i = 0; i < Sch->Forest->nb_orbits; i++) {
		l = Sch->Forest->orbit_len[i];
		cout << l << " ";
		if (i < Sch->Forest->nb_orbits - 1) {
			cout << "+ ";
		}
	}
	cout << endl;
	cout << "representatives for each of the "
			<< Sch->Forest->nb_orbits << " orbits:" << endl;
	for (i = 0; i < Sch->Forest->nb_orbits; i++) {
		f = Sch->Forest->orbit_first[i];
		l = Sch->Forest->orbit_len[i];
		a = Sch->Forest->orbit[f + 0];
		cout << setw(5) << a << " : " << setw(5) << l << " : ";
		A_given->Group_element->print_point(a, cout);
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

int strong_generators::test_if_normalizing(
		sims *S, int verbose_level)
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
			return false;
		}
	}
	if (f_v) {
		cout << "strong_generators::test_if_normalizing done, "
				"the given generators normalize the given group" << endl;
	}
	return true;
}


int strong_generators::test_if_subgroup(
		sims *S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int ret = true;

	if (f_v) {
		cout << "strong_generators::test_if_subgroup" << endl;
	}
	for (i = 0; i < gens->len; i++) {
		if (f_v) {
			cout << "strong_generators::test_if_subgroup "
					"testing generator " << i << " / "
					<< gens->len << endl;
		}
		if (!S->is_element_of(gens->ith(i), verbose_level)) {
			if (f_v) {
				cout << "strong_generators::test_if_subgroup "
						"generator " << i << " / " << gens->len
						<< " does not belong to the group" << endl;
			}
			ret = false;
			break;
		}
	}
	if (f_v) {
		cout << "strong_generators::test_if_subgroup done ret = " << ret << endl;
	}
	return ret;
}



void strong_generators::test_if_set_is_invariant_under_given_action(
		actions::action *A_given,
		long int *set, int set_sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "strong_generators::test_if_set_is_invariant_under_given_action" << endl;
	}
	for (i = 0; i < gens->len; i++) {

		if (!A_given->Group_element->test_if_set_stabilizes(
				gens->ith(i),
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

int strong_generators::test_if_they_stabilize_the_equation(
		int *equation,
		algebra::ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int ret;

	if (f_v) {
		cout << "strong_generators::test_if_they_stabilize_the_equation" << endl;
	}
	for (i = 0; i < gens->len; i++) {

		ret = A->Group_element->test_if_it_fixes_the_polynomial(
				gens->ith(i),
				equation,
				HPD,
				verbose_level - 1);

		if (!ret) {
			cout << "strong_generators::test_if_they_stabilize_the_equation "
					"the generator do not fix the equation" << endl;
			break;
		}

	}
	if (i < gens->len) {
		ret = false;
	}
	else {
		ret = true;
	}
	if (f_v) {
		cout << "strong_generators::test_if_they_stabilize_the_equation done" << endl;
	}
	return ret;
}

void strong_generators::set_of_coset_representatives(
		sims *S,
		data_structures_groups::vector_ge *&coset_reps,
		int verbose_level)
// computes a set of coset representatives for $H$ in $S$,
// where $S$ is an overgroup of $H$
// Here $H$, is the group generated by the strong generating set in the current orbject
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::set_of_coset_representatives" << endl;
	}

	algebra::ring_theory::longinteger_object G_order, H_order;
	long int subgroup_index, i, j, cur, len;
	//long int *Q;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	algebra::ring_theory::longinteger_domain D;
	sims *H;

	H = create_sims_in_different_action(S->A, verbose_level);
	S->group_order(G_order);
	group_order(H_order);

	subgroup_index = D.quotient_as_lint(G_order, H_order);


	coset_reps = NEW_OBJECT(data_structures_groups::vector_ge);

	coset_reps->init(S->A, verbose_level);
	coset_reps->allocate(subgroup_index, verbose_level);


	Elt1 = NEW_int(S->A->elt_size_in_int);
	Elt2 = NEW_int(S->A->elt_size_in_int);
	Elt3 = NEW_int(S->A->elt_size_in_int);

	for (i = 0; i < subgroup_index; i++) {

		S->A->Group_element->element_one(coset_reps->ith(i), 0);

	}

	//Q = NEW_lint(subgroup_index);

	//Q[0] = 0;

	//Q_len = 1;

	cur = 0;
	len = 1;

	for (cur = 0; cur < len; cur++) {

		if (f_v) {
			cout << "strong_generators::set_of_coset_representatives cur = " << cur << endl;
		}
		for (i = 0; i < S->gens.len; i++) {

			if (f_v) {
				cout << "strong_generators::set_of_coset_representatives "
						"cur = " << cur << " gen " << i << " / " << S->gens.len << endl;
			}
			S->A->Group_element->element_mult(
					coset_reps->ith(cur), S->gens.ith(i), Elt1, 0);

			S->A->Group_element->element_invert(
					Elt1, Elt2, 0);

			if (f_v) {
				cout << "strong_generators::set_of_coset_representatives "
						"searching for coset, len = " << len << endl;
			}

			for (j = 0; j < len; j++) {
				if (f_v) {
					cout << "strong_generators::set_of_coset_representatives "
							"searching for coset at position " << j << endl;
				}
				S->A->Group_element->element_mult(
						coset_reps->ith(j), Elt2, Elt3, 0);

				if (H->is_element_of(Elt3, 0 /* verbose_level */)) {
					break;
				}
			}
			if (j == len) {
				S->A->Group_element->element_move(
						Elt1, coset_reps->ith(len), 0);
				len++;
			}

		}
	}

	if (len != subgroup_index) {
		cout << "strong_generators::set_of_coset_representatives len != subgroup_index" << endl;
		exit(1);
	}

	FREE_OBJECT(H);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	//FREE_lint(Q);


	if (f_v) {
		cout << "strong_generators::set_of_coset_representatives done" << endl;
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
	algebra::ring_theory::longinteger_object G_order, stab_go;

	if (f_v) {
		cout << "strong_generators::point_stabilizer "
				"computing orbit of point " << pt << ":" << endl;
	}
	group_order(G_order);
	Sch = orbit_of_one_point_schreier(A, pt, 0 /*verbose_level*/);
	if (f_v) {
		cout << "strong_generators::point_stabilizer "
				"orbit of point " << pt << " has length "
				<< Sch->Forest->orbit_len[0] << endl;
	}
	Sch->point_stabilizer(
			A, G_order,
		Stab, 0 /* orbit_no */,
		0 /*verbose_level*/);
	Stab->group_order(stab_go);
	if (f_v) {
		cout << "strong_generators::point_stabilizer "
				"stabilizer of point " << pt << " has order "
				<< stab_go << endl;
	}
	Stab_gens = NEW_OBJECT(strong_generators);
	Stab_gens->init_from_sims(Stab, 0 /*verbose_level*/);
	if (f_v) {
		cout << "strong_generators::point_stabilizer "
				"generators for the stabilizer "
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
		int nb_fixpoints,
		actions::action *A_given, int verbose_level)
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
		cout << "strong_generators::find_cyclic_subgroup_with_exactly_n_fixpoints "
				"finding element with n fixpoints, "
				"where n = " << nb_fixpoints << ":" << endl;
	}

	H = create_sims(verbose_level - 2);

	Elt = NEW_int(A->elt_size_in_int);

	order = H->find_element_with_exactly_n_fixpoints_in_given_action(
			Elt, nb_fixpoints, A_given,
			verbose_level);

	Sub_gens = NEW_OBJECT(strong_generators);
	if (f_v) {
		cout << "strong_generators::find_cyclic_subgroup_with_exactly_n_fixpoints "
				"before init_single_with_target_go" << endl;
		cout << "Elt=" << endl;
		A->Group_element->element_print(Elt, cout);
	}
	Sub_gens->init_single_with_target_go(
			A, Elt, order, verbose_level);
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
		actions::action *A_given,
	int pt_A, int pt_B, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::make_element_which_moves_a_point_from_A_to_B" << endl;
	}

	schreier *Orb;
	int orbit_idx;
	int len;

	if (f_v) {
		cout << "strong_generators::make_element_which_moves_a_point_from_A_to_B "
				"before orbit_of_one_point_schreier" << endl;
	}


	Orb = orbit_of_one_point_schreier(
			A_given, pt_A,
			0 /*verbose_level */);

	if (f_v) {
		cout << "strong_generators::make_element_which_moves_a_point_from_A_to_B "
				"after orbit_of_one_point_schreier" << endl;
	}

	len = Orb->Forest->orbit_len[0];

	if (f_v) {
		cout << "strong_generators::make_element_which_moves_a_point_from_A_to_B "
				"orbit length = " << len << endl;
	}

	if (Orb->Forest->orbit_inv[pt_B] >= len) {
		cout << "strong_generators::make_element_which_moves_a_point_from_A_to_B "
				"the two points are not in the same orbit" << endl;
		exit(1);
	}
	Orb->transporter_from_orbit_rep_to_point(
			pt_B, orbit_idx, Elt,
			0 /* verbose_level */);

	if (A_given->Group_element->element_image_of(
			pt_A, Elt, 0 /* verbose_level*/)
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

void strong_generators::export_group_and_copy_to_latex(
		std::string &label_txt,
		std::ostream &ost,
		actions::action *A2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "strong_generators::export_group_and_copy_to_latex" << endl;
	}

	interfaces::magma_interface M;

	M.export_group_to_magma_and_copy_to_latex(
			label_txt, ost, A2, this,
			verbose_level);


	interfaces::l3_interface_gap GAP;

	GAP.export_group_to_GAP_and_copy_to_latex(
			ost,
			label_txt, this, A2,
			verbose_level);

	if (f_v) {
		cout << "strong_generators::export_group_and_copy_to_latex done" << endl;
	}

}


void strong_generators::report_fixed_objects_in_PG(
		std::ostream &ost,
		geometry::projective_geometry::projective_space *P,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "strong_generators::report_fixed_objects_in_PG" << endl;
	}

	ost << "\\begin{enumerate}[(1)]" << endl;
	for (i = 0; i < gens->len; i++) {

		ost << "\\item" << endl;
		A->Group_element->report_fixed_objects_in_PG(
				ost,
				P,
				gens->ith(i),
				verbose_level);
	}
	ost << "\\end{enumerate}" << endl;
	if (f_v) {
		cout << "strong_generators::report_fixed_objects_in_PG" << endl;
	}
}

void strong_generators::reverse_isomorphism_exterior_square(
		int verbose_level)
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

void strong_generators::get_gens_data(
		int *&data, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::get_gens_data" << endl;
	}
	int i;

	sz = gens->len * A->make_element_size;
	data = NEW_int(sz);
	for (i = 0; i < gens->len; i++) {
		Int_vec_copy(
				gens->ith(i),
				data + i * A->make_element_size,
				A->make_element_size);
	}
	if (f_v) {
		cout << "strong_generators::get_gens_data done" << endl;
	}
}

std::string strong_generators::stringify_gens_data(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::stringify_gens_data" << endl;
	}
	int *data;
	int sz;
	string str;

	get_gens_data(data, sz, verbose_level);


	str = Int_vec_stringify(data, sz);

	FREE_int(data);

	if (f_v) {
		cout << "strong_generators::stringify_gens_data done" << endl;
	}
	return str;
}

void strong_generators::export_to_orbiter_as_bsgs(
		actions::action *A2,
		std::string &fname,
		std::string &label, std::string &label_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int a;
	other::orbiter_kernel_system::file_io Fio;
	algebra::ring_theory::longinteger_object go;

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

		fname_generators = label + "_gens.csv";


#if 0
		for (i = 0; i < gens->len; i++) {
			fp << "GENERATOR_" << label << "_" << i << " = \\" << endl;
			fp << "\t\"";
			for (j = 0; j < A2->degree; j++) {
				if (false) {
					cout << "strong_generators::export_to_orbiter_as_bsgs "
							"computing image of " << j << " under generator " << i << endl;
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

				if (f_v) {
					cout << "strong_generators::export_to_orbiter_as_bsgs "
							"computing generator " << i << " / " << gens->len << endl;
				}

				for (j = 0; j < A2->degree; j++) {
					a = A2->Group_element->element_image_of(j, gens->ith(i), 0 /* verbose_level*/);
					Data[i * A2->degree + j] = a;
				}
			}


			if (f_v) {
				cout << "strong_generators::export_to_orbiter_as_bsgs "
						"writing csv file" << endl;
			}
			Fio.Csv_file_support->lint_matrix_write_csv(
					fname_generators, Data, gens->len, A2->degree);
			if (f_v) {
				cout << "strong_generators::export_to_orbiter_as_bsgs "
						"writing csv file done" << endl;
			}


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





void strong_generators::report_group(
		std::string &prefix, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i;

	if (f_v) {
		cout << "strong_generators::report_group" << endl;
	}

	string fname;

	fname = prefix + "_report.tex";

	{
		string title, author, extra_praeamble;

		title = "Group";


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
				cout << "strong_generators::report_group "
						"before report_group2" << endl;
			}
			report_group2(ost, verbose_level);
			if (f_v) {
				cout << "strong_generators::report_group "
						"after report_group2" << endl;
			}


			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "strong_generators::report_group done" << endl;
	}
}


void strong_generators::report_group2(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::report_group2" << endl;
	}

	print_generators_tex(ost);

	print_for_make_element(ost);

	if (f_v) {
		cout << "strong_generators::report_group2 done" << endl;
	}
}

void strong_generators::stringify(
		std::string &s_tl, std::string &s_gens, std::string &s_go)
{
	int f_v = false;

	if (f_v) {
		cout << "strong_generators::stringify "
				"before Int_vec_stringify" << endl;
	}
	s_tl = Int_vec_stringify(tl, A->base_len());
	if (f_v) {
		cout << "strong_generators::stringify "
				"after Int_vec_stringify" << endl;
	}

	if (f_v) {
		cout << "strong_generators::stringify "
				"before stringify_gens_data" << endl;
	}
	s_gens = stringify_gens_data(0 /*verbose_level*/);
	if (f_v) {
		cout << "strong_generators::stringify "
				"after stringify_gens_data" << endl;
	}

	algebra::ring_theory::longinteger_object go;


	group_order(go);
	if (f_v) {
		cout << "strong_generators::stringify "
				"before go.stringify" << endl;
	}
	s_go = go.stringify();
	if (f_v) {
		cout << "strong_generators::stringify "
				"after go.stringify" << endl;
	}

}

void strong_generators::compute_rank_vector(
		long int *&rank_vector, int &len, groups::sims *Sims,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "strong_generators::compute_rank_vector" << endl;
	}


	if (f_v) {
		cout << "strong_generators::compute_rank_vector "
				"before gens->compute_rank_vector" << endl;
	}
	gens->compute_rank_vector(
			rank_vector, Sims, verbose_level);
	if (f_v) {
		cout << "strong_generators::compute_rank_vector "
				"after gens->compute_rank_vector" << endl;
	}
	len = gens->len;

	if (f_v) {
		cout << "strong_generators::compute_rank_vector done" << endl;
	}
}


}}}


