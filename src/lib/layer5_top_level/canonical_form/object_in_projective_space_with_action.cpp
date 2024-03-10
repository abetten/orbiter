// object_in_projective_space_with_action.cpp
// 
// Anton Betten
//
// December 30, 2017
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace canonical_form {



object_in_projective_space_with_action::object_in_projective_space_with_action()
{
	OwCF = NULL;
	Aut_gens = NULL;
	ago = 0;
	nb_rows = nb_cols = 0;
	canonical_labeling = NULL;
}

object_in_projective_space_with_action::~object_in_projective_space_with_action()
{
	if (canonical_labeling) {
		FREE_int(canonical_labeling);
	}
}

void object_in_projective_space_with_action::init(
		canonical_form_classification::object_with_canonical_form *OwCF,
	long int ago,
	groups::strong_generators *Aut_gens,
	int *canonical_labeling,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space_with_action::init" << endl;
	}

	int nb_rows, nb_cols;

	OwCF->encoding_size(
			nb_rows, nb_cols,
			verbose_level);


	object_in_projective_space_with_action::OwCF = OwCF;
	object_in_projective_space_with_action::Aut_gens = Aut_gens;
	object_in_projective_space_with_action::ago = ago;
	object_in_projective_space_with_action::nb_rows = nb_rows;
	object_in_projective_space_with_action::nb_cols = nb_cols;
	object_in_projective_space_with_action::canonical_labeling = canonical_labeling;
	OwCF->f_has_known_ago = true;
	OwCF->known_ago = ago; //Aut_gens->group_order_as_lint();
	if (f_v) {
		cout << "object_in_projective_space_with_action::init done" << endl;
	}
}

void object_in_projective_space_with_action::print()
{
	cout << "object_in_projective_space_with_action" << endl;
	cout << "nb_rows=" << nb_rows << endl;
	cout << "nb_cols=" << nb_cols << endl;
	cout << "ago=" << ago << endl;
}

void object_in_projective_space_with_action::report(
		std::ostream &fp,
		projective_geometry::projective_space_with_action *PA,
		int max_TDO_depth, int verbose_level)
// includes the TDO
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "object_in_projective_space_with_action::report" << endl;
	}

	if (f_v) {
		cout << "OwCF:" << endl;
		OwCF->print(cout);
	}
	if (OwCF->type != t_PAC) {
		OwCF->print(cout);
	}



	groups::strong_generators *SG;
	ring_theory::longinteger_object go;


	data_structures::bitvector *Canonical_form;

	int nb_r, nb_c;

	OwCF->encoding_size(
			nb_r, nb_c,
			verbose_level);

#if 1
	if (f_v) {
		cout << "projective_space_object_classifier::latex_report "
				"before Nau.set_stabilizer_of_object" << endl;
	}

	long int nauty_complexity;


	{
		interfaces::nauty_interface_with_group Nau;
		l1_interfaces::nauty_output *NO;
		canonical_form_classification::encoded_combinatorial_object *Enc;

		NO = NEW_OBJECT(l1_interfaces::nauty_output);
		NO->nauty_output_allocate(nb_r + nb_c,
				0,
				nb_r + nb_c,
				verbose_level);

		SG = Nau.set_stabilizer_of_object(
				OwCF,
			PA->A,
			true /* f_compute_canonical_form */, Canonical_form,
			NO,
			Enc,
			verbose_level - 2);

		if (f_v) {
			cout << "projective_space_object_classifier::latex_report "
					"after Nau.set_stabilizer_of_object" << endl;
		}

		nauty_complexity = NO->nauty_complexity();

		FREE_OBJECT(NO);
		FREE_OBJECT(Enc);
	}

	SG->group_order(go);
#endif

	//if (OiP->type != t_PAC) {

	OwCF->print_tex(fp, verbose_level);
		fp << endl;
		fp << "\\bigskip" << endl;
		fp << endl;
	//	}

	fp << "Nauty complexity: " << nauty_complexity << "\\\\" << endl;

	if (OwCF->type == t_PAC) {
		long int *Sets;
		int nb_sets;
		int set_size;
		actions::action *A_on_spreads;
		groups::schreier *Sch;

		OwCF->get_packing_as_set_system(Sets, nb_sets, set_size, verbose_level);


		A_on_spreads = PA->A_on_lines->Induced_action->create_induced_action_on_sets(
				nb_sets,
				set_size, Sets,
				verbose_level);


		Sch = SG->compute_all_point_orbits_schreier(A_on_spreads, verbose_level);

		fp << "Orbits on spreads:\\\\" << endl;
		Sch->print_and_list_orbits_tex(fp);


		FREE_OBJECT(Sch);
		FREE_OBJECT(A_on_spreads);
		FREE_lint(Sets);
	}
	//int_vec_print(fp, OiP->set, OiP->sz);
	fp << "Group order " << go << "\\\\" << endl;

	//fp << "Stabilizer:" << endl;
	//SG->print_generators_tex(fp);





	fp << "Stabilizer:\\\\" << endl;
	SG->print_generators_tex(fp);




	canonical_form_classification::encoded_combinatorial_object *Enc;
	geometry::incidence_structure *Inc;
	data_structures::partitionstack *Stack;


	OwCF->encode_incma_and_make_decomposition(
		Enc,
		Inc,
		Stack,
		verbose_level);
	FREE_OBJECT(Enc);
#if 0
	cout << "set ";
	int_vec_print(cout, OiP->set, OiP->sz);
	cout << " go=" << go << endl;



	incidence_structure *Inc;
	partitionstack *Stack;

	int Sz[1];
	int *Subsets[1];

	Sz[0] = OiP->sz;
	Subsets[0] = OiP->set;

	cout << "computing decomposition:" << endl;
	PA->P->decomposition(1 /* nb_subsets */, Sz, Subsets,
		Inc,
		Stack,
		verbose_level);

#if 0
	cout << "the decomposition is:" << endl;
	Inc->get_and_print_decomposition_schemes(*Stack);
	Stack->print_classes(cout);
#endif




#if 0
	fp << "canonical form: ";
	for (i = 0; i < canonical_form_len; i++) {
		fp << (int)canonical_form[i];
		if (i < canonical_form_len - 1) {
			fp << ", ";
			}
		}
	fp << "\\\\" << endl;
#endif
#endif

	combinatorics::decomposition *Decomposition;


	Decomposition = NEW_OBJECT(combinatorics::decomposition);

	Decomposition->init_inc_and_stack(
			Inc, Stack,
			verbose_level);


	Decomposition->get_and_print_row_tactical_decomposition_scheme_tex(
		fp, true /* f_enter_math */,
		true /* f_print_subscripts */);



	int f_refine_prev, f_refine, h;
	int f_print_subscripts = true;

	f_refine_prev = true;
	for (h = 0; h < max_TDO_depth; h++) {
		if (EVEN(h)) {
			f_refine = Decomposition->refine_column_partition_safe(
					verbose_level - 3);
		}
		else {
			f_refine = Decomposition->refine_row_partition_safe(
					verbose_level - 3);
		}

		if (f_v) {
			cout << "projective_space_object_classifier::latex_report "
					"h=" << h << " after refine" << endl;
		}
		if (EVEN(h)) {
			Decomposition->get_and_print_column_tactical_decomposition_scheme_tex(
				fp, true /* f_enter_math */,
				f_print_subscripts);
		}
		else {
			Decomposition->get_and_print_row_tactical_decomposition_scheme_tex(
				fp, true /* f_enter_math */,
				f_print_subscripts);
		}

		if (!f_refine_prev && !f_refine) {
			break;
		}
		f_refine_prev = f_refine;
	}

	cout << "Classes of the partition:\\\\" << endl;
	Stack->print_classes_tex(fp);



	//OwCF->klein(verbose_level);

	FREE_OBJECT(SG);

	FREE_OBJECT(Decomposition);
	//FREE_OBJECT(Stack);
	FREE_OBJECT(Inc);

}

}}}


