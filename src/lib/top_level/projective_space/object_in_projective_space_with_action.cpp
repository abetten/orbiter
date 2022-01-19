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
namespace top_level {



object_in_projective_space_with_action::object_in_projective_space_with_action()
{
	OwCF = NULL;
	Aut_gens = NULL;
	ago = 0;
	nb_rows = nb_cols = 0;
	canonical_labeling = NULL;
	//null();
}

object_in_projective_space_with_action::~object_in_projective_space_with_action()
{
	freeself();
}

void object_in_projective_space_with_action::null()
{
}

void object_in_projective_space_with_action::freeself()
{
	if (canonical_labeling) {
		FREE_int(canonical_labeling);
	}
	null();
}

void object_in_projective_space_with_action::init(
	object_with_canonical_form *OwCF,
	long int ago,
	groups::strong_generators *Aut_gens,
	//int nb_rows, int nb_cols,
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
	OwCF->f_has_known_ago = TRUE;
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

void object_in_projective_space_with_action::report(std::ostream &fp,
		projective_space_with_action *PA, int max_TDO_depth, int verbose_level)
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
		cout << "projective_space_object_classifier::latex_report before Nau.set_stabilizer_of_object" << endl;
	}

	actions::nauty_interface_with_group Nau;
	data_structures::nauty_output *NO;

	NO = NEW_OBJECT(data_structures::nauty_output);
	NO->allocate(nb_r + nb_c, verbose_level);

	SG = Nau.set_stabilizer_of_object(
			OwCF,
		PA->A,
		TRUE /* f_compute_canonical_form */, Canonical_form,
		NO,
		verbose_level - 2);

	if (f_v) {
		cout << "projective_space_object_classifier::latex_report after Nau.set_stabilizer_of_object" << endl;
	}

	FREE_OBJECT(NO);

	SG->group_order(go);
#endif

	//if (OiP->type != t_PAC) {

	OwCF->print_tex(fp);
		fp << endl;
		fp << "\\bigskip" << endl;
		fp << endl;
	//	}

	if (OwCF->type == t_PAC) {
		long int *Sets;
		int nb_sets;
		int set_size;
		actions::action *A_on_spreads;
		groups::schreier *Sch;

		OwCF->get_packing_as_set_system(Sets, nb_sets, set_size, verbose_level);


		A_on_spreads = PA->A_on_lines->create_induced_action_on_sets(nb_sets,
				set_size, Sets,
				verbose_level);


		Sch = SG->orbits_on_points_schreier(A_on_spreads, verbose_level);

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




#if 0
	if (OiP->type == t_PTS) {
		//long int *set;
		//int sz;

		OiP->print_tex(fp);


		cout << "printing generators in restricted action:" << endl;
		action *A_restricted;

		A_restricted = SG->A->restricted_action(OiP->set, OiP->sz,
				verbose_level);
		SG->print_with_given_action(
				fp, A_restricted);
		FREE_OBJECT(A_restricted);
	}
#endif


	fp << "Stabilizer:\\\\" << endl;
	SG->print_generators_tex(fp);


#if 0
	//fp << "Stabilizer, all elements:\\\\" << endl;
	//SG->print_elements_ost(fp);
	//SG->print_elements_with_special_orthogonal_action_ost(fp);

	{
		action *A_conj;
		sims *Base_group;

		Base_group = SG->create_sims(verbose_level);

		A_conj = PA->A->create_induced_action_by_conjugation(
			Base_group, FALSE /* f_ownership */,
			verbose_level);

		fp << "Generators in conjugation action on the group itself:\\\\" << endl;
		SG->print_with_given_action(fp, A_conj);

		fp << "Elements in conjugation action on the group itself:\\\\" << endl;
		SG->print_elements_with_given_action(fp, A_conj);

		string fname_gap;
		char str[1000];

		fname_gap.assign("class_");

		sprintf(str, "%d", i);

		fname_gap.append(str);
		fname_gap.append(".gap");

		SG->export_permutation_group_to_GAP(fname_gap.c_str(), verbose_level);
		schreier *Sch;

		Sch = SG->orbits_on_points_schreier(A_conj, verbose_level);

		fp << "Orbits on itself by conjugation:\\\\" << endl;
		Sch->print_and_list_orbits_tex(fp);


		FREE_OBJECT(Sch);
		FREE_OBJECT(A_conj);
		FREE_OBJECT(Base_group);
	}
#endif


	combinatorics::encoded_combinatorial_object *Enc;
	incidence_structure *Inc;
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


	Inc->get_and_print_row_tactical_decomposition_scheme_tex(
		fp, TRUE /* f_enter_math */,
		TRUE /* f_print_subscripts */, *Stack);

#if 0
	Inc->get_and_print_tactical_decomposition_scheme_tex(
		fp, TRUE /* f_enter_math */,
		*Stack);
#endif



	int f_refine_prev, f_refine, h;
	int f_print_subscripts = TRUE;

	f_refine_prev = TRUE;
	for (h = 0; h < max_TDO_depth; h++) {
		if (EVEN(h)) {
			f_refine = Inc->refine_column_partition_safe(
					*Stack, verbose_level - 3);
		}
		else {
			f_refine = Inc->refine_row_partition_safe(
					*Stack, verbose_level - 3);
		}

		if (f_v) {
			cout << "incidence_structure::compute_TDO_safe "
					"h=" << h << " after refine" << endl;
		}
		if (EVEN(h)) {
			//int f_list_incidences = FALSE;
			Inc->get_and_print_column_tactical_decomposition_scheme_tex(
				fp, TRUE /* f_enter_math */,
				f_print_subscripts, *Stack);
			//get_and_print_col_decomposition_scheme(
			//PStack, f_list_incidences, FALSE);
			//PStack.print_classes_points_and_lines(cout);
		}
		else {
			//int f_list_incidences = FALSE;
			Inc->get_and_print_row_tactical_decomposition_scheme_tex(
				fp, TRUE /* f_enter_math */,
				f_print_subscripts, *Stack);
			//get_and_print_row_decomposition_scheme(
			//PStack, f_list_incidences, FALSE);
			//PStack.print_classes_points_and_lines(cout);
		}

		if (!f_refine_prev && !f_refine) {
			break;
		}
		f_refine_prev = f_refine;
	}

	cout << "Classes of the partition:\\\\" << endl;
	Stack->print_classes_tex(fp);



	OwCF->klein(verbose_level);

#if 0
	sims *Stab;
	int *Elt;
	int nb_trials;
	int max_trials = 100;

	Stab = SG->create_sims(verbose_level);
	Elt = NEW_int(PA->A->elt_size_in_int);

	for (h = 0; h < fixed_structure_order_list_sz; h++) {
		if (Stab->find_element_of_given_order_int(Elt,
				fixed_structure_order_list[h], nb_trials, max_trials,
				verbose_level)) {
			fp << "We found an element of order "
					<< fixed_structure_order_list[h] << ", which is:" << endl;
			fp << "$$" << endl;
			PA->A->element_print_latex(Elt, fp);
			fp << "$$" << endl;
			PA->report_fixed_points_lines_and_planes(
				Elt, fp,
				verbose_level);
		}
		else {
			fp << "We could not find an element of order "
				<< fixed_structure_order_list[h] << "\\\\" << endl;
		}
	}

	FREE_int(Elt);
#endif

	FREE_OBJECT(SG);

	FREE_OBJECT(Stack);
	FREE_OBJECT(Inc);

}

}}

