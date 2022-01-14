// projective_space_with_action.cpp
// 
// Anton Betten
//
// December 22, 2017
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



projective_space_with_action::projective_space_with_action()
{
	null();
}

projective_space_with_action::~projective_space_with_action()
{
	freeself();
}

void projective_space_with_action::null()
{
	q = 0;
	F = NULL;
	P = NULL;
	PA2 = NULL;
	Dom = NULL;
	QCDA = NULL;
	A = NULL;
	A_on_lines = NULL;

	f_has_action_on_planes = FALSE;
	A_on_planes = NULL;

	Elt1 = NULL;
}

void projective_space_with_action::freeself()
{
	if (P) {
		FREE_OBJECT(P);
	}
	if (PA2) {
		FREE_OBJECT(PA2);
	}
	if (Dom) {
		FREE_OBJECT(Dom);
	}
	if (QCDA) {
		FREE_OBJECT(QCDA);
	}
	if (A) {
		FREE_OBJECT(A);
	}
	if (A_on_lines) {
		FREE_OBJECT(A_on_lines);
	}
	if (f_has_action_on_planes) {
		FREE_OBJECT(A_on_planes);
	}
	if (Elt1) {
		FREE_int(Elt1);
	}
	null();
}

void projective_space_with_action::init(
	finite_field *F, int n, int f_semilinear,
	int f_init_incidence_structure,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::init" << endl;
	}
	projective_space_with_action::f_init_incidence_structure = f_init_incidence_structure;
	projective_space_with_action::n = n;
	projective_space_with_action::F = F;
	projective_space_with_action::f_semilinear = f_semilinear;
	d = n + 1;
	q = F->q;
	
	P = NEW_OBJECT(projective_space);
	P->init(n, F, 
		f_init_incidence_structure, 
		verbose_level);
	
	init_group(f_semilinear, verbose_level);

	if (n == 2) {
		Dom = NEW_OBJECT(quartic_curve_domain);

		if (f_v) {
			cout << "projective_space_with_action::init before Dom->init" << endl;
		}
		Dom->init(F, verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::init after Dom->init" << endl;
		}
		QCDA = NEW_OBJECT(quartic_curve_domain_with_action);
		if (f_v) {
			cout << "projective_space_with_action::init before QCDA->init" << endl;
		}
		QCDA->init(Dom, this, verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::init after QCDA->init" << endl;
		}
	}

	if (n >= 3) {
		if (f_v) {
			cout << "projective_space_with_action::init n >= 3, so we initialize a plane" << endl;
		}
		PA2 = NEW_OBJECT(projective_space_with_action);
		if (f_v) {
			cout << "projective_space_with_action::init before PA2->init" << endl;
		}
		PA2->init(F, 2, f_semilinear,
			f_init_incidence_structure,
			verbose_level - 2);
		if (f_v) {
			cout << "projective_space_with_action::init after PA2->init" << endl;
		}
	}


	
	Elt1 = NEW_int(A->elt_size_in_int);


	if (f_v) {
		cout << "projective_space_with_action::init done" << endl;
	}
}

void projective_space_with_action::init_group(
		int f_semilinear, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "projective_space_with_action::init_group" << endl;
	}
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating linear group" << endl;
	}

	vector_ge *nice_gens;

	A = NEW_OBJECT(action);
	A->init_linear_group(
		F, d, 
		TRUE /*f_projective*/,
		FALSE /* f_general*/,
		FALSE /* f_affine */,
		f_semilinear,
		FALSE /* f_special */,
		nice_gens,
		0 /* verbose_level*/);
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating linear group done" << endl;
	}
#if 0
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"before create_sims" << endl;
	}
	S = A->Strong_gens->create_sims(verbose_level - 2);

	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"after create_sims" << endl;
	}
#endif
	FREE_OBJECT(nice_gens);


	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating action on lines" << endl;
	}
	A_on_lines = A->induced_action_on_grassmannian(2, verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating action on lines done" << endl;
	}

	if (d >= 4) {
		if (f_v) {
			cout << "projective_space_with_action::init_group "
					"creating action on planes" << endl;
		}
		f_has_action_on_planes = TRUE;
		A_on_planes = A->induced_action_on_grassmannian(3, verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::init_group "
					"creating action on lines planes" << endl;
		}
	}
	else {
		f_has_action_on_planes = FALSE;
		A_on_planes = NULL;
	}

	if (f_v) {
		cout << "projective_space_with_action::init_group done" << endl;
	}
}


#if 0
void projective_space_with_action::canonical_form(
		projective_space_object_classifier_description *Canonical_form_PG_Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	classification_of_objects *OC;

	if (f_v) {
		cout << "projective_space_with_action::canonical_form" << endl;
	}

#if 1
	OC = NEW_OBJECT(classification_of_objects);

	data_input_stream_description *IS_Descr;

	IS_Descr = NEW_OBJECT(data_input_stream_description);

	IS_Descr->add_set_of_points(a);

	data_input_stream *IS;


	IS = NEW_OBJECT(data_input_stream);


	if (f_v) {
		cout << "projective_space_with_action::canonical_form before OC->do_the_work" << endl;
	}
	OC->do_the_work(
			Canonical_form_PG_Descr,
			TRUE,
			this,
			IS,
			verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::canonical_form after OC->do_the_work" << endl;
	}

	FREE_OBJECT(OC);
#endif

	if (f_v) {
		cout << "projective_space_with_action::canonical_form done" << endl;
	}
}
#endif

void projective_space_with_action::canonical_labeling(
	object_with_canonical_form *OiP,
	int *canonical_labeling,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling"
				<< endl;
	}

	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling "
				"before OiP->canonical_labeling" << endl;
	}

	int nb_rows, nb_cols;

	OiP->encoding_size(
			nb_rows, nb_cols,
			0 /* verbose_level */);

	nauty_output *NO;

	NO = NEW_OBJECT(nauty_output);
	NO->allocate(nb_rows + nb_cols, 0 /* verbose_level */);


	OiP->canonical_labeling(NO, verbose_level);

	int i;

	for (i = 0; i < NO->N; i++) {
		canonical_labeling[i] = NO->canonical_labeling[i];
	}
	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling "
				"after OiP->canonical_labeling" << endl;
	}

	FREE_OBJECT(NO);


	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling done" << endl;
	}
}

void projective_space_with_action::report_fixed_points_lines_and_planes(
	int *Elt, std::ostream &ost,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_fixed_points_lines_and_planes" << endl;
	}

	if (P->n < 3) {
		cout << "projective_space_with_action::report_fixed_points_lines_and_planes P->n < 3" << endl;
		exit(1);
	}
	projective_space *P3;
	int i, j, cnt;
	int v[4];

	P3 = P;
	
	ost << "Fixed Objects:\\\\" << endl;



	ost << "The element" << endl;
	ost << "$$" << endl;
	A->element_print_latex(Elt, ost);
	ost << "$$" << endl;
	ost << "has the following fixed objects:\\\\" << endl;


	ost << "Fixed points:\\\\" << endl;

	cnt = 0;
	for (i = 0; i < P3->N_points; i++) {
		j = A->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			cnt++;
		}
	}

	ost << "There are " << cnt << " fixed points, they are: \\\\" << endl;
	for (i = 0; i < P3->N_points; i++) {
		j = A->element_image_of(i, Elt, 0 /* verbose_level */);
		F->PG_element_unrank_modified(v, 1, 4, i);
		if (j == i) {
			ost << i << " : ";
			Orbiter->Int_vec.print(ost, v, 4);
			ost << "\\\\" << endl;
			cnt++;
		}
	}

	ost << "Fixed Lines:\\\\" << endl;

	{
		action *A2;

		A2 = A->induced_action_on_grassmannian(2, 0 /* verbose_level*/);
	
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
				}
			}

		ost << "There are " << cnt << " fixed lines, they are: \\\\" << endl;
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				ost << i << " : $\\left[";
				A2->G.AG->G->print_single_generator_matrix_tex(ost, i);
				ost << "\\right]$\\\\" << endl;
				cnt++;
				}
			}

		FREE_OBJECT(A2);
	}

	ost << "Fixed Planes:\\\\" << endl;

	{
		action *A2;

		A2 = A->induced_action_on_grassmannian(3, 0 /* verbose_level*/);
	
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
			}
		}

		ost << "There are " << cnt << " fixed planes, they are: \\\\" << endl;
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				ost << i << " : $\\left[";
				A2->G.AG->G->print_single_generator_matrix_tex(ost, i);
				ost << "\\right]$\\\\" << endl;
				cnt++;
			}
		}

		FREE_OBJECT(A2);
	}

	if (f_v) {
		cout << "projective_space_with_action::report_fixed_points_lines_and_planes done" << endl;
	}
}

void projective_space_with_action::report_orbits_on_points_lines_and_planes(
	int *Elt, std::ostream &ost,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_orbits_on_points_lines_and_planes" << endl;
	}

	if (P->n < 3) {
		cout << "projective_space_with_action::report_orbits_on_points_lines_and_planes P->n < 3" << endl;
		exit(1);
	}
	//projective_space *P3;
	int order;

	longinteger_object full_group_order;
	order = A->element_order(Elt);

	full_group_order.create(order, __FILE__, __LINE__);

	//P3 = P;

	ost << "Fixed Objects:\\\\" << endl;



	ost << "The group generated by the element" << endl;
	ost << "$$" << endl;
	A->element_print_latex(Elt, ost);
	ost << "$$" << endl;
	ost << "has the following orbits:\\\\" << endl;

	ost << "Orbits on points:\\\\" << endl;


	schreier *Sch;

	Sch = NEW_OBJECT(schreier);
	A->all_point_orbits_from_single_generator(*Sch,
			Elt,
			0 /*verbose_level*/);
	Sch->print_orbit_lengths_tex(ost);


	FREE_OBJECT(Sch);

	ost << "Orbits on lines:\\\\" << endl;

	{
		action *A2;
		schreier *Sch;

		A2 = A->induced_action_on_grassmannian(2, 0 /* verbose_level*/);

		Sch = NEW_OBJECT(schreier);
		A2->all_point_orbits_from_single_generator(*Sch,
				Elt,
				0 /*verbose_level*/);
		Sch->print_orbit_lengths_tex(ost);


		FREE_OBJECT(Sch);
		FREE_OBJECT(A2);
	}

	ost << "Orbits on planes:\\\\" << endl;

	{
		action *A2;
		schreier *Sch;


		A2 = A->induced_action_on_grassmannian(3, 0 /* verbose_level*/);

		Sch = NEW_OBJECT(schreier);
		A2->all_point_orbits_from_single_generator(*Sch,
				Elt,
				0 /*verbose_level*/);
		Sch->print_orbit_lengths_tex(ost);


		FREE_OBJECT(Sch);
		FREE_OBJECT(A2);
	}

	if (f_v) {
		cout << "projective_space_with_action::report_orbits_on_points_lines_and_planes done" << endl;
	}
}

void projective_space_with_action::report_decomposition_by_single_automorphism(
	int *Elt, ostream &ost, std::string &fname_base,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_single_automorphism" << endl;
	}

	top_level_geometry_global Geo;


	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_single_automorphism "
				"before Geo.report_decomposition_by_single_automorphism" << endl;
	}

	Geo.report_decomposition_by_single_automorphism(
			this,
			Elt, ost, fname_base,
			verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_single_automorphism "
				"after Geo.report_decomposition_by_single_automorphism" << endl;
	}


	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_single_automorphism done" << endl;
	}
}






void projective_space_with_action::compute_group_of_set(long int *set, int set_sz,
		strong_generators *&Sg,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	long int a;
	char str[1000];

	if (f_v) {
		cout << "projective_space_with_action::compute_group_of_set" << endl;
	}

#if 0
	projective_space_object_classifier_description *Descr;
	classification_of_objects *Classifier;

	Descr = NEW_OBJECT(projective_space_object_classifier_description);
	Classifier = NEW_OBJECT(classification_of_objects);

	Descr->f_input = TRUE;
	Descr->Data = NEW_OBJECT(data_input_stream_description);
	Descr->Data->input_type[Descr->Data->nb_inputs] = INPUT_TYPE_SET_OF_POINTS;
	Descr->Data->input_string[Descr->Data->nb_inputs].assign("");
	for (i = 0; i < set_sz; i++) {
		a = set[i];
		sprintf(str, "%ld", a);
		Descr->Data->input_string[Descr->Data->nb_inputs].append(str);
		if (i < set_sz - 1) {
			Descr->Data->input_string[Descr->Data->nb_inputs].append(",");
		}
	}
	Descr->Data->input_string2[Descr->Data->nb_inputs].assign("");
	Descr->Data->nb_inputs++;

	if (f_v) {
		cout << "projective_space_with_action::compute_group_of_set "
				"before Classifier->do_the_work" << endl;
	}

	Classifier->do_the_work(
			Descr,
			TRUE,
			this,
			verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::compute_group_of_set "
				"after Classifier->do_the_work" << endl;
	}

	int idx;
	long int ago;

	idx = Classifier->CB->type_of[Classifier->CB->n - 1];


	object_in_projective_space_with_action *OiPA;

	OiPA = (object_in_projective_space_with_action *) Classifier->CB->Type_extra_data[idx];


	ago = OiPA->ago;

	Sg = OiPA->Aut_gens;

	Sg->A = A;


	if (f_v) {
		cout << "projective_space_with_action::compute_group_of_set ago = " << ago << endl;
	}

#endif


	if (f_v) {
		cout << "projective_space_with_action::compute_group_of_set done" << endl;
	}
}


void projective_space_with_action::map(formula *Formula,
		std::string &evaluate_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::map" << endl;
	}

	if (f_v) {
		cout << "projective_space_activity::map" << endl;
		cout << "formula:" << endl;
		Formula->print();
	}

	if (!Formula->f_is_homogeneous) {
		cout << "Formula is not homogeneous" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "Formula is homogeneous of degree " << Formula->degree << endl;
		exit(1);
	}
	if (Formula->nb_managed_vars != P->n + 1) {
		cout << "Formula->nb_managed_vars != P->n + 1" << endl;
		exit(1);
	}

	homogeneous_polynomial_domain *Poly;

	Poly = NEW_OBJECT(homogeneous_polynomial_domain);

	if (f_v) {
		cout << "projective_space_with_action::map before Poly->init" << endl;
	}
	Poly->init(F,
			Formula->nb_managed_vars /* nb_vars */, Formula->degree,
			FALSE /* f_init_incidence_structure */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::map after Poly->init" << endl;
	}


	syntax_tree_node **Subtrees;
	int nb_monomials;

	if (f_v) {
		cout << "projective_space_with_action::map before Formula->get_subtrees" << endl;
	}
	Formula->get_subtrees(Poly, Subtrees, nb_monomials, verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::map after Formula->get_subtrees" << endl;
	}

	int i;

	for (i = 0; i < nb_monomials; i++) {
		cout << "Monomial " << i << " : ";
		if (Subtrees[i]) {
			Subtrees[i]->print_expression(cout);
			cout << " * ";
			Poly->print_monomial(cout, i);
			cout << endl;
		}
		else {
			cout << "no subtree" << endl;
		}
	}


	int *Coefficient_vector;

	Coefficient_vector = NEW_int(nb_monomials);

	Formula->evaluate(Poly,
			Subtrees, evaluate_text, Coefficient_vector,
			verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::map coefficient vector:" << endl;
		Orbiter->Int_vec.print(cout, Coefficient_vector, nb_monomials);
		cout << endl;
	}

#if 0
	del_pezzo_surface_of_degree_two_domain *del_Pezzo;

	del_Pezzo = NEW_OBJECT(del_pezzo_surface_of_degree_two_domain);

	del_Pezzo->init(P, Poly4_3, verbose_level);

	del_pezzo_surface_of_degree_two_object *del_Pezzo_surface;

	del_Pezzo_surface = NEW_OBJECT(del_pezzo_surface_of_degree_two_object);

	del_Pezzo_surface->init(del_Pezzo,
			Formula, Subtrees, Coefficient_vector,
			verbose_level);

	del_Pezzo_surface->enumerate_points_and_lines(verbose_level);

	del_Pezzo_surface->pal->write_points_to_txt_file(Formula->name_of_formula, verbose_level);

	del_Pezzo_surface->create_latex_report(Formula->name_of_formula, Formula->name_of_formula_latex, verbose_level);

	FREE_OBJECT(del_Pezzo_surface);
	FREE_OBJECT(del_Pezzo);
#endif

	FREE_int(Coefficient_vector);
	FREE_OBJECT(Poly);

	if (f_v) {
		cout << "projective_space_with_action::map done" << endl;
	}
}


void projective_space_with_action::analyze_del_Pezzo_surface(formula *Formula,
		std::string &evaluate_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::analyze_del_Pezzo_surface" << endl;
	}

	if (f_v) {
		cout << "projective_space_activity::analyze_del_Pezzo_surface" << endl;
		cout << "formula:" << endl;
		Formula->print();
	}

	if (!Formula->f_is_homogeneous) {
		cout << "Formula is not homogeneous" << endl;
		exit(1);
	}
	if (Formula->degree != 4) {
		cout << "Formula is not of degree 4. Degree is " << Formula->degree << endl;
		exit(1);
	}
	if (Formula->nb_managed_vars != 3) {
		cout << "Formula should have 3 managed variables. Has " << Formula->nb_managed_vars << endl;
		exit(1);
	}

	homogeneous_polynomial_domain *Poly4_3;

	Poly4_3 = NEW_OBJECT(homogeneous_polynomial_domain);

	if (f_v) {
		cout << "projective_space_with_action::analyze_del_Pezzo_surface before Poly->init" << endl;
	}
	Poly4_3->init(F,
			Formula->nb_managed_vars /* nb_vars */, Formula->degree,
			FALSE /* f_init_incidence_structure */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::analyze_del_Pezzo_surface after Poly->init" << endl;
	}


	syntax_tree_node **Subtrees;
	int nb_monomials;

	if (f_v) {
		cout << "projective_space_with_action::analyze_del_Pezzo_surface before Formula->get_subtrees" << endl;
	}
	Formula->get_subtrees(Poly4_3, Subtrees, nb_monomials, verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::analyze_del_Pezzo_surface after Formula->get_subtrees" << endl;
	}

	int i;

	for (i = 0; i < nb_monomials; i++) {
		cout << "Monomial " << i << " : ";
		if (Subtrees[i]) {
			Subtrees[i]->print_expression(cout);
			cout << " * ";
			Poly4_3->print_monomial(cout, i);
			cout << endl;
		}
		else {
			cout << "no subtree" << endl;
		}
	}


	int *Coefficient_vector;

	Coefficient_vector = NEW_int(nb_monomials);

	Formula->evaluate(Poly4_3,
			Subtrees, evaluate_text, Coefficient_vector,
			verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::analyze_del_Pezzo_surface coefficient vector:" << endl;
		Orbiter->Int_vec.print(cout, Coefficient_vector, nb_monomials);
		cout << endl;
	}

	del_pezzo_surface_of_degree_two_domain *del_Pezzo;

	del_Pezzo = NEW_OBJECT(del_pezzo_surface_of_degree_two_domain);

	del_Pezzo->init(P, Poly4_3, verbose_level);

	del_pezzo_surface_of_degree_two_object *del_Pezzo_surface;

	del_Pezzo_surface = NEW_OBJECT(del_pezzo_surface_of_degree_two_object);

	del_Pezzo_surface->init(del_Pezzo,
			Formula, Subtrees, Coefficient_vector,
			verbose_level);

	del_Pezzo_surface->enumerate_points_and_lines(verbose_level);

	del_Pezzo_surface->pal->write_points_to_txt_file(Formula->name_of_formula, verbose_level);

	del_Pezzo_surface->create_latex_report(Formula->name_of_formula, Formula->name_of_formula_latex, verbose_level);

	FREE_OBJECT(del_Pezzo_surface);
	FREE_OBJECT(del_Pezzo);

	FREE_int(Coefficient_vector);
	FREE_OBJECT(Poly4_3);

	if (f_v) {
		cout << "projective_space_with_action::analyze_del_Pezzo_surface done" << endl;
	}
}


void projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG(
		int decomposition_by_element_power,
		std::string &decomposition_by_element_data, std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG verbose_level="
				<< verbose_level << endl;
	}


	finite_field *F;

	F = P->F;


	{
		char title[1000];
		char author[1000];

		snprintf(title, 1000, "Decomposition of PG($%d,%d$)", n, F->q);
		//strcpy(author, "");
		author[0] = 0;


		string fname_tex;

		fname_tex.assign(fname_base);
		fname_tex.append(".tex");

		{
			ofstream ost(fname_tex);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG f_decomposition_by_element" << endl;
			}

			int *Elt;

			Elt = NEW_int(A->elt_size_in_int);


			A->make_element_from_string(Elt,
					decomposition_by_element_data, verbose_level);


			A->element_power_int_in_place(Elt,
					decomposition_by_element_power, verbose_level);

			report_decomposition_by_single_automorphism(
					Elt, ost, fname_base,
					verbose_level);

			FREE_int(Elt);


			L.foot(ost);

		}
		file_io Fio;

		if (f_v) {
			cout << "written file " << fname_tex << " of size "
					<< Fio.file_size(fname_tex) << endl;
		}
	}

	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG done" << endl;
	}

}

void projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup(std::string &label,
		linear_group_description * subgroup_Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup" << endl;
	}


	linear_group *H_LG;

	H_LG = NEW_OBJECT(linear_group);

	subgroup_Descr->F = P->F;

	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup H_LG->init, "
				"creating the group" << endl;
	}

	H_LG->linear_group_init(subgroup_Descr, verbose_level - 2);

	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup after H_LG->linear_group_init" << endl;
	}


	action *A;

	A = H_LG->A2;


	string fname;

	fname.assign(H_LG->label);
	fname.append("_decomp.tex");


	{
		char title[1000];
		char author[1000];

		snprintf(title, 1000, "Decomposition of PG($%d,%d$)", n, F->q);
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG f_decomposition_by_element" << endl;
			}


			report_decomposition_by_group(
					H_LG->Strong_gens, ost, H_LG->label,
					verbose_level);



			L.foot(ost);

		}
		file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
	}

	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup done" << endl;
	}
}


void projective_space_with_action::report(
	ostream &ost,
	layered_graph_draw_options *O,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report" << endl;
	}



	if (f_v) {
		cout << "projective_space_with_action::report done" << endl;
	}
}


void projective_space_with_action::create_quartic_curve(
		quartic_curve_create_description *Quartic_curve_descr,
		quartic_curve_create *&QC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::create_quartic_curve" << endl;
	}
	if (n != 2) {
		cout << "projective_space_with_action::create_quartic_curve we need a two-dimensional projective space" << endl;
		exit(1);
	}

	if (Quartic_curve_descr->get_q() != q) {
		cout << "projective_space_activity::do_create_quartic_curve Quartic_curve_descr->get_q() != q" << endl;
		exit(1);
	}

	QC = NEW_OBJECT(quartic_curve_create);

	if (f_v) {
		cout << "projective_space_with_action::create_quartic_curve before SC->init" << endl;
	}
	QC->init(Quartic_curve_descr, this, QCDA, verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::create_quartic_curve after SC->init" << endl;
	}


	if (f_v) {
		cout << "projective_space_with_action::create_quartic_curve "
				"before SC->apply_transformations" << endl;
	}
	QC->apply_transformations(Quartic_curve_descr->transform_coeffs,
			Quartic_curve_descr->f_inverse_transform,
			verbose_level - 2);

	if (f_v) {
		cout << "projective_space_with_action::create_quartic_curve "
				"after SC->apply_transformations" << endl;
	}

	QC->F->PG_element_normalize(QC->QO->eqn15, 1, 15);

	if (f_v) {
		cout << "projective_space_with_action::create_quartic_curve done" << endl;
	}
}

void projective_space_with_action::canonical_form_of_code(
		std::string &label,
		int *genma, int m, int n,
		classification_of_objects_description *Canonical_form_codes_Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code" << endl;
	}
	int i, j;
	int *v;
	long int *set;

	if (f_v) {
		cout << "Generator matrix: " << endl;
		Orbiter->Int_vec.matrix_print(genma, m, n);
		cout << endl;
	}
	v = NEW_int(m);
	set = NEW_lint(n);
	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
			v[i] = genma[i * n + j];
		}
		if (f_v) {
			cout << "projective_space_with_action::canonical_form_of_code "
					"before PA->P->rank_point" << endl;
			Orbiter->Int_vec.print(cout, v, m);
			cout << endl;
		}
		if (P == NULL) {
			cout << "P == NULL" << endl;
			exit(1);
		}
		set[j] = P->rank_point(v);
	}
	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code set=";
		Orbiter->Lint_vec.print(cout, set, n);
		cout << endl;
	}

	string_tools ST;
	string points_as_string;

	ST.create_comma_separated_list(points_as_string, set, n);
	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code "
				"points_as_string=" << points_as_string << endl;
	}

	data_input_stream_description ISD;
	data_input_stream_description_element E;

	E.init_set_of_points(points_as_string);
	ISD.Input.push_back(E);
	ISD.nb_inputs++;

	data_input_stream IS;

	IS.init(&ISD, verbose_level);



	combinatorial_object_activity_description COAD;

#if 0
	int f_save;

	int f_save_as;
	std::string save_as_fname;

	int f_extract_subset;
	std::string extract_subset_set;
	std::string extract_subset_fname;

	int f_line_type;

	int f_conic_type;
	int conic_type_threshold;

	int f_non_conical_type;

	int f_ideal;
	int ideal_degree;


	// options that apply to IS = data_input_stream

	int f_canonical_form_PG;
	std::string canonical_form_PG_PG_label;
	classification_of_objects_description *Canonical_form_PG_Descr;

	int f_canonical_form;
	classification_of_objects_description *Canonical_form_Descr;

	int f_report;
	classification_of_objects_report_options *Classification_of_objects_report_options;

#endif

	COAD.f_canonical_form_PG = TRUE;
	COAD.f_canonical_form_PG_has_PA = TRUE;
	COAD.Canonical_form_PG_PA = this;
	COAD.Canonical_form_PG_Descr = Canonical_form_codes_Descr;

	COAD.f_report = TRUE;
	COAD.Classification_of_objects_report_options = NEW_OBJECT(classification_of_objects_report_options);
	COAD.Classification_of_objects_report_options->f_prefix = TRUE;
	COAD.Classification_of_objects_report_options->prefix.assign(COAD.Canonical_form_PG_Descr->label);
	COAD.Classification_of_objects_report_options->f_export_flag_orbits = TRUE;
	COAD.Classification_of_objects_report_options->f_show_incidence_matrices = TRUE;
	COAD.Classification_of_objects_report_options->f_show_TDO = TRUE;
	COAD.Classification_of_objects_report_options->f_show_TDA = TRUE;
	COAD.Classification_of_objects_report_options->f_export_group = TRUE;



	combinatorial_object_activity COA;

	COA.init_input_stream(&COAD,
			&IS,
			verbose_level);


	COA.perform_activity(verbose_level);



	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code after PA->canonical_form" << endl;
	}


	FREE_int(v);
	FREE_lint(set);
	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code done" << endl;
	}

}

void projective_space_with_action::table_of_quartic_curves(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::table_of_quartic_curves" << endl;
	}

	if (n != 2) {
		cout << "projective_space_with_action::table_of_quartic_curves "
				"we need a two-dimensional projective space" << endl;
		exit(1);
	}

	knowledge_base K;

	int nb_quartic_curves;
	int h;
	quartic_curve_create **QC;
	int *nb_K;
	long int *Table;
	int nb_cols = 6;

	nb_quartic_curves = K.quartic_curves_nb_reps(q);

	QC = (quartic_curve_create **) NEW_pvoid(nb_quartic_curves);

	nb_K = NEW_int(nb_quartic_curves);

	Table = NEW_lint(nb_quartic_curves * nb_cols);

	for (h = 0; h < nb_quartic_curves; h++) {

		if (f_v) {
			cout << "projective_space_with_action::table_of_quartic_curves "
					<< h << " / " << nb_quartic_curves << endl;
		}
		quartic_curve_create_description Quartic_curve_descr;

		Quartic_curve_descr.f_q = TRUE;
		Quartic_curve_descr.q = q;
		Quartic_curve_descr.f_catalogue = TRUE;
		Quartic_curve_descr.iso = h;


		int *data;
		int nb_gens;
		int data_size;
		string stab_order;

		long int ago;
		string_tools ST;

		if (f_v) {
			cout << "projective_space_with_action::table_of_quartic_curves "
					<< h << " / " << nb_quartic_curves << " before K.quartic_curves_stab_gens" << endl;
		}
		K.quartic_curves_stab_gens(q, h, data, nb_gens, data_size, stab_order);

		if (f_v) {
			cout << "projective_space_with_action::table_of_quartic_curves "
					<< h << " / " << nb_quartic_curves << " stab_order=" << stab_order << endl;
		}
		ago = ST.strtolint(stab_order);

		if (ago > 0) {

			create_quartic_curve(
						&Quartic_curve_descr,
						QC[h],
						verbose_level);

			nb_K[h] = QC[h]->QO->QP->nb_Kovalevski;


			Table[h * nb_cols + 0] = h;
			Table[h * nb_cols + 1] = nb_K[h];
			Table[h * nb_cols + 2] = QC[h]->QO->QP->nb_Kovalevski_on;
			Table[h * nb_cols + 3] = QC[h]->QO->QP->nb_Kovalevski_off;
			Table[h * nb_cols + 4] = QC[h]->QOA->Aut_gens->group_order_as_lint();
			Table[h * nb_cols + 5] = QC[h]->QO->nb_pts;
		}
		else {
			Table[h * nb_cols + 0] = h;
			Table[h * nb_cols + 1] = -1;
			Table[h * nb_cols + 2] = -1;
			Table[h * nb_cols + 3] = -1;
			Table[h * nb_cols + 4] = -1;
			Table[h * nb_cols + 5] = -1;

		}
		if (f_v) {
			cout << "projective_space_with_action::table_of_quartic_curves "
					<< h << " / " << nb_quartic_curves << " done" << endl;
		}

	}

	file_io Fio;
	char str[1000];

	sprintf(str, "_q%d", q);

	string fname;
	fname.assign("quartic_curves");
	fname.append(str);
	fname.append("_info.csv");

	//Fio.lint_matrix_write_csv(fname, Table, nb_quartic_curves, nb_cols);

	{
		ofstream f(fname);
		int i, j;

		f << "Row,OCN,K,Kon,Koff,Ago,NbPts,BisecantType,Eqn15,Eqn,Pts,Bitangents28";
		f << endl;
		for (i = 0; i < nb_quartic_curves; i++) {
			f << i;


			if (Table[i * nb_cols + 1] == -1) {
				for (j = 0; j < nb_cols; j++) {
					f << ",";
				}
				f << ",,,,,";
			}
			else {


				for (j = 0; j < nb_cols; j++) {
					f << "," << Table[i * nb_cols + j];
				}
				{
					string str;
					f << ",";
					Orbiter->Int_vec.create_string_with_quotes(str, QC[i]->QO->QP->line_type_distribution, 3);
					f << str;
				}
				{
					string str;
					f << ",";
					Orbiter->Int_vec.create_string_with_quotes(str, QC[i]->QO->eqn15, 15);
					f << str;
				}

				{
					stringstream sstr;
					string str;
					QC[i]->QCDA->Dom->print_equation_maple(sstr, QC[i]->QO->eqn15);
					str.assign(sstr.str());
					f << ",";
					f << "\"$";
					f << str;
					f << "$\"";
				}

				{
					string str;
					f << ",";
					Orbiter->Lint_vec.create_string_with_quotes(str, QC[i]->QO->Pts, QC[i]->QO->nb_pts);
					f << str;
				}
				{
					string str;
					f << ",";
					Orbiter->Lint_vec.create_string_with_quotes(str, QC[i]->QO->bitangents28, 28);
					f << str;
				}
			}
			f << endl;
		}
		f << "END" << endl;
	}


	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "projective_space_with_action::table_of_quartic_curves done" << endl;
	}

}

void projective_space_with_action::table_of_cubic_surfaces(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::table_of_cubic_surfaces" << endl;
	}

	if (n != 3) {
		cout << "projective_space_with_action::table_of_cubic_surfaces "
				"we need a three-dimensional projective space" << endl;
		exit(1);
	}

	surface_with_action *Surf_A;

	setup_surface_with_action(
			Surf_A,
			verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::table_of_cubic_surfaces before Surf_A->table_of_cubic_surfaces" << endl;
	}
	Surf_A->table_of_cubic_surfaces(verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::table_of_cubic_surfaces after Surf_A->table_of_cubic_surfaces" << endl;
	}


	if (f_v) {
		cout << "projective_space_with_action::table_of_cubic_surfaces done" << endl;
	}

}

void projective_space_with_action::conic_type(
		long int *Pts, int nb_pts, int threshold,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::conic_type threshold = " << threshold << endl;
	}


	long int **Pts_on_conic;
	int **Conic_eqn;
	int *nb_pts_on_conic;
	int len;
	int h;


	if (f_v) {
		cout << "projective_space_with_action::conic_type before PA->P->conic_type" << endl;
	}

	P->conic_type(Pts, nb_pts,
			threshold,
			Pts_on_conic, Conic_eqn, nb_pts_on_conic, len,
			verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::conic_type after PA->P->conic_type" << endl;
	}


	cout << "We found the following conics:" << endl;
	for (h = 0; h < len; h++) {
		cout << h << " : " << nb_pts_on_conic[h] << " : ";
		Orbiter->Int_vec.print(cout, Conic_eqn[h], 6);
		cout << " : ";
		Orbiter->Lint_vec.print(cout, Pts_on_conic[h], nb_pts_on_conic[h]);
		cout << endl;
	}

	cout << "computing intersection types with bisecants of the first 11 points:" << endl;
	int Line_P1[55];
	int Line_P2[55];
	int P1, P2;
	long int p1, p2, line_rk;
	long int *pts_on_line;
	long int pt;
	int *Conic_line_intersection_sz;
	int cnt;
	int i, j, q, u, v;
	int nb_pts_per_line;

	q = P->F->q;
	nb_pts_per_line = q + 1;
	pts_on_line = NEW_lint(55 * nb_pts_per_line);

	cnt = 0;
	for (i = 0; i < 11; i++) {
		for (j = i + 1; j < 11; j++) {
			Line_P1[cnt] = i;
			Line_P2[cnt] = j;
			cnt++;
		}
	}
	if (cnt != 55) {
		cout << "cnt != 55" << endl;
		cout << "cnt = " << cnt << endl;
		exit(1);
	}
	for (u = 0; u < 55; u++) {
		P1 = Line_P1[u];
		P2 = Line_P2[u];
		p1 = Pts[P1];
		p2 = Pts[P2];
		line_rk = P->line_through_two_points(p1, p2);
		P->create_points_on_line(line_rk, pts_on_line + u * nb_pts_per_line, 0 /*verbose_level*/);
	}

	Conic_line_intersection_sz = NEW_int(len * 55);
	Orbiter->Int_vec.zero(Conic_line_intersection_sz, len * 55);

	for (h = 0; h < len; h++) {
		for (u = 0; u < 55; u++) {
			for (v = 0; v < nb_pts_per_line; v++) {
				if (P->test_if_conic_contains_point(Conic_eqn[h], pts_on_line[u * nb_pts_per_line + v])) {
					Conic_line_intersection_sz[h * 55 + u]++;
				}

			}
		}
	}

	sorting Sorting;
	int idx;

	cout << "We found the following conics and their intersections with the 55 bisecants:" << endl;
	for (h = 0; h < len; h++) {
		cout << h << " : " << nb_pts_on_conic[h] << " : ";
		Orbiter->Int_vec.print(cout, Conic_eqn[h], 6);
		cout << " : ";
		Orbiter->Int_vec.print_fully(cout, Conic_line_intersection_sz + h * 55, 55);
		cout << " : ";
		Orbiter->Lint_vec.print(cout, Pts_on_conic[h], nb_pts_on_conic[h]);
		cout << " : ";
		cout << endl;
	}

	for (u = 0; u < 55; u++) {
		cout << "line " << u << " : ";
		int str[55];

		Orbiter->Int_vec.zero(str, 55);
		for (v = 0; v < nb_pts; v++) {
			pt = Pts[v];
			if (Sorting.lint_vec_search_linear(pts_on_line + u * nb_pts_per_line, nb_pts_per_line, pt, idx)) {
				str[v] = 1;
			}
		}
		Orbiter->Int_vec.print_fully(cout, str, 55);
		cout << endl;
	}

	if (f_v) {
		cout << "projective_space_with_action::conic_type done" << endl;
	}

}

void projective_space_with_action::cheat_sheet(
		layered_graph_draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::cheat_sheet" << endl;
	}



	{
		char fname[1000];
		char title[1000];
		char author[1000];

		snprintf(fname, 1000, "PG_%d_%d.tex", n, F->q);
		snprintf(title, 1000, "Cheat Sheet ${\\rm PG}(%d,%d)$", n, F->q);
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_PG before A->report" << endl;
			}

			A->report(ost, A->f_has_sims, A->Sims,
					A->f_has_strong_generators, A->Strong_gens,
					O,
					verbose_level);

			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_PG after PA->A->report" << endl;
			}

			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_PG before PA->P->report" << endl;
			}



			P->report(ost, O, verbose_level);

			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_PG after PP->report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}

	}

	if (f_v) {
		cout << "projective_space_with_action::cheat_sheet done" << endl;
	}


}


void projective_space_with_action::do_spread_classify(int k,
		poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify" << endl;
	}
	spread_classify *SC;

	SC = NEW_OBJECT(spread_classify);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify before SC->init" << endl;
	}

	SC->init(
			this,
			k,
			TRUE /* f_recoordinatize */,
			verbose_level - 1);
	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify after SC->init" << endl;
	}

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify before SC->init2" << endl;
	}
	SC->init2(Control, verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify after SC->init2" << endl;
	}


	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify before SC->compute" << endl;
	}

	SC->compute(verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify after SC->compute" << endl;
	}


	FREE_OBJECT(SC);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify done" << endl;
	}
}

void projective_space_with_action::setup_surface_with_action(
		surface_with_action *&Surf_A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::setup_surface_with_action" << endl;
		cout << "projective_space_with_action::setup_surface_with_action verbose_level=" << verbose_level << endl;
	}


	surface_domain *Surf;


	if (f_v) {
		cout << "projective_space_with_action::setup_surface_with_action before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_with_action::setup_surface_with_action after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "projective_space_with_action::setup_surface_with_action before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, this, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_with_action::setup_surface_with_action after Surf_A->init" << endl;
	}

}


void projective_space_with_action::report_decomposition_by_group(
	strong_generators *SG, std::ostream &ost, std::string &fname_base,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_group" << endl;
	}


	top_level_geometry_global Geo;


	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_group "
				"before Geo.report_decomposition_by_group" << endl;
	}
	Geo.report_decomposition_by_group(
			this,
			SG, ost, fname_base,
			verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_group "
				"after Geo.report_decomposition_by_group" << endl;
	}



	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_group done" << endl;
	}
}





// #############################################################################
// globals:
// #############################################################################


#if 0

void OiPA_encode(void *extra_data,
		long int *&encoding, int &encoding_sz, void *global_data)
{
	//cout << "OiPA_encode" << endl;
	object_in_projective_space_with_action *OiPA;
	object_with_canonical_form *OwCF;

	OiPA = (object_in_projective_space_with_action *) extra_data;
	OwCF = OiPA->OwCF;
	//OiP->print(cout);
	OwCF->encode_object(encoding, encoding_sz, 1 /* verbose_level*/);
	//cout << "OiPA_encode done" << endl;

}

void OiPA_group_order(void *extra_data,
		longinteger_object &go, void *global_data)
{
	//cout << "OiPA_group_order" << endl;
	object_in_projective_space_with_action *OiPA;
	//object_in_projective_space *OiP;

	OiPA = (object_in_projective_space_with_action *) extra_data;
	//OiP = OiPA->OiP;
	go.create(OiPA->ago, __FILE__, __LINE__);
	//OiPA->Aut_gens->group_order(go);
	//cout << "OiPA_group_order done" << endl;

}
#endif



#if 0
void compute_ago_distribution(
	classify_bitvectors *CB, tally *&C_ago, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_ago_distribution" << endl;
	}
	long int *Ago;
	int i;

	Ago = NEW_lint(CB->nb_types);
	for (i = 0; i < CB->nb_types; i++) {
		object_in_projective_space_with_action *OiPA;

		OiPA = (object_in_projective_space_with_action *)
				CB->Type_extra_data[i];
		Ago[i] = OiPA->ago; //OiPA->Aut_gens->group_order_as_lint();
	}
	C_ago = NEW_OBJECT(tally);
	C_ago->init_lint(Ago, CB->nb_types, FALSE, 0);
	FREE_lint(Ago);
	if (f_v) {
		cout << "compute_ago_distribution done" << endl;
	}
}

void compute_ago_distribution_permuted(
	classify_bitvectors *CB, tally *&C_ago, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_ago_distribution_permuted" << endl;
	}
	long int *Ago;
	int i;

	Ago = NEW_lint(CB->nb_types);
	for (i = 0; i < CB->nb_types; i++) {
		object_in_projective_space_with_action *OiPA;

		OiPA = (object_in_projective_space_with_action *)
				CB->Type_extra_data[CB->perm[i]];
		Ago[i] = OiPA->ago; //OiPA->Aut_gens->group_order_as_lint();
	}
	C_ago = NEW_OBJECT(tally);
	C_ago->init_lint(Ago, CB->nb_types, FALSE, 0);
	FREE_lint(Ago);
	if (f_v) {
		cout << "compute_ago_distribution_permuted done" << endl;
	}
}

void compute_and_print_ago_distribution(ostream &ost,
	classify_bitvectors *CB, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_and_print_ago_distribution" << endl;
	}
	tally *C_ago;
	compute_ago_distribution(CB, C_ago, verbose_level);
	ost << "ago distribution: " << endl;
	ost << "$$" << endl;
	C_ago->print_naked_tex(ost, TRUE /* f_backwards */);
	ost << endl;
	ost << "$$" << endl;
	FREE_OBJECT(C_ago);
}

void compute_and_print_ago_distribution_with_classes(ostream &ost,
	classify_bitvectors *CB, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	latex_interface L;

	if (f_v) {
		cout << "compute_and_print_ago_distribution_with_classes" << endl;
	}
	tally *C_ago;
	compute_ago_distribution_permuted(CB, C_ago, verbose_level);
	ost << "Ago distribution: " << endl;
	ost << "$$" << endl;
	C_ago->print_naked_tex(ost, TRUE /* f_backwards */);
	ost << endl;
	ost << "$$" << endl;
	set_of_sets *SoS;
	int *types;
	int nb_types;

	SoS = C_ago->get_set_partition_and_types(types,
			nb_types, verbose_level);


	// go backwards to show large group orders first:
	for (i = SoS->nb_sets - 1; i >= 0; i--) {
		ost << "Group order $" << types[i]
			<< "$ appears for the following $" << SoS->Set_size[i]
			<< "$ classes: $" << endl;
		L.lint_set_print_tex(ost, SoS->Sets[i], SoS->Set_size[i]);
		ost << "$\\\\" << endl;
		//int_vec_print_as_matrix(ost, SoS->Sets[i],
		//SoS->Set_size[i], 10 /* width */, TRUE /* f_tex */);
		//ost << "$$" << endl;

	}

	FREE_int(types);
	FREE_OBJECT(SoS);
	FREE_OBJECT(C_ago);
}
#endif


int table_of_sets_compare_func(void *data, int i,
		void *search_object,
		void *extra_data)
{
	long int *Data = (long int *) data;
	long int *p = (long int *) extra_data;
	long int len = p[0];
	int ret;

	ret = lint_vec_compare(Data + i * len, (long int *) search_object, len);
	return ret;
}



}}
