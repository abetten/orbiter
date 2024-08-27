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
namespace layer5_applications {
namespace projective_geometry {



projective_space_with_action::projective_space_with_action()
{
	Descr = NULL;

	n = 0;
	d = 0;
	q = 0;
	F = NULL;
	f_semilinear = false;
	f_init_incidence_structure = false;
	P = NULL;
	PA2 = NULL;
	Surf_A = NULL;
	Dom = NULL;
	QCDA = NULL;
	A = NULL;
	A_on_lines = NULL;

	f_has_action_on_planes = false;
	A_on_planes = NULL;

	Elt1 = NULL;
}





projective_space_with_action::~projective_space_with_action()
{
	if (P) {
		FREE_OBJECT(P);
	}
	if (PA2) {
		FREE_OBJECT(PA2);
	}
	if (Surf_A) {
		FREE_OBJECT(Surf_A);
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
}

void projective_space_with_action::init_from_description(
		projective_space_with_action_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::init_from_description, "
				"verbose_level=" << verbose_level << endl;
	}

	projective_space_with_action::Descr = Descr;

	field_theory::finite_field *F;

	if (Descr->f_field_label) {

		F = Get_finite_field(
				Descr->field_label);
	}
	else if (Descr->f_field_pointer) {
		F = Descr->F;
	}
	else if (Descr->f_q) {

		if (f_v) {
			cout << "projective_space_with_action::init_from_description "
					"creating the finite field of order " << q << endl;
		}
		F = NEW_OBJECT(field_theory::finite_field);
		F->finite_field_init_small_order(Descr->q,
				false /* f_without_tables */,
				true /* f_compute_related_fields */,
				verbose_level - 1);
		if (f_v) {
			cout << "projective_space_with_action::init_from_description "
					"the finite field of order " << Descr->q
					<< " has been created" << endl;
		}

	}
	else {
		cout << "projective_space_with_action::init_from_description, "
				"I need a field by label or by pointer or by its order" << endl;
		exit(1);
	}


	Descr->F = F;
	Descr->q = F->q;

	if (!Descr->f_n) {
		cout << "projective_space_with_action::init_from_description, "
				"I need a dimension n" << endl;
		exit(1);
	}



	int f_semilinear = true;

	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "projective_space_with_action::init_from_description, "
				"q=" << Descr->q << endl;
	}

	if (NT.is_prime(Descr->q)) {
		f_semilinear = false;
	}
	else {
		f_semilinear = true;
	}

	if (Descr->f_use_projectivity_subgroup) {
		f_semilinear = false;
	}

	if (f_v) {
		cout << "projective_space_with_action::init_from_description, "
				"f_semilinear=" << f_semilinear << endl;
	}


	if (f_v) {
		cout << "projective_space_with_action::init_from_description, "
				"before init" << endl;
	}
	init(
			F,
			Descr->n, f_semilinear,
			true /* f_init_incidence_structure */,
			verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::init_from_description, "
				"after init" << endl;
	}

	if (f_v) {
		cout << "projective_space_with_action::init_from_description done" << endl;
	}
}


void projective_space_with_action::init(
		field_theory::finite_field *F,
		int n, int f_semilinear,
	int f_init_incidence_structure,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::init, "
				"verbose_level=" << verbose_level << endl;
		cout << "projective_space_with_action::init, "
				"f_semilinear=" << f_semilinear << endl;
	}
	projective_space_with_action::f_init_incidence_structure
		= f_init_incidence_structure;
	projective_space_with_action::n = n;
	projective_space_with_action::F = F;
	projective_space_with_action::f_semilinear = f_semilinear;
	d = n + 1;
	q = F->q;
	
	P = NEW_OBJECT(geometry::projective_space);

	if (f_v) {
		cout << "projective_space_with_action::init "
				"before P->projective_space_init" << endl;
	}

	P->projective_space_init(n, F,
		f_init_incidence_structure, 
		verbose_level);
	

	if (f_v) {
		cout << "projective_space_with_action::init "
				"after P->projective_space_init" << endl;
	}

	if (f_v) {
		cout << "projective_space_with_action::init "
				"before init_group" << endl;
	}

	init_group(f_semilinear, verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::init "
				"after init_group" << endl;
	}


	if (n == 2) {
		if (f_v) {
			cout << "projective_space_with_action::init "
					"n == 2" << endl;
		}
		Dom = NEW_OBJECT(algebraic_geometry::quartic_curve_domain);

		if (f_v) {
			cout << "projective_space_with_action::init "
					"before Dom->init" << endl;
		}
		Dom->init(F, verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::init "
					"after Dom->init" << endl;
		}
		QCDA = NEW_OBJECT(applications_in_algebraic_geometry::
				quartic_curves::quartic_curve_domain_with_action);
		if (f_v) {
			cout << "projective_space_with_action::init "
					"before QCDA->init" << endl;
		}
		QCDA->init(Dom, this, verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::init "
					"after QCDA->init" << endl;
		}
	}

	if (n >= 3) {
		if (f_v) {
			cout << "projective_space_with_action::init "
					"n >= 3, so we initialize a plane" << endl;
		}
		PA2 = NEW_OBJECT(projective_space_with_action);
		if (f_v) {
			cout << "projective_space_with_action::init "
					"before PA2->init" << endl;
		}
		PA2->init(F, 2, f_semilinear,
			f_init_incidence_structure,
			verbose_level - 2);
		if (f_v) {
			cout << "projective_space_with_action::init "
					"after PA2->init" << endl;
		}
	}
	if (n == 3) {
		if (f_v) {
			cout << "projective_space_with_action::init "
					"n == 3, so we initialize a Surf_A object" << endl;
		}

		algebraic_geometry::surface_domain *Surf;

		Surf = NEW_OBJECT(algebraic_geometry::surface_domain);
		if (f_v) {
			cout << "projective_space_with_action::init "
					"before Surf->init_surface_domain" << endl;
		}
		Surf->init_surface_domain(F, verbose_level - 1);
		if (f_v) {
			cout << "projective_space_with_action::init "
					"after Surf->init_surface_domain" << endl;
		}

		Surf_A = NEW_OBJECT(applications_in_algebraic_geometry::
				cubic_surfaces_in_general::surface_with_action);

		if (f_v) {
			cout << "projective_space_with_action::init before Surf_A->init" << endl;
		}
		Surf_A->init(Surf, this, true /* f_recoordinatize */, verbose_level - 1);
		if (f_v) {
			cout << "projective_space_with_action::init after Surf_A->init" << endl;
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

	data_structures_groups::vector_ge *nice_gens;

	A = NEW_OBJECT(actions::action);

	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"before A->Known_groups->init_linear_group" << endl;
	}
	A->Known_groups->init_linear_group(
		F, d, 
		true /*f_projective*/,
		false /* f_general*/,
		false /* f_affine */,
		f_semilinear,
		false /* f_special */,
		nice_gens,
		verbose_level - 2);
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"after A->Known_groups->init_linear_group" << endl;
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
	A_on_lines = A->Induced_action->induced_action_on_grassmannian(
			2, verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating action on lines done" << endl;
	}

	if (d >= 4) {
		if (f_v) {
			cout << "projective_space_with_action::init_group "
					"creating action on planes" << endl;
		}
		f_has_action_on_planes = true;
		A_on_planes = A->Induced_action->induced_action_on_grassmannian(
				3, verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::init_group "
					"creating action on lines planes" << endl;
		}
	}
	else {
		f_has_action_on_planes = false;
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
		cout << "projective_space_with_action::canonical_form "
				"before OC->do_the_work" << endl;
	}
	OC->do_the_work(
			Canonical_form_PG_Descr,
			true,
			this,
			IS,
			verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::canonical_form "
				"after OC->do_the_work" << endl;
	}

	FREE_OBJECT(OC);
#endif

	if (f_v) {
		cout << "projective_space_with_action::canonical_form done" << endl;
	}
}
#endif

#if 0
void projective_space_with_action::canonical_labeling(
		canonical_form_classification::object_with_canonical_form *OwCF,
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
				"before OwCF->encoding_size" << endl;
	}

	int nb_rows, nb_cols;

	OwCF->encoding_size(
			nb_rows, nb_cols,
			0 /* verbose_level */);
	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling "
				"after OwCF->encoding_size" << endl;
	}

	l1_interfaces::nauty_output *NO;

#if 0
	NO = NEW_OBJECT(l1_interfaces::nauty_output);
	NO->nauty_output_allocate(nb_rows + nb_cols,
			0,
			nb_rows + nb_cols,
			0 /* verbose_level */);
#endif

	//int f_save_nauty_input_graphs = false;

	//OwCF->canonical_labeling(f_save_nauty_input_graphs, NO, verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling "
				"before OwCF->run_nauty_basic" << endl;
	}
	OwCF->run_nauty_basic(
			NO,
			verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling "
				"after OwCF->run_nauty_basic" << endl;
	}


	Int_vec_copy(NO->canonical_labeling, canonical_labeling, NO->N);



#if 0
	int i;

	for (i = 0; i < NO->N; i++) {
		canonical_labeling[i] = NO->canonical_labeling[i];
	}
#endif

	FREE_OBJECT(NO);


	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling done" << endl;
	}
}
#endif

#if 0
void projective_space_with_action::report_fixed_points_lines_and_planes(
	int *Elt, std::ostream &ost,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_fixed_points_lines_and_planes" << endl;
	}

	if (P->Subspaces->n < 3) {
		cout << "projective_space_with_action::report_fixed_points_lines_and_planes "
				"P->Subspaces->n < 3" << endl;
		exit(1);
	}
	geometry::projective_space *P3;
	int i, j, cnt;
	int v[4];

	P3 = P;
	
	ost << "Fixed Objects:\\\\" << endl;



	ost << "The element" << endl;
	ost << "$$" << endl;
	A->Group_element->element_print_latex(Elt, ost);
	ost << "$$" << endl;
	ost << "has the following fixed objects:\\\\" << endl;


	ost << "Fixed points:\\\\" << endl;

	cnt = 0;
	for (i = 0; i < P3->Subspaces->N_points; i++) {
		j = A->Group_element->element_image_of(
				i, Elt, 0 /* verbose_level */);
		if (j == i) {
			cnt++;
		}
	}

	ost << "There are " << cnt << " fixed points, they are: \\\\" << endl;
	for (i = 0; i < P3->Subspaces->N_points; i++) {
		j = A->Group_element->element_image_of(
				i, Elt, 0 /* verbose_level */);
		F->Projective_space_basic->PG_element_unrank_modified(v, 1, 4, i);
		if (j == i) {
			ost << i << " : ";
			Int_vec_print(ost, v, 4);
			ost << "\\\\" << endl;
			cnt++;
		}
	}

	ost << "Fixed Lines:\\\\" << endl;

	{
		actions::action *A2;

		A2 = A->Induced_action->induced_action_on_grassmannian(
				2, 0 /* verbose_level*/);
	
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->Group_element->element_image_of(
					i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
				}
			}

		ost << "There are " << cnt << " fixed lines, they are: \\\\" << endl;
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->Group_element->element_image_of(
					i, Elt, 0 /* verbose_level */);
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
		actions::action *A2;

		A2 = A->Induced_action->induced_action_on_grassmannian(
				3, 0 /* verbose_level*/);
	
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->Group_element->element_image_of(
					i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
			}
		}

		ost << "There are " << cnt << " fixed planes, they are: \\\\" << endl;
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->Group_element->element_image_of(
					i, Elt, 0 /* verbose_level */);
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
#endif

void projective_space_with_action::report_orbits_on_points_lines_and_planes(
	int *Elt, std::ostream &ost,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_orbits_on_points_lines_and_planes" << endl;
	}

	if (P->Subspaces->n < 3) {
		cout << "projective_space_with_action::report_orbits_on_points_lines_and_planes "
				"P->Subspaces->n < 3" << endl;
		exit(1);
	}
	//projective_space *P3;
	int order;

	ring_theory::longinteger_object full_group_order;
	order = A->Group_element->element_order(Elt);

	full_group_order.create(order);

	//P3 = P;

	ost << "Fixed Objects:\\\\" << endl;



	ost << "The group generated by the element" << endl;
	ost << "$$" << endl;
	A->Group_element->element_print_latex(Elt, ost);
	ost << "$$" << endl;
	ost << "has the following orbits:\\\\" << endl;

	ost << "Orbits on points:\\\\" << endl;


	groups::schreier *Sch;
	actions::action_global AcGl;

	Sch = NEW_OBJECT(groups::schreier);
	AcGl.all_point_orbits_from_single_generator(A,
			*Sch,
			Elt,
			0 /*verbose_level*/);
	Sch->print_orbit_lengths_tex(ost);


	FREE_OBJECT(Sch);

	ost << "Orbits on lines:\\\\" << endl;

	{
		actions::action *A2;
		groups::schreier *Sch;

		A2 = A->Induced_action->induced_action_on_grassmannian(
				2, 0 /* verbose_level*/);

		Sch = NEW_OBJECT(groups::schreier);
		AcGl.all_point_orbits_from_single_generator(A2,
				*Sch,
				Elt,
				0 /*verbose_level*/);
		Sch->print_orbit_lengths_tex(ost);


		FREE_OBJECT(Sch);
		FREE_OBJECT(A2);
	}

	ost << "Orbits on planes:\\\\" << endl;

	{
		actions::action *A2;
		groups::schreier *Sch;


		A2 = A->Induced_action->induced_action_on_grassmannian(
				3, 0 /* verbose_level*/);

		Sch = NEW_OBJECT(groups::schreier);
		AcGl.all_point_orbits_from_single_generator(A2,
				*Sch,
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




#if 0
void projective_space_with_action::compute_group_of_set(long int *set, int set_sz,
		groups::strong_generators *&Sg,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::compute_group_of_set" << endl;
	}

#if 0
	int i;
	long int a;

	projective_space_object_classifier_description *Descr;
	classification_of_objects *Classifier;

	Descr = NEW_OBJECT(projective_space_object_classifier_description);
	Classifier = NEW_OBJECT(classification_of_objects);

	Descr->f_input = true;
	Descr->Data = NEW_OBJECT(data_input_stream_description);
	Descr->Data->input_type[Descr->Data->nb_inputs] = INPUT_TYPE_SET_OF_POINTS;
	Descr->Data->input_string[Descr->Data->nb_inputs].assign("");
	for (i = 0; i < set_sz; i++) {
		a = set[i];
		str += std::to_string(a);
		Descr->Data->input_string[Descr->Data->nb_inputs] += str;
		if (i < set_sz - 1) {
			Descr->Data->input_string[Descr->Data->nb_inputs] += ",";
		}
	}
	Descr->Data->input_string2[Descr->Data->nb_inputs] = "";
	Descr->Data->nb_inputs++;

	if (f_v) {
		cout << "projective_space_with_action::compute_group_of_set "
				"before Classifier->do_the_work" << endl;
	}

	Classifier->do_the_work(
			Descr,
			true,
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
#endif



void projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG(
		int decomposition_by_element_power,
		std::string &decomposition_by_element_data,
		std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG "
				"verbose_level=" << verbose_level << endl;
	}


	field_theory::finite_field *F;

	F = P->Subspaces->F;


	{
		string title, author, extra_praeamble;

		title = "Decomposition of PG($" + std::to_string(n) + "," + std::to_string(F->q) + "$)";



		string fname_tex;

		fname_tex = fname_base + ".tex";

		{
			ofstream ost(fname_tex);
			l1_interfaces::latex_interface L;

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
				cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG "
						"f_decomposition_by_element" << endl;
			}

			int *Elt;

			Elt = NEW_int(A->elt_size_in_int);


			A->Group_element->make_element_from_string(Elt,
					decomposition_by_element_data, verbose_level);


			A->Group_element->element_power_int_in_place(Elt,
					decomposition_by_element_power, verbose_level);

			apps_geometry::top_level_geometry_global Geo;


			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG "
						"before Geo.report_decomposition_by_single_automorphism" << endl;
			}

			Geo.report_decomposition_by_single_automorphism(
					this,
					Elt, ost, fname_base,
					verbose_level);

			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG "
						"after Geo.report_decomposition_by_single_automorphism" << endl;
			}
			FREE_int(Elt);


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname_tex << " of size "
					<< Fio.file_size(fname_tex) << endl;
		}
	}

	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG done" << endl;
	}

}

void projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup(
		std::string &label,
		group_constructions::linear_group_description * subgroup_Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup" << endl;
	}


	group_constructions::linear_group *H_LG;

	H_LG = NEW_OBJECT(group_constructions::linear_group);

	subgroup_Descr->F = P->Subspaces->F;

	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup "
				"H_LG->init, "
				"creating the group" << endl;
	}

	H_LG->linear_group_init(subgroup_Descr, verbose_level - 2);

	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup "
				"after H_LG->linear_group_init" << endl;
	}


	string fname;

	fname = H_LG->label + "_decomp.tex";


	{
		string title, author, extra_praeamble;


		title = "Decomposition of PG($" + std::to_string(n) + "," + std::to_string(F->q) + "$)";



		{
			ofstream ost(fname);
			l1_interfaces::latex_interface L;

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
				cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG "
						"f_decomposition_by_element" << endl;
			}


			report_decomposition_by_group(
					H_LG->Strong_gens, ost, H_LG->label,
					verbose_level);



			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

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
	std::ostream &ost,
	graphics::layered_graph_draw_options *O,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report" << endl;
	}


	cout << "projective_space_with_action::report not yet implemented" << endl;
	exit(1);


	if (f_v) {
		cout << "projective_space_with_action::report done" << endl;
	}
}



void projective_space_with_action::canonical_form_of_code(
		std::string &label,
		int *genma, int m, int n,
		canonical_form_classification::classification_of_objects_description
			*Canonical_form_codes_Descr,
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
		Int_matrix_print(genma, m, n);
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
			Int_vec_print(cout, v, m);
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
		Lint_vec_print(cout, set, n);
		cout << endl;
	}

	data_structures::string_tools ST;
	string points_as_string;

	ST.create_comma_separated_list(points_as_string, set, n);
	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code "
				"points_as_string=" << points_as_string << endl;
	}

	canonical_form_classification::data_input_stream_description ISD;
	canonical_form_classification::data_input_stream_description_element E;

	E.init_set_of_points(points_as_string);
	ISD.Input.push_back(E);
	ISD.nb_inputs++;

	//data_structures::data_input_stream IS;

	//IS.init(&ISD, verbose_level);



	apps_combinatorics::combinatorial_object_activity_description COAD;

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

	COAD.f_canonical_form_PG = true;
	COAD.f_canonical_form_PG_has_PA = true;
	COAD.Canonical_form_PG_PA = this;
	COAD.Canonical_form_PG_Descr = Canonical_form_codes_Descr;

	COAD.f_report = true;
	COAD.Classification_of_objects_report_options =
			NEW_OBJECT(canonical_form_classification::classification_of_objects_report_options);

#if 0
	COAD.Classification_of_objects_report_options->f_prefix = true;
	COAD.Classification_of_objects_report_options->prefix.assign(COAD.Canonical_form_PG_Descr->label);
#endif

	COAD.Classification_of_objects_report_options->f_export_flag_orbits = true;
	COAD.Classification_of_objects_report_options->f_show_incidence_matrices = true;
	COAD.Classification_of_objects_report_options->f_show_TDO = true;
	COAD.Classification_of_objects_report_options->f_show_TDA = true;
	COAD.Classification_of_objects_report_options->f_export_group_GAP = true;
	COAD.Classification_of_objects_report_options->f_export_group_orbiter = true;


	apps_combinatorics::combinatorial_object_stream *Combo;

	Combo = NEW_OBJECT(apps_combinatorics::combinatorial_object_stream);
	Combo->init(
			&ISD,
			verbose_level);


	apps_combinatorics::combinatorial_object_activity COA;

	COA.init_combo(&COAD,
			Combo,
			verbose_level);


	orbiter_kernel_system::activity_output *AO;

	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code "
				"before COA.perform_activity" << endl;
	}
	COA.perform_activity(AO, verbose_level);

    if (AO) {
        FREE_OBJECT(AO);
    }


	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code "
				"after COA.perform_activity" << endl;
	}


	FREE_int(v);
	FREE_lint(set);
	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code done" << endl;
	}

}



void projective_space_with_action::cheat_sheet(
		graphics::layered_graph_draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::cheat_sheet" << endl;
	}



	{

		string fname, title, author, extra_praeamble;

		fname = "PG_" + std::to_string(n) + "_" + std::to_string(F->q) + ".tex";
		title = "Cheat Sheet ${\\rm PG}(" + std::to_string(n) + "," + std::to_string(F->q) + ")$";





		{
			ofstream ost(fname);
			l1_interfaces::latex_interface L;

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
				cout << "projective_space_with_action::do_cheat_sheet_PG "
						"before A->report" << endl;
			}

			A->report(ost, A->f_has_sims, A->Sims,
					A->f_has_strong_generators, A->Strong_gens,
					O,
					verbose_level);

			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_PG "
						"after A->report" << endl;
			}

			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_PG "
						"before P->Reporting->report" << endl;
			}



			P->Reporting->report(ost, O, verbose_level);

			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_PG "
						"after P->Reporting->report" << endl;
			}

			if (n == 3) {

				// ToDo PA now has a Surf_A

#if 0
				applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_with_action *Surf_A;

				setup_surface_with_action(
						Surf_A,
						verbose_level);
#endif

				Surf_A->Surf->Schlaefli->print_Steiner_and_Eckardt(ost);

				FREE_OBJECT(Surf_A);

			}



			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}

	}

	if (f_v) {
		cout << "projective_space_with_action::cheat_sheet done" << endl;
	}


}


void projective_space_with_action::do_spread_classify(
		int k,
		poset_classification::poset_classification_control
			*Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify" << endl;
	}

	int n;

	n = A->matrix_group_dimension();

	geometry::spread_domain *SD;

	SD = NEW_OBJECT(geometry::spread_domain);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify "
				"before SD->init_spread_domain" << endl;
	}

	SD->init_spread_domain(
			F,
			n, k,
			verbose_level - 1);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify "
				"after SD->init_spread_domain" << endl;
	}

	spreads::spread_classify_description *Descr;

	Descr = NEW_OBJECT(spreads::spread_classify_description);


	Descr->f_recoordinatize = true;


	spreads::spread_classify *SC;

	SC = NEW_OBJECT(spreads::spread_classify);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify "
				"before SC->init" << endl;
	}

	SC->init(
			Descr,
			SD,
			this,
			verbose_level - 1);
	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify "
				"after SC->init" << endl;
	}

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify "
				"before SC->classify_partial_spreads" << endl;
	}

	SC->classify_partial_spreads(verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify "
				"after SC->classify_partial_spreads" << endl;
	}


	FREE_OBJECT(SC);
	FREE_OBJECT(SD);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify done" << endl;
	}
}

void projective_space_with_action::report_decomposition_by_group(
		groups::strong_generators *SG,
		std::ostream &ost, std::string &fname_base,
	int verbose_level)
// used by projective_space_with_action::do_cheat_sheet_for_decomposition_by_subgroup
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_group" << endl;
	}


	apps_geometry::top_level_geometry_global Geo;


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




void projective_space_with_action::report_fixed_objects(
		std::string &Elt_text,
		std::string &fname_latex, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_fixed_objects" << endl;
	}

	int *data;
	int sz;
	int *Elt;

	Get_int_vector_from_label(Elt_text, data, sz, 0 /* verbose_level */);
	if (sz != A->make_element_size) {
		cout << "projective_space_with_action::report_fixed_objects "
				"the size of the input does not match" << endl;
		cout << "expected: " << A->make_element_size << endl;
		cout << "seeing: " << sz << endl;
		exit(1);
	}
	Elt = NEW_int(A->elt_size_in_int);
	A->Group_element->make_element(Elt, data, verbose_level);


	{
		ofstream fp(fname_latex);
		string title, author, extra_praeamble;

		l1_interfaces::latex_interface L;

		title = "Fixed Objects";
		author = "";

		L.head(fp,
			false /* f_book */, true /* f_title */,
			title, author /* const char *author */,
			false /* f_toc */, false /* f_landscape */, true /* f_12pt */,
			true /* f_enlarged_page */, true /* f_pagenumbers */,
			extra_praeamble /* extra_praeamble */);
		//latex_head_easy(fp);



		if (f_v) {
			cout << "projective_space_with_action::report_fixed_objects "
					"before A->Group_element->report_fixed_objects_in_PG" << endl;
		}
		A->Group_element->report_fixed_objects_in_PG(fp,
				P,
				Elt,
				verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::report_fixed_objects "
					"after A->Group_element->report_fixed_objects_in_PG" << endl;
		}


		L.foot(fp);
	}
	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname_latex << " of size "
			<< Fio.file_size(fname_latex) << endl;

	FREE_int(Elt);

	if (f_v) {
		cout << "projective_space_with_action::report_fixed_objects done" << endl;
	}
}





}}}

