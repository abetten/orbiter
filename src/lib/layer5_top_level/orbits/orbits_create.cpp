/*
 * orbits_create.cpp
 *
 *  Created on: Nov 5, 2022
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orbits {


orbits_create::orbits_create()
{
	Record_birth();
	Descr = NULL;

	Group = NULL;
	Group_action = NULL;

	f_has_Orb = false;
	Orb = NULL;

	f_has_On_subsets = false;
	On_subsets = NULL;

	f_has_On_Subspaces = false;
	On_Subspaces = NULL;

	f_has_On_tensors = false;
	On_tensors = NULL;

	f_has_Cascade = false;
	Cascade = NULL;

	f_has_On_polynomials = false;
	On_polynomials = NULL;

	f_has_Of_One_polynomial = false;
	Of_One_polynomial = NULL;

	f_has_on_cubic_curves = false;
	Arc_generator_description = NULL;
	CC = NULL;
	CCA = NULL;
	CCC = NULL;

	f_has_cubic_surfaces = false;
	SCW = NULL;

	f_has_arcs = false;
	Arc_generator_description_for_arcs = NULL;
	Arc_generator = NULL;

	f_has_semifields = false;
	Semifields = NULL;

	f_has_boolean_functions = false;
	BF = NULL;
	BFC = NULL;

	f_has_classification_by_canonical_form = false;
	Canonical_form_classifier = NULL;

	//std::string prefix;
	//std::string label_txt;
	//std::string label_tex;

}


orbits_create::~orbits_create()
{
	Record_death();
}

void orbits_create::init(
		orbits_create_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "orbits_create::init" << endl;
	}
	orbits_create::Descr = Descr;

	if (Descr->f_group) {

		Group = Get_any_group(Descr->group_label);
		prefix.assign(Group->label);
	}

	if (Descr->f_group_action) {

		Group_action = Get_any_group(Descr->group_action_label);
		prefix += "_in_action_" + Group_action->label;
	}



	if (Descr->f_on_points) {

		if (f_v) {
			cout << "orbits_create::init f_on_points" << endl;
		}
		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
		}

		orbits_global Orbits;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"print_interval = " << Descr->print_interval << endl;
		}
		cout << "group_theoretic_activity::perform_activity "
				"before Orbits.orbits_on_points" << endl;

		Orbits.orbits_on_points(Group, Orb, Descr->print_interval, verbose_level);



		f_has_Orb = true;


		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after Orbits.orbits_on_points" << endl;
		}


	}

	if (Descr->f_on_points_with_generators) {

		if (f_v) {
			cout << "orbits_create::init f_on_points_with_generators" << endl;
		}
		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
		}


		apps_algebra::vector_ge_builder *Gens;

		Gens = Get_object_of_type_vector_ge(Descr->on_points_with_generators_gens_label);

		orbits_global Orbits;


		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before Orbits.orbits_on_points_from_generators" << endl;
		}

		Orbits.orbits_on_points_from_generators(
				Group, Gens->V, Descr->print_interval, Orb, verbose_level);


		f_has_Orb = true;


		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after Orbits.orbits_on_points_from_generators" << endl;
		}
	}

	if (Descr->f_on_subsets) {

		if (f_v) {
			cout << "orbits_create::init f_on_subsets" << endl;
		}
		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
		}
		if (!Descr->f_group_action) {
			cout << "orbits_create::init please specify the group using -group_action <label>" << endl;
			exit(1);
		}

		poset_classification::poset_classification_control *Control =
				Get_poset_classification_control(
						Descr->on_subsets_poset_classification_control_label);


		orbits::orbits_global Orbits;

		if (f_v) {
			cout << "Strong generators Group->Subgroup_gens:" << endl;
			Group->Subgroup_gens->print_generators(cout, 0 /* verbose_level */);
			cout << "Strong generators Group->Subgroup_gens in tex:" << endl;
			Group->Subgroup_gens->print_generators_tex(cout);
		}

		if (f_v) {
			cout << "orbits_create::init before Orbits.orbits_on_subsets" << endl;
		}

		Orbits.orbits_on_subsets(
				Group, Group_action, Group->Subgroup_gens,
				Control, On_subsets,
				Descr->on_subsets_size, verbose_level);

		f_has_On_subsets = true;

		if (f_v) {
			cout << "orbits_create::init after Group->orbits_on_subsets" << endl;
		}

	}
	if (Descr->f_of_one_subset) {

		if (f_v) {
			cout << "orbits_create::init f_of_one_subset" << endl;
		}
		if (!Descr->f_group) {
			cout << "orbits_create::init please "
					"specify the group using -group <label>" << endl;
			exit(1);
		}

		long int *set;
		int sz;
		std::string label_set;

		Get_lint_vector_from_label(Descr->of_one_subset_label, set, sz, verbose_level - 2);

		label_set = Descr->of_one_subset_label;

		long int *Table;
		int size;
		orbits::orbits_global Orbits;


		if (f_v) {
			cout << "orbits_create::init before Orbits.orbits_of_one_subset" << endl;
		}
		Orbits.orbits_of_one_subset(
				Group,
				set, sz,
				label_set,
				Group->A, Group->A,
				Table, size,
				verbose_level);
		if (f_v) {
			cout << "orbits_create::init after Orbits.orbits_of_one_subset" << endl;
		}


		if (f_v) {
			cout << "orbits_create::init after f_of_one_subset" << endl;
		}
	}

	if (Descr->f_on_subspaces) {

		if (f_v) {
			cout << "orbits_create::init f_on_subspaces" << endl;
		}
		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
		}

		poset_classification::poset_classification_control *Control =
				Get_poset_classification_control(
						Descr->on_subspaces_poset_classification_control_label);


		orbits::orbits_global Orbits_global;

		if (f_v) {
			cout << "orbits_create::init "
					"before Orbits_global.do_orbits_on_subspaces" << endl;
		}

		Orbits_global.do_orbits_on_subspaces(
				Group,
				Control,
				On_Subspaces,
				Descr->on_subspaces_dimension,
				verbose_level);


		if (f_v) {
			cout << "orbits_create::init "
					"after Orbits_global.do_orbits_on_subspaces" << endl;
		}

		f_has_On_Subspaces = true;
		prefix = On_Subspaces->orbits_on_subspaces_PC->get_problem_label();
		label_txt = prefix;
		if (f_v) {
			cout << "orbits_create::init prefix = " << prefix << endl;
		}


		if (f_v) {
			cout << "orbits_create::init after Group->do_orbits_on_subspaces" << endl;
		}

	}
	if (Descr->f_on_tensors) {

		if (f_v) {
			cout << "orbits_create::init f_on_tensors" << endl;
		}
		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
		}

		orbits::orbits_global Orbits_global;


		if (f_v) {
			cout << "orbits_create::init "
					"before Orbits_global.do_tensor_classify" << endl;
		}

		Orbits_global.do_tensor_classify(
				Group,
				Descr->on_tensors_poset_classification_control_label,
				On_tensors,
				Descr->on_tensors_dimension,
				verbose_level);

		f_has_On_tensors = true;

		if (f_v) {
			cout << "orbits_create::init "
					"after Orbits_global.do_tensor_classify" << endl;
		}

	}
	if (Descr->f_on_partition) {

		if (f_v) {
			cout << "orbits_create::init f_on_partition" << endl;
		}
		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
		}



		Cascade = NEW_OBJECT(orbit_cascade);


		if (f_v) {
			cout << "orbits_create::init before Cascade->init" << endl;
		}

		Cascade->init(Group->A->degree,
				Descr->on_partition_k,
				Group,
				Descr->on_partition_poset_classification_control_label,
				verbose_level);

		f_has_Cascade = true;

		if (f_v) {
			cout << "orbits_create::init after Cascade->init" << endl;
		}

	}


	if (Descr->f_on_polynomials) {


		if (f_v) {
			cout << "orbits_create::init f_on_polynomials" << endl;
		}
		if (f_v) {
			cout << "orbits_create::init ring = " << Descr->on_polynomials_ring << endl;
		}

		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
		}

		if (!Group->f_linear_group) {
			cout << "orbits_create::init group must be linear" << endl;
			exit(1);
		}

		On_polynomials = NEW_OBJECT(orbits_on_polynomials);



		algebra::ring_theory::homogeneous_polynomial_domain *HPD;


		HPD = Get_ring(Descr->on_polynomials_ring);

		if (f_v) {
			cout << "orbits_create::init "
					"before On_polynomials->init" << endl;
		}
		On_polynomials->init(
				Group->LG,
				HPD,
				Descr->print_interval,
				verbose_level);

		if (f_v) {
			cout << "orbits_create::init "
					"after On_polynomials->init" << endl;
		}

		f_has_On_polynomials = true;




	}


	if (Descr->f_of_one_polynomial) {


		if (f_v) {
			cout << "orbits_create::init f_of_one_polynomial" << endl;
		}
		if (f_v) {
			cout << "orbits_create::init ring = " << Descr->of_one_polynomial_ring << endl;
		}

		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
		}

		if (!Group->f_linear_group) {
			cout << "orbits_create::init group must be linear" << endl;
			exit(1);
		}



		algebra::ring_theory::homogeneous_polynomial_domain *HPD;


		HPD = Get_ring(Descr->of_one_polynomial_ring);


		algebra::expression_parser::symbolic_object_builder *Symbol;

		Symbol = Get_symbol(Descr->of_one_polynomial_equation);


		Of_One_polynomial = NEW_OBJECT(orbits_on_polynomials);

		if (f_v) {
			cout << "orbits_create::init "
					"before Of_One_polynomial->orbit_of_one_polynomial" << endl;
		}
		Of_One_polynomial->orbit_of_one_polynomial(
				Group->LG,
				HPD,
				Symbol,
				verbose_level);
		if (f_v) {
			cout << "orbits_create::init "
					"after Of_One_polynomial->orbit_of_one_polynomial" << endl;
		}

		f_has_Of_One_polynomial = true;

	}


	if (Descr->f_on_cubic_curves) {


		if (f_v) {
			cout << "orbits_create::init f_on_cubic_curves" << endl;
		}
		if (f_v) {
			cout << "orbits_create::init control = " << Descr->on_cubic_curves_control << endl;
		}



		Arc_generator_description = Get_object_of_type_arc_generator_control(
				Descr->on_cubic_curves_control);


		projective_geometry::projective_space_with_action *PA;

		if (!Arc_generator_description->f_projective_space) {
			cout << "Please use option -projective_space in arc_generator" << endl;
			exit(1);
		}
		PA = Get_projective_space(Arc_generator_description->projective_space_label);

		CC = NEW_OBJECT(geometry::algebraic_geometry::cubic_curve);

		if (f_v) {
			cout << "orbits_create::init "
					"before CC->init" << endl;
		}
		CC->init(PA->F, verbose_level);
		if (f_v) {
			cout << "orbits_create::init "
					"after CC->init" << endl;
		}



		CCA = NEW_OBJECT(apps_geometry::cubic_curve_with_action);

		if (f_v) {
			cout << "orbits_create::init "
					"before CCA->init" << endl;
		}
		CCA->init(CC, PA->A, verbose_level);
		if (f_v) {
			cout << "orbits_create::init "
					"after CCA->init" << endl;
		}



		CCC = NEW_OBJECT(apps_geometry::classify_cubic_curves);


		if (f_v) {
			cout << "orbits_create::init "
					"before CCC->init" << endl;
		}
		CCC->init(
				PA,
				CCA,
				Arc_generator_description,
				verbose_level);
		if (f_v) {
			cout << "orbits_create::init "
					"after CCC->init" << endl;
		}

		if (f_v) {
			cout << "orbits_create::init "
					"before CCC->compute_starter" << endl;
		}
		CCC->compute_starter(verbose_level);
		if (f_v) {
			cout << "orbits_create::init "
					"after CCC->compute_starter" << endl;
		}

	#if 0
		if (f_v) {
			cout << "orbits_create::init "
					"before CCC->test_orbits" << endl;
		}
		CCC->test_orbits(verbose_level);
		if (f_v) {
			cout << "orbits_create::init "
					"after CCC->test_orbits" << endl;
		}
	#endif

		if (f_v) {
			cout << "orbits_create::init "
					"before CCC->do_classify" << endl;
		}
		CCC->do_classify(verbose_level);
		if (f_v) {
			cout << "orbits_create::init "
					"after CCC->do_classify" << endl;
		}


		f_has_on_cubic_curves = true;


	}

	if (Descr->f_on_cubic_surfaces) {


		if (f_v) {
			cout << "orbits_create::init f_on_cubic_surfaces" << endl;
		}
		if (f_v) {
			cout << "orbits_create::init classify = " << Descr->on_cubic_surfaces_classify << endl;
		}

		orbits::classify_cubic_surfaces_description *Classify_cubic_surfaces_description;

		Classify_cubic_surfaces_description = Get_classify_cubic_surfaces(Descr->on_cubic_surfaces_classify);



		if (!Classify_cubic_surfaces_description->f_use_double_sixes) {
			cout << "please use the double six algorithm" << endl;
			exit(1);
		}

		projective_geometry::projective_space_with_action *PA;

		if (Classify_cubic_surfaces_description->f_projective_space == false) {
			cout << "please specify a projective space" << endl;
			exit(1);
		}

		PA = Get_projective_space(Classify_cubic_surfaces_description->projective_space_label);



		if (Classify_cubic_surfaces_description->f_poset_classification_control == false) {
			cout << "please specify a poset classification control object" << endl;
			exit(1);
		}

		poset_classification::poset_classification_control *Control =
				Get_poset_classification_control(
						Classify_cubic_surfaces_description->poset_classification_control_object);

		if (f_v) {
			cout << "orbits_create::init "
					"before classify_surfaces, control=" << endl;
			Control->print();
		}


		SCW = NEW_OBJECT(applications_in_algebraic_geometry::cubic_surfaces_and_double_sixes::surface_classify_wedge);

		if (f_v) {
			cout << "orbits_create::init "
					"before SCW->init" << endl;
		}

		SCW->init(PA,
				Control,
				verbose_level - 1);

		if (f_v) {
			cout << "orbits_create::init "
					"after SCW->init" << endl;
		}


		if (f_v) {
			cout << "orbits_create::init "
					"before SCW->do_classify_double_sixes" << endl;
		}
		SCW->do_classify_double_sixes(verbose_level);
		if (f_v) {
			cout << "orbits_create::init "
					"after SCW->do_classify_double_sixes" << endl;
		}

		if (f_v) {
			cout << "orbits_create::init "
					"before SCW->do_classify_surfaces" << endl;
		}
		SCW->do_classify_surfaces(verbose_level);
		if (f_v) {
			cout << "orbits_create::init "
					"after SCW->do_classify_surfaces" << endl;
		}

		if (f_v) {
			cout << "orbits_create::init "
					"after classify_surfaces" << endl;
		}


		f_has_cubic_surfaces = true;


	}

	if (Descr->f_on_arcs) {


		if (f_v) {
			cout << "orbits_create::init f_on_arcs" << endl;
		}
		if (f_v) {
			cout << "orbits_create::init control = " << Descr->on_arcs_control << endl;
		}


		Arc_generator_description_for_arcs = Get_object_of_type_arc_generator_control(
				Descr->on_arcs_control);


		projective_geometry::projective_space_with_action *PA;

		if (!Arc_generator_description_for_arcs->f_projective_space) {
			cout << "Please use option -projective_space in arc_generator" << endl;
			exit(1);
		}

		PA = Get_projective_space(Arc_generator_description_for_arcs->projective_space_label);


		Arc_generator = NEW_OBJECT(apps_geometry::arc_generator);


		if (f_v) {
			cout << "orbits_create::init before Arc_generator->init" << endl;
		}
		Arc_generator->init(
				Arc_generator_description_for_arcs,
				PA,
				PA->A->Strong_gens,
				verbose_level);
		if (f_v) {
			cout << "orbits_create::init after Arc_generator->init" << endl;
		}

		if (f_v) {
			cout << "orbits_create::init before Arc_generator->compute_starter" << endl;
		}
		Arc_generator->compute_starter(
				verbose_level);
		if (f_v) {
			cout << "orbits_create::init after Arc_generator->compute_starter" << endl;
		}

		f_has_arcs = true;



	}


	if (Descr->f_classify_semifields) {
		if (f_v) {
			cout << "orbits_create::init f_classify_semifields" << endl;
		}
		projective_geometry::projective_space_with_action *PA;

		PA = Get_projective_space(Descr->classify_semifields_PA);

		poset_classification::poset_classification_control *Control =
				Get_poset_classification_control(
						Descr->classify_semifields_control);

		if (f_v) {
			cout << "orbits_create::init "
					"before classify_surfaces, control=" << endl;
			Control->print();
		}

		Semifields = NEW_OBJECT(semifields::semifield_classify_with_substructure);

		if (f_v) {
			cout << "orbits_create::init "
					"before Semifields->init" << endl;
		}
		Semifields->init(
				Descr->Classify_semifields_description,
				PA,
				Control,
				verbose_level);
		if (f_v) {
			cout << "orbits_create::init "
					"after Semifields->init" << endl;
		}

		if (f_v) {
			cout << "orbits_create::init "
					"before Semifields->classify_semifields" << endl;
		}
		Semifields->classify_semifields(verbose_level);
		if (f_v) {
			cout << "orbits_create::init "
					"after Semifields->classify_semifields" << endl;
		}

		f_has_semifields = true;
	}


	if (Descr->f_on_boolean_functions) {
		if (f_v) {
			cout << "orbits_create::init f_on_boolean_functions" << endl;
		}
		projective_geometry::projective_space_with_action *PA;

		PA = Get_projective_space(Descr->on_boolean_functions_PA);

		if (PA->P->Subspaces->F->q != 2) {
			cout << "orbits_create::init "
					"the field must have order 2" << endl;
			exit(1);
		}
		if (PA->A->matrix_group_dimension() != PA->n + 1) {
			cout << "orbits_create::init "
					"the dimension of the matrix group must be PA->n + 1" << endl;
			exit(1);
		}


		BF = NEW_OBJECT(combinatorics::special_functions::boolean_function_domain);

		if (f_v) {
			cout << "orbits_create::init before BF->init" << endl;
		}
		BF->init(PA->P->Subspaces->F, PA->n, verbose_level);
		if (f_v) {
			cout << "orbits_create::init after BF->init" << endl;
		}


		BFC = NEW_OBJECT(apps_combinatorics::boolean_function_classify);

		if (f_v) {
			cout << "orbits_create::init "
					"before BFC->init_group" << endl;
		}
		BFC->init_group(BF, PA->A, verbose_level);
		if (f_v) {
			cout << "orbits_create::init "
					"after BFC->init_group" << endl;
		}

		if (f_v) {
			cout << "orbits_create::init "
					"before BFC->search_for_bent_functions" << endl;
		}
		BFC->search_for_bent_functions(verbose_level);
		if (f_v) {
			cout << "orbits_create::init "
					"after BFC->search_for_bent_functions" << endl;
		}

		f_has_boolean_functions = true;


	}

	if (Descr->f_classification_by_canonical_form) {


		if (f_v) {
			cout << "orbits_create::init f_classification_by_canonical_form" << endl;
		}



		if (!Descr->Canonical_form_classifier_description->f_output_fname) {
			cout << "Please specify the output file name using -output_fname <fname>" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "orbits_create::init getting projective space "
					<< Descr->Canonical_form_classifier_description->space_label << endl;
		}

		Canonical_form_classifier = NEW_OBJECT(canonical_form::canonical_form_classifier);

		if (f_v) {
			cout << "orbits_create::init "
					"before Canonical_form_classifier->init_objects_from_list_of_csv_files" << endl;
		}
		Canonical_form_classifier->init_objects_from_list_of_csv_files(
				Descr->Canonical_form_classifier_description,
				verbose_level - 2);
		if (f_v) {
			cout << "orbits_create::init "
					"after Canonical_form_classifier->init_objects_from_list_of_csv_files" << endl;
		}

		// now Input is set


		std::string fname_base;

		if (Descr->Canonical_form_classifier_description->f_output_fname) {
			fname_base = Descr->Canonical_form_classifier_description->fname_base_out;
		}
		else {
			fname_base = "classification_";
		}

		if (f_v) {
			cout << "orbits_create::init "
					"fname_base = " << fname_base << endl;
		}

		if (f_v) {
			cout << "orbits_create::init "
					"before Canonical_form_classifier->classify" << endl;
		}
		Canonical_form_classifier->classify(
				Canonical_form_classifier->Input,
				fname_base,
				verbose_level - 2);
		if (f_v) {
			cout << "orbits_create::init "
					"after Classifier.classify" << endl;
		}

		//Descr->Canonical_form_classifier_description->Canonical_form_classifier = Canonical_form_classifier;


		f_has_classification_by_canonical_form = true;


		if (f_v) {
			cout << "orbits_create::init f_classification_by_canonical_form done" << endl;
		}

	}


	if (f_v) {
		cout << "orbits_create::init done" << endl;
	}
}


}}}





