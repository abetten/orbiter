/*
 *  modified_group_init_layer5.cpp
 *
 *  Created on: Mar 20, 2025
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


modified_group_init_layer5::modified_group_init_layer5()
{
	Record_birth();

}

modified_group_init_layer5::~modified_group_init_layer5()
{
	Record_death();

}


void modified_group_init_layer5::modified_group_init(
		group_constructions::modified_group_create *Modified_group_create,
		group_constructions::group_modification_description *Descr,
		int verbose_level)
// Some group modifications need to be performed at level 5.
// This is because they rely on stuff that is in the application layer.
// Examples: Orbit computations, variety stabilizer
// So, this function serves as a front end for the actual function in
// group_constructions::modified_group_create
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_init" << endl;
	}
	Modified_group_create->Descr = Descr;

	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_init "
				"initializing group" << endl;
	}


	if (Descr->f_point_stabilizer) {

		if (f_v) {
			cout << "modified_group_init_layer5::modified_group_init "
					"before create_point_stabilizer_subgroup" << endl;
		}

		create_point_stabilizer_subgroup(Modified_group_create, Descr, verbose_level);

		if (f_v) {
			cout << "modified_group_init_layer5::modified_group_init "
					"after create_point_stabilizer_subgroup" << endl;
		}
	}

	else if (Descr->f_set_stabilizer) {

		if (f_v) {
			cout << "modified_group_init_layer5::modified_group_init "
					"before create_set_stabilizer_subgroup" << endl;
		}

		create_set_stabilizer_subgroup(Modified_group_create, Descr, verbose_level);

		if (f_v) {
			cout << "modified_group_init_layer5::modified_group_init "
					"after create_set_stabilizer_subgroup" << endl;
		}
	}
	else if (Descr->f_stabilizer_of_variety) {

		if (f_v) {
			cout << "modified_group_init_layer5::modified_group_init "
					"f_stabilizer_of_variety" << endl;
		}

		if (f_v) {
			cout << "modified_group_init_layer5::modified_group_init "
					"before modified_group_create_stabilizer_of_variety" << endl;
		}
		modified_group_create_stabilizer_of_variety(
				Modified_group_create,
				Descr,
				Descr->stabilizer_of_variety_label,
				verbose_level);

		if (f_v) {
			cout << "modified_group_init_layer5::modified_group_init "
					"after modified_group_create_stabilizer_of_variety" << endl;
		}

		// output in A_modified

	}
	else if (Descr->f_subgroup_by_generators) {

		if (f_v) {
			cout << "modified_group_init_layer5::modified_group_init "
					"f_subgroup_by_generators label=" << Descr->subgroup_by_generators_label << endl;
		}

		if (f_v) {
			cout << "modified_group_init_layer5::modified_group_init "
					"before create_subgroup_by_generators" << endl;
		}
		create_subgroup_by_generators(
				Modified_group_create,
				Descr,
				Descr->subgroup_by_generators_label,
				verbose_level);

		if (f_v) {
			cout << "modified_group_init_layer5::modified_group_init "
					"after create_subgroup_by_generators" << endl;
		}

		// output in A_modified

	}




	else {

		if (f_v) {
			cout << "modified_group_init_layer5::modified_group_init "
					"before Modified_group_create->modified_group_init" << endl;
		}
		Modified_group_create->modified_group_init(Descr, verbose_level);
		if (f_v) {
			cout << "algebra_global_with_action::modified_group_init "
					"after Modified_group_create->modified_group_init" << endl;
		}
		//cout << "algebra_global_with_action::modified_group_init "
		//		"unknown operation" << endl;
		//exit(1);

	}

	if (f_v) {

		algebra::ring_theory::longinteger_object go;

		Modified_group_create->Strong_gens->group_order(go);

		cout << "modified_group_init_layer5::modified_group_init "
				"created a group of order " << go
				<< " and degree " << Modified_group_create->A_modified->degree << endl;

	}



	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_init done" << endl;
	}
}




void modified_group_init_layer5::create_point_stabilizer_subgroup(
		group_constructions::modified_group_create *Modified_group_create,
		group_constructions::group_modification_description *Descr,
		int verbose_level)
// needs orbits::orbits_global, hence level 5
// output in A_modified and Strong_gens
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_init_layer5::create_point_stabilizer_subgroup" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "modified_group_init_layer5::create_point_stabilizer_subgroup "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	groups::any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	Modified_group_create->A_base = AG->A_base;
	Modified_group_create->A_previous = AG->A;

	Modified_group_create->label = AG->label;
	Modified_group_create->label_tex = AG->label_tex;

	if (f_v) {
		cout << "modified_group_init_layer5::create_point_stabilizer_subgroup "
				"A_base=";
		Modified_group_create->A_base->print_info();
		cout << endl;
		cout << "modified_group_init_layer5::create_point_stabilizer_subgroup "
				"A_previous=";
		Modified_group_create->A_previous->print_info();
		cout << endl;
	}

	Modified_group_create->A_modified = Modified_group_create->A_previous; // ToDo!



	Modified_group_create->f_has_strong_generators = true;
	if (f_v) {
		cout << "modified_group_init_layer5::create_point_stabilizer_subgroup "
				"before Strong_gens = AG->Subgroup_gens" << endl;
	}

	//Strong_gens = NEW_OBJECT(groups::strong_generators);

	{
		groups::orbits_on_something *Orb;

		if (f_v) {
			cout << "modified_group_init_layer5::create_point_stabilizer_subgroup "
					"before Orbits.orbits_on_points" << endl;
		}

		orbits::orbits_global Orbits;
		int print_interval = 10000;

		Orbits.orbits_on_points(AG, Orb, print_interval, verbose_level);

		if (f_v) {
			cout << "modified_group_init_layer5::create_point_stabilizer_subgroup "
					"after Orbits.orbits_on_points" << endl;
		}

		Orb->stabilizer_any_point(
				Descr->point_stabilizer_point,
				Modified_group_create->Strong_gens, verbose_level);


		FREE_OBJECT(Orb);
	}



	if (f_v) {
		cout << "modified_group_init_layer5::create_point_stabilizer_subgroup "
				"action A_modified created: ";
		Modified_group_create->A_modified->print_info();
	}


	Modified_group_create->label += "_Stab" + std::to_string(Descr->point_stabilizer_point);
	Modified_group_create->label_tex += "{\\rm Stab " + std::to_string(Descr->point_stabilizer_point) + "}";



	if (f_v) {
		cout << "modified_group_init_layer5::create_point_stabilizer_subgroup "
				"done" << endl;
	}
}


void modified_group_init_layer5::create_set_stabilizer_subgroup(
		group_constructions::modified_group_create *Modified_group_create,
		group_constructions::group_modification_description *Descr,
		int verbose_level)
// needs poset_classification::poset_classification, hence level 5
// output in A_modified and Strong_gens
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_init_layer5::create_set_stabilizer_subgroup" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "modified_group_init_layer5::create_set_stabilizer_subgroup "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	groups::any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	Modified_group_create->A_base = AG->A_base;
	Modified_group_create->A_previous = AG->A;

	Modified_group_create->label = AG->label;
	Modified_group_create->label_tex = AG->label_tex;

	if (f_v) {
		cout << "modified_group_init_layer5::create_set_stabilizer_subgroup "
				"A_base=";
		Modified_group_create->A_base->print_info();
		cout << endl;
		cout << "modified_group_init_layer5::create_set_stabilizer_subgroup "
				"A_previous=";
		Modified_group_create->A_previous->print_info();
		cout << endl;
	}

	Modified_group_create->A_modified = Modified_group_create->A_previous; // ToDo !!!




#if 0
	//Strong_gens = NEW_OBJECT(groups::strong_generators);

	{
		groups::orbits_on_something *Orb;

		if (f_v) {
			cout << "modified_group_init_layer5::create_set_stabilizer_subgroup "
					"before Orbits.orbits_on_points" << endl;
		}

		orbits::orbits_global Orbits;

		Orbits.orbits_on_points(AG, Orb, verbose_level);

		if (f_v) {
			cout << "modified_group_init_layer5::create_set_stabilizer_subgroup "
					"after Orbits.orbits_on_points" << endl;
		}

		Orb->stabilizer_any_point(
				Descr->point_stabilizer_point,
				Strong_gens, verbose_level);


		FREE_OBJECT(Orb);
	}
#endif
	{

		orbits::orbits_global Orbits_global;
		poset_classification::poset_classification_control *Control;
		long int *the_set;
		long int *canonical_set;
		int *Elt1;
		int the_set_sz;
		int local_idx;

		Lint_vec_scan(Descr->set_stabilizer_the_set, the_set, the_set_sz);


		canonical_set = NEW_lint(the_set_sz);
		Elt1 = NEW_int(Modified_group_create->A_base->elt_size_in_int);

		Control = Get_poset_classification_control(Descr->set_stabilizer_control);

		poset_classification::poset_classification *PC;

		if (f_v) {
			cout << "modified_group_init_layer5::create_set_stabilizer_subgroup "
					"before Orbits_global.orbits_on_subsets" << endl;
		}

		// ToDo:
		Orbits_global.orbits_on_subsets(
				AG,
				AG,
				AG->Subgroup_gens,
				Control,
				PC,
				the_set_sz,
				verbose_level - 2);
		if (f_v) {
			cout << "modified_group_init_layer5::create_set_stabilizer_subgroup "
					"after Orbits_global.orbits_on_subsets" << endl;
		}


		// trace the subset:

		if (f_v) {
			cout << "modified_group_init_layer5::create_set_stabilizer_subgroup "
					"before trace_set" << endl;
		}


		local_idx = PC->trace_set(
				the_set, the_set_sz, the_set_sz,
				canonical_set, Elt1,
			verbose_level - 2);


		// Elt1 maps the_set to canonical_set.


		if (f_v) {
			cout << "modified_group_init_layer5::create_set_stabilizer_subgroup "
					"after trace_set local_idx=" << local_idx << endl;
			cout << "modified_group_init_layer5::create_set_stabilizer_subgroup "
					"canonical_set=";
			Lint_vec_print(cout, canonical_set, the_set_sz);
			cout << endl;
		}

		groups::strong_generators *stab_gens_canonical_set;

		PC->get_stabilizer_generators_cleaned_up(
				stab_gens_canonical_set,
				the_set_sz, local_idx, verbose_level - 2);

		groups::group_theory_global Group_theory_global;

		if (f_v) {
			cout << "modified_group_init_layer5::create_set_stabilizer_subgroup "
					"before Group_theory_global.strong_generators_conjugate_aGav" << endl;
		}

		Group_theory_global.strong_generators_conjugate_aGav(
				stab_gens_canonical_set,
				Elt1,
				Modified_group_create->Strong_gens,
				verbose_level - 2);

		if (f_v) {
			cout << "modified_group_init_layer5::create_set_stabilizer_subgroup "
					"after Group_theory_global.strong_generators_conjugate_aGav" << endl;
		}



		FREE_OBJECT(stab_gens_canonical_set);
		FREE_OBJECT(PC);
		FREE_lint(canonical_set);
		FREE_int(Elt1);
	}

	Modified_group_create->f_has_strong_generators = true;

	if (f_v) {
		cout << "modified_group_init_layer5::create_set_stabilizer_subgroup "
				"strong generators created" << endl;
	}


	Modified_group_create->label += "_SetStab" + Descr->set_stabilizer_the_set;
	Modified_group_create->label_tex += "{\\rm SetStab " + Descr->set_stabilizer_the_set + "}";



	if (f_v) {
		cout << "modified_group_init_layer5::create_set_stabilizer_subgroup "
				"done" << endl;
	}
}


void modified_group_init_layer5::modified_group_create_stabilizer_of_variety(
		group_constructions::modified_group_create *Modified_group_create,
		group_constructions::group_modification_description *Descr,
		std::string &variety_label,
		int verbose_level)
// needs canonical_form::canonical_form_classifier, hence level 5
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety" << endl;
	}

	canonical_form::variety_object_with_action *Input_Variety;

	Input_Variety = Get_variety(variety_label);

	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"Input_Variety = " << Input_Variety->Variety_object->label_txt << endl;
	}

	std::string fname_base;

	fname_base = Input_Variety->Variety_object->label_txt + "_c";


	canonical_form::canonical_form_classifier *Classifier;

	Classifier = NEW_OBJECT(canonical_form::canonical_form_classifier);


	if (!Descr->f_nauty_control) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"Please use -nauty_control" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"before Classifier->init_direct" << endl;
	}

	Classifier->init_direct(
			1 /*nb_input_Vo*/,
			Input_Variety,
			fname_base,
			Descr->f_nauty_control,
			Descr->Nauty_interface_control,
			verbose_level);

	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"after Classifier->init_direct" << endl;
	}

	canonical_form::canonical_form_classifier_description *Descr1;

	Descr1 = NEW_OBJECT(canonical_form::canonical_form_classifier_description);



	//Descr1->f_save_nauty_input_graphs = true;

	Classifier->set_description(Descr1);



	canonical_form::canonical_form_global Canonical_form_global;
	canonical_form::classification_of_varieties_nauty *Classification_of_varieties_nauty;

	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"before Canonical_form_global.compute_group_and_tactical_decomposition" << endl;
	}
	Canonical_form_global.compute_group_and_tactical_decomposition(
			Classifier,
			Input_Variety,
			Classification_of_varieties_nauty,
			verbose_level);
	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"after Canonical_form_global.compute_group_and_tactical_decomposition" << endl;
	}


	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"before FREE_OBJECT(Classification_of_varieties_nauty)" << endl;
	}

	//FREE_OBJECT(Classification_of_varieties_nauty);
	// Classification_of_varieties_nauty is freed in FREE_OBJECT(Classifier);

	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"after FREE_OBJECT(Classification_of_varieties_nauty)" << endl;
	}

	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"before FREE_OBJECT(Descr1)" << endl;
	}
	FREE_OBJECT(Descr1);
	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"after FREE_OBJECT(Descr1)" << endl;
	}
	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"before FREE_OBJECT(Classifier)" << endl;
	}
	FREE_OBJECT(Classifier);
	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"after FREE_OBJECT(Classifier)" << endl;
	}


	//Input_Vo[0].Stab_gens;


	Modified_group_create->A_base = Input_Variety->PA->A;
	Modified_group_create->A_previous = Input_Variety->PA->A;





	Modified_group_create->label = Input_Variety->PA->A->label + "_stab_of_" + Input_Variety->Variety_object->label_txt;
	Modified_group_create->label_tex = Input_Variety->PA->A->label_tex + "{\\rm \\_stab\\_of\\_}" + Input_Variety->Variety_object->label_tex;
	if (f_v) {
		cout << "algebra_global_with_action::do_stabilizer_of_variety "
				"label = " << Modified_group_create->label << endl;
		cout << "algebra_global_with_action::do_stabilizer_of_variety "
				"label_tex = " << Modified_group_create->label_tex << endl;
	}

	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"A_base=";
		Modified_group_create->A_base->print_info();
		cout << endl;
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"A_previous=";
		Modified_group_create->A_previous->print_info();
		cout << endl;
	}

	Modified_group_create->f_has_strong_generators = true;

	groups::strong_generators *Strong_gens_temp;
	Strong_gens_temp = Input_Variety->Stab_gens->create_copy(verbose_level - 4);

	actions::action_global Action_global;

	Modified_group_create->A_modified = Action_global.init_subgroup_from_strong_generators(
			Modified_group_create->A_base,
			Strong_gens_temp,
			verbose_level - 1);

	Modified_group_create->A_modified->label = Modified_group_create->label;
	Modified_group_create->A_modified->label_tex = Modified_group_create->label_tex;

	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"A_modified->label = " << Modified_group_create->A_modified->label << endl;
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety "
				"A_modified->label_tex = " << Modified_group_create->A_modified->label_tex << endl;
	}


	// Strong_gens should be in the new action.

	Modified_group_create->f_has_strong_generators = true;
	Modified_group_create->Strong_gens = Modified_group_create->A_modified->Strong_gens->create_copy(verbose_level - 4);
	//Strong_gens = AG->class_data->Conjugacy_class[orbit_index]->gens->create_copy(verbose_level - 4);



	if (f_v) {
		cout << "modified_group_init_layer5::modified_group_create_stabilizer_of_variety done" << endl;
	}
}



void modified_group_init_layer5::create_subgroup_by_generators(
		group_constructions::modified_group_create *Modified_group_create,
		group_constructions::group_modification_description *Descr,
		std::string &subgroup_by_generators_label,
		int verbose_level)
// needs vector_ge_builder, hence level5
// output in A_modified and Strong_gens
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_init_layer5::create_subgroup_by_generators" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "modified_group_init_layer5::create_subgroup_by_generators "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	groups::any_group *AG;

	AG = Get_any_group(Descr->from[0]);


	apps_algebra::vector_ge_builder *VB;

	VB = Get_object_of_type_vector_ge(subgroup_by_generators_label);

	data_structures_groups::vector_ge *nice_gens;


	nice_gens = VB->V;


	Modified_group_create->A_base = AG->A_base;
	Modified_group_create->A_previous = AG->A;

	Modified_group_create->label = AG->label;
	Modified_group_create->label_tex = AG->label_tex;

	if (f_v) {
		cout << "modified_group_init_layer5::create_subgroup_by_generators "
				"A_base=";
		Modified_group_create->A_base->print_info();
		cout << endl;
		cout << "modified_group_init_layer5::create_subgroup_by_generators "
				"A_previous=";
		Modified_group_create->A_previous->print_info();
		cout << endl;
	}

	Modified_group_create->A_modified = Modified_group_create->A_previous;

	actions::action_global Action_global;


	groups::strong_generators *SG;

	SG = NEW_OBJECT(groups::strong_generators);

	algebra::ring_theory::longinteger_object target_go;

	if (f_v) {
		cout << "modified_group_init_layer5::create_subgroup_by_generators "
				"before AG->A_base->generators_to_strong_generators" << endl;
	}
	AG->A_base->generators_to_strong_generators(
		false /* f_target_go */, target_go,
		nice_gens, SG,
		verbose_level - 1);
	if (f_v) {
		cout << "modified_group_init_layer5::create_subgroup_by_generators "
				"after AG->A_base->generators_to_strong_generators" << endl;
	}

	if (false) {
		cout << "modified_group_init_layer5::create_subgroup_by_generators "
				"strong generators are:" << endl;
		SG->print_generators(cout, verbose_level - 1);
	}


	Modified_group_create->f_has_strong_generators = true;
	Modified_group_create->Strong_gens = SG;
	if (f_v) {
		cout << "modified_group_init_layer5::create_subgroup_by_generators "
				"before Strong_gens = AG->Subgroup_gens" << endl;
	}



	if (f_v) {
		cout << "modified_group_init_layer5::create_subgroup_by_generators "
				"action A_modified created: ";
		Modified_group_create->A_modified->print_info();
	}


	Modified_group_create->label += "_SubgroupGens" + subgroup_by_generators_label;
	Modified_group_create->label_tex += "{\\rm SubgroupGens " + subgroup_by_generators_label + "}";



	if (f_v) {
		cout << "modified_group_init_layer5::create_subgroup_by_generators "
				"done" << endl;
	}
}






}}}

