/*
 * modified_group_create.cpp
 *
 *  Created on: Dec 1, 2021
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


modified_group_create::modified_group_create()
{
		Descr = NULL;

		//std::string label;
		//std::string label_tex;

		//initial_strong_gens = NULL;

		A_base = NULL;
		A_previous = NULL;
		A_modified = NULL;

		f_has_strong_generators = false;
		Strong_gens = NULL;

		action_on_self_by_right_multiplication_sims = NULL;
		Action_by_right_multiplication = NULL;
}


modified_group_create::~modified_group_create()
{
		Descr = NULL;
}


void modified_group_create::modified_group_init(
		group_modification_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::modified_group_init" << endl;
	}
	modified_group_create::Descr = description;

	if (f_v) {
		cout << "modified_group_create::modified_group_init "
				"initializing group" << endl;
	}


	if (Descr->f_restricted_action) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_restricted_action" << endl;
		}

		create_restricted_action(description, verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_restricted_action" << endl;
		}
	}

	else if (Descr->f_on_k_subspaces) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_action_on_k_subspaces" << endl;
		}

		create_action_on_k_subspaces(description, verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_action_on_k_subspaces" << endl;
		}
	}

	else if (Descr->f_on_k_subsets) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_action_on_k_subsets" << endl;
		}

		create_action_on_k_subsets(description, verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_action_on_k_subsets" << endl;
		}
	}

	else if (Descr->f_on_wedge_product) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_action_on_wedge_product" << endl;
		}

		create_action_on_wedge_product(description, verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_action_on_wedge_product" << endl;
		}
	}

	else if (Descr->f_create_special_subgroup) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_special_subgroup" << endl;
		}

		create_special_subgroup(description, verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_special_subgroup" << endl;
		}
	}

	else if (Descr->f_point_stabilizer) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_point_stabilizer_subgroup" << endl;
		}

		create_point_stabilizer_subgroup(description, verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_point_stabilizer_subgroup" << endl;
		}
	}

	else if (Descr->f_projectivity_subgroup) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_projectivity_subgroup" << endl;
		}

		create_projectivity_subgroup(description, verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_projectivity_subgroup" << endl;
		}
	}

	else if (Descr->f_subfield_subgroup) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_subfield_subgroup" << endl;
		}

		create_subfield_subgroup(description, verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_subfield_subgroup" << endl;
		}
	}
	else if (Descr->f_action_on_self_by_right_multiplication) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_action_on_self_by_right_multiplication" << endl;
		}

		create_action_on_self_by_right_multiplication(description, verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_action_on_self_by_right_multiplication" << endl;
		}
	}
	else if (Descr->f_direct_product) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"f_direct_product" << endl;
		}

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_product_action" << endl;
		}
		create_product_action(
					description,
					verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_product_action" << endl;
		}
	}
	else if (Descr->f_polarity_extension) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"f_polarity_extension" << endl;
		}

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_polarity_extension" << endl;
		}
		create_polarity_extension(
					description->polarity_extension_input,
					description->polarity_extension_PA,
					description->f_on_middle_layer_grassmannian,
					description->f_on_points_and_hyperplanes,
					verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_polarity_extension" << endl;
		}
	}


	else {
		cout << "modified_group_create::modified_group_init "
				"unknown operation" << endl;
		exit(1);

	}




	if (f_v) {
		cout << "modified_group_create::modified_group_init done" << endl;
	}
}


void modified_group_create::create_restricted_action(
		group_modification_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_restricted_action" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "modified_group_create::create_restricted_action "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	any_group *AG;

	AG = Get_any_group(Descr->from[0]);


	A_base = AG->A_base;
	A_previous = AG->A;

	label.assign(AG->label);
	label_tex.assign(AG->label_tex);

	long int *points;
	int nb_points;

	Get_vector_or_set(Descr->restricted_action_set_text,
			points, nb_points);

	if (f_v) {
		cout << "modified_group_create::create_restricted_action "
				"before A_previous->Induced_action->restricted_action" << endl;
	}
	A_modified = A_previous->Induced_action->restricted_action(
			points, nb_points,
			Descr->restricted_action_set_text /* label_of_set */,
			Descr->restricted_action_set_text_tex /* label_of_set */,
			verbose_level);
	if (f_v) {
		cout << "modified_group_create::create_restricted_action "
				"after A_previous->Induced_action->restricted_action" << endl;
	}
	A_modified->f_is_linear = A_previous->f_is_linear;

	f_has_strong_generators = true;
	if (f_v) {
		cout << "modified_group_create::create_restricted_action "
				"before Strong_gens = AG->Subgroup_gens" << endl;
	}
	Strong_gens = AG->Subgroup_gens;

#if 0
	A_modified->Strong_gens->print_generators_in_latex_individually(cout);
	A_modified->Strong_gens->print_generators_in_source_code();
	A_modified->print_base();
#endif
	A_modified->print_info();

	if (f_v) {
		cout << "modified_group_create::create_restricted_action "
				"before assigning label" << endl;
	}
	label.assign(A_modified->label);
	label_tex.assign(A_modified->label_tex);

	if (f_v) {
		cout << "modified_group_create::create_restricted_action done" << endl;
	}
}


void modified_group_create::create_action_on_k_subspaces(
		group_modification_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "modified_group_create::create_action_on_k_subspaces "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	A_base = AG->A_base;
	A_previous = AG->A;

	label.assign(AG->label);
	label_tex.assign(AG->label_tex);




	actions::action_global AGlobal;

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces "
				"before AGlobal.create_action_on_k_subspaces" << endl;
	}
	A_modified = AGlobal.create_action_on_k_subspaces(
			A_previous,
			description->on_k_subspaces_k,
			verbose_level - 1);
	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces "
				"before AGlobal.create_action_on_k_subspaces" << endl;
	}


#if 0
	algebra::matrix_group *M;
	field_theory::finite_field *Fq;
	int n;

	M = A_previous->get_matrix_group();

	n = M->n;
	Fq = M->GFq;

	induced_actions::action_on_grassmannian *AonG;
	geometry::grassmann *Grass;

	AonG = NEW_OBJECT(induced_actions::action_on_grassmannian);

	Grass = NEW_OBJECT(geometry::grassmann);


	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces "
				"before Grass->init" << endl;
	}

	Grass->init(n,
			description->on_k_subspaces_k,
			Fq, 0 /* verbose_level */);

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces "
				"after Grass->init" << endl;
	}


	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces "
				"before AonG->init" << endl;
	}

	AonG->init(*A_previous, Grass, verbose_level - 2);

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces "
				"after AonG->init" << endl;
	}


	//A_modified = NEW_OBJECT(actions::action);

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces "
				"before induced_action_on_grassmannian_preloaded" << endl;
	}

	A_modified = A_previous->Induced_action->induced_action_on_grassmannian_preloaded(AonG,
		false /* f_induce_action */, NULL /*sims *old_G */,
		verbose_level - 2);

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces "
				"after induced_action_on_grassmannian_preloaded" << endl;
	}
#endif


	f_has_strong_generators = true;

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces "
				"before Strong_gens = AG->Subgroup_gens" << endl;
	}
	Strong_gens = AG->Subgroup_gens;

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces "
				"action A_modified created: ";
		A_modified->print_info();
	}

	algebra::matrix_group *M;
	field_theory::finite_field *Fq;
	int n;

	M = A_previous->get_matrix_group();

	n = M->n;
	Fq = M->GFq;


	label += "_OnGr_" + std::to_string(n) + "_" + std::to_string(description->on_k_subspaces_k) + "_" + std::to_string(Fq->q);
	label_tex += " {\\rm Gr}_{" + std::to_string(n) + "," + std::to_string(description->on_k_subspaces_k) + "}(" + std::to_string(Fq->q) + ")";



	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subspaces "
				"done" << endl;
	}
}

void modified_group_create::create_action_on_k_subsets(
		group_modification_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subsets" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "modified_group_create::create_action_on_k_subsets "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	A_base = AG->A_base;
	A_previous = AG->A;


	label.assign(AG->label);
	label_tex.assign(AG->label_tex);



	//A_modified = NEW_OBJECT(actions::action);

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subsets "
				"before A_previous->Induced_action->induced_action_on_k_subsets" << endl;
	}


	A_modified = A_previous->Induced_action->induced_action_on_k_subsets(
			description->on_k_subsets_k,
			verbose_level);


	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subsets "
				"after A_previous->Induced_action->induced_action_on_k_subsets" << endl;
	}


	A_modified->f_is_linear = false;

	f_has_strong_generators = true;

	A_modified->f_is_linear = A_previous->f_is_linear;
	A_modified->dimension = A_previous->dimension;

	f_has_strong_generators = true;
	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subsets "
				"before Strong_gens = AG->Subgroup_gens" << endl;
	}
	Strong_gens = AG->Subgroup_gens;

	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subsets "
				"action A_modified created: ";
		A_modified->print_info();
	}


	label += "_OnSubsets_" + std::to_string(description->on_k_subsets_k);
	label_tex += " {\\rm OnSubsets}_{" + std::to_string(description->on_k_subsets_k) + "}";


	if (f_v) {
		cout << "modified_group_create::create_action_on_k_subsets "
				"done" << endl;
	}
}


void modified_group_create::create_action_on_wedge_product(
		group_modification_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_action_on_wedge_product" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "modified_group_create::create_action_on_wedge_product "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	A_base = AG->A_base;
	A_previous = AG->A;


	label.assign(AG->label);
	label_tex.assign(AG->label_tex);




	if (f_v) {
		cout << "modified_group_create::create_action_on_wedge_product "
				"before A_previous->Induced_action->induced_action_on_wedge_product" << endl;
	}
	A_modified = A_previous->Induced_action->induced_action_on_wedge_product(verbose_level);
	if (f_v) {
		cout << "modified_group_create::create_action_on_wedge_product "
				"after A_previous->Induced_action->induced_action_on_wedge_product" << endl;
	}
	if (f_v) {
		cout << "modified_group_create::create_action_on_wedge_product "
				"action A_wedge:" << endl;
		A_modified->print_info();
	}



	f_has_strong_generators = true;

	//A_modified->f_is_linear = A_previous->f_is_linear;
	//A_modified->dimension = A_previous->dimension;

	if (f_v) {
		cout << "modified_group_create::create_action_on_wedge_product "
				"before Strong_gens = AG->Subgroup_gens" << endl;
	}
	Strong_gens = AG->Subgroup_gens;

	if (f_v) {
		cout << "modified_group_create::create_action_on_wedge_product "
				"action A_modified created: ";
		A_modified->print_info();
	}


	label += "_OnWedge";
	label_tex += " {\\rm OnWedge}";



	if (f_v) {
		cout << "modified_group_create::create_action_on_wedge_product "
				"done" << endl;
	}
}





void modified_group_create::create_special_subgroup(
		group_modification_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_special_subgroup" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "modified_group_create::create_special_subgroup "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	A_base = AG->A_base;
	A_previous = AG->A;

	label.assign(AG->label);
	label_tex.assign(AG->label_tex);


	A_modified = A_previous;



	f_has_strong_generators = true;
	if (f_v) {
		cout << "modified_group_create::create_special_subgroup "
				"before Strong_gens = AG->Subgroup_gens" << endl;
	}

	Strong_gens = NEW_OBJECT(groups::strong_generators);

	{
		actions::action *A_on_det;
		ring_theory::longinteger_object go;


		groups::sims *Sims;


		if (f_v) {
			cout << "modified_group_create::create_special_subgroup "
					"before AG->Subgroup_gens->create_sims" << endl;
		}
		Sims = AG->Subgroup_gens->create_sims(verbose_level);
		if (f_v) {
			cout << "modified_group_create::create_special_subgroup "
					"after AG->Subgroup_gens->create_sims" << endl;
		}

		if (f_v) {
			cout << "modified_group_create::create_special_subgroup "
					"before Sims->A->Induced_action->induced_action_on_determinant" << endl;
		}
		A_on_det = Sims->A->Induced_action->induced_action_on_determinant(
				Sims, verbose_level);
		if (f_v) {
			cout << "modified_group_create::create_special_subgroup "
					"after Sims->A->Induced_action->induced_action_on_determinant" << endl;
		}
		A_on_det->Kernel->group_order(go);
		if (f_v) {
			cout << "modified_group_create::create_special_subgroup "
					"kernel has order " << go << endl;
		}


		Strong_gens->init_from_sims(A_on_det->Kernel, verbose_level);

		FREE_OBJECT(A_on_det);
		FREE_OBJECT(Sims);
	}



	if (f_v) {
		cout << "modified_group_create::create_special_subgroup "
				"action A_modified created: ";
		A_modified->print_info();
	}


	label += "_SpecialSub";
	label_tex += " {\\rm SpecialSub}";



	if (f_v) {
		cout << "modified_group_create::create_special_subgroup "
				"done" << endl;
	}
}



void modified_group_create::create_point_stabilizer_subgroup(
		group_modification_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_point_stabilizer_subgroup" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "modified_group_create::create_point_stabilizer_subgroup "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	A_base = AG->A_base;
	A_previous = AG->A;

	label.assign(AG->label);
	label_tex.assign(AG->label_tex);

	if (f_v) {
		cout << "modified_group_create::create_point_stabilizer_subgroup "
				"A_base=";
		A_base->print_info();
		cout << endl;
		cout << "modified_group_create::create_point_stabilizer_subgroup "
				"A_previous=";
		A_previous->print_info();
		cout << endl;
	}

	A_modified = A_previous;



	f_has_strong_generators = true;
	if (f_v) {
		cout << "modified_group_create::create_point_stabilizer_subgroup "
				"before Strong_gens = AG->Subgroup_gens" << endl;
	}

	//Strong_gens = NEW_OBJECT(groups::strong_generators);

	{
		groups::orbits_on_something *Orb;

		if (f_v) {
			cout << "modified_group_create::create_point_stabilizer_subgroup "
					"before Orbits.orbits_on_points" << endl;
		}

		orbits::orbits_global Orbits;

		Orbits.orbits_on_points(AG, Orb, verbose_level);

		if (f_v) {
			cout << "modified_group_create::create_point_stabilizer_subgroup "
					"after Orbits.orbits_on_points" << endl;
		}

		Orb->stabilizer_any_point(Descr->point_stabilizer_index,
				Strong_gens, verbose_level);


		FREE_OBJECT(Orb);
	}



	if (f_v) {
		cout << "modified_group_create::create_point_stabilizer_subgroup "
				"action A_modified created: ";
		A_modified->print_info();
	}


	label += "_Stab" + std::to_string(Descr->point_stabilizer_index);
	label_tex += " {\\rm Stab " + std::to_string(Descr->point_stabilizer_index) + "}";



	if (f_v) {
		cout << "modified_group_create::create_point_stabilizer_subgroup "
				"done" << endl;
	}
}


void modified_group_create::create_projectivity_subgroup(
		group_modification_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_projectivity_subgroup" << endl;
	}

	any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	A_base = AG->A_base;
	A_previous = AG->A;

	label.assign(AG->label);
	label_tex.assign(AG->label_tex);

	if (f_v) {
		cout << "modified_group_create::create_projectivity_subgroup "
				"A_base=";
		A_base->print_info();
		cout << endl;
		cout << "modified_group_create::create_projectivity_subgroup "
				"A_previous=";
		A_previous->print_info();
		cout << endl;
	}

	A_modified = A_previous;



	if (f_v) {
		cout << "modified_group_create::create_projectivity_subgroup "
				"before A_previous->compute_projectivity_subgroup" << endl;
	}

	A_previous->compute_projectivity_subgroup(
			Strong_gens,
			AG->Subgroup_gens,
			verbose_level);
	if (f_v) {
		cout << "modified_group_create::create_projectivity_subgroup "
				"after A_previous->compute_projectivity_subgroup" << endl;
	}

	f_has_strong_generators = true;


	if (f_v) {
		cout << "modified_group_create::create_projectivity_subgroup "
				"action A_modified created: ";
		A_modified->print_info();
	}


	label += "_ProjectivitySubgroup";
	label_tex += " {\\rm\\_ProjectivitySubgroup}";



	if (f_v) {
		cout << "modified_group_create::create_projectivity_subgroup "
				"done" << endl;
	}
}




void modified_group_create::create_subfield_subgroup(
		group_modification_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_subfield_subgroup" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "modified_group_create::create_subfield_subgroup "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	//int index;

	//index = description->subfield_subgroup_index;

	any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	A_base = AG->A_base;
	A_previous = AG->A;

	label.assign(AG->label);
	label_tex.assign(AG->label_tex);

	if (f_v) {
		cout << "modified_group_create::create_subfield_subgroup "
				"A_base=";
		A_base->print_info();
		cout << endl;
		cout << "modified_group_create::create_subfield_subgroup "
				"A_previous=";
		A_previous->print_info();
		cout << endl;
	}

	A_modified = A_previous;



	f_has_strong_generators = true;
	if (f_v) {
		cout << "modified_group_create::create_subfield_subgroup "
				"before Strong_gens = AG->Subgroup_gens" << endl;
	}

	//Strong_gens = NEW_OBJECT(groups::strong_generators);

	// ToDo


#if 0
	{
		groups::orbits_on_something *Orb;

		if (f_v) {
			cout << "modified_group_create::create_subfield_subgroup "
					"before AG->orbits_on_points" << endl;
		}

		AG->orbits_on_points(Orb, verbose_level);

		if (f_v) {
			cout << "modified_group_create::create_subfield_subgroup "
					"after AG->orbits_on_points" << endl;
		}

		Orb->stabilizer_any_point(Descr->point_stabilizer_index,
				Strong_gens, verbose_level);


		FREE_OBJECT(Orb);
	}
#endif


	if (f_v) {
		cout << "modified_group_create::create_subfield_subgroup "
				"action A_modified created: ";
		A_modified->print_info();
	}


	label += "_SubfieldOfIndex" + std::to_string(Descr->subfield_subgroup_index);
	label_tex +=" {\\rm SubfieldOfIndex " + std::to_string(Descr->subfield_subgroup_index) + "}";



	if (f_v) {
		cout << "modified_group_create::create_subfield_subgroup "
				"done" << endl;
	}
}



void modified_group_create::create_action_on_self_by_right_multiplication(
		group_modification_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_action_on_self_by_right_multiplication" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "modified_group_create::create_action_on_self_by_right_multiplication "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	A_base = AG->A_base;
	A_previous = AG->A;

	label.assign(AG->label);
	label_tex.assign(AG->label_tex);

	if (f_v) {
		cout << "modified_group_create::create_action_on_self_by_right_multiplication "
				"A_base=";
		A_base->print_info();
		cout << endl;
		cout << "modified_group_create::create_action_on_self_by_right_multiplication "
				"A_previous=";
		A_previous->print_info();
		cout << endl;
	}

	//A_modified = A_previous;


	if (f_v) {
		cout << "modified_group_create::create_action_on_self_by_right_multiplication "
				"before AG->Subgroup_gens->create_sims" << endl;
	}
	action_on_self_by_right_multiplication_sims = AG->Subgroup_gens->create_sims(verbose_level);
	if (f_v) {
		cout << "modified_group_create::create_action_on_self_by_right_multiplication "
				"after AG->Subgroup_gens->create_sims" << endl;
	}


	A_modified = A_previous->Induced_action->induced_action_by_right_multiplication(
			false /* f_basis */, NULL,
			action_on_self_by_right_multiplication_sims, false /* f_ownership */,
			verbose_level);




	A_modified->f_is_linear = false;

	f_has_strong_generators = true;

	//A_modified->f_is_linear = A_previous->f_is_linear;
	//A_modified->dimension = A_previous->dimension;

	f_has_strong_generators = true;
	if (f_v) {
		cout << "modified_group_create::create_action_on_self_by_right_multiplication "
				"before Strong_gens = AG->Subgroup_gens" << endl;
	}
	Strong_gens = AG->Subgroup_gens;




	if (f_v) {
		cout << "modified_group_create::create_action_on_self_by_right_multiplication "
				"action A_modified created: ";
		A_modified->print_info();
	}


	label += "_ByRightMult";
	label_tex += " {\\rm ByRightMult}";



	if (f_v) {
		cout << "modified_group_create::create_action_on_self_by_right_multiplication "
				"done" << endl;
	}
}

void modified_group_create::create_product_action(
		group_modification_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_product_action" << endl;
	}
#if 0
	if (Descr->from.size() != 1) {
		cout << "modified_group_create::create_product_action "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}
#endif

	data_structures::string_tools ST;
	std::vector<std::string> Input;

	ST.parse_comma_separated_strings(
			description->direct_product_input, Input);


	if (Input.size() != 2) {
		cout << "modified_group_create::create_product_action "
				"need exactly two input actions" << endl;
		exit(1);
	}


	any_group *AG1, *AG2;

	AG1 = Get_any_group(Input[0]);
	AG2 = Get_any_group(Input[1]);

	algebra::matrix_group *M1;
	algebra::matrix_group *M2;


	if (!AG1->A->is_matrix_group()) {
		cout << "modified_group_create::create_product_action "
				"group 1 is not a matrix group" << endl;
		exit(1);
	}
	M1 = AG1->A->get_matrix_group();

	if (!AG2->A->is_matrix_group()) {
		cout << "modified_group_create::create_product_action "
				"group 2 is not a matrix group" << endl;
		exit(1);
	}
	M2 = AG2->A->get_matrix_group();

	actions::action_global AG;

	//actions::action *A;

	if (f_v) {
		cout << "modified_group_create::create_product_action "
				"before AG.init_direct_product_group_and_restrict" << endl;
	}
	A_modified = AG.init_direct_product_group_and_restrict(
			M1, M2,
			verbose_level);
	if (f_v) {
		cout << "modified_group_create::create_product_action "
				"after AG.init_direct_product_group_and_restrict" << endl;
	}

	A_modified->f_is_linear = false;
	f_has_strong_generators = false;

	actions::action *A0;
	//groups::direct_product *P;

	A0 = A_modified->subaction;

	//P = A0->G.direct_product_group;


	if (f_v) {
		cout << "modified_group_create::create_product_action "
				"before AG.scan_generators" << endl;
	}
	Strong_gens = AG.scan_generators(
			A0,
			Descr->direct_product_subgroup_gens,
			Descr->direct_product_subgroup_order,
			verbose_level);
	if (f_v) {
		cout << "modified_group_create::create_product_action "
				"after AG.scan_generators" << endl;
	}


	f_has_strong_generators = true;





	label += "product_" + AG1->label + "_" + AG2->label;
	label_tex += "product(" + AG1->label_tex + "," + AG2->label_tex + ")";


	if (f_v) {
		cout << "modified_group_create::create_product_action "
				"done" << endl;
	}
}



void modified_group_create::create_polarity_extension(
		std::string &input_group_label,
		std::string &input_projective_space_label,
		int f_on_middle_layer_grassmannian,
		int f_on_points_and_hyperplanes,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_polarity_extension" << endl;
		cout << "modified_group_create::create_polarity_extension input_group_label = " << input_group_label << endl;
		cout << "modified_group_create::create_polarity_extension f_on_middle_layer_grassmannian = " << f_on_middle_layer_grassmannian << endl;
		cout << "modified_group_create::create_polarity_extension f_on_points_and_hyperplanes = " << f_on_points_and_hyperplanes << endl;
	}

	any_group *AG;

	AG = Get_any_group(input_group_label);

	//algebra::matrix_group *M;


	if (!AG->A->is_matrix_group()) {
		cout << "modified_group_create::create_polarity_extension "
				"the given group is not a matrix group" << endl;
		exit(1);
	}
	//M = AG->A->get_matrix_group();

	actions::action_global AGlobal;


	projective_geometry::projective_space_with_action *PA;

	PA = Get_projective_space(input_projective_space_label);

	geometry::polarity *Standard_polarity;

	Standard_polarity = PA->P->Subspaces->Standard_polarity;


	if (f_v) {
		cout << "modified_group_create::create_polarity_extension before creating extension" << endl;
	}

	if (f_on_middle_layer_grassmannian || f_on_points_and_hyperplanes) {
		if (f_v) {
			cout << "modified_group_create::create_polarity_extension "
					"before AGlobal.init_polarity_extension_group_and_restrict" << endl;
		}
		A_modified = AGlobal.init_polarity_extension_group_and_restrict(
				AG->A,
				PA->P,
				Standard_polarity,
				f_on_middle_layer_grassmannian,
				f_on_points_and_hyperplanes,
				verbose_level);
		if (f_v) {
			cout << "modified_group_create::create_polarity_extension "
					"after AGlobal.init_polarity_extension_group_and_restrict" << endl;
		}

		if (A_modified->subaction->Strong_gens == NULL) {
			cout << "modified_group_create::create_polarity_extension A_modified->subaction->Strong_gens == NULL" << endl;
			exit(1);
		}
		A_modified->Strong_gens = A_modified->subaction->Strong_gens;

	}
	else {
		if (f_v) {
			cout << "modified_group_create::create_polarity_extension "
					"before AGlobal.init_polarity_extension_group" << endl;
		}
		A_modified = AGlobal.init_polarity_extension_group(
				AG->A,
				PA->P,
				Standard_polarity,
				verbose_level);
		if (f_v) {
			cout << "modified_group_create::create_polarity_extension "
					"after AGlobal.init_polarity_extension_group" << endl;
		}
	}

	if (f_v) {
		cout << "modified_group_create::create_polarity_extension after creating extension" << endl;
	}

	// test if it has strong generators


	if (A_modified->Strong_gens == NULL) {
		cout << "modified_group_create::create_polarity_extension A_modified->Strong_gens == NULL" << endl;
		exit(1);
	}

	A_modified->f_is_linear = false;


	f_has_strong_generators = true;
	Strong_gens = A_modified->Strong_gens;

	A_base = A_modified;
	A_previous = A_modified;


	label += AG->label + "_polarity_ext";
	label_tex += AG->label_tex + " {\\rm polarity extension}";
	if (f_on_middle_layer_grassmannian) {
		label += "_on_middle_layer_grassmannian";
		label_tex += "{\\rm \\_on\\_middle\\_layer\\_grassmannian}";
	}
	if (f_on_points_and_hyperplanes) {
		label += "_on_points_and_hyperplanes";
		label_tex += "{\\rm \\_on\\_points\\_and\\_hyperplanes}";
	}

	if (f_v) {
		cout << "modified_group_create::create_polarity_extension "
				"done" << endl;
	}
}




}}}




