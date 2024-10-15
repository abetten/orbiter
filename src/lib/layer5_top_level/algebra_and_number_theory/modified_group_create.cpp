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

	else if (Descr->f_create_even_subgroup) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_even_subgroup" << endl;
		}

		create_even_subgroup(description, verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_even_subgroup" << endl;
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

	else if (Descr->f_set_stabilizer) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_set_stabilizer_subgroup" << endl;
		}

		create_set_stabilizer_subgroup(description, verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_set_stabilizer_subgroup" << endl;
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

		// output in A_modified

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

		// output in A_modified

	}
	else if (Descr->f_holomorph) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"f_holomorph" << endl;
		}

		cout << "modified_group_create::modified_group_init f_holomorph not yet implemented" << endl;
		exit(1);

#if 0
		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_holomorph" << endl;
		}
		create_holomorph(
					verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_holomorph" << endl;
		}
#endif

		// output in A_modified

	}
	else if (Descr->f_automorphism_group) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"f_automorphism_group" << endl;
		}

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_automorphism_group" << endl;
		}
		create_automorphism_group(
					verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_automorphism_group" << endl;
		}

		// output in A_modified

	}
	else if (Descr->f_subgroup_by_lattice) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"f_subgroup_by_lattice" << endl;
		}

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before create_subgroup_by_lattice" << endl;
		}
		create_subgroup_by_lattice(
				Descr->subgroup_by_lattice_orbit_index,
				verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after create_subgroup_by_lattice" << endl;
		}

		// output in A_modified

	}
	else if (Descr->f_stabilizer_of_variety) {

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"f_stabilizer_of_variety" << endl;
		}

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"before do_stabilizer_of_variety" << endl;
		}
		do_stabilizer_of_variety(
				Descr->stabilizer_of_variety_label,
				verbose_level);

		if (f_v) {
			cout << "modified_group_create::modified_group_init "
					"after do_stabilizer_of_variety" << endl;
		}

		// output in A_modified

	}




	else {
		cout << "modified_group_create::modified_group_init "
				"unknown operation" << endl;
		exit(1);

	}

	if (f_v) {

		ring_theory::longinteger_object go;

		Strong_gens->group_order(go);

		cout << "modified_group_create::modified_group_init "
				"created a group of order " << go
				<< " and degree " << A_modified->degree << endl;

	}



	if (f_v) {
		cout << "modified_group_create::modified_group_init done" << endl;
	}
}


void modified_group_create::create_restricted_action(
		group_modification_description *description,
		int verbose_level)
// output in A_modified and Strong_gens
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

	label = AG->label;
	label_tex = AG->label_tex;

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
	label = A_modified->label;
	label_tex = A_modified->label_tex;

	if (f_v) {
		cout << "modified_group_create::create_restricted_action done" << endl;
	}
}


void modified_group_create::create_action_on_k_subspaces(
		group_modification_description *description,
		int verbose_level)
// output in A_modified and Strong_gens
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

	label = AG->label;
	label_tex = AG->label_tex;


	int n;
	n = A_previous->dimension;
	// should also work in case the previou action was a wedge action

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

	M = A_previous->get_matrix_group();

	//n = M->n;
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
// output in A_modified and Strong_gens
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


	label = AG->label;
	label_tex = AG->label_tex;



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
// output in A_modified and Strong_gens
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


	label = AG->label;
	label_tex = AG->label_tex;




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
// output in A_modified and Strong_gens
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

	label = AG->label;
	label_tex = AG->label_tex;


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


void modified_group_create::create_even_subgroup(
		group_modification_description *description,
		int verbose_level)
// output in A_modified and Strong_gens
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_even_subgroup" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "modified_group_create::create_even_subgroup "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	A_base = AG->A_base;
	A_previous = AG->A;

	label = AG->label;
	label_tex = AG->label_tex;


	A_modified = A_previous; // ToDo !!



	f_has_strong_generators = true;
	if (f_v) {
		cout << "modified_group_create::create_even_subgroup "
				"before Strong_gens = AG->Subgroup_gens" << endl;
	}

	Strong_gens = NEW_OBJECT(groups::strong_generators);

	{
		actions::action *A_on_sign;
		ring_theory::longinteger_object go;


		groups::sims *Sims;


		if (f_v) {
			cout << "modified_group_create::create_even_subgroup "
					"before AG->Subgroup_gens->create_sims" << endl;
		}
		Sims = AG->Subgroup_gens->create_sims(verbose_level);
		if (f_v) {
			cout << "modified_group_create::create_even_subgroup "
					"after AG->Subgroup_gens->create_sims" << endl;
		}

		if (f_v) {
			cout << "modified_group_create::create_even_subgroup "
					"before Sims->A->Induced_action->induced_action_on_sign" << endl;
		}
		A_on_sign = Sims->A->Induced_action->induced_action_on_sign(
				Sims, verbose_level);
		if (f_v) {
			cout << "modified_group_create::create_even_subgroup "
					"after Sims->A->Induced_action->induced_action_on_sign" << endl;
		}
		A_on_sign->Kernel->group_order(go);
		if (f_v) {
			cout << "modified_group_create::create_even_subgroup "
					"kernel has order " << go << endl;
		}


		Strong_gens->init_from_sims(A_on_sign->Kernel, verbose_level);

		FREE_OBJECT(A_on_sign);
		FREE_OBJECT(Sims);
	}



	if (f_v) {
		cout << "modified_group_create::create_even_subgroup "
				"action A_modified created: ";
		A_modified->print_info();
	}


	label += "_EvenSub";
	label_tex += " {\\rm EvenSub}";



	if (f_v) {
		cout << "modified_group_create::create_even_subgroup "
				"done" << endl;
	}
}



void modified_group_create::create_point_stabilizer_subgroup(
		group_modification_description *description,
		int verbose_level)
// output in A_modified and Strong_gens
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

	label = AG->label;
	label_tex = AG->label_tex;

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

	A_modified = A_previous; // ToDo!



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

		Orb->stabilizer_any_point(
				Descr->point_stabilizer_point,
				Strong_gens, verbose_level);


		FREE_OBJECT(Orb);
	}



	if (f_v) {
		cout << "modified_group_create::create_point_stabilizer_subgroup "
				"action A_modified created: ";
		A_modified->print_info();
	}


	label += "_Stab" + std::to_string(Descr->point_stabilizer_point);
	label_tex += " {\\rm Stab " + std::to_string(Descr->point_stabilizer_point) + "}";



	if (f_v) {
		cout << "modified_group_create::create_point_stabilizer_subgroup "
				"done" << endl;
	}
}


void modified_group_create::create_set_stabilizer_subgroup(
		group_modification_description *description,
		int verbose_level)
// output in A_modified and Strong_gens
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_set_stabilizer_subgroup" << endl;
	}
	if (Descr->from.size() != 1) {
		cout << "modified_group_create::create_set_stabilizer_subgroup "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	A_base = AG->A_base;
	A_previous = AG->A;

	label = AG->label;
	label_tex = AG->label_tex;

	if (f_v) {
		cout << "modified_group_create::create_set_stabilizer_subgroup "
				"A_base=";
		A_base->print_info();
		cout << endl;
		cout << "modified_group_create::create_set_stabilizer_subgroup "
				"A_previous=";
		A_previous->print_info();
		cout << endl;
	}

	A_modified = A_previous; // ToDo !!!




#if 0
	//Strong_gens = NEW_OBJECT(groups::strong_generators);

	{
		groups::orbits_on_something *Orb;

		if (f_v) {
			cout << "modified_group_create::create_set_stabilizer_subgroup "
					"before Orbits.orbits_on_points" << endl;
		}

		orbits::orbits_global Orbits;

		Orbits.orbits_on_points(AG, Orb, verbose_level);

		if (f_v) {
			cout << "modified_group_create::create_set_stabilizer_subgroup "
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

		Lint_vec_scan(description->set_stabilizer_the_set, the_set, the_set_sz);


		canonical_set = NEW_lint(the_set_sz);
		Elt1 = NEW_int(A_base->elt_size_in_int);

		Control = Get_poset_classification_control(description->set_stabilizer_control);

		poset_classification::poset_classification *PC;

		if (f_v) {
			cout << "modified_group_create::create_set_stabilizer_subgroup "
					"before Orbits_global.orbits_on_subsets" << endl;
		}
		Orbits_global.orbits_on_subsets(
				AG,
				Control,
				PC,
				the_set_sz,
				verbose_level - 2);
		if (f_v) {
			cout << "modified_group_create::create_set_stabilizer_subgroup "
					"after Orbits_global.orbits_on_subsets" << endl;
		}


		// trace the subset:

		if (f_v) {
			cout << "modified_group_create::create_set_stabilizer_subgroup "
					"before trace_set" << endl;
		}


		local_idx = PC->trace_set(
				the_set, the_set_sz, the_set_sz,
				canonical_set, Elt1,
			verbose_level - 2);


		// Elt1 maps the_set to canonical_set.


		if (f_v) {
			cout << "modified_group_create::create_set_stabilizer_subgroup "
					"after trace_set local_idx=" << local_idx << endl;
			cout << "modified_group_create::create_set_stabilizer_subgroup "
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
			cout << "modified_group_create::create_set_stabilizer_subgroup "
					"before Group_theory_global.strong_generators_conjugate_aGav" << endl;
		}

		Group_theory_global.strong_generators_conjugate_aGav(
				stab_gens_canonical_set,
				Elt1,
				Strong_gens,
				verbose_level - 2);

		if (f_v) {
			cout << "modified_group_create::create_set_stabilizer_subgroup "
					"after Group_theory_global.strong_generators_conjugate_aGav" << endl;
		}



		FREE_OBJECT(stab_gens_canonical_set);
		FREE_OBJECT(PC);
		FREE_lint(canonical_set);
		FREE_int(Elt1);
	}

	f_has_strong_generators = true;

	if (f_v) {
		cout << "modified_group_create::create_set_stabilizer_subgroup "
				"strong generators created" << endl;
	}


	label += "_SetStab" + Descr->set_stabilizer_the_set;
	label_tex += " {\\rm SetStab " + Descr->set_stabilizer_the_set + "}";



	if (f_v) {
		cout << "modified_group_create::create_set_stabilizer_subgroup "
				"done" << endl;
	}
}


void modified_group_create::create_projectivity_subgroup(
		group_modification_description *description,
		int verbose_level)
// output in A_modified and Strong_gens
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_projectivity_subgroup" << endl;
	}

	any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	A_base = AG->A_base;
	A_previous = AG->A;

	label = AG->label;
	label_tex = AG->label_tex;

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

	A_modified = A_previous; // ToDo !!


	actions::action_global Action_global;

	if (f_v) {
		cout << "modified_group_create::create_projectivity_subgroup "
				"before Action_global.compute_projectivity_subgroup" << endl;
	}

	Action_global.compute_projectivity_subgroup(
			A_previous,
			Strong_gens,
			AG->Subgroup_gens,
			verbose_level);
	if (f_v) {
		cout << "modified_group_create::create_projectivity_subgroup "
				"after Action_global.compute_projectivity_subgroup" << endl;
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
// ToDo
// output in A_modified but not yet in Strong_gens

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

	label = AG->label;
	label_tex = AG->label_tex;

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

	A_modified = A_previous; // ToDo !!



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
// output in A_modified and Strong_gens
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

	label = AG->label;
	label_tex = AG->label_tex;

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
// output in A_modified and Strong_gens
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
// output in A_modified and Strong_gens
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_polarity_extension" << endl;
		cout << "modified_group_create::create_polarity_extension "
				"input_group_label = " << input_group_label << endl;
		cout << "modified_group_create::create_polarity_extension "
				"f_on_middle_layer_grassmannian = " << f_on_middle_layer_grassmannian << endl;
		cout << "modified_group_create::create_polarity_extension "
				"f_on_points_and_hyperplanes = " << f_on_points_and_hyperplanes << endl;
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
		cout << "modified_group_create::create_polarity_extension "
				"before creating extension" << endl;
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
			cout << "modified_group_create::create_polarity_extension "
					"A_modified->subaction->Strong_gens == NULL" << endl;
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
		cout << "modified_group_create::create_polarity_extension "
				"after creating extension" << endl;
	}

	// test if it has strong generators


	if (A_modified->Strong_gens == NULL) {
		cout << "modified_group_create::create_polarity_extension "
				"A_modified->Strong_gens == NULL" << endl;
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


void modified_group_create::create_automorphism_group(
		int verbose_level)
// output in A_modified and Strong_gens
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_automorphism_group" << endl;
	}

	if (Descr->from.size() != 1) {
		cout << "modified_group_create::create_automorphism_group "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	A_base = AG->A_base;
	A_previous = AG->A;

	label = AG->label + "_aut";
	label_tex = AG->label_tex + "\\_aut";

	if (f_v) {
		cout << "modified_group_create::create_automorphism_group "
				"A_base=";
		A_base->print_info();
		cout << endl;
		cout << "modified_group_create::create_automorphism_group "
				"A_previous=";
		A_previous->print_info();
		cout << endl;
	}


	//actions::action_global AGlobal;


	groups::sims *Sims;


	if (f_v) {
		cout << "modified_group_create::create_automorphism_group "
				"before AG->Subgroup_gens->create_sims" << endl;
	}
	Sims = AG->Subgroup_gens->create_sims(verbose_level);
	if (f_v) {
		cout << "modified_group_create::create_automorphism_group "
				"after AG->Subgroup_gens->create_sims" << endl;
	}



	//A_modified = NEW_OBJECT(actions::action);
	interfaces::magma_interface Magma;



	//int *Table, int group_order, int *gens, int nb_gens,

	actions::action *A_perm;

	data_structures_groups::group_table_and_generators *Table;

	Table = NEW_OBJECT(data_structures_groups::group_table_and_generators);


	if (f_v) {
		cout << "modified_group_create::create_automorphism_group "
				"before Table->init" << endl;
	}

	Table->init(
			Sims,
			AG->Subgroup_gens->gens,
			verbose_level);

	if (f_v) {
		cout << "modified_group_create::create_automorphism_group "
				"after Table->init" << endl;
	}


	if (f_v) {
		cout << "modified_group_create::create_automorphism_group "
				"before Magma.compute_automorphism_group_from_group_table" << endl;
	}


	Magma.compute_automorphism_group_from_group_table(
			label,
		Table,
		A_perm,
		Strong_gens /* Aut_gens */,
		verbose_level);

	if (f_v) {
		cout << "modified_group_create::create_automorphism_group "
				"after Magma.compute_automorphism_group_from_group_table" << endl;
	}


	FREE_OBJECT(Table);

	ring_theory::longinteger_object Aut_order;

	f_has_strong_generators = true;
	Strong_gens->group_order(Aut_order);

	A_modified = A_perm;
	A_modified->f_is_linear = false;

	if (f_v) {
		cout << "modified_group_create::create_automorphism_group "
				"order of automorphism group = " << Aut_order << endl;
	}

	A_base = A_modified;
	A_previous = A_modified;

	FREE_OBJECT(Sims);


	if (f_v) {
		cout << "modified_group_create::create_automorphism_group done" << endl;
	}
}

void modified_group_create::create_subgroup_by_lattice(
		int orbit_index,
		int verbose_level)
// output in A_modified and Strong_gens
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::create_subgroup_by_lattice" << endl;
	}

	if (Descr->from.size() != 1) {
		cout << "modified_group_create::create_subgroup_by_lattice "
				"need exactly one argument of type -from" << endl;
		exit(1);
	}

	any_group *AG;

	AG = Get_any_group(Descr->from[0]);

	A_base = AG->A_base;
	A_previous = AG->A;





	label = AG->label + "_subgroup_by_lattice_" + std::to_string(orbit_index);
	label_tex = AG->label_tex + "\\_subgroup\\_by\\_lattice\\_" + std::to_string(orbit_index);
	if (f_v) {
		cout << "modified_group_create::create_subgroup_by_lattice label = " << label << endl;
		cout << "modified_group_create::create_subgroup_by_lattice label_tex = " << label_tex << endl;
	}

	if (f_v) {
		cout << "modified_group_create::create_subgroup_by_lattice "
				"A_base=";
		A_base->print_info();
		cout << endl;
		cout << "modified_group_create::create_subgroup_by_lattice "
				"A_previous=";
		A_previous->print_info();
		cout << endl;
	}

	if (!AG->f_has_class_data) {
		cout << "modified_group_create::create_subgroup_by_lattice "
				"the subgroup lattice has not been computed yet" << endl;
		exit(1);
	}

	if (orbit_index >= AG->class_data->nb_classes) {
		cout << "modified_group_create::create_subgroup_by_lattice orbit_index is out of range" << endl;
		exit(1);
	}

	f_has_strong_generators = true;

	groups::strong_generators *Strong_gens_temp;
	Strong_gens_temp = AG->class_data->Conjugacy_class[orbit_index]->gens->create_copy(verbose_level - 4);

	actions::action_global Action_global;

	A_modified = Action_global.init_subgroup_from_strong_generators(
			AG->A_base,
			Strong_gens_temp,
			verbose_level - 1);

	A_modified->label = label;
	A_modified->label_tex = label_tex;

	if (f_v) {
		cout << "modified_group_create::create_subgroup_by_lattice A_modified->label = " << A_modified->label << endl;
		cout << "modified_group_create::create_subgroup_by_lattice A_modified->label_tex = " << A_modified->label_tex << endl;
	}


	// Strong_gens should be in the new action.

	f_has_strong_generators = true;
	Strong_gens = A_modified->Strong_gens->create_copy(verbose_level - 4);
	//Strong_gens = AG->class_data->Conjugacy_class[orbit_index]->gens->create_copy(verbose_level - 4);

	if (f_v) {
		cout << "modified_group_create::create_subgroup_by_lattice done" << endl;
	}
}

void modified_group_create::do_stabilizer_of_variety(
		std::string &variety_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "modified_group_create::do_stabilizer_of_variety" << endl;
	}

	canonical_form::variety_object_with_action *Input_Variety;

	Input_Variety = Get_variety(variety_label);

	if (f_v) {
		cout << "modified_group_create::do_stabilizer_of_variety "
				"Input_Variety = " << Input_Variety->Variety_object->label_txt << endl;
	}

	std::string fname_base;

	fname_base = Input_Variety->Variety_object->label_txt + "_c";


	canonical_form::canonical_form_classifier *Classifier;

	Classifier = NEW_OBJECT(canonical_form::canonical_form_classifier);

	if (f_v) {
		cout << "modified_group_create::do_stabilizer_of_variety "
				"before getting PA" << endl;
	}
	projective_geometry::projective_space_with_action *PA = Input_Variety->PA;
	if (f_v) {
		cout << "modified_group_create::do_stabilizer_of_variety "
				"after getting PA" << endl;
	}

	if (f_v) {
		cout << "modified_group_create::do_stabilizer_of_variety "
				"before getting Poly_ring" << endl;
	}
	ring_theory::homogeneous_polynomial_domain *Poly_ring = Input_Variety->Variety_object->Ring;
	if (f_v) {
		cout << "modified_group_create::do_stabilizer_of_variety "
				"after getting Poly_ring" << endl;
	}


	if (f_v) {
		cout << "modified_group_create::do_stabilizer_of_variety "
				"before Classifier->init_direct" << endl;
	}

	Classifier->init_direct(
			PA,
			Poly_ring,
			1 /*nb_input_Vo*/,
			Input_Variety,
			fname_base,
			verbose_level);

	if (f_v) {
		cout << "modified_group_create::do_stabilizer_of_variety "
				"after Classifier->init_direct" << endl;
	}



	canonical_form::canonical_form_global Canonical_form_global;


	Canonical_form_global.compute_group_and_tactical_decomposition(
			Classifier,
			Input_Variety,
			fname_base,
			verbose_level);



#if 0
	geometry::projective_space *Projective_space;

	ring_theory::homogeneous_polynomial_domain *Ring;


	std::string label_txt;
	std::string label_tex;


#if 0
	std::string eqn_txt;

	int f_second_equation;
	std::string eqn2_txt;
#endif


	int *eqn; // [Ring->get_nb_monomials()]
	//int *eqn2; // [Ring->get_nb_monomials()]


	// the partition into points and lines
	// must be invariant under the group.
	// must be sorted if find_point() or identify_lines() is invoked.

	data_structures::set_of_sets *Point_sets;

	data_structures::set_of_sets *Line_sets;

	int f_has_singular_points;
	std::vector<long int> Singular_points;
#endif

#if 0
	projective_geometry::projective_space_with_action *PA;

	int cnt;
	int po_go;
	int po_index;
	int po;
	int so;

	int f_has_nauty_output;
	int nauty_output_index_start;
	std::vector<std::string> Carrying_through;

	algebraic_geometry::variety_object *Variety_object;

	int f_has_automorphism_group;
	groups::strong_generators *Stab_gens;

	apps_combinatorics::variety_with_TDO_and_TDA *TD;
#endif


	//Input_Vo[0].Stab_gens;


	A_base = PA->A;
	A_previous = PA->A;





	label = PA->A->label + "_stab_of_" + Input_Variety->Variety_object->label_txt;
	label_tex = PA->A->label_tex + "\\_stab\\_of\\_" + Input_Variety->Variety_object->label_tex;
	if (f_v) {
		cout << "modified_group_create::do_stabilizer_of_variety label = " << label << endl;
		cout << "modified_group_create::do_stabilizer_of_variety label_tex = " << label_tex << endl;
	}

	if (f_v) {
		cout << "modified_group_create::do_stabilizer_of_variety "
				"A_base=";
		A_base->print_info();
		cout << endl;
		cout << "modified_group_create::do_stabilizer_of_variety "
				"A_previous=";
		A_previous->print_info();
		cout << endl;
	}

	f_has_strong_generators = true;

	groups::strong_generators *Strong_gens_temp;
	Strong_gens_temp = Input_Variety->Stab_gens->create_copy(verbose_level - 4);

	actions::action_global Action_global;

	A_modified = Action_global.init_subgroup_from_strong_generators(
			A_base,
			Strong_gens_temp,
			verbose_level - 1);

	A_modified->label = label;
	A_modified->label_tex = label_tex;

	if (f_v) {
		cout << "modified_group_create::do_stabilizer_of_variety A_modified->label = " << A_modified->label << endl;
		cout << "modified_group_create::do_stabilizer_of_variety A_modified->label_tex = " << A_modified->label_tex << endl;
	}


	// Strong_gens should be in the new action.

	f_has_strong_generators = true;
	Strong_gens = A_modified->Strong_gens->create_copy(verbose_level - 4);
	//Strong_gens = AG->class_data->Conjugacy_class[orbit_index]->gens->create_copy(verbose_level - 4);



	if (f_v) {
		cout << "modified_group_create::do_stabilizer_of_variety done" << endl;
	}
}



}}}




