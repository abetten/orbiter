/*
 * induced_action.cpp
 *
 *  Created on: Feb 1, 2023
 *      Author: betten
 */


#include "layer1_foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {



induced_action::induced_action()
{
	A_old = NULL;
}

induced_action::~induced_action()
{
}

void induced_action::init(
		action *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "induced_action::init" << endl;
	}

	induced_action::A_old = A;

	if (f_v) {
		cout << "induced_action::init done" << endl;
	}
}



action *induced_action::induced_action_on_interior_direct_product(
		int nb_rows,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_interior_direct_product *IDP;
	action *A;

	if (f_v) {
		cout << "induced_action::induced_action_on_interior_direct_product" << endl;
	}
	A = NEW_OBJECT(action);



	A->label = A_old->label + "_on_interior_direct_product_" + std::to_string(A_old->degree) + "_" + std::to_string(nb_rows);
	A->label_tex = A_old->label_tex + " {\\rm OnIntDirectProduct}_{" + std::to_string(A_old->degree) + "," + std::to_string(nb_rows) + "}";



	if (f_v) {
		cout << "the old_action " << A_old->label
				<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}
	A->f_has_subaction = true;
	A->subaction = A_old;
	IDP = NEW_OBJECT(induced_actions::action_on_interior_direct_product);

	IDP->init(A_old, nb_rows, verbose_level);

	A->type_G = action_on_interior_direct_product_t;
	A->G.OnInteriorDirectProduct = IDP;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;
	A->low_level_point_size = 0;

	A->f_has_strong_generators = false;

	A->degree = IDP->degree;
	//A->base_len = 0;
	if (f_v) {
		cout << "induced_action::induced_action_on_interior_direct_product "
				"before init_function_pointers_induced_action" << endl;
	}
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();



	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	if (f_v) {
		cout << "induced_action::induced_action_on_interior_direct_product "
				"before allocate_element_data" << endl;
	}
	A->Group_element->allocate_element_data();


	if (f_v) {
		cout << "induced_action::induced_action_on_interior_direct_product "
				"finished, created action " << A->label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		A->print_info();
	}
	return A;
}


action *induced_action::induced_action_on_set_partitions(
		int partition_class_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_set_partitions *OSP;
	action *A;

	if (f_v) {
		cout << "induced_action::induced_action_on_set_partitions" << endl;
	}
	A = NEW_OBJECT(action);


	A->label = A_old->label + "_on_set_partitions_" + std::to_string(A_old->degree) + "_" + std::to_string(partition_class_size);
	A->label_tex = A_old->label_tex + " {\\rm OnSetPart}_{" + std::to_string(A_old->degree) + "," + std::to_string(partition_class_size) + "}";





	if (f_v) {
		cout << "the old_action " << A_old->label
				<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}
	A->f_has_subaction = true;
	A->subaction = A_old;
	OSP = NEW_OBJECT(induced_actions::action_on_set_partitions);

	OSP->init(partition_class_size,
			A_old, verbose_level);
	A->type_G = action_on_set_partitions_t;
	A->G.OnSetPartitions = OSP;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;
	A->low_level_point_size = 0;

	A->f_has_strong_generators = false;

	A->degree = OSP->nb_set_partitions;
	//A->base_len = 0;
	if (f_v) {
		cout << "induced_action::induced_action_on_set_partitions "
				"before init_function_pointers_induced_action" << endl;
	}
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();



	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	if (f_v) {
		cout << "induced_action::induced_action_on_set_partitions "
				"before allocate_element_data" << endl;
	}
	A->Group_element->allocate_element_data();


	if (f_v) {
		cout << "induced_action::induced_action_on_set_partitions "
				"finished, created action " << A->label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		A->print_info();
	}
	return A;
}


#if 0
void action::init_action_on_lines(
		action *A,
		field_theory::finite_field *F, int n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	induced_actions::action_on_grassmannian *A_lines;
	geometry::grassmann *Grass_lines;

	if (f_v) {
		cout << "induced_action::init_action_on_lines" << endl;
		}

	A_lines = NEW_OBJECT(induced_actions::action_on_grassmannian);

	Grass_lines = NEW_OBJECT(geometry::grassmann);


	if (f_v) {
		cout << "induced_action::init_action_on_lines "
				"before Grass_lines->init" << endl;
		}
	Grass_lines->init(n, 2, F, verbose_level - 2);

	if (f_v) {
		cout << "induced_action::init_action_on_lines "
				"before A_lines->init" << endl;
		}
	A_lines->init(*A, Grass_lines, verbose_level - 5);


	if (f_v) {
		cout << "induced_action::init_action_on_lines "
				"action on grassmannian established" << endl;
		}

	if (f_v) {
		cout << "induced_action::init_action_on_lines "
				"initializing A2" << endl;
		}
	groups::sims S;
	ring_theory::longinteger_object go1;

	S.init(A, verbose_level - 2);
	S.init_generators(*A->Strong_gens->gens, 0/*verbose_level*/);
	S.compute_base_orbits_known_length(
			A->get_transversal_length(), 0/*verbose_level - 1*/);
	S.group_order(go1);
	if (f_v) {
		cout << "induced_action::init_action_on_lines "
				"group order " << go1 << endl;
		}

	if (f_v) {
		cout << "induced_action::init_action_on_lines "
				"initializing action on grassmannian" << endl;
		}
	induced_action_on_grassmannian(
			A, A_lines,
		true /* f_induce_action */, &S,
		verbose_level);
	if (f_v) {
		cout << "induced_action::init_action_on_lines "
				"after induced_action_on_grassmannian" << endl;
		}
	if (f_vv) {
		print_info();
		}

	if (f_v) {
		cout << "induced_action::init_action_on_lines done" << endl;
		}
}
#endif

action *induced_action::induced_action_by_representation_on_conic(
	int f_induce_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "induced_action::induced_action_by_representation_on_conic "
				"f_induce_action=" << f_induce_action << endl;
		}

	action *A;

	A = NEW_OBJECT(action);
	induced_actions::action_by_representation *Rep;


	A->label = A_old->label + "_OnConic";
	A->label_tex = A_old->label_tex + " {\\rm OnConic}";




	if (f_v) {
		cout << "the old_action " << A_old->label
				<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}
	A->f_has_subaction = true;
	A->subaction = A_old;
	if (A_old->type_G != matrix_group_t) {
		cout << "induced_action::induced_action_by_representation_on_conic "
				"action not of matrix group type" << endl;
		cout << "symmetry group type: ";
		action_global AG;
		AG.action_print_symmetry_group_type(
					cout, A_old->type_G);
		cout << endl;
		exit(1);
	}
	//M = A->G.matrix_grp;

	Rep = NEW_OBJECT(induced_actions::action_by_representation);
	Rep->init_action_on_conic(A_old, verbose_level);

	A->type_G = action_by_representation_t;
	A->G.Rep = Rep;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;
	A->low_level_point_size = Rep->low_level_point_size;

	A->f_has_strong_generators = false;

	A->degree = Rep->degree;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();



	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (f_induce_action) {

		action_global AG;

		if (f_v) {
			cout << "induced_action::induced_action_by_representation_on_conic "
					"before AG.induced_action_override_sims" << endl;
		}
		AG.induced_action_override_sims(A_old, A, old_G, 0/*verbose_level - 2*/);
		if (f_v) {
			cout << "induced_action::induced_action_by_representation_on_conic "
					"after AG.induced_action_override_sims" << endl;
		}
	}

	if (f_v) {
		cout << "induced_action::induced_action_by_representation_on_conic "
				"finished, created action " << A->label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		A->print_info();
	}
	return A;
}


action *induced_action::induced_action_on_cosets(
		induced_actions::action_on_cosets *A_on_cosets,
	int f_induce_action,
	groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;

	if (f_v) {
		cout << "induced_action::induced_action_on_cosets "
				"f_induce_action=" << f_induce_action << endl;
	}

	A = NEW_OBJECT(action);


	A->label = A_old->label + "_Cosets_" + std::to_string(A_on_cosets->dimension_of_subspace);
	A->label_tex = A_old->label_tex + " {\\rm OnCosets}_{" + std::to_string(A_on_cosets->dimension_of_subspace) + "}";



	if (f_v) {
		cout << "the old_action is " << A_old->label << endl;
		//		<< " has base_length = " << A->base_len()
		//	<< " and degree " << A->degree << endl;
	}
	A->f_has_subaction = true;
	A->subaction = A_old;
	if (!A_old->f_is_linear) {
		cout << "induced_action::induced_action_on_cosets "
				"action not of linear type" << endl;
		exit(1);
	}
	A->type_G = action_on_cosets_t;
	A->G.OnCosets = A_on_cosets;
	A->f_allocated = false;
	A->make_element_size = A_old->make_element_size;
	A->low_level_point_size = A_old->low_level_point_size;

	A->f_has_strong_generators = false;

	A->degree = A_on_cosets->nb_points;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();



	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (f_induce_action) {

		action_global AG;

		if (f_v) {
			cout << "induced_action::induced_action_on_cosets "
					"before AG.induced_action_override_sims" << endl;
		}
		AG.induced_action_override_sims(
				A_old, A, old_G, verbose_level - 2);
		if (f_v) {
			cout << "induced_action::induced_action_on_cosets "
					"after AG.induced_action_override_sims" << endl;
		}
	}

	if (f_v) {
		cout << "induced_action::induced_action_on_cosets "
				"finished, created action " << A->label
				<< " of degree=" << A->degree << endl;
		A->print_info();
	}
	return A;
}


action *induced_action::induced_action_on_factor_space(
		induced_actions::action_on_factor_space *AF,
	int f_induce_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;

	if (f_v) {
		cout << "induced_action::induced_action_on_factor_space "
				"f_induce_action=" << f_induce_action << endl;
	}
	A = NEW_OBJECT(action);


	A->label = A_old->label + "_Factor_" + std::to_string(AF->VS->dimension) + "_" + std::to_string(AF->factor_space_len);
	A->label_tex = A_old->label_tex + " {\\rm OnFactor}_{" + std::to_string(AF->VS->dimension) + "," + std::to_string(AF->factor_space_len) + "}";


	if (f_v) {
		cout << "the old_action " << A_old->label
				<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}
	A->f_has_subaction = true;
	A->subaction = A_old;
	if (!A_old->f_is_linear) {
		cout << "induced_action::induced_action_on_factor_space "
				"action not of linear type" << endl;
		cout << "the old action is:" << endl;
		A_old->print_info();
		exit(1);
	}

	A->type_G = action_on_factor_space_t;
	A->G.AF = AF;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;
	A->low_level_point_size = A_old->low_level_point_size;

	A->f_has_strong_generators = false;

	A->degree = AF->degree;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();



	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (f_induce_action) {

		action_global AG;

		if (f_v) {
			cout << "induced_action::induced_action_on_factor_space "
					"before AG.induced_action_override_sims" << endl;
		}
		AG.induced_action_override_sims(A_old, A, old_G, verbose_level - 2);
		if (f_v) {
			cout << "induced_action::induced_action_on_factor_space "
					"after AG.induced_action_override_sims" << endl;
		}
	}

	if (f_v) {
		cout << "induced_action::induced_action_on_factor_space "
				"finished, created action " << A->label << endl;
		cout << "degree=" << A->degree << endl;
		A->print_info();
	}
	return A;
}

action *induced_action::induced_action_on_grassmannian(
		int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_grassmannian *AG;
	action *A;
	algebra::matrix_group *M;

	if (f_v) {
		cout << "induced_action::induced_action_on_grassmannian" << endl;
	}
	A = NEW_OBJECT(action);

	A->label = A_old->label + "_Gr_" + std::to_string(k);
	A->label_tex = A_old->label_tex + " {\\rm OnGr}_{" + std::to_string(k) + "}";



	if (f_v) {
		cout << "the old_action " << A_old->label
				<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}
	A->f_has_subaction = true;
	A->subaction = A_old;
	if (A_old->type_G != matrix_group_t) {
		cout << "induced_action::induced_action_on_grassmannian "
				"old action not of matrix group type" << endl;
		cout << "symmetry group type: ";
		action_global AG;
		AG.action_print_symmetry_group_type(
					cout, A_old->type_G);
		cout << endl;
		exit(1);
	}
	M = A_old->G.matrix_grp;
	AG = NEW_OBJECT(induced_actions::action_on_grassmannian);

	geometry::grassmann *Gr;

	Gr = NEW_OBJECT(geometry::grassmann);
	Gr->init(M->n, k, M->GFq, verbose_level);
	AG->init(*A_old, Gr, verbose_level);
	A->type_G = action_on_grassmannian_t;
	A->G.AG = AG;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;
	A->low_level_point_size = AG->low_level_point_size;

	A->f_has_strong_generators = false;

	A->degree = AG->degree_as_text.as_int();
	//A->base_len = 0;
	if (f_v) {
		cout << "induced_action::induced_action_on_grassmannian "
				"before init_function_pointers_induced_action" << endl;
	}
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();



	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	if (f_v) {
		cout << "induced_action::induced_action_on_grassmannian "
				"before A->allocate_element_data" << endl;
	}
	A->Group_element->allocate_element_data();


	if (f_v) {
		cout << "induced_action::induced_action_on_grassmannian "
				"finished, created action " << A->label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		A->print_info();
	}
	return A;
}


action *induced_action::induced_action_on_grassmannian_preloaded(
		induced_actions::action_on_grassmannian *AG,
	int f_induce_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;

	if (f_v) {
		cout << "induced_action::induced_action_on_grassmannian_preloaded "
				"f_induce_action=" << f_induce_action << endl;
	}
	A = NEW_OBJECT(action);

	A->label = A_old->label + "_Gr_" + std::to_string(AG->n) + "_" + std::to_string(AG->k);
	A->label_tex = A_old->label_tex + " {\\rm OnGr}_{" + std::to_string(AG->n) + "," + std::to_string(AG->k) + "}";


	if (f_v) {
		cout << "induced_action::induced_action_on_grassmannian_preloaded "
			"the old_action " << A_old->label
			<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}
	A->f_has_subaction = true;
	if (f_v) {
		cout << "induced_action::induced_action_on_grassmannian_preloaded "
				"before subaction = A_old" << endl;
	}
	A->subaction = A_old;
	if (!A_old->f_is_linear) {
		cout << "induced_action::induced_action_on_grassmannian_preloaded "
				"action not of linear type" << endl;
		cout << "symmetry group type: ";
		action_global AG;
		AG.action_print_symmetry_group_type(
					cout, A_old->type_G);
		cout << endl;
		exit(1);
	}

	if (f_v) {
		cout << "induced_action::induced_action_on_grassmannian_preloaded "
				"action is of linear type" << endl;
	}
	A->type_G = action_on_grassmannian_t;
	A->G.AG = AG;
	A->f_allocated = false;
	A->make_element_size = A_old->make_element_size;
	A->low_level_point_size = AG->low_level_point_size;

	A->f_has_strong_generators = false;

	A->degree = AG->degree_as_text.as_int();

	if (f_v) {
		cout << "induced_action::induced_action_on_grassmannian_preloaded "
				"before init_function_pointers_induced_action" << endl;
	}
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();
	if (f_v) {
		cout << "induced_action::induced_action_on_grassmannian_preloaded "
				"after init_function_pointers_induced_action" << endl;
	}



	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	if (f_v) {
		cout << "induced_action::induced_action_on_grassmannian_preloaded "
				"before allocate_element_data" << endl;
	}
	A->Group_element->allocate_element_data();
	if (f_v) {
		cout << "induced_action::induced_action_on_grassmannian_preloaded "
				"after allocate_element_data" << endl;
	}

	if (f_induce_action) {

		action_global AG;

		if (f_v) {
			cout << "induced_action::induced_action_on_grassmannian_preloaded "
					"before AG.induced_action_override_sims" << endl;
		}
		AG.induced_action_override_sims(A_old, A, old_G, verbose_level);
		if (f_v) {
			cout << "action::induced_action_on_grassmannian_preloaded "
					"after AG.induced_action_override_sims" << endl;
		}
	}

	if (f_v) {
		cout << "induced_action::induced_action_on_grassmannian_preloaded finished, "
				"created action " << A->label << endl;
		cout << "induced_action::induced_action_on_grassmannian_preloaded "
				"degree=" << A->degree << endl;
		cout << "induced_action::induced_action_on_grassmannian_preloaded "
				"make_element_size=" << A->make_element_size << endl;
		cout << "induced_action::induced_action_on_grassmannian_preloaded "
				"low_level_point_size=" << A->low_level_point_size << endl;
		A->print_info();
		cout << "induced_action::induced_action_on_grassmannian_preloaded finished, "
				"after print_info" << endl;
	}
	if (f_v) {
		cout << "induced_action::induced_action_on_grassmannian_preloaded done" << endl;
	}
	return A;
}

action *induced_action::induced_action_on_spread_set(
		induced_actions::action_on_spread_set *AS,
	int f_induce_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;

	if (f_v) {
		cout << "induced_action::induced_action_on_spread_set "
				"f_induce_action=" << f_induce_action << endl;
	}

	A = NEW_OBJECT(action);

	A->label = A_old->label + "_SpreadSet_" + std::to_string(AS->k) + "_" + std::to_string(AS->q);
	A->label_tex = A_old->label_tex + " {\\rm OnSpreadSet}_{" + std::to_string(AS->k) + "," + std::to_string(AS->q) + "}";



	if (f_v) {
		cout << "the old_action " << A_old->label
			<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}
	A->f_has_subaction = true;
	A->subaction = A_old;
	if (!A_old->f_is_linear) {
		cout << "induced_action::induced_action_on_spread_set "
				"action not of linear type" << endl;
		cout << "symmetry group type: ";
		action_global AG;
		AG.action_print_symmetry_group_type(
					cout, A_old->type_G);
		cout << endl;
		exit(1);
	}

	A->f_is_linear = true;


	A->type_G = action_on_spread_set_t;
	A->G.AS = AS;
	A->f_allocated = false;
	A->make_element_size = A_old->make_element_size;
	A->low_level_point_size = AS->low_level_point_size;

	A->f_has_strong_generators = false;

	A->degree = AS->degree;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();



	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (f_induce_action) {
		action_global AG;


		if (f_v) {
			cout << "induced_action::induced_action_on_spread_set "
					"before AG.induced_action_override_sims" << endl;
		}
		AG.induced_action_override_sims(
				A_old, A,
					old_G, 0/*verbose_level - 2*/);
		if (f_v) {
			cout << "induced_action::induced_action_on_spread_set "
					"after AG.induced_action_override_sims" << endl;
		}
	}


	if (f_v) {
		cout << "induced_action::induced_action_on_spread_set finished, "
				"created action " << A->label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		A->print_info();
	}
	return A;
}



action *induced_action::induced_action_on_wedge_product(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	//groups::matrix_group *M;

	if (f_v) {
		cout << "induced_action::induced_action_on_wedge_product" << endl;
	}
	A = NEW_OBJECT(action);


	A->label = A_old->label + "_Wedge";

	A->label_tex = A_old->label_tex + " {\\rm OnWedge}";


	if (f_v) {
		cout << "the old_action " << A_old->label
				<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}
	A->f_has_subaction = true;
	A->subaction = A_old;
	if (A_old->type_G != matrix_group_t) {
		cout << "induced_action::induced_action_on_wedge_product "
				"old action not of matrix group type" << endl;
		cout << "symmetry group type: ";
		action_global AG;
		AG.action_print_symmetry_group_type(
					cout, A_old->type_G);
		cout << endl;
		exit(1);
	}
	//M = G.matrix_grp;

	induced_actions::action_on_wedge_product *AW;

	AW = NEW_OBJECT(induced_actions::action_on_wedge_product);




	if (f_v) {
		cout << "induced_action::induced_action_on_wedge_product "
				"before AW->init" << endl;
	}
	AW->init(A_old, verbose_level);
	if (f_v) {
		cout << "induced_action::induced_action_on_wedge_product "
				"after AW->init" << endl;
	}

	A->type_G = action_on_wedge_product_t;
	A->G.AW = AW;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;
	A->low_level_point_size = AW->low_level_point_size;

	A->f_has_strong_generators = false;

	A->degree = AW->degree;
	//A->base_len = 0;
	if (f_v) {
		cout << "induced_action::induced_action_on_wedge_product "
				"before init_function_pointers_induced_action" << endl;
	}
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();



	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->f_is_linear = true;
	A->dimension = AW->wedge_dimension;

	if (f_v) {
		cout << "induced_action::induced_action_on_wedge_product "
				"before A->allocate_element_data" << endl;
	}
	A->Group_element->allocate_element_data();


	if (f_v) {
		cout << "induced_action::induced_action_on_wedge_product "
				"finished, created action " << A->label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		A->print_info();
	}
	return A;
}


action *induced_action::induced_action_on_Galois_group(
		groups::sims *old_G, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "induced_action::induced_action_on_Galois_group" << endl;
		}
	induced_actions::action_on_galois_group *AG;
	action *A;
	algebra::matrix_group *M;

	A = NEW_OBJECT(action);

	if (A_old != old_G->A) {
		cout << "induced_action::induced_action_on_Galois_group A_old != old_G->A" << endl;
		cout << "A_old = ";
		A_old->print_info();
		cout << endl;
		cout << "old_G->A = ";
		old_G->A->print_info();
		cout << endl;
		exit(1);
	}


	A->label = A_old->label + "_gal";
	A->label_tex = A_old->label_tex + " {\\rm OnGal}";

	if (f_v) {
		cout << "the old_action " << A_old->label
			<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
		}
	A->f_has_subaction = true;
	A->subaction = A_old;
	if (A_old->type_G != matrix_group_t) {
		cout << "action::induced_action_on_Galois_group "
				"action not of matrix group type" << endl;
		exit(1);
		}
	M = A_old->G.matrix_grp;

	AG = NEW_OBJECT(induced_actions::action_on_galois_group);
	if (f_v) {
		cout << "induced_action::induced_action_on_Galois_group "
				"before AG->init" << endl;
	}
	AG->init(A_old, M->n, verbose_level);
	if (f_v) {
		cout << "induced_action::induced_action_on_Galois_group "
				"after AG->init" << endl;
	}

	A->type_G = action_on_galois_group_t;
	A->G.on_Galois_group = AG;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;

	A->f_has_strong_generators = false;

	A->degree = AG->degree;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();

	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A_old, 0, verbose_level);
	//allocate_base_data(0);


	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();


	action_global AGl;

	if (f_v) {
		cout << "induced_action::induced_action_on_Galois_group "
				"before AGl.induced_action_override_sims" << endl;
	}
	AGl.induced_action_override_sims(A_old, A, old_G, verbose_level - 2);
	if (f_v) {
		cout << "induced_action::induced_action_on_Galois_group "
				"after AGl.induced_action_override_sims" << endl;
	}
	if (f_v) {
		cout << "induced_action::induced_action_on_Galois_group "
				"finished, created action " << A->label << endl;
		A->print_info();
	}
	return A;
}


action *induced_action::induced_action_on_determinant(
		groups::sims *old_G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_determinant *AD;
	action *A;
	algebra::matrix_group *M;

	if (f_v) {
		cout << "induced_action::induced_action_on_determinant" << endl;
	}
	A = NEW_OBJECT(action);

	ring_theory::longinteger_object go1;
	ring_theory::longinteger_object go2;

	old_G->group_order(go1);
	if (f_v) {
		cout << "induced_action::induced_action_on_determinant "
				"old group order = " << go1 << endl;
	}

	A->label = A_old->label + "_det";
	A->label_tex = A_old->label_tex + " {\\rm OnDet}";


	if (f_v) {
		cout << "the old_action " << A_old->label
				<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}
	A->f_has_subaction = true;
	A->subaction = A_old;
	if (A_old->type_G != matrix_group_t) {
		cout << "induced_action::induced_action_on_determinant "
				"action not of matrix group type" << endl;
		cout << "symmetry group type: ";
		action_global AG;
		AG.action_print_symmetry_group_type(
					cout, A_old->type_G);
		cout << endl;
		exit(1);
	}
	M = A_old->G.matrix_grp;
	AD = NEW_OBJECT(induced_actions::action_on_determinant);

	if (f_v) {
		cout << "induced_action::induced_action_on_determinant "
				"before AD->init" << endl;
	}

	AD->init(*A_old, M->f_projective, M->n, verbose_level);

	if (f_v) {
		cout << "induced_action::induced_action_on_determinant "
				"after AD->init" << endl;
	}

	A->type_G = action_on_determinant_t;
	A->G.AD = AD;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;

	A->f_has_strong_generators = false;

	A->degree = AD->degree;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();

	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A_old, 0, verbose_level);
	//allocate_base_data(0);


	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();


	action_global AG;

	if (f_v) {
		cout << "induced_action::induced_action_on_determinant "
				"before AG.induced_action_override_sims" << endl;
	}
	AG.induced_action_override_sims(A_old, A, old_G, verbose_level - 2);
	if (f_v) {
		cout << "induced_action::induced_action_on_determinant "
				"after AG.induced_action_override_sims" << endl;
	}
	if (f_v) {
		cout << "induced_action::induced_action_on_determinant "
				"finished, created action " << A->label << endl;
		A->print_info();
		A->group_order(go2);
		cout << "induced_action::induced_action_on_determinant "
				"old group order = " << go1 << endl;
		cout << "induced_action::induced_action_on_determinant "
				"new group order = " << go2 << endl;
	}
	return A;
}


action *induced_action::induced_action_on_sign(
		groups::sims *old_G, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "induced_action::induced_action_on_sign" << endl;
	}

	induced_actions::action_on_sign *OnSign;
	action *A;

	A = NEW_OBJECT(action);


	A->label = A_old->label + "_OnSign";
	A->label_tex = A_old->label_tex + " {\\rm OnSign}";


	if (f_v) {
		cout << "the old_action " << A_old->label
			<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
		}
	A->f_has_subaction = true;
	A->subaction = A_old;
	OnSign = NEW_OBJECT(induced_actions::action_on_sign);
	OnSign->init(A_old, verbose_level);
	A->type_G = action_on_sign_t;
	A->G.OnSign = OnSign;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;

	A->f_has_strong_generators = false;

	A->degree = OnSign->degree;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();

	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A_old, 0, verbose_level);
	//allocate_base_data(0);


	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	action_global AG;

	if (f_v) {
		cout << "induced_action::induced_action_on_sign "
				"before AG.induced_action_override_sims" << endl;
	}
	AG.induced_action_override_sims(A_old, A, old_G, verbose_level - 2);
	if (f_v) {
		cout << "induced_action::induced_action_on_sign "
				"after AG.induced_action_override_sims" << endl;
	}
	if (f_v) {
		cout << "induced_action::induced_action_on_sign finished, "
				"created action " << A->label << endl;
		A->print_info();
	}
	return A;
}


action *induced_action::create_induced_action_by_conjugation(
		groups::sims *Base_group, int f_ownership,
		int f_basis, groups::sims *old_G,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;

	if (f_v) {
		cout << "induced_action::create_induced_action_by_conjugation" << endl;
		}
	A = NEW_OBJECT(action);


	induced_actions::action_by_conjugation *ABC;
	ring_theory::longinteger_object go;
	long int goi;

	if (f_v) {
		cout << "induced_action::create_induced_action_by_conjugation" << endl;
		cout << "the old_action " << A_old->label
				<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}
	Base_group->group_order(go);
	goi = go.as_lint();
	if (f_v) {
		cout << "induced_action::create_induced_action_by_conjugation "
				"we are acting on a group of order " << goi << endl;
	}



	A->label = A_old->label + "_Conj" + std::to_string(goi);
	A->label_tex = A_old->label_tex + " {\\rm ByConj " + std::to_string(goi) + "}";


	A->f_has_subaction = true;
	A->subaction = A_old;
	ABC = NEW_OBJECT(induced_actions::action_by_conjugation);
	ABC->init(Base_group, f_ownership, verbose_level);
	A->type_G = action_by_conjugation_t;
	A->G.ABC = ABC;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;

	A->f_has_strong_generators = false;

	A->degree = goi;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();

	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A_old, 0, verbose_level);
	//allocate_base_data(0);


	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (f_basis) {
		if (f_v) {
			cout << "induced_action::create_induced_action_by_conjugation "
					"before AG.induced_action_override_sims" << endl;
		}

		action_global AG;

		AG.induced_action_override_sims(A_old, A, old_G, verbose_level - 2);
		if (f_v) {
			cout << "induced_action::create_induced_action_by_conjugation "
					"after AG.induced_action_override_sims" << endl;
		}
	}
	if (f_v) {
		cout << "induced_action::create_induced_action_by_conjugation "
				"finished, created action " << A->label << endl;
		A->print_info();
	}


	if (f_v) {
		cout << "induced_action::create_induced_action_by_conjugation done" << endl;
	}
	return A;
}


action *induced_action::induced_action_by_right_multiplication(
	int f_basis, groups::sims *old_G,
	groups::sims *Base_group, int f_ownership,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_by_right_multiplication *ABRM;
	ring_theory::longinteger_object go;
	int goi;
	action *A;

	//A = Base_group->A;
	if (f_v) {
		cout << "induced_action::induced_action_by_right_multiplication" << endl;
		cout << "the old_action " << A_old->label
				<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}
	Base_group->group_order(go);
	goi = go.as_int();


	A = NEW_OBJECT(action);

	A->label = A_old->label + "_E" + std::to_string(goi);
	A->label_tex = A_old->label_tex + " {\\rm RightMult" + std::to_string(goi) + "}";


	if (f_v) {
		cout << "we are acting on a group of order " << goi << endl;
	}
	A->f_has_subaction = true;
	A->subaction = A_old;
	ABRM = NEW_OBJECT(induced_actions::action_by_right_multiplication);
	ABRM->init(Base_group, f_ownership, verbose_level);
	A->type_G = action_by_right_multiplication_t;
	A->G.ABRM = ABRM;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;

	A->f_has_strong_generators = false;

	A->degree = goi;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();

	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A_old, 0, verbose_level);
	//allocate_base_data(0);


	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;
	A->low_level_point_size = A_old->make_element_size;

	A->Group_element->allocate_element_data();

	if (f_basis) {
		actions::action_global AG;

		if (f_v) {
			cout << "induced_action::induced_action_by_right_multiplication "
					"before induced_action_override_sims" << endl;
		}
		AG.induced_action_override_sims(
				A_old, A, old_G, verbose_level - 2);
		if (f_v) {
			cout << "induced_action::induced_action_by_right_multiplication "
					"after induced_action_override_sims" << endl;
		}
	}
	if (f_v) {
		cout << "induced_action::induced_action_by_right_multiplication "
				"finished, created action " << A->label << endl;
		A->print_info();
	}
	return A;
}

action *induced_action::create_induced_action_on_sets(
		int nb_sets, int set_size, long int *sets,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;

	if (f_v) {
		cout << "induced_action::create_induced_action_on_sets" << endl;
	}
	if (f_v) {
		cout << "induced_action::create_induced_action_on_sets "
				"before induced_action_on_sets" << endl;
	}
	A = induced_action_on_sets(NULL,
		nb_sets, set_size, sets,
		false /*f_induce_action*/,
		verbose_level);
	if (f_v) {
		cout << "induced_action::create_induced_action_on_sets "
				"after A->induced_action_on_sets" << endl;
	}
	if (f_v) {
		cout << "induced_action::create_induced_action_on_sets done" << endl;
	}
	return A;
}


action *induced_action::induced_action_on_sets(
	groups::sims *old_G,
	int nb_sets, int set_size, long int *sets,
	int f_induce_action, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_sets *AOS;

	if (f_v) {
		cout << "induced_action::induced_action_on_sets" << endl;
		cout << "induced_action::induced_action_on_sets "
				"f_induce_action=" << f_induce_action << endl;

		cout << "induced_action::induced_action_on_sets "
				"the old_action " << A_old->label
				//<< " has base_length = " << old_action.base_len()
			<< " has degree " << A_old->degree << endl;

		cout << "induced_action::induced_action_on_sets "
				"verbose_level = " << verbose_level << endl;
	}

	action *A;

	A = NEW_OBJECT(action);

	A->label = A_old->label + "_S" + std::to_string(set_size);
	A->label_tex = A_old->label_tex + " {\\rm S" + std::to_string(set_size) + "}";


	A->f_has_subaction = true;
	A->subaction = A_old;
	if (f_v) {
		cout << "induced_action::induced_action_on_sets "
				"allocating action_on_sets" << endl;
	}
	AOS = NEW_OBJECT(induced_actions::action_on_sets);
	if (f_v) {
		cout << "induced_action::induced_action_on_sets before AOS->init" << endl;
	}
	AOS->init(nb_sets, set_size, sets, verbose_level - 1);
	if (f_v) {
		cout << "induced_action::induced_action_on_sets after AOS->init" << endl;
	}
	A->type_G = action_on_sets_t;
	A->G.on_sets = AOS;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;
	A->low_level_point_size = 0;

	A->f_has_strong_generators = false;

	A->degree = nb_sets;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();
	if (f_v) {
		cout << "induced_action::induced_action_on_sets "
				"calling allocate_base_data" << endl;
	}
	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A, 0, verbose_level); // we need to pass A here!
	//allocate_base_data(0);


	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (f_induce_action) {
		if (f_v) {
			cout << "induced_action::induced_action_on_sets "
					"before induced_action_override_sims" << endl;
		}
		action_global AG;

		AG.induced_action_override_sims(
				A_old, A,
				old_G, verbose_level /*- 2*/);
		if (f_v) {
			cout << "induced_action::induced_action_on_sets "
					"after induced_action_override_sims" << endl;
		}
	}
	if (f_v) {
		cout << "induced_action::induced_action_on_sets finished, "
				"created action " << A->label << endl;
		//Sims->print_transversal_lengths();
		//cout << endl;
		A->print_info();
	}
	if (f_v) {
		cout << "induced_action::induced_action_on_sets done" << endl;
	}
	return A;
}

action *induced_action::create_induced_action_on_subgroups(
		groups::sims *S,
		data_structures_groups::hash_table_subgroups *Hash_table_subgroups,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;

	if (f_v) {
		cout << "induced_action::create_induced_action_on_subgroups" << endl;
	}
	A = induced_action_on_subgroups(
			A_old, S,
			Hash_table_subgroups,
		verbose_level - 1);
	if (f_v) {
		cout << "induced_action::create_induced_action_on_subgroups done" << endl;
	}
	return A;
}


action *induced_action::induced_action_on_subgroups(
	action *old_action, groups::sims *S,
	data_structures_groups::hash_table_subgroups *Hash_table_subgroups,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_subgroups *AOS;

	if (f_v) {
		cout << "induced_action::induced_action_on_subgroups" << endl;
		cout << "induced_action::induced_action_on_sets "
				"the old_action " << old_action->label
				<< " has base_length = " << old_action->base_len()
			<< " and degree " << old_action->degree << endl;
		cout << "induced_action::induced_action_on_subgroups "
				"verbose_level = " << verbose_level << endl;
	}

	action *A;

	A = NEW_OBJECT(action);

	A->label = A_old->label + "_on_subgroups_nb_" + std::to_string(Hash_table_subgroups->nb_groups());
	A->label_tex = A_old->label_tex + " {\\rm OnSubgroups" + std::to_string(Hash_table_subgroups->nb_groups()) + "}";

	A->f_has_subaction = true;
	A->subaction = old_action;
	if (f_v) {
		cout << "induced_action::induced_action_on_subgroups "
				"allocating action_on_subgroups" << endl;
	}
	AOS = NEW_OBJECT(induced_actions::action_on_subgroups);
	AOS->init(old_action, S,
			Hash_table_subgroups,
			//nb_subgroups,
			//group_order, Subgroups,
			verbose_level - 1);
	if (f_v) {
		cout << "induced_action::induced_action_on_subgroups "
				"after action_on_subgroups init" << endl;
	}
	A->type_G = action_on_subgroups_t;
	A->G.on_subgroups = AOS;
	A->f_allocated = true;
	A->make_element_size = old_action->make_element_size;
	A->low_level_point_size = 0;

	A->f_has_strong_generators = false;

	A->degree = Hash_table_subgroups->nb_groups();
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();
	if (f_v) {
		cout << "induced_action::induced_action_on_subgroups "
				"calling allocate_base_data" << endl;
	}
	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A_old, 0, verbose_level);
	//allocate_base_data(0);


	A->elt_size_in_int = old_action->elt_size_in_int;
	A->coded_elt_size_in_char = old_action->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (f_v) {
		cout << "induced_action::induced_action_on_subgroups "
				"finished, created action " << A->label << endl;
		//Sims->print_transversal_lengths();
		//cout << endl;
		A->print_info();
	}
	return A;
}

action *induced_action::induced_action_by_restriction_on_orbit_with_schreier_vector(
	int f_induce_action, groups::sims *old_G,
	data_structures_groups::schreier_vector *Schreier_vector,
	int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_by_restriction *ABR;

	if (f_v) {
		cout << "induced_action::induced_action_by_restriction_on_orbit_with_schreier_vector" << endl;
		cout << "old_action ";
		A_old->print_info();
		cout << "pt = " << pt << endl;
		cout << "f_induce_action = " << f_induce_action << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	action *A;

	A = NEW_OBJECT(action);

	A->label = A_old->label + "_res_sv" + std::to_string(pt);
	A->label_tex = A_old->label_tex + " {\\rm res sv" + std::to_string(pt) + "}";

	A->f_has_subaction = true;
	A->subaction = A_old;
	ABR = NEW_OBJECT(induced_actions::action_by_restriction);

	if (f_v) {
		cout << "induced_action::induced_action_by_restriction_on_orbit_with_schreier_vector "
				"before ABR->init_single_orbit_from_schreier_vector" << endl;
	}
	ABR->init_single_orbit_from_schreier_vector(Schreier_vector, pt, verbose_level - 1);
	if (f_v) {
		cout << "induced_action::induced_action_by_restriction_on_orbit_with_schreier_vector "
				"after ABR->init_single_orbit_from_schreier_vector" << endl;
	}

	A->type_G = action_by_restriction_t;
	A->G.ABR = ABR;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;
	A->low_level_point_size = A_old->low_level_point_size;

	A->f_has_strong_generators = false;

	A->degree = ABR->nb_points;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();
	if (false) {
		cout << "induced_action::induced_action_by_restriction_on_orbit_with_schreier_vector "
				"calling allocate_base_data" << endl;
	}
	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A_old, 0, verbose_level);
	//allocate_base_data(0);


	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (f_induce_action) {
		if (f_v) {
			cout << "induced_action::induced_action_by_restriction_on_orbit_with_schreier_vector "
					"calling induced_action_override_sims" << endl;
		}
		action_global AG;

		AG.induced_action_override_sims(
				A_old, A, old_G, verbose_level - 2);
	}
	if (f_v) {
		cout << "induced_action::induced_action_by_restriction_on_orbit_with_schreier_vector "
				"finished, created action " << A->label << endl;
		//Sims->print_transversal_lengths();
		//cout << endl;
		A->print_info();
	}
	return A;
}

void induced_action::original_point_labels(
		long int *points, int nb_points,
		long int *&original_points, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "induced_action::original_point_labels" << endl;
	}
	if (A_old->type_G == action_by_restriction_t) {
		induced_actions::action_by_restriction *ABR;

		original_points = NEW_lint(nb_points);

		ABR = A_old->G.ABR;


		int i;
		long int a, b;

		for (i = 0; i < nb_points; i++) {
			a = points[i];
			b = ABR->original_point(a);
			original_points[i] = b;
		}
	}
	else {
		cout << "induced_action::original_point_labels "
				"type must be action_by_restriction_t" << endl;
		exit(1);
	}

}

action *induced_action::restricted_action(
		long int *points, int nb_points,
		std::string &label_of_set,
		std::string &label_of_set_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	induced_actions::action_by_restriction *ABR;

	if (f_v) {
		cout << "induced_action::restricted_action" << endl;
		cout << "old_action ";
		A_old->print_info();
		cout << "nb_points = " << nb_points << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}
	A = NEW_OBJECT(action);

	A->label = A_old->label + "_res_" + label_of_set;
	A->label_tex = A_old->label_tex + " {\\rm res}" + label_of_set_tex;


	A->f_has_subaction = true;
	A->subaction = A_old;
	ABR = NEW_OBJECT(induced_actions::action_by_restriction);
	ABR->init(nb_points, points, verbose_level);
	A->type_G = action_by_restriction_t;
	A->G.ABR = ABR;
	A->f_allocated = true;
	A->make_element_size =A_old->make_element_size;
	A->low_level_point_size = A_old->low_level_point_size;

	A->f_has_strong_generators = false;

	A->degree = nb_points;
	//A->base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();
	if (false) {
		cout << "induced_action::restricted_action "
				"calling allocate_base_data" << endl;
	}
	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A_old, 0, verbose_level);
	//A->allocate_base_data(0);


	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (f_v) {
		cout << "induced_action::restricted_action finished, "
				"created action " << A->label << endl;
		A->print_info();
	}
	return A;
}

action *induced_action::create_induced_action_by_restriction(
		groups::sims *old_G, int size,
		long int *set,
		std::string &label_of_set,
		std::string &label_of_set_tex,
		int f_induce,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;

	if (f_v) {
		cout << "induced_action::create_induced_action_by_restriction" << endl;
	}
	A = induced_action_by_restriction(
			A_old,
			f_induce, old_G, size, set, label_of_set, label_of_set_tex,
			verbose_level - 1);
	if (f_v) {
		cout << "induced_action::create_induced_action_by_restriction done" << endl;
	}
	return A;
}

action *induced_action::induced_action_by_restriction(
	action *old_action,
	int f_induce_action, groups::sims *old_G,
	int nb_points, long int *points,
	std::string &label_of_set,
	std::string &label_of_set_tex,
	int verbose_level)
// uses action_by_restriction data type
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_by_restriction *ABR;

	if (f_v) {
		cout << "induced_action::induced_action_by_restriction" << endl;
		cout << "old_action ";
		old_action->print_info();
		cout << "nb_points = " << nb_points << endl;
		cout << "f_induce_action = " << f_induce_action << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	action *A;

	A = NEW_OBJECT(action);



	A->label = old_action->label + "_res_" + label_of_set;
	A->label_tex = old_action->label_tex + " {\\rm res}" + label_of_set_tex;


	A->f_has_subaction = true;
	A->subaction = old_action;
	ABR = NEW_OBJECT(induced_actions::action_by_restriction);
	ABR->init(nb_points, points, verbose_level);
	A->type_G = action_by_restriction_t;
	A->G.ABR = ABR;
	A->f_allocated = true;
	A->make_element_size = old_action->make_element_size;
	A->low_level_point_size = old_action->low_level_point_size;

	A->f_has_strong_generators = false;

	A->degree = nb_points;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();
	if (false) {
		cout << "induced_action::induced_action_by_restriction "
				"calling allocate_base_data" << endl;
	}
	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A_old, 0, verbose_level);
	//allocate_base_data(0);


	A->elt_size_in_int = old_action->elt_size_in_int;
	A->coded_elt_size_in_char = old_action->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (f_induce_action) {
		if (f_v) {
			cout << "induced_action::induced_action_by_restriction "
					"calling induced_action_override_sims" << endl;
		}

		action_global AG;

		AG.induced_action_override_sims(old_action, A,
				old_G, verbose_level - 2);
	}
	if (f_v) {
		cout << "induced_action::induced_action_by_restriction "
				"finished, created action " << A->label << endl;
		//Sims->print_transversal_lengths();
		//cout << endl;
		A->print_info();
	}
	return A;
}


action *induced_action::induced_action_on_pairs(
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "induced_action::induced_action_on_pairs" << endl;
		cout << "the old_action " << A_old->label
			<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}

	action *A;


	if (f_v) {
		cout << "induced_action::induced_action_on_pairs "
				"before induced_action_on_k_subsets" << endl;
	}
	A = induced_action_on_k_subsets(
			2,
			verbose_level - 1);
	if (f_v) {
		cout << "induced_action::induced_action_on_pairs "
				"after induced_action_on_k_subsets" << endl;
	}

	A->label = A_old->label + "_on_pairs";
	A->label_tex = A_old->label_tex + " {\\rm OnPairs}";


	if (f_v) {
		cout << "induced_action::induced_action_on_pairs done" << endl;
	}
	return A;
}



action *induced_action::induced_action_on_ordered_pairs(
	groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "induced_action::induced_action_on_ordered_pairs" << endl;
		cout << "the old_action " << A_old->label
				<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}

	action *A;

	A = NEW_OBJECT(action);


	A->label = A_old->label + "_on_ordered_pairs";
	A->label_tex = A_old->label_tex + " {\\rm OnOrderedPairs}";


	A->f_has_subaction = true;
	A->subaction = A_old;
	A->type_G = action_on_ordered_pairs_t;
	A->f_allocated = false;

	A->f_has_strong_generators = false;

	A->degree = A_old->degree * (A_old->degree - 1);
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();
	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A_old, 0, verbose_level);
	//allocate_base_data(0);


	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (old_G) {

		action_global AG;

		if (f_v) {
			cout << "induced_action::induced_action_on_ordered_pairs "
					"before AG.induced_action_override_sims" << endl;
		}

		AG.induced_action_override_sims(A_old, A,
				old_G, verbose_level - 2);
		if (f_v) {
			cout << "induced_action::induced_action_on_ordered_pairs "
					"after AG.induced_action_override_sims" << endl;
		}
	}
	if (f_v) {
		cout << "induced_action::induced_action_on_ordered_pairs "
				"finished, created action " << A->label << endl;
		A->print_info();
	}
	return A;
}

action *induced_action::induced_action_on_k_subsets(
	int k,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_k_subsets *On_k_subsets;

	if (f_v) {
		cout << "induced_action::induced_action_on_k_subsets" << endl;
		cout << "the old_action " << A_old->label
				//<< " has base_length = " << old_action.base_len()
			<< " has degree " << A_old->degree << endl;
	}

	action *A;

	A = NEW_OBJECT(action);

	A->label = A_old->label + "_on_" + std::to_string(k) + "_subsets";
	A->label_tex = A_old->label_tex + " ^{[" + std::to_string(k) + "]}";

	On_k_subsets = NEW_OBJECT(induced_actions::action_on_k_subsets);
	On_k_subsets->init(A_old, k, verbose_level);


	A->f_has_subaction = true;
	A->subaction = A_old;
	A->type_G = action_on_k_subsets_t;
	A->G.on_k_subsets = On_k_subsets;
	A->f_allocated = true;


	A->f_has_strong_generators = false;

	A->degree = On_k_subsets->degree;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();
	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A_old, 0, verbose_level);
	//allocate_base_data(0);


	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (f_v) {
		cout << "induced_action::induced_action_on_k_subsets "
				"finished, created action " << A->label << endl;
		A->print_info();
	}
	return A;
}

action *induced_action::induced_action_on_orbits(
		groups::schreier *Sch, int f_play_it_safe,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_orbits *On_orbits;

	if (f_v) {
		cout << "induced_action::induced_action_on_orbits" << endl;
		cout << "the old_action " << A_old->label
				<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}

	action *A;

	A = NEW_OBJECT(action);

	A->label = A_old->label + "_on_orbits_" + std::to_string(Sch->nb_orbits);
	A->label_tex = A_old->label_tex + " {\\rm OnOrbits}_{" + std::to_string(Sch->nb_orbits) + "}";


	On_orbits = NEW_OBJECT(induced_actions::action_on_orbits);
	On_orbits->init(A_old, Sch, f_play_it_safe, verbose_level);


	A->f_has_subaction = true;
	A->subaction = A_old;
	A->type_G = action_on_orbits_t;
	A->G.OnOrbits = On_orbits;
	A->f_allocated = true;


	A->f_has_strong_generators = false;

	A->degree = On_orbits->degree;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();
	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A_old, 0, verbose_level);
	//allocate_base_data(0);


	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (f_v) {
		cout << "induced_action::induced_action_on_orbits "
				"finished, created action " << A->label << endl;
		A->print_info();
	}
	return A;
}


action *induced_action::induced_action_on_andre(
		action *An,
	action *An1,
	geometry::andre_construction *Andre,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_andre *On_andre;

	if (f_v) {
		cout << "induced_action::induced_action_on_andre" << endl;
		cout << "action An = " << An->label
				<< " has degree " << An->degree << endl;
		cout << "action An1 = " << An1->label
				<< " has degree " << An1->degree << endl;
	}

	action *A;

	A = NEW_OBJECT(action);

	A->label = An1->label + "_on_andre";
	A->label_tex = An1->label_tex + " {\\rm OnAndre}";

	On_andre = NEW_OBJECT(induced_actions::action_on_andre);
	On_andre->init(An, An1, Andre, verbose_level);


	A->f_has_subaction = true;
	A->subaction = An1;
	A->type_G = action_on_andre_t;
	A->G.OnAndre = On_andre;
	A->f_allocated = true;


	A->f_has_strong_generators = false;

	A->degree = On_andre->degree;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();
	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(A_old, 0, verbose_level);
	//allocate_base_data(0);


	A->elt_size_in_int = An1->elt_size_in_int;
	A->coded_elt_size_in_char = An1->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (f_v) {
		cout << "induced_action::induced_action_on_andre "
				"finished, created action " << A->label << " of degree " << A->degree << endl;
		A->print_info();
	}
	return A;
}


action *induced_action::induced_action_on_homogeneous_polynomials(
	ring_theory::homogeneous_polynomial_domain *HPD,
	int f_induce_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_homogeneous_polynomials *OnHP;

	if (f_v) {
		cout << "induced_action::induced_action_on_homogeneous_polynomials "
				"f_induce_action=" << f_induce_action << endl;
	}
	action *A;

	A = NEW_OBJECT(action);

	OnHP = NEW_OBJECT(induced_actions::action_on_homogeneous_polynomials);

	A->label = A_old->label + "_on_homog_poly_"
			+ std::to_string(HPD->nb_variables)
			+ "_" + std::to_string(HPD->degree);
	A->label_tex = A_old->label_tex
			+ " {\\rm OnHomPoly}_{" + std::to_string(HPD->nb_variables)
			+ "," + std::to_string(HPD->degree) + "}";


	if (f_v) {
		cout << "the old_action " << A_old->label
			<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}
	A->f_has_subaction = true;
	A->subaction = A_old;
	if (A_old->type_G != matrix_group_t) {
		cout << "induced_action::induced_action_on_homogeneous_polynomials "
				"action not of matrix group type" << endl;
		action_global AG;
		AG.action_print_symmetry_group_type(
					cout, A_old->type_G);
		cout << endl;
		exit(1);
	}

	if (f_v) {
		cout << "induced_action::induced_action_on_homogeneous_polynomials "
				"before OnHP->init" << endl;
	}
	OnHP->init(A_old, HPD, verbose_level);
	if (f_v) {
		cout << "induced_action::induced_action_on_homogeneous_polynomials "
				"after OnHP->init" << endl;
	}

	A->type_G = action_on_homogeneous_polynomials_t;
	A->G.OnHP = OnHP;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;
	A->low_level_point_size = OnHP->low_level_point_size;

	A->f_has_strong_generators = false;

	A->degree = OnHP->degree;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();
	A->f_is_linear = true;
	A->dimension = OnHP->dimension;



	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (f_induce_action) {

		action_global AG;

		if (f_v) {
			cout << "induced_action::induced_action_on_homogeneous_polynomials "
					"before AG.induced_action_override_sims" << endl;
		}
		AG.induced_action_override_sims(A_old, A, old_G, 0/*verbose_level - 2*/);
		if (f_v) {
			cout << "induced_action::induced_action_on_homogeneous_polynomials "
					"after AG.induced_action_override_sims" << endl;
		}
	}

	if (f_v) {
		cout << "induced_action::induced_action_on_homogeneous_polynomials "
				"finished, created action " << A->label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		A->print_info();
	}
	return A;
}

action *induced_action::induced_action_on_homogeneous_polynomials_given_by_equations(
	ring_theory::homogeneous_polynomial_domain *HPD,
	int *Equations, int nb_equations,
	int f_induce_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "induced_action::induced_action_on_homogeneous_polynomials_given_by_equations "
				"f_induce_action=" << f_induce_action << endl;
	}
	action *A;
	induced_actions::action_on_homogeneous_polynomials *OnHP;

	A = NEW_OBJECT(action);

	OnHP = NEW_OBJECT(induced_actions::action_on_homogeneous_polynomials);

	A->label = A_old->label + "_on_homog_poly_" + std::to_string(HPD->nb_variables) + "_" + std::to_string(HPD->degree) + "_eqn" + std::to_string(nb_equations);
	A->label_tex = A_old->label_tex + " {\\rm OnHomPolyEqn}_{" + std::to_string(HPD->nb_variables) + "," + std::to_string(HPD->degree)  + "," + std::to_string(nb_equations) + "}";


	if (f_v) {
		cout << "the old_action " << A_old->label
			<< " has base_length = " << A_old->base_len()
			<< " and degree " << A_old->degree << endl;
	}
	A->f_has_subaction = true;
	A->subaction = A_old;
	if (A_old->type_G != matrix_group_t) {
		cout << "induced_action::induced_action_on_homogeneous_polynomials_given_by_equations "
				"action not of matrix group type" << endl;
		action_global AG;
		AG.action_print_symmetry_group_type(
					cout, A_old->type_G);
		cout << endl;
		exit(1);
	}

	if (f_v) {
		cout << "induced_action::induced_action_on_homogeneous_polynomials_given_by_equations "
				"before OnHP->init" << endl;
	}
	OnHP->init(A_old, HPD, verbose_level);
	if (f_v) {
		cout << "induced_action::induced_action_on_homogeneous_polynomials_given_by_equations "
				"after OnHP->init" << endl;
	}

	if (f_v) {
		cout << "induced_action::induced_action_on_homogeneous_polynomials_given_by_equations "
				"before OnHP->init_invariant_set_of_equations" << endl;
	}
	OnHP->init_invariant_set_of_equations(
			Equations, nb_equations, verbose_level);
	if (f_v) {
		cout << "induced_action::induced_action_on_homogeneous_"
				"polynomials_given_by_equations "
				"after OnHP->init_invariant_set_of_equations" << endl;
	}

	A->type_G = action_on_homogeneous_polynomials_t;
	A->G.OnHP = OnHP;
	A->f_allocated = true;
	A->make_element_size = A_old->make_element_size;
	A->low_level_point_size = OnHP->low_level_point_size;

	A->f_has_strong_generators = false;

	A->degree = OnHP->degree;
	//base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();
	A->f_is_linear = true;
	A->dimension = OnHP->dimension;



	A->elt_size_in_int = A_old->elt_size_in_int;
	A->coded_elt_size_in_char = A_old->coded_elt_size_in_char;

	A->Group_element->allocate_element_data();

	if (f_induce_action) {

		action_global AG;

		if (f_v) {
			cout << "induced_action::induced_action_on_homogeneous_polynomials_given_by_equations "
					"before AG.induced_action_override_sims" << endl;
		}
		AG.induced_action_override_sims(A_old, A,
				old_G, 0/*verbose_level - 2*/);
		if (f_v) {
			cout << "induced_action::induced_action_on_homogeneous_polynomials_given_by_equations "
					"after AG.induced_action_override_sims" << endl;
		}
	}

	if (f_v) {
		cout << "induced_action::induced_action_on_homogeneous_polynomials_given_by_equations "
				"finished, created action " << A->label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		A->print_info();
	}
	if (f_v) {
		cout << "induced_action::induced_action_on_homogeneous_polynomials_given_by_equations done" << endl;
	}
	return A;
}



action *induced_action::base_change(
	int size, long int *set, groups::sims *old_Sims,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "induced_action::base_change" << endl;
	}

	action *old_action = A_old;
	action *new_action;

	if (f_v) {
		cout << "induced_action::base_change to the following set:" << endl;
		Lint_vec_print(cout, set, size);
		cout << endl;
	}
#if 0
	if (!old_action->f_has_sims) {
		cout << "induced_action::base_change old_action does not have sims" << endl;
		exit(1);
	}
#endif

	new_action = NEW_OBJECT(action);

	new_action->f_has_subaction = true;
	new_action->subaction = old_action;
	new_action->type_G = base_change_t;
	new_action->f_allocated = false;

	new_action->f_has_strong_generators = false;

	new_action->degree = old_action->degree;
	//base_len = 0;
	new_action->ptr = NEW_OBJECT(action_pointer_table);
	new_action->ptr->init_function_pointers_induced_action();

	new_action->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	new_action->Stabilizer_chain->allocate_base_data(A_old, 0 /* base_len */, verbose_level);
	//allocate_base_data(0);


	new_action->elt_size_in_int = old_action->elt_size_in_int;
	new_action->coded_elt_size_in_char = old_action->coded_elt_size_in_char;
	new_action->make_element_size = old_action->make_element_size;
	new_action->low_level_point_size = old_action->low_level_point_size;


	new_action->Group_element->allocate_element_data();

	action_global AG;

	if (f_v) {
		cout << "induced_action::base_change before AG.induce" << endl;
	}
	AG.induce(old_action,
			new_action,
			old_Sims /*old_action->Sims */,
			size, set,
			verbose_level - 1);
	if (f_v) {
		cout << "induced_action::base_change after AG.induce" << endl;
	}

	new_action->label = old_action->label + "_base_change";
	new_action->label_tex = old_action->label_tex + " {\\rm BaseChange}";



	if (f_v) {
		ring_theory::longinteger_object go, K_go;
		new_action->group_order(go);
		new_action->Kernel->group_order(K_go);
		cout << "induced_action::base_change finished" << endl;
		cout << "induced action has order " << go << endl;
		cout << "kernel has order " << K_go << endl;
		//cout << "generators are:" << endl;
		//Sims->print_generators();
	}
	if (false) {
		new_action->Sims->print_generators();
		new_action->Sims->print_generators_as_permutations();
		new_action->Sims->print_basic_orbits();
	}
	if (f_v) {
		cout << "induced_action::base_change done" << endl;
	}
	return new_action;
}







#if 0
void induced_action::induced_action_recycle_sims(
		action &old_action,
	int verbose_level)
{
	groups::sims *old_G;

	if (!old_action.f_has_sims) {
		cout << "induced_action::induced_action_recycle_sims: "
				"old action must have sims" << endl;
		exit(1);
	}
	old_G = old_action.Sims;

	action_global AG;

	induce(&old_action, old_G,
		0 /* base_of_choice_len */, NULL /* base_of_choice */,
		verbose_level);
}
#endif




}}}



