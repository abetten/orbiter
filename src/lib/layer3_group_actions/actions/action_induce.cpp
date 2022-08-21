// action_induce.cpp
//
// Anton Betten
// 1/1/2009

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {


action *action::induced_action_on_interior_direct_product(
		int nb_rows,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_interior_direct_product *IDP;
	action *A;

	if (f_v) {
		cout << "action::induced_action_on_interior_direct_product" << endl;
	}
	A = NEW_OBJECT(action);



	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_on_interior_direct_product_%ld_%d", A->degree, nb_rows);
	snprintf(str2, 1000, " {\\rm OnIntDirectProduct}_{%ld,%d}", A->degree, nb_rows);

	A->label.assign(label);
	A->label.append(str1);

	A->label_tex.assign(label_tex);
	A->label_tex.append(str2);



	if (f_v) {
		cout << "the old_action " << label
				<< " has base_length = " << base_len()
			<< " and degree " << degree << endl;
	}
	A->f_has_subaction = TRUE;
	A->subaction = this;
	IDP = NEW_OBJECT(induced_actions::action_on_interior_direct_product);

	IDP->init(this, nb_rows, verbose_level);

	A->type_G = action_on_interior_direct_product_t;
	A->G.OnInteriorDirectProduct = IDP;
	A->f_allocated = TRUE;
	A->make_element_size = make_element_size;
	A->low_level_point_size = 0;

	A->f_has_strong_generators = FALSE;

	A->degree = IDP->degree;
	//A->base_len = 0;
	if (f_v) {
		cout << "action::induced_action_on_interior_direct_product "
				"before init_function_pointers_induced_action" << endl;
	}
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();



	A->elt_size_in_int = elt_size_in_int;
	A->coded_elt_size_in_char = coded_elt_size_in_char;

	if (f_v) {
		cout << "action::induced_action_on_interior_direct_product "
				"before allocate_element_data" << endl;
	}
	allocate_element_data();


	if (f_v) {
		cout << "action::induced_action_on_interior_direct_product "
				"finished, created action " << A->label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		print_info();
	}
	return A;
}


action *action::induced_action_on_set_partitions(
		int partition_class_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_set_partitions *OSP;
	action *A;

	if (f_v) {
		cout << "action::induced_action_on_set_partitions" << endl;
		}
	A = NEW_OBJECT(action);



	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_on_set_partitions_%ld_%d", A->degree, partition_class_size);
	snprintf(str2, 1000, " {\\rm OnSetPart}_{%ld,%d}", A->degree, partition_class_size);

	A->label.assign(label);
	A->label.append(str1);

	A->label_tex.assign(label_tex);
	A->label_tex.append(str2);





	if (f_v) {
		cout << "the old_action " << label
				<< " has base_length = " << base_len()
			<< " and degree " << degree << endl;
		}
	A->f_has_subaction = TRUE;
	A->subaction = this;
	OSP = NEW_OBJECT(induced_actions::action_on_set_partitions);

	OSP->init(partition_class_size,
			this, verbose_level);
	A->type_G = action_on_set_partitions_t;
	A->G.OnSetPartitions = OSP;
	A->f_allocated = TRUE;
	A->make_element_size = make_element_size;
	A->low_level_point_size = 0;

	A->f_has_strong_generators = FALSE;

	A->degree = OSP->nb_set_partitions;
	//A->base_len = 0;
	if (f_v) {
		cout << "action::induced_action_on_set_partitions "
				"before init_function_pointers_induced_action" << endl;
		}
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();



	A->elt_size_in_int = elt_size_in_int;
	A->coded_elt_size_in_char = coded_elt_size_in_char;

	if (f_v) {
		cout << "action::induced_action_on_set_partitions "
				"before allocate_element_data" << endl;
		}
	allocate_element_data();


	if (f_v) {
		cout << "action::induced_action_on_set_partitions "
				"finished, created action " << A->label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		print_info();
		}
	return A;
}


void action::init_action_on_lines(action *A,
		field_theory::finite_field *F, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	induced_actions::action_on_grassmannian *A_lines;
	geometry::grassmann *Grass_lines;

	if (f_v) {
		cout << "action::init_action_on_lines" << endl;
		}

	A_lines = NEW_OBJECT(induced_actions::action_on_grassmannian);

	Grass_lines = NEW_OBJECT(geometry::grassmann);


	if (f_v) {
		cout << "action::init_action_on_lines "
				"before Grass_lines->init" << endl;
		}
	Grass_lines->init(n, 2, F, verbose_level - 2);

	if (f_v) {
		cout << "action::init_action_on_lines "
				"before A_lines->init" << endl;
		}
	A_lines->init(*A, Grass_lines, verbose_level - 5);
	
	
	if (f_v) {
		cout << "action::init_action_on_lines "
				"action on grassmannian established" << endl;
		}

	if (f_v) {
		cout << "action::init_action_on_lines "
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
		cout << "action::init_action_on_lines "
				"group order " << go1 << endl;
		}
	
	if (f_v) {
		cout << "action::init_action_on_lines "
				"initializing action on grassmannian" << endl;
		}
	induced_action_on_grassmannian(A, A_lines, 
		TRUE /* f_induce_action */, &S, verbose_level);
	if (f_v) {
		cout << "action::init_action_on_lines "
				"after induced_action_on_grassmannian" << endl;
		}
	if (f_vv) {
		print_info();
		}

	if (f_v) {
		cout << "action::init_action_on_lines done" << endl;
		}
}


void action::induced_action_by_representation_on_conic(
	action *A_old,
	int f_induce_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	induced_actions::action_by_representation *Rep; // do not free
	
	if (f_v) {
		cout << "action::induced_action_by_representation_on_conic "
				"f_induce_action=" << f_induce_action << endl;
		}
	
	A = A_old;


	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_OnConic");
	snprintf(str2, 1000, " {\\rm OnConic}");

	label.assign(A->label);
	label_tex.assign(A->label_tex);
	label.append(str1);
	label_tex.append(str2);



	if (f_v) {
		cout << "the old_action " << A->label
				<< " has base_length = " << A->base_len()
			<< " and degree " << A->degree << endl;
		}
	f_has_subaction = TRUE;
	subaction = A;
	if (A->type_G != matrix_group_t) {
		cout << "action::induced_action_by_representation_on_conic "
				"action not of matrix group type" << endl;
		exit(1);
		}
	//M = A->G.matrix_grp;

	Rep = NEW_OBJECT(induced_actions::action_by_representation);
	Rep->init_action_on_conic(*A_old, verbose_level);

	type_G = action_by_representation_t;
	G.Rep = Rep;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	low_level_point_size = Rep->low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = Rep->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_induce_action) {
		induced_action_override_sims(*A, old_G, 0/*verbose_level - 2*/);
		}

	if (f_v) {
		cout << "action::induced_action_by_representation_on_conic "
				"finished, created action " << label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		print_info();
		}
}


void action::induced_action_on_cosets(
		induced_actions::action_on_cosets *A_on_cosets,
	int f_induce_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	
	if (f_v) {
		cout << "action::induced_action_on_cosets "
				"f_induce_action=" << f_induce_action << endl;
		}
	A = A_on_cosets->A_linear;


	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_Cosets_%d", A_on_cosets->dimension_of_subspace);
	snprintf(str2, 1000, " {\\rm OnCosets}_{%d}", A_on_cosets->dimension_of_subspace);

	label.assign(A->label);
	label_tex.assign(A->label_tex);
	label.append(str1);
	label_tex.append(str2);


	if (f_v) {
		cout << "the old_action is " << A->label << endl;
		//		<< " has base_length = " << A->base_len()
		//	<< " and degree " << A->degree << endl;
		}
	f_has_subaction = TRUE;
	subaction = A;
	if (!A->f_is_linear) {
		cout << "action::induced_action_on_cosets "
				"action not of linear type" << endl;
		exit(1);
		}
#if 0
	if (A->type_G == matrix_group_t) {
		M = A->G.matrix_grp;
		}
	else {
		action *sub = A->subaction;
		M = sub->G.matrix_grp;
		}
#endif
	type_G = action_on_cosets_t;
	G.OnCosets = A_on_cosets;
	f_allocated = FALSE;
	make_element_size = A->make_element_size;
	low_level_point_size = A->low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = A_on_cosets->nb_points;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_induce_action) {
		induced_action_override_sims(*A, old_G, verbose_level - 2);
		}

	if (f_v) {
		cout << "action::induced_action_on_cosets "
				"finished, created action " << label << " of degree=" << degree << endl;
		print_info();
		}
}



void action::induced_action_on_factor_space(action *A_old, 
		induced_actions::action_on_factor_space *AF,
	int f_induce_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;

	if (f_v) {
		cout << "action::induced_action_on_factor_space "
				"f_induce_action=" << f_induce_action << endl;
	}
	A = A_old;

	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_Factor_%d_%d", AF->VS->dimension, AF->factor_space_len);
	snprintf(str2, 1000, " {\\rm OnFactor}_{%d,%d}", AF->VS->dimension, AF->factor_space_len);

	label.assign(A->label);
	label_tex.assign(A->label_tex);
	label.append(str1);
	label_tex.append(str2);


	if (f_v) {
		cout << "the old_action " << A->label
				<< " has base_length = " << A->base_len()
			<< " and degree " << A->degree << endl;
	}
	f_has_subaction = TRUE;
	subaction = A;
	if (!A->f_is_linear) {
		cout << "action::induced_action_on_factor_space "
				"action not of linear type" << endl;
		cout << "the old action is:" << endl;
		A->print_info();
		exit(1);
	}

	type_G = action_on_factor_space_t;
	G.AF = AF;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	low_level_point_size = A->low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = AF->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_induce_action) {
		induced_action_override_sims(*A, old_G, verbose_level - 2);
	}

	if (f_v) {
		cout << "action::induced_action_on_factor_space "
				"finished, created action " << label << endl;
		cout << "degree=" << A->degree << endl;
		print_info();
	}
}

action *action::induced_action_on_grassmannian(int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_grassmannian *AG;
	action *A;
	groups::matrix_group *M;
	
	if (f_v) {
		cout << "action::induced_action_on_grassmannian" << endl;
		}
	A = NEW_OBJECT(action);


	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_Gr_%d", k);
	snprintf(str2, 1000, " {\\rm OnGr}_{%d}", k);

	A->label.assign(label);
	A->label_tex.assign(label_tex);
	A->label.append(str1);
	A->label_tex.append(str2);


	if (f_v) {
		cout << "the old_action " << label
				<< " has base_length = " << base_len()
			<< " and degree " << degree << endl;
		}
	A->f_has_subaction = TRUE;
	A->subaction = this;
	if (type_G != matrix_group_t) {
		cout << "action::induced_action_on_grassmannian "
				"old action not of matrix group type" << endl;
		exit(1);
		}
	M = G.matrix_grp;
	AG = NEW_OBJECT(induced_actions::action_on_grassmannian);

	geometry::grassmann *Gr;

	Gr = NEW_OBJECT(geometry::grassmann);
	Gr->init(M->n, k, M->GFq, verbose_level);
	AG->init(*this, Gr, verbose_level);
	A->type_G = action_on_grassmannian_t;
	A->G.AG = AG;
	A->f_allocated = TRUE;
	A->make_element_size = make_element_size;
	A->low_level_point_size = AG->low_level_point_size;
	
	A->f_has_strong_generators = FALSE;
	
	A->degree = AG->degree.as_int();
	//A->base_len = 0;
	if (f_v) {
		cout << "action::induced_action_on_grassmannian "
				"before init_function_pointers_induced_action" << endl;
		}
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();
	
	
	
	A->elt_size_in_int = elt_size_in_int;
	A->coded_elt_size_in_char = coded_elt_size_in_char;
	
	if (f_v) {
		cout << "action::induced_action_on_grassmannian "
				"before allocate_element_data" << endl;
		}
	allocate_element_data();
	

	if (f_v) {
		cout << "action::induced_action_on_grassmannian "
				"finished, created action " << A->label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		print_info();
		}
	return A;
}


void action::induced_action_on_grassmannian(action *A_old, 
		induced_actions::action_on_grassmannian *AG,
	int f_induce_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	
	if (f_v) {
		cout << "action::induced_action_on_grassmannian "
				"f_induce_action=" << f_induce_action << endl;
	}
	A = A_old;

	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_Gr_%d_%d", AG->n, AG->k);
	snprintf(str2, 1000, " {\\rm OnGr}_{%d,%d}", AG->n, AG->k);

	label.assign(A->label);
	label_tex.assign(A->label_tex);
	label.append(str1);
	label_tex.append(str2);


	if (f_v) {
		cout << "action::induced_action_on_grassmannian "
			"the old_action " << A->label
			<< " has base_length = " << A->base_len()
			<< " and degree " << A->degree << endl;
	}
	f_has_subaction = TRUE;
	if (f_v) {
		cout << "action::induced_action_on_grassmannian "
				"before subaction = A" << endl;
	}
	subaction = A;
	if (!A->f_is_linear) {
		cout << "action::induced_action_on_grassmannian "
				"action not of linear type" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "action::induced_action_on_grassmannian "
				"action is of linear type" << endl;
	}
	type_G = action_on_grassmannian_t;
	G.AG = AG;
	f_allocated = FALSE;
	make_element_size = A->make_element_size;
	low_level_point_size = AG->low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = AG->degree.as_int();

	if (f_v) {
		cout << "action::induced_action_on_grassmannian "
				"before init_function_pointers_induced_action" << endl;
	}
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	if (f_v) {
		cout << "action::induced_action_on_grassmannian "
				"after init_function_pointers_induced_action" << endl;
	}
	
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	if (f_v) {
		cout << "action::induced_action_on_grassmannian "
				"before allocate_element_data" << endl;
	}
	allocate_element_data();
	if (f_v) {
		cout << "action::induced_action_on_grassmannian "
				"after allocate_element_data" << endl;
	}
	
	if (f_induce_action) {
		if (f_v) {
			cout << "action::induced_action_on_grassmannian "
					"before induced_action_override_sims" << endl;
		}
		induced_action_override_sims(*A, old_G, verbose_level);
		if (f_v) {
			cout << "action::induced_action_on_grassmannian "
					"after induced_action_override_sims" << endl;
		}
	}

	if (f_v) {
		cout << "action::induced_action_on_grassmannian finished, "
				"created action " << label << endl;
		cout << "action::induced_action_on_grassmannian "
				"degree=" << degree << endl;
		cout << "action::induced_action_on_grassmannian "
				"make_element_size=" << make_element_size << endl;
		cout << "action::induced_action_on_grassmannian "
				"low_level_point_size=" << low_level_point_size << endl;
		print_info();
		cout << "action::induced_action_on_grassmannian finished, "
				"after print_info()" << endl;
	}
	if (f_v) {
		cout << "action::induced_action_on_grassmannian done" << endl;
	}
}

void action::induced_action_on_spread_set(action *A_old, 
		induced_actions::action_on_spread_set *AS,
	int f_induce_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	
	if (f_v) {
		cout << "action::induced_action_on_spread_set "
				"f_induce_action=" << f_induce_action << endl;
		}
	A = A_old;

	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_SpreadSet_%d_%d", AS->k, AS->q);
	snprintf(str2, 1000, " {\\rm OnSpreadSet %d,%d}", AS->k, AS->q);

	label.assign(A->label);
	label_tex.assign(A->label_tex);
	label.append(str1);
	label_tex.append(str2);



	if (f_v) {
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len()
			<< " and degree " << A->degree << endl;
		}
	f_has_subaction = TRUE;
	subaction = A;
	if (!A->f_is_linear) {
		cout << "action::induced_action_on_spread_set "
				"action not of linear type" << endl;
		exit(1);
		}

	f_is_linear = TRUE;

	
	type_G = action_on_spread_set_t;
	G.AS = AS;
	f_allocated = FALSE;
	make_element_size = A->make_element_size;
	low_level_point_size = AS->low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = AS->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_induce_action) {
		induced_action_override_sims(*A,
				old_G, 0/*verbose_level - 2*/);
		}

	if (f_v) {
		cout << "action::induced_action_on_spread_set finished, "
				"created action " << label << endl;
		cout << "degree=" << degree << endl;
		cout << "make_element_size=" << make_element_size << endl;
		cout << "low_level_point_size=" << low_level_point_size << endl;
		print_info();
		}
}

void action::induced_action_on_orthogonal(action *A_old, 
		induced_actions::action_on_orthogonal *AO,
	int f_induce_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	
	if (f_v) {
		cout << "action::induced_action_on_orthogonal "
				"f_induce_action=" << f_induce_action << endl;
		}
	A = A_old;

	char str1[1000];
	char str2[1000];

	if (AO->f_on_points) {
		snprintf(str1, 1000, "_Opts_%d_%d_%d", AO->O->epsilon, AO->O->n, AO->O->q);
		snprintf(str2, 1000, " {\\rm OnOpts %d,%d,%d}", AO->O->epsilon, AO->O->n, AO->O->q);
		}
	else if (AO->f_on_lines) {
		snprintf(str1, 1000, "_Olines_%d_%d_%d", AO->O->epsilon, AO->O->n, AO->O->q);
		snprintf(str2, 1000, " {\\rm OnOlines %d,%d,%d}", AO->O->epsilon, AO->O->n, AO->O->q);
		}
	else if (AO->f_on_points_and_lines) {
		snprintf(str1, 1000, "_Optslines_%d_%d_%d", AO->O->epsilon, AO->O->n, AO->O->q);
		snprintf(str2, 1000, " {\\rm OnOptslines %d,%d,%d}", AO->O->epsilon, AO->O->n, AO->O->q);
		}



	label.assign(A->label);
	label_tex.assign(A->label_tex);
	label.append(str1);
	label_tex.append(str2);



	if (f_v) {
		cout << "the old_action " << A->label
				<< " has base_length = " << A->base_len()
			<< " and degree " << A->degree << endl;
		}
	f_has_subaction = TRUE;
	subaction = A;
	if (!A->f_is_linear) {
		cout << "action::induced_action_on_orthogonal "
				"action not of linear type" << endl;
		exit(1);
		}
	type_G = action_on_orthogonal_t;
	G.AO = AO;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	low_level_point_size = AO->low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = AO->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_induce_action) {
		induced_action_override_sims(*A, old_G, 0/*verbose_level - 2*/);
		}

	if (f_v) {
		cout << "action::induced_action_on_orthogonal "
				"finished, created action " << label << endl;
		cout << "degree=" << degree << endl;
		cout << "make_element_size=" << make_element_size << endl;
		cout << "low_level_point_size=" << low_level_point_size << endl;
		print_info();
		}
}


action *action::induced_action_on_wedge_product(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	groups::matrix_group *M;

	if (f_v) {
		cout << "action::induced_action_on_wedge_product" << endl;
	}
	A = NEW_OBJECT(action);


	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_Wedge");
	snprintf(str2, 1000, " {\\rm OnWedge}");

	A->label.assign(label);
	A->label_tex.assign(label_tex);
	A->label.append(str1);
	A->label_tex.append(str2);


	if (f_v) {
		cout << "the old_action " << label
				<< " has base_length = " << base_len()
			<< " and degree " << degree << endl;
	}
	A->f_has_subaction = TRUE;
	A->subaction = this;
	if (type_G != matrix_group_t) {
		cout << "action::induced_action_on_wedge_product "
				"old action not of matrix group type" << endl;
		exit(1);
	}
	M = G.matrix_grp;

	induced_actions::action_on_wedge_product *AW;

	AW = NEW_OBJECT(induced_actions::action_on_wedge_product);




	if (f_v) {
		cout << "action::induced_action_on_wedge_product before AW->init" << endl;
	}
	AW->init(*this, verbose_level);
	if (f_v) {
		cout << "action::induced_action_on_wedge_product after AW->init" << endl;
	}

	A->type_G = action_on_wedge_product_t;
	A->G.AW = AW;
	A->f_allocated = TRUE;
	A->make_element_size = make_element_size;
	A->low_level_point_size = AW->low_level_point_size;

	A->f_has_strong_generators = FALSE;

	A->degree = AW->degree;
	//A->base_len = 0;
	if (f_v) {
		cout << "action::induced_action_on_wedge_product "
				"before init_function_pointers_induced_action" << endl;
		}
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();



	A->elt_size_in_int = elt_size_in_int;
	A->coded_elt_size_in_char = coded_elt_size_in_char;

	A->f_is_linear = TRUE;
	A->dimension = AW->wedge_dimension;

	if (f_v) {
		cout << "action::induced_action_on_wedge_product "
				"before allocate_element_data" << endl;
		}
	allocate_element_data();


	if (f_v) {
		cout << "action::induced_action_on_wedge_product "
				"finished, created action " << A->label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		print_info();
		}
	return A;
}

#if 0
void action::induced_action_on_wedge_product(action *A_old, 
	action_on_wedge_product *AW, 
	int f_induce_action, sims *old_G, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;

	if (f_v) {
		cout << "action::induced_action_on_wedge_product "
				"f_induce_action=" << f_induce_action << endl;
		}
	A = A_old;


	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_Wedge");
	snprintf(str2, 1000, " {\\rm OnWedge}");

	label.assign(A->label);
	label_tex.assign(A->label_tex);
	label.append(str1);
	label_tex.append(str2);



	if (f_v) {
		cout << "the old_action " << A->label
				<< " has base_length = " << A->base_len()
			<< " and degree " << A->degree << endl;
		}
	f_has_subaction = TRUE;
	subaction = A;
	if (A->type_G != matrix_group_t) {
		cout << "action::induced_action_on_wedge_product "
				"action not of matrix group type" << endl;
		exit(1);
		}
	//M = A->G.matrix_grp;
	type_G = action_on_wedge_product_t;
	G.AW = AW;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	low_level_point_size = AW->low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = AW->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	f_is_linear = TRUE;
	dimension = AW->wedge_dimension;
	
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_induce_action) {
		induced_action_override_sims(*A, old_G, 0/*verbose_level - 2*/);
		}

	if (f_v) {
		cout << "action::induced_action_on_wedge_product "
				"finished, created action " << label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		print_info();
		}
}
#endif

void action::induced_action_by_subfield_structure(action *A_old, 
		induced_actions::action_by_subfield_structure *SubfieldStructure,
	int f_induce_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;

	if (f_v) {
		cout << "action::induced_action_by_subfield_structure "
				"f_induce_action=" << f_induce_action << endl;
		}
	A = A_old;


	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_subfield_%d", SubfieldStructure->q);
	snprintf(str2, 1000, " {\\rm OnSubfield F%d}", SubfieldStructure->q);

	label.assign(A->label);
	label_tex.assign(A->label_tex);
	label.append(str1);
	label_tex.append(str2);


	if (f_v) {
		cout << "the old_action " << A->label
				<< " has base_length = " << A->base_len()
			<< " and degree " << A->degree << endl;
		}
	f_has_subaction = TRUE;
	subaction = A;
	if (A->type_G != matrix_group_t) {
		cout << "action::induced_action_by_subfield_structure "
				"action not of matrix group type" << endl;
		exit(1);
		}
	//M = A->G.matrix_grp;
	type_G = action_by_subfield_structure_t;
	G.SubfieldStructure = SubfieldStructure;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	low_level_point_size = SubfieldStructure->low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = SubfieldStructure->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	f_is_linear = TRUE;
	dimension = SubfieldStructure->m;
	
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_induce_action) {
		induced_action_override_sims(*A,
				old_G, 0/*verbose_level - 2*/);
		}

	if (f_v) {
		cout << "action::induced_action_by_subfield_structure "
				"finished, created action " << label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		print_info();
		}
}


void action::induced_action_on_Galois_group(
		groups::sims *old_G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_galois_group *AG;
	action *A;
	groups::matrix_group *M;

	if (f_v) {
		cout << "action::induced_action_on_Galois_group" << endl;
		}
	A = old_G->A;


	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_gal");
	snprintf(str2, 1000, " {\\rm OnGal}");

	label.assign(A->label);
	label_tex.assign(A->label_tex);
	label.append(str1);
	label_tex.append(str2);

	if (f_v) {
		cout << "the old_action " << A->label
				<< " has base_length = " << A->base_len()
			<< " and degree " << A->degree << endl;
		}
	f_has_subaction = TRUE;
	subaction = A;
	if (A->type_G != matrix_group_t) {
		cout << "action::induced_action_on_Galois_group "
				"action not of matrix group type" << endl;
		exit(1);
		}
	M = A->G.matrix_grp;
	AG = NEW_OBJECT(induced_actions::action_on_galois_group);
	AG->init(A, M->n, verbose_level);
	type_G = action_on_galois_group_t;
	G.on_Galois_group = AG;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;

	f_has_strong_generators = FALSE;

	degree = AG->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();

	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);


	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;

	allocate_element_data();

	if (f_v) {
		cout << "action::induced_action_on_Galois_group before induced_action_override_sims" << endl;
		}
	induced_action_override_sims(*A, old_G, verbose_level - 2);
	if (f_v) {
		cout << "action::induced_action_on_Galois_group "
				"finished, created action " << label << endl;
		print_info();
		}
}

void action::induced_action_on_determinant(
		groups::sims *old_G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_determinant *AD;
	action *A;
	groups::matrix_group *M;
	
	if (f_v) {
		cout << "action::induced_action_on_determinant" << endl;
		}
	A = old_G->A;


	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_det");
	snprintf(str2, 1000, " {\\rm OnDet}");

	label.assign(A->label);
	label_tex.assign(A->label_tex);
	label.append(str1);
	label_tex.append(str2);


	if (f_v) {
		cout << "the old_action " << A->label
				<< " has base_length = " << A->base_len()
			<< " and degree " << A->degree << endl;
		}
	f_has_subaction = TRUE;
	subaction = A;
	if (A->type_G != matrix_group_t) {
		cout << "action::induced_action_on_determinant "
				"action not of matrix group type" << endl;
		exit(1);
		}
	M = A->G.matrix_grp;
	AD = NEW_OBJECT(induced_actions::action_on_determinant);
	AD->init(*A, M->f_projective, M->n, verbose_level);
	type_G = action_on_determinant_t;
	G.AD = AD;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	
	f_has_strong_generators = FALSE;
	
	degree = AD->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	allocate_element_data();
	
	induced_action_override_sims(*A, old_G, verbose_level - 2);
	if (f_v) {
		cout << "action::induced_action_on_determinant "
				"finished, created action " << label << endl;
		print_info();
		}
}

void action::induced_action_on_sign(
		groups::sims *old_G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_sign *OnSign;
	action *A;
	
	if (f_v) {
		cout << "action::induced_action_on_sign" << endl;
		}
	A = old_G->A;


	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_OnSign");
	snprintf(str2, 1000, " {\\rm OnSign}");

	label.assign(A->label);
	label_tex.assign(A->label_tex);
	label.append(str1);
	label_tex.append(str2);


	if (f_v) {
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len()
			<< " and degree " << A->degree << endl;
		}
	f_has_subaction = TRUE;
	subaction = A;
	OnSign = NEW_OBJECT(induced_actions::action_on_sign);
	OnSign->init(A, verbose_level);
	type_G = action_on_sign_t;
	G.OnSign = OnSign;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	
	f_has_strong_generators = FALSE;
	
	degree = OnSign->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	allocate_element_data();
	
	induced_action_override_sims(*A, old_G, verbose_level - 2);
	if (f_v) {
		cout << "action::induced_action_on_sign finished, "
				"created action " << label << endl;
		print_info();
		}
}

action *action::create_induced_action_by_conjugation(
		groups::sims *Base_group, int f_ownership,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;

	if (f_v) {
		cout << "action::create_induced_action_by_conjugation" << endl;
		}
	A = NEW_OBJECT(action);
	if (f_v) {
		cout << "action::create_induced_action_by_conjugation "
				"before A->induced_action_on_sets" << endl;
		}
	A->induced_action_by_conjugation(NULL,
			Base_group, f_ownership, FALSE /* f_basis */,
			verbose_level);
	if (f_v) {
		cout << "action::create_induced_action_by_conjugation "
				"after A->induced_action_by_conjugation" << endl;
		}
	if (f_v) {
		cout << "action::create_induced_action_by_conjugation done" << endl;
		}
	return A;
}

void action::induced_action_by_conjugation(groups::sims *old_G,
		groups::sims *Base_group, int f_ownership,
	int f_basis, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_by_conjugation *ABC;
	ring_theory::longinteger_object go;
	long int goi;
	action *A;
	
	A = Base_group->A;
	if (f_v) {
		cout << "action::induced_action_by_conjugation" << endl;
		cout << "the old_action " << A->label
				<< " has base_length = " << A->base_len()
			<< " and degree " << A->degree << endl;
	}
	Base_group->group_order(go);
	goi = go.as_lint();
	if (f_v) {
		cout << "we are acting on a group of order " << goi << endl;
	}



	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_C%ld", goi);
	snprintf(str2, 1000, " {\\rm ByConj%ld}", goi);

	label.assign(A->label);
	label_tex.assign(A->label_tex);
	label.append(str1);
	label_tex.append(str2);



	f_has_subaction = TRUE;
	subaction = A;
	ABC = NEW_OBJECT(induced_actions::action_by_conjugation);
	ABC->init(Base_group, f_ownership, verbose_level);
	type_G = action_by_conjugation_t;
	G.ABC = ABC;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	
	f_has_strong_generators = FALSE;
	
	degree = goi;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_basis) {
		if (f_v) {
			cout << "action::induced_action_by_conjugation "
					"calling induced_action_override_sims" << endl;
		}
		induced_action_override_sims(*A, old_G, verbose_level - 2);
	}
	if (f_v) {
		cout << "action::induced_action_by_conjugation "
				"finished, created action " << label << endl;
		print_info();
	}
}

void action::induced_action_by_right_multiplication(
	int f_basis, groups::sims *old_G,
	groups::sims *Base_group, int f_ownership, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_by_right_multiplication *ABRM;
	ring_theory::longinteger_object go;
	int goi;
	action *A;
	
	A = Base_group->A;
	if (f_v) {
		cout << "action::induced_action_by_right_multiplication" << endl;
		cout << "the old_action " << A->label
				<< " has base_length = " << A->base_len()
			<< " and degree " << A->degree << endl;
	}
	Base_group->group_order(go);
	goi = go.as_int();


	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_R%d", goi);
	snprintf(str2, 1000, " {\\rm RightMult%d}", goi);

	label.assign(A->label);
	label_tex.assign(A->label_tex);
	label.append(str1);
	label_tex.append(str2);


	if (f_v) {
		cout << "we are acting on a group of order " << goi << endl;
	}
	f_has_subaction = TRUE;
	subaction = A;
	ABRM = NEW_OBJECT(induced_actions::action_by_right_multiplication);
	ABRM->init(Base_group, f_ownership, verbose_level);
	type_G = action_by_right_multiplication_t;
	G.ABRM = ABRM;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	
	f_has_strong_generators = FALSE;
	
	degree = goi;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_basis) {
		induced_action_override_sims(*A, old_G, verbose_level - 2);
	}
	if (f_v) {
		cout << "action::induced_action_by_right_multiplication "
				"finished, created action " << label << endl;
		print_info();
	}
}

action *action::create_induced_action_on_sets(
		int nb_sets, int set_size, long int *sets,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	
	if (f_v) {
		cout << "action::create_induced_action_on_sets" << endl;
	}
	A = NEW_OBJECT(action);
	if (f_v) {
		cout << "action::create_induced_action_on_sets "
				"before A->induced_action_on_sets" << endl;
	}
	A->induced_action_on_sets(*this, NULL, 
		nb_sets, set_size, sets, 
		FALSE /*f_induce_action*/, verbose_level);
	if (f_v) {
		cout << "action::create_induced_action_on_sets "
				"after A->induced_action_on_sets" << endl;
	}
	if (f_v) {
		cout << "action::create_induced_action_on_sets done" << endl;
	}
	return A;
}


void action::induced_action_on_sets(
	action &old_action, groups::sims *old_G,
	int nb_sets, int set_size, long int *sets,
	int f_induce_action, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_sets *AOS;
	
	if (f_v) {
		cout << "action::induced_action_on_sets" << endl;
		cout << "action::induced_action_on_sets "
				"f_induce_action=" << f_induce_action << endl;

		cout << "action::induced_action_on_sets "
				"the old_action " << old_action.label
				//<< " has base_length = " << old_action.base_len()
			<< " has degree " << old_action.degree << endl;

		cout << "action::induced_action_on_sets "
				"verbose_level = " << verbose_level << endl;
	}

	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_S%d", set_size);
	snprintf(str2, 1000, " {\\rm S%d}", set_size);

	label.assign(old_action.label);
	label_tex.assign(old_action.label_tex);
	label.append(str1);
	label_tex.append(str2);


	f_has_subaction = TRUE;
	subaction = &old_action;
	if (f_v) {
		cout << "action::induced_action_on_sets "
				"allocating action_on_sets" << endl;
	}
	AOS = NEW_OBJECT(induced_actions::action_on_sets);
	if (f_v) {
		cout << "action::induced_action_on_sets before AOS->init" << endl;
	}
	AOS->init(nb_sets, set_size, sets, verbose_level - 1);
	if (f_v) {
		cout << "action::induced_action_on_sets after AOS->init" << endl;
	}
	type_G = action_on_sets_t;
	G.on_sets = AOS;
	f_allocated = TRUE;
	make_element_size = old_action.make_element_size;
	low_level_point_size = 0;
	
	f_has_strong_generators = FALSE;
	
	degree = nb_sets;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	if (f_v) {
		cout << "action::induced_action_on_sets "
				"calling allocate_base_data" << endl;
	}
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = old_action.elt_size_in_int;
	coded_elt_size_in_char = old_action.coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_induce_action) {
		if (f_v) {
			cout << "action::induced_action_on_sets "
					"calling induced_action_override_sims" << endl;
		}
		induced_action_override_sims(old_action,
				old_G, verbose_level /*- 2*/);
		if (f_v) {
			cout << "action::induced_action_on_sets "
					"induced_action_override_sims done" << endl;
		}
	}
	if (f_v) {
		cout << "action::induced_action_on_sets finished, "
				"created action " << label << endl;
		//Sims->print_transversal_lengths();
		//cout << endl;
		print_info();
	}
}

action *action::create_induced_action_on_subgroups(groups::sims *S,
	int nb_subgroups, int group_order, groups::subgroup **Subgroups,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	
	if (f_v) {
		cout << "action::create_induced_action_on_subgroups" << endl;
	}
	A = NEW_OBJECT(action);
	A->induced_action_on_subgroups(this, S, 
		nb_subgroups, group_order, Subgroups, 
		0 /* verbose_level*/);
	if (f_v) {
		cout << "action::create_induced_action_on_subgroups done" << endl;
	}
	return A;
}


void action::induced_action_on_subgroups(
	action *old_action, groups::sims *S,
	int nb_subgroups, int group_order, groups::subgroup **Subgroups,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_subgroups *AOS;
	
	if (f_v) {
		cout << "action::induced_action_on_subgroups" << endl;
		cout << "action::induced_action_on_sets "
				"the old_action " << old_action->label
				<< " has base_length = " << old_action->base_len()
			<< " and degree " << old_action->degree << endl;
		cout << "action::induced_action_on_subgroups "
				"verbose_level = " << verbose_level << endl;
	}

	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_on_subgroups%d_%d", nb_subgroups, group_order);
	snprintf(str2, 1000, " {\\rm OnSubgroups%d,%d}", nb_subgroups, group_order);

	label.assign(old_action->label);
	label_tex.assign(old_action->label_tex);
	label.append(str1);
	label_tex.append(str2);

	f_has_subaction = TRUE;
	subaction = old_action;
	if (f_v) {
		cout << "action::induced_action_on_subgroups "
				"allocating action_on_subgroups" << endl;
	}
	AOS = NEW_OBJECT(induced_actions::action_on_subgroups);
	AOS->init(old_action, S, nb_subgroups,
			group_order, Subgroups, verbose_level - 1);
	if (f_v) {
		cout << "action::induced_action_on_subgroups "
				"after action_on_subgroups init" << endl;
	}
	type_G = action_on_subgroups_t;
	G.on_subgroups = AOS;
	f_allocated = TRUE;
	make_element_size = old_action->make_element_size;
	low_level_point_size = 0;
	
	f_has_strong_generators = FALSE;
	
	degree = nb_subgroups;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	if (f_v) {
		cout << "action::induced_action_on_subgroups "
				"calling allocate_base_data" << endl;
	}
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = old_action->elt_size_in_int;
	coded_elt_size_in_char = old_action->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_v) {
		cout << "action::induced_action_on_subgroups "
				"finished, created action " << label << endl;
		//Sims->print_transversal_lengths();
		//cout << endl;
		print_info();
	}
}

void action::induced_action_by_restriction_on_orbit_with_schreier_vector(
	action &old_action,
	int f_induce_action, groups::sims *old_G,
	data_structures_groups::schreier_vector *Schreier_vector,
	int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_by_restriction *ABR;
	
	if (f_v) {
		cout << "action::induced_action_by_restriction_on_orbit_with_schreier_vector" << endl;
		cout << "old_action ";
		old_action.print_info();
		cout << "pt = " << pt << endl;
		cout << "f_induce_action = " << f_induce_action << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_res_sv%d", pt);
	snprintf(str2, 1000, " {\\rm res sv%d}", pt);

	label.assign(old_action.label);
	label_tex.assign(old_action.label_tex);
	label.append(str1);
	label_tex.append(str2);

	f_has_subaction = TRUE;
	subaction = &old_action;
	ABR = NEW_OBJECT(induced_actions::action_by_restriction);
	
	if (f_v) {
		cout << "action::induced_action_by_restriction_on_orbit_with_schreier_vector "
				"before ABR->init_single_orbit_from_schreier_vector" << endl;
	}
	ABR->init_single_orbit_from_schreier_vector(Schreier_vector, pt, verbose_level - 1);
	if (f_v) {
		cout << "action::induced_action_by_restriction_on_orbit_with_schreier_vector "
				"after ABR->init_single_orbit_from_schreier_vector" << endl;
	}
	
	type_G = action_by_restriction_t;
	G.ABR = ABR;
	f_allocated = TRUE;
	make_element_size = old_action.make_element_size;
	low_level_point_size = old_action.low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = ABR->nb_points;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	if (FALSE) {
		cout << "action::induced_action_by_restriction_on_orbit_with_schreier_vector "
				"calling allocate_base_data" << endl;
	}
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = old_action.elt_size_in_int;
	coded_elt_size_in_char = old_action.coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_induce_action) {
		if (f_v) {
			cout << "action::induced_action_by_restriction_on_orbit_with_schreier_vector "
					"calling induced_action_override_sims" << endl;
		}
		induced_action_override_sims(old_action, old_G, verbose_level - 2);
	}
	if (f_v) {
		cout << "action::induced_action_by_restriction_on_orbit_with_schreier_vector "
				"finished, created action " << label << endl;
		//Sims->print_transversal_lengths();
		//cout << endl;
		print_info();
	}
}

void action::original_point_labels(long int *points, int nb_points,
		long int *&original_points, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::original_point_labels" << endl;
	}
	if (type_G == action_by_restriction_t) {
		induced_actions::action_by_restriction *ABR;

		original_points = NEW_lint(nb_points);

		ABR = G.ABR;


		int i;
		long int a, b;

		for (i = 0; i < nb_points; i++) {
			a = points[i];
			b = ABR->original_point(a);
			original_points[i] = b;
		}
	}
	else {
		cout << "action::original_point_labels type must be action_by_restriction_t" << endl;
		exit(1);
	}

}

action *action::restricted_action(
		long int *points, int nb_points, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	induced_actions::action_by_restriction *ABR;
	
	if (f_v) {
		cout << "action::restricted_action" << endl;
		cout << "old_action ";
		print_info();
		cout << "nb_points = " << nb_points << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	A = NEW_OBJECT(action);

	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_res%d", nb_points);
	snprintf(str2, 1000, " {\\rm res%d}", nb_points);

	A->label.assign(label);
	A->label_tex.assign(label_tex);
	A->label.append(str1);
	A->label_tex.append(str2);


	A->f_has_subaction = TRUE;
	A->subaction = this;
	ABR = NEW_OBJECT(induced_actions::action_by_restriction);
	ABR->init(nb_points, points, verbose_level);
	A->type_G = action_by_restriction_t;
	A->G.ABR = ABR;
	A->f_allocated = TRUE;
	A->make_element_size = make_element_size;
	A->low_level_point_size = low_level_point_size;
	
	A->f_has_strong_generators = FALSE;
	
	A->degree = nb_points;
	//A->base_len = 0;
	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();
	if (FALSE) {
		cout << "action::restricted_action "
				"calling allocate_base_data" << endl;
	}
	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//A->allocate_base_data(0);
	
	
	A->elt_size_in_int = elt_size_in_int;
	A->coded_elt_size_in_char = coded_elt_size_in_char;
	
	A->allocate_element_data();
	
	if (f_v) {
		cout << "action::restricted_action finished, "
				"created action " << A->label << endl;
		A->print_info();
	}
	return A;
}

action *action::create_induced_action_by_restriction(
		groups::sims *S, int size, long int *set, int f_induce,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A2;

	if (f_v) {
		cout << "action::create_induced_action_by_restriction" << endl;
	}
	A2 = NEW_OBJECT(action);
	A2->induced_action_by_restriction_internal_function(*this,
			f_induce,  S, size, set, verbose_level - 1);
	if (f_v) {
		cout << "action::create_induced_action_by_restriction done" << endl;
	}
	return A2;
}

void action::induced_action_by_restriction_internal_function(
	action &old_action,
	int f_induce_action, groups::sims *old_G,
	int nb_points, long int *points, int verbose_level)
// uses action_by_restriction data type
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_by_restriction *ABR;
	
	if (f_v) {
		cout << "action::induced_action_by_restriction_internal_function" << endl;
		cout << "old_action ";
		old_action.print_info();
		cout << "nb_points = " << nb_points << endl;
		cout << "f_induce_action = " << f_induce_action << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}


	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_res%d", nb_points);
	snprintf(str2, 1000, " {\\rm res%d}", nb_points);

	label.assign(old_action.label);
	label_tex.assign(old_action.label_tex);
	label.append(str1);
	label_tex.append(str2);


	f_has_subaction = TRUE;
	subaction = &old_action;
	ABR = NEW_OBJECT(induced_actions::action_by_restriction);
	ABR->init(nb_points, points, verbose_level);
	type_G = action_by_restriction_t;
	G.ABR = ABR;
	f_allocated = TRUE;
	make_element_size = old_action.make_element_size;
	low_level_point_size = old_action.low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = nb_points;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	if (FALSE) {
		cout << "action::induced_action_by_restriction_internal_function "
				"calling allocate_base_data" << endl;
	}
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = old_action.elt_size_in_int;
	coded_elt_size_in_char = old_action.coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_induce_action) {
		if (f_v) {
			cout << "action::induced_action_by_restriction_internal_function "
					"calling induced_action_override_sims" << endl;
		}
		induced_action_override_sims(old_action,
				old_G, verbose_level - 2);
	}
	if (f_v) {
		cout << "action::induced_action_by_restriction_internal_function "
				"finished, created action " << label << endl;
		//Sims->print_transversal_lengths();
		//cout << endl;
		print_info();
	}
}

void action::induced_action_on_pairs(
	action &old_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "action::induced_action_on_pairs" << endl;
		cout << "the old_action " << old_action.label
			<< " has base_length = " << old_action.base_len()
			<< " and degree " << old_action.degree << endl;
	}

	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_on_pairs");
	snprintf(str2, 1000, " {\\rm OnPairs}");

	label.assign(old_action.label);
	label_tex.assign(old_action.label_tex);
	label.append(str1);
	label_tex.append(str2);


	f_has_subaction = TRUE;
	subaction = &old_action;
	type_G = action_on_pairs_t;
	f_allocated = FALSE;
	
	f_has_strong_generators = FALSE;
	
	degree = Combi.int_n_choose_k(old_action.degree, 2);
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = old_action.elt_size_in_int;
	coded_elt_size_in_char = old_action.coded_elt_size_in_char;
	
	allocate_element_data();
	
	induced_action_override_sims(old_action,
			old_G, verbose_level - 2);
	if (f_v) {
		cout << "action::induced_action_on_pairs "
				"finished, created action " << label << endl;
		print_info();
	}
}

action *action::create_induced_action_on_ordered_pairs(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	
	if (f_v) {
		cout << "action::create_induced_action_on_ordered_pairs" << endl;
	}
	A = NEW_OBJECT(action);
	A->induced_action_on_ordered_pairs(*this,
			NULL, 0 /* verbose_level*/);
	if (f_v) {
		cout << "action::create_induced_action_on_ordered_pairs done" << endl;
	}
	return A;
}


void action::induced_action_on_ordered_pairs(
	action &old_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action::induced_action_on_ordered_pairs" << endl;
		cout << "the old_action " << old_action.label
				<< " has base_length = " << old_action.base_len()
			<< " and degree " << old_action.degree << endl;
	}

	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_on_ordered_pairs");
	snprintf(str2, 1000, " {\\rm OnOrderedPairs}");

	label.assign(old_action.label);
	label_tex.assign(old_action.label_tex);
	label.append(str1);
	label_tex.append(str2);


	f_has_subaction = TRUE;
	subaction = &old_action;
	type_G = action_on_ordered_pairs_t;
	f_allocated = FALSE;
	
	f_has_strong_generators = FALSE;
	
	degree = old_action.degree * (old_action.degree - 1);
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = old_action.elt_size_in_int;
	coded_elt_size_in_char = old_action.coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (old_G) {
		induced_action_override_sims(old_action,
				old_G, verbose_level - 2);
	}
	if (f_v) {
		cout << "action::induced_action_on_ordered_pairs "
				"finished, created action " << label << endl;
		print_info();
	}
}

void action::induced_action_on_k_subsets(
	action &old_action, int k,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_k_subsets *On_k_subsets;
	
	if (f_v) {
		cout << "action::induced_action_on_k_subsets" << endl;
		cout << "the old_action " << old_action.label
				//<< " has base_length = " << old_action.base_len()
			<< " has degree " << old_action.degree << endl;
	}
	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_on_%d_subsets",k);
	snprintf(str2, 1000, "^{[%d]}", k);

	label.assign(old_action.label);
	label_tex.assign(old_action.label_tex);
	label.append(str1);
	label_tex.append(str2);

	On_k_subsets = NEW_OBJECT(induced_actions::action_on_k_subsets);
	On_k_subsets->init(&old_action, k, verbose_level);


	f_has_subaction = TRUE;
	subaction = &old_action;
	type_G = action_on_k_subsets_t;
	G.on_k_subsets = On_k_subsets;
	f_allocated = TRUE;
	
	
	f_has_strong_generators = FALSE;
	
	degree = On_k_subsets->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = old_action.elt_size_in_int;
	coded_elt_size_in_char = old_action.coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_v) {
		cout << "action::induced_action_on_k_subsets "
				"finished, created action " << label << endl;
		print_info();
	}
}

void action::induced_action_on_orbits(action *old_action,
		groups::schreier *Sch, int f_play_it_safe,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_orbits *On_orbits;
	
	if (f_v) {
		cout << "action::induced_action_on_orbits" << endl;
		cout << "the old_action " << old_action->label
				<< " has base_length = " << old_action->base_len()
			<< " and degree " << old_action->degree << endl;
	}


	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_on_orbits_%d", Sch->nb_orbits);
	snprintf(str2, 1000, " {\\rm OnOrbits}_{%d}", Sch->nb_orbits);

	label.assign(old_action->label);
	label_tex.assign(old_action->label_tex);
	label.append(str1);
	label_tex.append(str2);


	On_orbits = NEW_OBJECT(induced_actions::action_on_orbits);
	On_orbits->init(old_action, Sch, f_play_it_safe, verbose_level);


	f_has_subaction = TRUE;
	subaction = old_action;
	type_G = action_on_orbits_t;
	G.OnOrbits = On_orbits;
	f_allocated = TRUE;
	
	
	f_has_strong_generators = FALSE;
	
	degree = On_orbits->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = old_action->elt_size_in_int;
	coded_elt_size_in_char = old_action->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_v) {
		cout << "action::induced_action_on_orbits "
				"finished, created action " << label << endl;
		print_info();
	}
}

void action::induced_action_on_flags(action *old_action,
	int *type, int type_len,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_flags *On_flags;
	
	if (f_v) {
		cout << "action::induced_action_on_flags" << endl;
		cout << "the old_action " << old_action->label
			<< " has base_length = " << old_action->base_len()
			<< " and degree " << old_action->degree << endl;
	}


	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_on_flags");
	snprintf(str2, 1000, " {\\rm OnFlags}");

	label.assign(old_action->label);
	label_tex.assign(old_action->label_tex);
	label.append(str1);
	label_tex.append(str2);


	On_flags = NEW_OBJECT(induced_actions::action_on_flags);
	On_flags->init(old_action, type,
			type_len, verbose_level);


	f_has_subaction = TRUE;
	subaction = old_action;
	type_G = action_on_flags_t;
	G.OnFlags = On_flags;
	f_allocated = TRUE;
	
	
	f_has_strong_generators = FALSE;
	
	degree = On_flags->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = old_action->elt_size_in_int;
	coded_elt_size_in_char = old_action->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_v) {
		cout << "action::induced_action_on_flags "
				"finished, created action " << label << endl;
		print_info();
	}
}

void action::induced_action_on_bricks(action &old_action,
		combinatorics::brick_domain *B, int f_linear_action,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_bricks *On_bricks;
	
	if (f_v) {
		cout << "action::induced_action_on_bricks" << endl;
		cout << "the old_action " << old_action.label
			<< " has base_length = " << old_action.base_len()
			<< " and degree " << old_action.degree << endl;
	}

	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_on_bricks");
	snprintf(str2, 1000, " {\\rm OnBricks}");

	label.assign(old_action.label);
	label_tex.assign(old_action.label_tex);
	label.append(str1);
	label_tex.append(str2);


	On_bricks = NEW_OBJECT(induced_actions::action_on_bricks);
	On_bricks->init(&old_action, B, f_linear_action, verbose_level);


	f_has_subaction = TRUE;
	subaction = &old_action;
	type_G = action_on_bricks_t;
	G.OnBricks = On_bricks;
	f_allocated = TRUE;
	
	
	f_has_strong_generators = FALSE;
	
	degree = B->nb_bricks;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = old_action.elt_size_in_int;
	coded_elt_size_in_char = old_action.coded_elt_size_in_char;
	
	allocate_element_data();
	
	//induced_action_override_sims(old_action, old_G, verbose_level - 2);
	if (f_v) {
		cout << "action::induced_action_on_bricks finished, "
				"created action " << label << endl;
		print_info();
	}
}

void action::induced_action_on_andre(action *An,
	action *An1, geometry::andre_construction *Andre,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	induced_actions::action_on_andre *On_andre;
	
	if (f_v) {
		cout << "action::induced_action_on_andre" << endl;
		cout << "action An = " << An->label
				<< " has degree " << An->degree << endl;
		cout << "action An1 = " << An1->label
				<< " has degree " << An1->degree << endl;
	}

	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_on_andre");
	snprintf(str2, 1000, " {\\rm OnAndre}");

	label.assign(An1->label);
	label_tex.assign(An1->label_tex);
	label.append(str1);
	label_tex.append(str2);

	On_andre = NEW_OBJECT(induced_actions::action_on_andre);
	On_andre->init(An, An1, Andre, verbose_level);


	f_has_subaction = TRUE;
	subaction = An1;
	type_G = action_on_andre_t;
	G.OnAndre = On_andre;
	f_allocated = TRUE;
	
	
	f_has_strong_generators = FALSE;
	
	degree = On_andre->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = An1->elt_size_in_int;
	coded_elt_size_in_char = An1->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_v) {
		cout << "action::induced_action_on_andre "
				"finished, created action " << label << endl;
		print_info();
	}
}

void action::setup_product_action(action *A1, action *A2,
	int f_use_projections, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	induced_actions::product_action *PA;
	int i;
	
	if (f_v) {
		cout << "action::setup_product_action" << endl;
	}

	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_product_action");
	snprintf(str2, 1000, " {\\rm ProductAction}");

	label.assign(A1->label);
	label_tex.assign(A1->label_tex);
	label.assign("_");
	label_tex.assign(",");
	label.assign(A2->label);
	label_tex.assign(A2->label_tex);
	label.append(str1);
	label_tex.append(str2);


	PA = NEW_OBJECT(induced_actions::product_action);
	PA->init(A1, A2, f_use_projections, verbose_level);
	f_has_subaction = TRUE;
	subaction = NULL;
	type_G = product_action_t;
	G.product_action_data = PA;
	f_allocated = TRUE;
	
	f_has_strong_generators = FALSE;
	
	degree = PA->degree;
	
	//base_len = 0;
	
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	
	//Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	//Stabilizer_chain->allocate_base_data(this, 0);
	//allocate_base_data(0);
	
	
	elt_size_in_int = PA->elt_size_in_int;
	coded_elt_size_in_char = PA->coded_elt_size_in_char;
	
	make_element_size = A1->make_element_size + A2->make_element_size;
	
	allocate_element_data();

	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	set_base_len(A1->base_len() + A2->base_len());
	if (f_use_projections) {
		for (i = 0; i < A1->base_len(); i++) {
			base_i(i) = A1->base_i(i);
		}
		for (i = 0; i < A2->base_len(); i++) {
			base_i(A1->base_len() + i) = A1->degree + A2->base_i(i);
		}
	}
	else {
		for (i = 0; i < A1->base_len(); i++) {
			base_i(i) = A1->base_i(i) * A2->degree;
		}
		for (i = 0; i < A2->base_len(); i++) {
			base_i(A1->base_len() + i) = A2->base_i(i);
		}
	}
	
	if (f_vv) {
		cout << "make_element_size=" << make_element_size << endl;
		cout << "base_len=" << base_len() << endl;
	}
	if (f_v) {
		cout << "action::setup_product_action finished" << endl;
		print_info();
	}
}


void action::induced_action_on_homogeneous_polynomials(
	action *A_old,
	ring_theory::homogeneous_polynomial_domain *HPD,
	int f_induce_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	induced_actions::action_on_homogeneous_polynomials *OnHP;
	
	if (f_v) {
		cout << "action::induced_action_on_homogeneous_polynomials "
				"f_induce_action=" << f_induce_action << endl;
	}
	A = A_old;
	OnHP = NEW_OBJECT(induced_actions::action_on_homogeneous_polynomials);

	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_on_homog_poly_%d_%d", HPD->nb_variables, HPD->degree);
	snprintf(str2, 1000, " {\\rm OnHomPoly}_{%d,%d}", HPD->nb_variables, HPD->degree);

	label.assign(A->label);
	label_tex.assign(A->label_tex);
	label.append(str1);
	label_tex.append(str2);


	if (f_v) {
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len()
			<< " and degree " << A->degree << endl;
	}
	f_has_subaction = TRUE;
	subaction = A;
	if (A->type_G != matrix_group_t) {
		cout << "action::induced_action_on_homogeneous_polynomials "
				"action not of matrix group type" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "action::induced_action_on_homogeneous_polynomials "
				"before OnHP->init" << endl;
	}
	OnHP->init(A, HPD, verbose_level);
	if (f_v) {
		cout << "action::induced_action_on_homogeneous_polynomials "
				"after OnHP->init" << endl;
	}

	type_G = action_on_homogeneous_polynomials_t;
	G.OnHP = OnHP;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	low_level_point_size = OnHP->low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = OnHP->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	f_is_linear = TRUE;
	dimension = OnHP->dimension;
	
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_induce_action) {
		induced_action_override_sims(*A, old_G, 0/*verbose_level - 2*/);
	}

	if (f_v) {
		cout << "action::induced_action_on_homogeneous_polynomials "
				"finished, created action " << label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		print_info();
	}
}

void action::induced_action_on_homogeneous_polynomials_given_by_equations(
	action *A_old,
	ring_theory::homogeneous_polynomial_domain *HPD,
	int *Equations, int nb_equations, 
	int f_induce_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	induced_actions::action_on_homogeneous_polynomials *OnHP;
	
	if (f_v) {
		cout << "action::induced_action_on_homogeneous_"
				"polynomials_given_by_equations "
				"f_induce_action=" << f_induce_action << endl;
	}
	A = A_old;
	OnHP = NEW_OBJECT(induced_actions::action_on_homogeneous_polynomials);

	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_on_homog_poly_%d_%d_eqn%d", HPD->nb_variables, HPD->degree, nb_equations);
	snprintf(str2, 1000, " {\\rm OnHomPolyEqn}_{%d,%d%d}", HPD->nb_variables, HPD->degree, nb_equations);

	label.assign(A->label);
	label_tex.assign(A->label_tex);
	label.append(str1);
	label_tex.append(str2);


	if (f_v) {
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len()
			<< " and degree " << A->degree << endl;
	}
	f_has_subaction = TRUE;
	subaction = A;
	if (A->type_G != matrix_group_t) {
		cout << "action::induced_action_on_homogeneous_"
				"polynomials_given_by_equations "
				"action not of matrix group type" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "action::induced_action_on_homogeneous_"
				"polynomials_given_by_equations "
				"before OnHP->init" << endl;
	}
	OnHP->init(A, HPD, verbose_level);
	if (f_v) {
		cout << "action::induced_action_on_homogeneous_"
				"polynomials_given_by_equations "
				"after OnHP->init" << endl;
	}

	if (f_v) {
		cout << "action::induced_action_on_homogeneous_"
				"polynomials_given_by_equations "
				"before OnHP->init_invariant_set_of_equations" << endl;
	}
	OnHP->init_invariant_set_of_equations(
			Equations, nb_equations, verbose_level);
	if (f_v) {
		cout << "action::induced_action_on_homogeneous_"
				"polynomials_given_by_equations "
				"after OnHP->init_invariant_set_of_equations" << endl;
	}

	type_G = action_on_homogeneous_polynomials_t;
	G.OnHP = OnHP;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	low_level_point_size = OnHP->low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = OnHP->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();
	f_is_linear = TRUE;
	dimension = OnHP->dimension;
	
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_induce_action) {
		induced_action_override_sims(*A,
				old_G, 0/*verbose_level - 2*/);
	}

	if (f_v) {
		cout << "action::induced_action_on_homogeneous_"
				"polynomials_given_by_equations "
				"finished, created action " << label << endl;
		cout << "degree=" << A->degree << endl;
		cout << "make_element_size=" << A->make_element_size << endl;
		cout << "low_level_point_size=" << A->low_level_point_size << endl;
		print_info();
	}
}




void action::induced_action_recycle_sims(action &old_action, 
	int verbose_level)
{
	groups::sims *old_G;
	
	if (!old_action.f_has_sims) {
		cout << "action::induced_action_recycle_sims: "
				"old action must have sims" << endl;
		exit(1);
	}
	old_G = old_action.Sims;
	induce(&old_action, old_G, 
		0 /* base_of_choice_len */, NULL /* base_of_choice */, 
		verbose_level);
}

void action::induced_action_override_sims(
	action &old_action, groups::sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action::induced_action_override_sims before induce" << endl;
	}
	induce(&old_action, old_G, 
		0 /* base_of_choice_len */, NULL /* base_of_choice */, 
		verbose_level);
	if (f_v) {
		cout << "action::induced_action_override_sims done" << endl;
	}
}

void action::induce(action *old_action, groups::sims *old_G,
	int base_of_choice_len, long int *base_of_choice,
	int verbose_level)
// after this procedure, action will have
// a sims for the group and the kernel
// it will also have strong generators
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	action *subaction;
	groups::sims *G, *K;
		// will become part of the action object
		// 'this' by the end of this procedure
	ring_theory::longinteger_object go, go1, go2, go3;
	ring_theory::longinteger_object G_order, K_order;
	ring_theory::longinteger_domain D;
	int b, i, old_base_len;
	action *fallback_action;
	
	if (f_v) {
		cout << "action::induce verbose_level=" << verbose_level << endl;
	}
	if (f_v) {
		cout << "inducing from action:" << endl;
		old_action->print_info();
		cout << "the old group is in action:" << endl;
		old_G->A->print_info();
	}
	
	if (old_action->subaction) {
		if (f_vv) {
			cout << "action::induce has subaction" << endl;
		}
		subaction = old_action->subaction;
		if (f_vv) {
			cout << "subaction is ";
			subaction->print_info();
		}
	}
	else {
		if (f_vv) {
			cout << "action::induce does not have subaction" << endl;
		}
		subaction = old_action;
	}
	old_G->group_order(go);
	old_action->group_order(go1);
	subaction->group_order(go2);
	if (f_v) {
		cout << "action::induce" << endl;
		cout << "from old action " << old_action->label << endl;
		cout << "subaction " << subaction->label << endl;
		cout << "target order = " << go << endl;
		cout << "old_action order = " << go1 << endl;
		cout << "subaction order = " << go2 << endl;
		cout << "degree = " << old_action->degree << endl;
		cout << "subaction->degree = " << subaction->degree << endl;
		cout << "base_length = " << old_action->base_len() << endl;
		cout << "subaction->base_len = " << subaction->base_len() << endl;
		if (base_of_choice_len) {
			cout << "base of choice:" << endl;
			Lint_vec_print(cout, base_of_choice, base_of_choice_len);
			cout << endl;
		}
		else {
			cout << "no base of choice" << endl;
		}
	}
	
	G = NEW_OBJECT(groups::sims);
	K = NEW_OBJECT(groups::sims);
	if (f_v) {
		cout << "action::induce: before G->init_without_base(this);" << endl;
	}
	G->init_without_base(this, verbose_level - 2);
	if (f_v) {
		cout << "action::induce: after G->init_without_base(this);" << endl;
	}
	
	
	if (base_of_choice_len) {
		if (f_vv) {
			cout << "action::induce: initializing base of choice" << endl;
		}
		for (i = 0; i < base_of_choice_len; i++) {
			b = base_of_choice[i];
			if (f_vv) {
				cout << i << "-th base point is " << b << endl;
			}
			old_base_len = base_len();
			Stabilizer_chain->reallocate_base(b);
			G->reallocate_base(old_base_len, verbose_level - 2);
		}
		if (f_vv) {
			cout << "action::induce initializing base of choice finished"
					<< endl;
		}
	}

	fallback_action = subaction; // changed A. Betten Dec 27, 2011 !!!
	//fallback_action = old_action; // changed back A. Betten, May 27, 2012 !!!
		// The BLT search needs old_action
		// the translation plane search needs subaction
	if (fallback_action->base_len() == 0) {
		if (f_vv) {
			cout << "WARNING: action::induce fallback_action->base_len == 0"
					<< endl;
			cout << "fallback_action=" << fallback_action->label << endl;
			cout << "subaction=" << subaction->label << endl;
			cout << "old_action=" << old_action->label << endl;
			cout << "old_G->A=" << old_G->A->label << endl;
		}
		fallback_action = old_G->A;
		if (f_vv) {
			cout << "changing fallback action to " << fallback_action->label
					<< endl;
		}
	}
	if (f_v) {
		cout << "action::induce: before K->init" << endl;
	}
	K->init(fallback_action, verbose_level - 2);
	if (f_v) {
		cout << "action::induce: after K->init" << endl;
	}

	if (f_v) {
		cout << "action::induce before G->init_trivial_group" << endl;
	}
		
	G->init_trivial_group(verbose_level - 2);

	if (f_v) {
		cout << "action::induce before K->init_trivial_group" << endl;
	}
	K->init_trivial_group(verbose_level - 2);
	if (f_v) {
		cout << "action::induce "
				"after init_trivial_group" << endl;
		cout << "action::induce "
				"before G->build_up_group_random_process" << endl;
	}

	G->build_up_group_random_process(K, old_G, go, 
		FALSE /*f_override_chose_next_base_point*/,
		NULL /*choose_next_base_point_method*/, 
		verbose_level - 3);
	if (f_v) {
		cout << "action::induce "
				"after G->build_up_group_random_process" << endl;
	}

	G->group_order(G_order);
	K->group_order(K_order);
	if (f_v) {
		cout << "action::induce: ";
		cout << "found a group in action " << G->A->label
				<< " of order " << G_order << " ";
		cout << "transversal lengths:" << endl;
		for (int t = 0; t < G->A->base_len(); t++) {
			cout << G->get_orbit_length(t) << ", ";
		}
		//int_vec_print(cout, G->get_orbit_length(i), G->A->base_len());
		cout << endl;

		cout << "kernel in action " << K->A->label
				<< " of order " << K_order << " ";
		cout << "transversal lengths:" << endl;
		for (int t = 0; t < G->A->base_len(); t++) {
			cout << K->get_orbit_length(t) << ", ";
		}
		//int_vec_print(cout, K->get_orbit_length(), K->A->base_len());
		cout << endl;
	}
	D.mult(G_order, K_order, go3);
	if (D.compare(go3, go) != 0) {
		cout << "action::induce group orders do not match: "
				<< go3 << " != " << go << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "action::induce product of group orders equals "
				"old group order" << endl;
	}
	if (f_vv) {
		cout << "action::induce before init_sims_only" << endl;
	}
	init_sims_only(G, verbose_level - 2);
	f_has_kernel = TRUE;
	Kernel = K;
	
	//init_transversal_reps_from_stabilizer_chain(G, verbose_level - 2);
	if (f_vv) {
		cout << "action::induce after init_sims, "
				"calling compute_strong_generators_from_sims" << endl;
	}
	compute_strong_generators_from_sims(verbose_level - 2);
	if (f_v) {
		cout << "action::induce done" << endl;
	}
}

int action::least_moved_point_at_level(int level, int verbose_level)
{
	return Sims->least_moved_point_at_level(level, verbose_level);
}

void action::lex_least_base_in_place(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *set;
	long int *old_base;
	int i, lmp, old_base_len;

	if (f_v) {
		cout << "action::lex_least_base_in_place action "
				<< label << " base=";
		//int_vec_print(cout, Stabilizer_chain->base, base_len());
		cout << endl;
		print_info();
		//cout << "the generators are:" << endl;
		//Sims->print_generators();
	}

	set = NEW_lint(degree);
	old_base = NEW_lint(base_len());
	old_base_len = base_len();
	for (i = 0; i < base_len(); i++) {
		old_base[i] = base_i(i);
	}
	
	
	
	for (i = 0; i < base_len(); i++) {
		set[i] = base_i(i);
		if (f_v) {
			cout << "action::lex_least_base_in_place "
					"i=" << i << " computing the least moved point" << endl;
		}
		lmp = least_moved_point_at_level(i, verbose_level - 2);
		if (f_v) {
			cout << "action::lex_least_base_in_place "
					"i=" << i << " the least moved point is " << lmp << endl;
		}
		if (lmp >= 0 && lmp < base_i(i)) {
			if (f_v) {
				cout << "action::lex_least_base_in_place "
						"i=" << i << " least moved point = " << lmp
					<< " less than base point " << base_i(i) << endl;
				cout << "doing a base change:" << endl;
			}
			set[i] = lmp;
			base_change_in_place(i + 1, set, verbose_level);
			if (f_v) {
				cout << "action::lex_least_base_in_place "
						"after base_change_in_place: action:" << endl;
				print_info();
			}
 		}
	}
	if (f_v) {
		cout << "action::lex_least_base_in_place "
				"done, action " << label << " base=";
		//int_vec_print(cout, Stabilizer_chain->base, base_len());
		cout << endl;
		print_info();
		//cout << "the generators are:" << endl;
		//Sims->print_generators();
		int f_changed = FALSE;

		if (old_base_len != base_len()) {
			f_changed = TRUE;
		}
		if (!f_changed) {
			for (i = 0; i < base_len(); i++) {
				if (old_base[i] != base_i(i)) {
					f_changed = TRUE;
					break;
				}
			}
		}
		if (f_changed) {
			cout << "The base has changed !!!" << endl;
			cout << "old base: ";
			Lint_vec_print(cout, old_base, old_base_len);
			cout << endl;
			cout << "new base: ";
			//int_vec_print(cout, Stabilizer_chain->base, base_len());
			cout << endl;
		}
	}
	FREE_lint(old_base);
	FREE_lint(set);
}

void action::lex_least_base(action *old_action, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *set;
	action *A, *old_A;
	int i, lmp;

	if (f_v) {
		cout << "action::lex_least_base action "
				<< old_action->label << " base=";
		//int_vec_print(cout, old_action->Stabilizer_chain->base, old_action->base_len());
		cout << endl;
	}
#if 0
	if (!f_has_sims) {
		cout << "action::lex_least_base fatal: does not have sims" << endl;
		exit(1);
	}
#endif
	

	if (f_v) {
		//cout << "the generators are:" << endl;
		//old_action->Sims->print_generators();
	}
	A = NEW_OBJECT(action);

	set = NEW_lint(old_action->degree);
	
	old_A = old_action;
	
	if (!old_action->f_has_sims) {
		cout << "action::lex_least_base does not have Sims" << endl;
		exit(1);
	}
	
	for (i = 0; i < old_A->base_len(); i++) {
		set[i] = old_A->base_i(i);
		if (f_v) {
			cout << "action::lex_least_base "
					"calling least_moved_point_at_level " << i << endl;
		}
		lmp = old_A->least_moved_point_at_level(i, verbose_level - 2);
		if (lmp < old_A->base_i(i)) {
			if (f_v) {
				cout << "action::lex_least_base least moved point = " << lmp 
					<< " less than base point " << old_A->base_i(i) << endl;
				cout << "doing a base change:" << endl;
			}
			set[i] = lmp;
			A = NEW_OBJECT(action);
			A->base_change(old_A, i + 1, set, verbose_level - 2);
			old_A = A;
		}
	}
	base_change(old_A, old_A->base_len(),
			old_A->get_base(), verbose_level - 1);
	FREE_lint(set);
	if (f_v) {
		cout << "action::lex_least_base action " << label << " base=";
		//int_vec_print(cout, Stabilizer_chain->base, base_len());
		cout << endl;
	}
}

int action::test_if_lex_least_base(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *AA;
	int i;
	
	if (f_v) {
		cout << "action::test_if_lex_least_base:" << endl;
		print_info();
	}

	AA = NEW_OBJECT(action);

	AA->lex_least_base(this, verbose_level);
	for (i = 0; i < base_len(); i++) {
		if (AA->base_len() >= i) {
			if (base_i(i) > AA->base_i(i)) {
				cout << "action::test_if_lex_least_base "
						"returns FALSE" << endl;
				cout << "base[i]=" << base_i(i) << endl;
				cout << "AA->base[i]=" << AA->base_i(i) << endl;
				FREE_OBJECT(AA);
				return FALSE;
			}
		}
	}
	FREE_OBJECT(AA);
	return TRUE;
}

void action::base_change_in_place(int size, long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_v5 = (verbose_level >= 5);
	action *A;
	int i;

	if (f_v) {
		cout << "action::base_change_in_place" << endl;
	}
	A = NEW_OBJECT(action);
	A->base_change(this, size, set, verbose_level);
	if (f_v) {
		cout << "action::base_change_in_place after base_change" << endl;
	}
	Stabilizer_chain->free_base_data();
	if (f_v5) {
		cout << "action::base_change_in_place after free_base_data" << endl;
	}
	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, A->base_len(), verbose_level);
	//allocate_base_data(A->base_len);
	if (f_v5) {
		cout << "action::base_change_in_place after allocate_base_data"
				<< endl;
	}
	set_base_len(A->base_len());
	for (i = 0; i < A->base_len(); i++) {
		base_i(i) = A->base_i(i);
	}
	if (f_v5) {
		cout << "action::base_change_in_place after copying base" << endl;
	}
	

	A->Sims->A = this; 
	A->Sims->gens.A = this; 
	A->Sims->gens_inv.A = this; 
		// not to forget: the sims also has an action pointer in it 
		// and this one has to be changed to the old action
	if (f_v5) {
		cout << "action::base_change_in_place "
				"after changing action pointer in A->Sims" << endl;
	}
	
	if (f_has_sims) {
		if (f_v5) {
			cout << "action::base_change_in_place "
					"before FREE_OBJECT Sims" << endl;
			cout << "Sims=" << Sims << endl;
		}
		FREE_OBJECT(Sims);
		if (f_v5) {
			cout << "action::base_change_in_place "
					"after FREE_OBJECT Sims" << endl;
		}
		Sims = NULL;
		f_has_sims = FALSE;
	}

	if (f_v5) {
		cout << "action::base_change_in_place after deleting sims" << endl;
	}

	if (f_v5) {
		cout << "action::base_change_in_place before init_sims_only" << endl;
	}
	init_sims_only(A->Sims, verbose_level);
	if (f_v5) {
		cout << "action::base_change_in_place after init_sims_only" << endl;
	}

	if (f_has_strong_generators) {
		f_has_strong_generators = FALSE;
		FREE_OBJECT(Strong_gens);
		Strong_gens = NULL;
	}

	A->f_has_sims = FALSE;
	A->Sims = NULL;
	
	if (f_v5) {
		cout << "action::base_change_in_place before FREE_OBJECT(A)" << endl;
	}
	FREE_OBJECT(A);
	if (f_v5) {
		cout << "action::base_change_in_place after FREE_OBJECT(A)" << endl;
	}

	compute_strong_generators_from_sims(verbose_level - 3);
	
	if (f_v) {
		cout << "action::base_change_in_place finished, created action"
				<< endl;
		print_info();
		//cout << "generators are:" << endl;
		//Sims->print_generators();
		//cout << "Sims:" << endl;
		//Sims->print(3);
	}
}

void action::base_change(action *old_action, 
	int size, long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "action::base_change to the following set:" << endl;
		Lint_vec_print(cout, set, size);
		cout << endl;
	}
	if (!old_action->f_has_sims) {
		cout << "action::base_change old_action does not have sims" << endl;
		exit(1);
	}
	f_has_subaction = TRUE;
	subaction = old_action;
	type_G = base_change_t;
	f_allocated = FALSE;
	
	f_has_strong_generators = FALSE;
	
	degree = old_action->degree;
	//base_len = 0;
	ptr = NEW_OBJECT(action_pointer_table);
	ptr->init_function_pointers_induced_action();

	Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	Stabilizer_chain->allocate_base_data(this, 0, verbose_level);
	//allocate_base_data(0);
	
	
	elt_size_in_int = old_action->elt_size_in_int;
	coded_elt_size_in_char = old_action->coded_elt_size_in_char;
	make_element_size = old_action->make_element_size;
	low_level_point_size = old_action->low_level_point_size;
	
	
	allocate_element_data();
		
	if (f_v) {
		cout << "action::base_change calling induce" << endl;
	}
	induce(old_action,
			old_action->Sims,
			size, set,
			verbose_level - 1);

	char str1[1000];
	char str2[1000];
	snprintf(str1, 1000, "_base_change");
	snprintf(str2, 1000, " {\\rm BaseChange}");

	label.assign(old_action->label);
	label_tex.assign(old_action->label_tex);
	label.append(str1);
	label_tex.append(str2);


	
	if (f_v) {
		ring_theory::longinteger_object go, K_go;
		group_order(go);
		Kernel->group_order(K_go);
		cout << "action::base_change finished" << endl;
		cout << "induced action has order " << go << endl;
		cout << "kernel has order " << K_go << endl;
		//cout << "generators are:" << endl;
		//Sims->print_generators();
	}
	if (FALSE) {
		Sims->print_generators();
		Sims->print_generators_as_permutations();
		Sims->print_basic_orbits();
	}
}


void action::create_orbits_on_subset_using_restricted_action(
		action *&A_by_restriction,
		groups::schreier *&Orbits, groups::sims *S,
		int size, long int *set,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_induce = FALSE;

	if (f_v) {
		cout << "action::create_orbits_on_subset_using_restricted_action" << endl;
	}
	A_by_restriction = create_induced_action_by_restriction(
			S,
			size, set,
			f_induce,
			verbose_level - 1);
	Orbits = NEW_OBJECT(groups::schreier);

	A_by_restriction->compute_all_point_orbits(*Orbits,
			S->gens, verbose_level - 2);
	if (f_v) {
		cout << "action::create_orbits_on_subset_using_restricted_action "
				"done" << endl;
	}
}

void action::create_orbits_on_sets_using_action_on_sets(
		action *&A_on_sets,
		groups::schreier *&Orbits, groups::sims *S,
		int nb_sets, int set_size, long int *sets,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_induce = FALSE;

	if (f_v) {
		cout << "action::create_orbits_on_sets_using_action_on_sets" << endl;
	}

	A_on_sets = create_induced_action_on_sets(
			nb_sets, set_size, sets,
			verbose_level);

	Orbits = NEW_OBJECT(groups::schreier);

	A_on_sets->compute_all_point_orbits(*Orbits, S->gens, verbose_level - 2);
	if (f_v) {
		cout << "action::create_orbits_on_sets_using_action_on_sets "
				"done" << endl;
	}
}




int action::choose_next_base_point_default_method(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int b;

	if (f_v) {
		cout << "action::choose_next_base_point_default_method" << endl;
		cout << "calling A->find_non_fixed_point" << endl;
	}
	b = find_non_fixed_point(Elt, verbose_level - 1);
	if (b == -1) {
		if (f_v) {
			cout << "action::choose_next_base_point_default_method "
					"cannot find another base point" << endl;
		}
		return -1;
	}
	if (f_v) {
		cout << "action::choose_next_base_point_default_method current base: ";
		//int_vec_print(cout, Stabilizer_chain->base, base_len());
		cout << " choosing next base point to be " << b << endl;
	}
	return b;
}

void action::generators_to_strong_generators(
	int f_target_go, ring_theory::longinteger_object &target_go,
	data_structures_groups::vector_ge *gens,
	groups::strong_generators *&Strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::generators_to_strong_generators" << endl;
		if (f_target_go) {
			cout << "action::generators_to_strong_generators "
					"trying to create a group of order " << target_go << endl;
		}
	}

	groups::sims *S;

	if (f_v) {
		cout << "action::generators_to_strong_generators "
				"before create_sims_from_generators_randomized" << endl;
	}

	S = create_sims_from_generators_randomized(
		gens, f_target_go,
		target_go, verbose_level - 2);

	if (f_v) {
		cout << "action::generators_to_strong_generators "
				"after create_sims_from_generators_randomized" << endl;
	}

	Strong_gens = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "action::generators_to_strong_generators "
				"before Strong_gens->init_from_sims" << endl;
	}
	Strong_gens->init_from_sims(S, verbose_level - 5);

	FREE_OBJECT(S);

	if (f_v) {
		cout << "action::generators_to_strong_generators done" << endl;
	}
}

void action::orbits_on_equations(
		ring_theory::homogeneous_polynomial_domain *HPD,
	int *The_equations, int nb_equations, groups::strong_generators *gens,
	action *&A_on_equations, groups::schreier *&Orb, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::orbits_on_equations" << endl;
	}

	A_on_equations = NEW_OBJECT(action);

	if (f_v) {
		cout << "action::orbits_on_equations "
				"creating the induced action on the equations:" << endl;
	}
	A_on_equations->induced_action_on_homogeneous_polynomials_given_by_equations(
		this,
		HPD,
		The_equations, nb_equations,
		FALSE /* f_induce_action */, NULL /* sims *old_G */,
		verbose_level);
	if (f_v) {
		cout << "action::orbits_on_equations "
				"The induced action on the equations has been created, "
				"degree = " << A_on_equations->degree << endl;
	}

	if (f_v) {
		cout << "action::orbits_on_equations "
				"computing orbits on the equations:" << endl;
	}
	Orb = gens->orbits_on_points_schreier(A_on_equations,
			verbose_level);

	if (f_v) {
		cout << "action::orbits_on_equations "
				"We found " << Orb->nb_orbits
				<< " orbits on the equations:" << endl;
		Orb->print_and_list_orbits_tex(cout);
	}

	if (f_v) {
		cout << "action::orbits_on_equations done" << endl;
	}
}



}}}

