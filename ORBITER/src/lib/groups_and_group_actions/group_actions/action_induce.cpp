// action_induce.C
//
// Anton Betten
// 1/1/2009

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

void action::init_action_on_lines(action *A,
		finite_field *F, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	action_on_grassmannian *A_lines;
	grassmann *Grass_lines;

	if (f_v) {
		cout << "action::init_action_on_lines" << endl;
		}

	A_lines = NEW_OBJECT(action_on_grassmannian);

	Grass_lines = NEW_OBJECT(grassmann);


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
	sims S;
	longinteger_object go1;

	S.init(A);
	S.init_generators(*A->Strong_gens->gens, 0/*verbose_level*/);
	S.compute_base_orbits_known_length(
			A->transversal_length, 0/*verbose_level - 1*/);
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
	int f_induce_action, sims *old_G, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	action_by_representation *Rep; // do not free
	//matrix_group *M;
	
	if (f_v) {
		cout << "action::induced_action_by_representation_on_conic "
				"f_induce_action=" << f_induce_action << endl;
		}
	
	A = A_old;
	sprintf(group_prefix, "%s_RepOnConic", A->label);
	sprintf(label, "%s_RepOnConic", A->label);
	sprintf(label_tex, "%s RepOnConic", A->label_tex);
	if (f_v) {
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len
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

	Rep = NEW_OBJECT(action_by_representation);
	Rep->init_action_on_conic(*A_old, verbose_level);

	type_G = action_by_representation_t;
	G.Rep = Rep;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	low_level_point_size = Rep->low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = Rep->degree;
	base_len = 0;
	init_function_pointers_induced_action();
	
	
	
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
	action_on_cosets *A_on_cosets,
	int f_induce_action, sims *old_G, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	//matrix_group *M;
	
	if (f_v) {
		cout << "action::induced_action_on_cosets "
				"f_induce_action=" << f_induce_action << endl;
		}
	A = A_on_cosets->A_linear;
	sprintf(group_prefix, "%s_Cosets_%d",
			A->label, A_on_cosets->dimension_of_subspace);
	sprintf(label, "%s_Cosets_%d",
			A->label, A_on_cosets->dimension_of_subspace);
	sprintf(label_tex, "%s Cosets_%d",
			A->label_tex, A_on_cosets->dimension_of_subspace);
	if (f_v) {
		cout << "the old_action " << A->label
				<< " has base_length = " << A->base_len
			<< " and degree " << A->degree << endl;
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
	base_len = 0;
	init_function_pointers_induced_action();
	
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_induce_action) {
		induced_action_override_sims(*A, old_G, verbose_level - 2);
		}

	if (f_v) {
		cout << "action::induced_action_on_cosets "
				"finished, created action " << label << endl;
		cout << "degree=" << A->degree << endl;
		print_info();
		}
}



void action::induced_action_on_factor_space(action *A_old, 
	action_on_factor_space *AF, 
	int f_induce_action, sims *old_G, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	//matrix_group *M;
	
	if (f_v) {
		cout << "action::induced_action_on_factor_space "
				"f_induce_action=" << f_induce_action << endl;
		}
	A = A_old;
	sprintf(group_prefix, "%s_Factorspace_%d_%d_%d",
			A->label, AF->len, AF->factor_space_len, AF->F->q);
	sprintf(label, "%s_Factorspace_%d_%d_%d",
			A->label, AF->len, AF->factor_space_len, AF->F->q);
	sprintf(label_tex, "%s Factorspace_%d_%d_%d",
			A->label_tex, AF->len, AF->factor_space_len, AF->F->q);
	if (f_v) {
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len
			<< " and degree " << A->degree << endl;
		}
	f_has_subaction = TRUE;
	subaction = A;
	if (!A->f_is_linear) {
		cout << "action::induced_action_on_factor_space "
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
	type_G = action_on_factor_space_t;
	G.AF = AF;
	f_allocated = FALSE;
	make_element_size = A->make_element_size;
	low_level_point_size = A->low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = AF->degree;
	base_len = 0;
	init_function_pointers_induced_action();
	
	
	
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
	action_on_grassmannian *AG;
	action *A;
	matrix_group *M;
	
	if (f_v) {
		cout << "action::induced_action_on_grassmannian" << endl;
		}
	A = NEW_OBJECT(action);

	sprintf(A->group_prefix, "%s_on_%d_subspaces", label, k);
	sprintf(A->label, "%s_on_%d_subspaces", label, k);
	sprintf(A->label_tex, "%s on %d subspaces", label, k);
	if (f_v) {
		cout << "the old_action " << label
			<< " has base_length = " << base_len
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
	AG = NEW_OBJECT(action_on_grassmannian);

	grassmann *Gr;

	Gr = NEW_OBJECT(grassmann);
	Gr->init(M->n, k, M->GFq, verbose_level);
	AG->init(*this, Gr, verbose_level);
	A->type_G = action_on_grassmannian_t;
	A->G.AG = AG;
	A->f_allocated = TRUE;
	A->make_element_size = make_element_size;
	A->low_level_point_size = AG->low_level_point_size;
	
	A->f_has_strong_generators = FALSE;
	
	A->degree = AG->degree.as_int();
	A->base_len = 0;
	if (f_v) {
		cout << "action::induced_action_on_grassmannian "
				"before init_function_pointers_induced_action" << endl;
		}
	A->init_function_pointers_induced_action();
	
	
	
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
	action_on_grassmannian *AG, 
	int f_induce_action, sims *old_G, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	//matrix_group *M;
	
	if (f_v) {
		cout << "action::induced_action_on_grassmannian "
				"f_induce_action=" << f_induce_action << endl;
		}
	A = A_old;
	sprintf(group_prefix, "%s_Grassmann_%d_%d_%d",
			A->label, AG->n, AG->k, AG->q);
	sprintf(label, "%s_Grassmann_%d_%d_%d",
			A->label, AG->n, AG->k, AG->q);
	sprintf(label_tex, "%s Grassmann_%d_%d_%d",
			A->label_tex, AG->n, AG->k, AG->q);
	if (f_v) {
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len
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
#if 0
	if (A->type_G == matrix_group_t) {
		if (f_v) {
			cout << "action::induced_action_on_grassmannian "
					"A->type_G == matrix_group_t" << endl;
			}
		M = A->G.matrix_grp;
		}
	else {
		if (f_v) {
			cout << "action::induced_action_on_grassmannian "
					"A->type_G != matrix_group_t" << endl;
			}
		action *sub = A->subaction;
		M = sub->G.matrix_grp;
		}
#endif
	type_G = action_on_grassmannian_t;
	G.AG = AG;
	f_allocated = FALSE;
	make_element_size = A->make_element_size;
	low_level_point_size = AG->low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = AG->degree.as_int();
	base_len = 0;
	if (f_v) {
		cout << "action::induced_action_on_grassmannian "
				"before init_function_pointers_induced_action" << endl;
		}
	init_function_pointers_induced_action();
	
	
	
	elt_size_in_int = A->elt_size_in_int;
	coded_elt_size_in_char = A->coded_elt_size_in_char;
	
	if (f_v) {
		cout << "action::induced_action_on_grassmannian "
				"before allocate_element_data" << endl;
		}
	allocate_element_data();
	
	if (f_induce_action) {
		if (f_v) {
			cout << "action::induced_action_on_grassmannian "
					"before induced_action_override_sims" << endl;
			}
		induced_action_override_sims(*A, old_G, verbose_level);
		}

	if (f_v) {
		cout << "action::induced_action_on_grassmannian finished, "
				"created action " << label << endl;
		cout << "degree=" << degree << endl;
		cout << "make_element_size=" << make_element_size << endl;
		cout << "low_level_point_size=" << low_level_point_size << endl;
		print_info();
		}
}

void action::induced_action_on_spread_set(action *A_old, 
	action_on_spread_set *AS, 
	int f_induce_action, sims *old_G, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	//matrix_group *M;
	
	if (f_v) {
		cout << "action::induced_action_on_spread_set "
				"f_induce_action=" << f_induce_action << endl;
		}
	A = A_old;
	sprintf(group_prefix, "%s_SpreadSet_%d_%d",
			A->label, AS->k, AS->q);
	sprintf(label, "%s_SpreadSet_%d_%d",
			A->label, AS->k, AS->q);
	sprintf(label_tex, "%s SpreadSet_%d_%d",
			A->label_tex, AS->k, AS->q);
	if (f_v) {
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len
			<< " and degree " << A->degree << endl;
		}
	f_has_subaction = TRUE;
	subaction = A;
	if (!A->f_is_linear) {
		cout << "action::induced_action_on_spread_set "
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

	f_is_linear = TRUE;

	
	type_G = action_on_spread_set_t;
	G.AS = AS;
	f_allocated = FALSE;
	make_element_size = A->make_element_size;
	low_level_point_size = AS->low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = AS->degree;
	base_len = 0;
	init_function_pointers_induced_action();
	
	
	
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
	action_on_orthogonal *AO, 
	int f_induce_action, sims *old_G, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	//matrix_group *M;
	
	if (f_v) {
		cout << "action::induced_action_on_orthogonal "
				"f_induce_action=" << f_induce_action << endl;
		}
	A = A_old;
	if (AO->f_on_points) {
		sprintf(group_prefix, "%s_orthogonal_on_points_%d_%d_%d",
				A->label, AO->O->epsilon, AO->O->n, AO->O->q);
		sprintf(label, "%s_orthogonal_on_points_%d_%d_%d",
				A->label, AO->O->epsilon, AO->O->n, AO->O->q);
		sprintf(label_tex, "%s orthogonal_on_points_%d_%d_%d",
				A->label_tex, AO->O->epsilon, AO->O->n, AO->O->q);
		}
	else if (AO->f_on_lines) {
		sprintf(group_prefix, "%s_orthogonal_on_lines_%d_%d_%d",
				A->label, AO->O->epsilon, AO->O->n, AO->O->q);
		sprintf(label, "%s_orthogonal_on_lines_%d_%d_%d",
				A->label, AO->O->epsilon, AO->O->n, AO->O->q);
		sprintf(label_tex, "%s orthogonal_on_lines_%d_%d_%d",
				A->label_tex, AO->O->epsilon, AO->O->n, AO->O->q);
		}
	else if (AO->f_on_points_and_lines) {
		sprintf(group_prefix, "%s_orthogonal_on_points_and_lines_%d_%d_%d",
				A->label, AO->O->epsilon, AO->O->n, AO->O->q);
		sprintf(label, "%s_orthogonal_on_points_and_lines_%d_%d_%d",
				A->label, AO->O->epsilon, AO->O->n, AO->O->q);
		sprintf(label_tex, "%s orthogonal_on_points_and_lines_%d_%d_%d",
				A->label_tex, AO->O->epsilon, AO->O->n, AO->O->q);
		}
	if (f_v) {
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len
			<< " and degree " << A->degree << endl;
		}
	f_has_subaction = TRUE;
	subaction = A;
	if (!A->f_is_linear) {
		cout << "action::induced_action_on_orthogonal "
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
	type_G = action_on_orthogonal_t;
	G.AO = AO;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	low_level_point_size = AO->low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = AO->degree;
	base_len = 0;
	init_function_pointers_induced_action();
	
	
	
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

void action::induced_action_on_wedge_product(action *A_old, 
	action_on_wedge_product *AW, 
	int f_induce_action, sims *old_G, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	//matrix_group *M;
	
	if (f_v) {
		cout << "action::induced_action_on_wedge_product "
				"f_induce_action=" << f_induce_action << endl;
		}
	A = A_old;
	sprintf(group_prefix, "%s_Wedge", A->label);
	sprintf(label, "%s_Wedge", A->label);
	sprintf(label_tex, "%s Wedge", A->label_tex);
	if (f_v) {
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len
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
	base_len = 0;
	init_function_pointers_induced_action();
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

void action::induced_action_by_subfield_structure(action *A_old, 
	action_by_subfield_structure *SubfieldStructure, 
	int f_induce_action, sims *old_G, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	//matrix_group *M;
	
	if (f_v) {
		cout << "action::induced_action_by_subfield_structure "
				"f_induce_action=" << f_induce_action << endl;
		}
	A = A_old;
	sprintf(group_prefix, "%s_subfield_%d",
			A->label, SubfieldStructure->q);
	sprintf(label, "%s_subfield_%d",
			A->label, SubfieldStructure->q);
	sprintf(label_tex, "%s subfield %d",
			A->label_tex, SubfieldStructure->q);
	if (f_v) {
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len
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
	base_len = 0;
	init_function_pointers_induced_action();
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


void action::induced_action_on_determinant(
		sims *old_G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action_on_determinant *AD;
	action *A;
	matrix_group *M;
	
	if (f_v) {
		cout << "action::induced_action_on_determinant" << endl;
		}
	A = old_G->A;
	sprintf(group_prefix, "%s_det", A->label);
	sprintf(label, "%s_det", A->label);
	sprintf(label_tex, "%s det", A->label_tex);
	if (f_v) {
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len
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
	AD = NEW_OBJECT(action_on_determinant);
	AD->init(*A, M->f_projective, M->n, verbose_level);
	type_G = action_on_determinant_t;
	G.AD = AD;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	
	f_has_strong_generators = FALSE;
	
	degree = AD->degree;
	base_len = 0;
	init_function_pointers_induced_action();
	
	allocate_base_data(0);
	
	
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
		sims *old_G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action_on_sign *OnSign;
	action *A;
	
	if (f_v) {
		cout << "action::induced_action_on_sign" << endl;
		}
	A = old_G->A;
	sprintf(group_prefix, "%s_OnSign", A->label);
	sprintf(label, "%s_OnSign", A->label);
	sprintf(label_tex, "%s sign", A->label_tex);
	if (f_v) {
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len
			<< " and degree " << A->degree << endl;
		}
	f_has_subaction = TRUE;
	subaction = A;
	OnSign = NEW_OBJECT(action_on_sign);
	OnSign->init(A, verbose_level);
	type_G = action_on_sign_t;
	G.OnSign = OnSign;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	
	f_has_strong_generators = FALSE;
	
	degree = OnSign->degree;
	base_len = 0;
	init_function_pointers_induced_action();
	
	allocate_base_data(0);
	
	
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

void action::induced_action_by_conjugation(sims *old_G, 
	sims *Base_group, int f_ownership,
	int f_basis, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action_by_conjugation *ABC;
	longinteger_object go;
	int goi;
	action *A;
	
	A = Base_group->A;
	if (f_v) {
		cout << "action::induced_action_by_conjugation" << endl;
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len
			<< " and degree " << A->degree << endl;
		}
	Base_group->group_order(go);
	goi = go.as_int();
	if (f_v) {
		cout << "we are acting on a group of order " << goi << endl;
		}
	sprintf(group_prefix, "%s_C%d", A->label, goi);
	sprintf(label, "%s_C%d", A->label, goi);
	sprintf(label_tex, "%s C%d", A->label_tex, goi);
	f_has_subaction = TRUE;
	subaction = A;
	ABC = NEW_OBJECT(action_by_conjugation);
	ABC->init(Base_group, f_ownership, verbose_level);
	type_G = action_by_conjugation_t;
	G.ABC = ABC;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	
	f_has_strong_generators = FALSE;
	
	degree = goi;
	base_len = 0;
	init_function_pointers_induced_action();
	
	allocate_base_data(0);
	
	
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
	int f_basis, sims *old_G,
	sims *Base_group, int f_ownership, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action_by_right_multiplication *ABRM;
	longinteger_object go;
	int goi;
	action *A;
	
	A = Base_group->A;
	if (f_v) {
		cout << "action::induced_action_by_right_multiplication" << endl;
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len
			<< " and degree " << A->degree << endl;
		}
	Base_group->group_order(go);
	goi = go.as_int();
	sprintf(group_prefix, "%s_R%d", A->label, goi);
	sprintf(label, "%s_R%d", A->label, goi);
	sprintf(label_tex, "%s R%d", A->label_tex, goi);
	if (f_v) {
		cout << "we are acting on a group of order " << goi << endl;
		}
	f_has_subaction = TRUE;
	subaction = A;
	ABRM = NEW_OBJECT(action_by_right_multiplication);
	ABRM->init(Base_group, f_ownership, verbose_level);
	type_G = action_by_right_multiplication_t;
	G.ABRM = ABRM;
	f_allocated = TRUE;
	make_element_size = A->make_element_size;
	
	f_has_strong_generators = FALSE;
	
	degree = goi;
	base_len = 0;
	init_function_pointers_induced_action();
	
	allocate_base_data(0);
	
	
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
		int nb_sets, int set_size, int *sets,
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
	action &old_action, sims *old_G,
	int nb_sets, int set_size, int *sets, 
	int f_induce_action, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = FALSE; // (verbose_level >= 2);
	action_on_sets *AOS;
	
	if (f_v) {
		cout << "action::induced_action_on_sets" << endl;
		cout << "action::induced_action_on_sets "
				"f_induce_action=" << f_induce_action << endl;
		cout << "action::induced_action_on_sets "
				"the old_action " << old_action.label
				<< " has base_length = " << old_action.base_len
			<< " and degree " << old_action.degree << endl;
		cout << "action::induced_action_on_sets "
				"verbose_level = " << verbose_level << endl;
		}
	sprintf(group_prefix, "%s_S%d",
			old_action.label, set_size);
	sprintf(label, "%s_S%d",
			old_action.label, set_size);
	sprintf(label_tex, "%s S%d",
			old_action.label_tex, set_size);
	f_has_subaction = TRUE;
	subaction = &old_action;
	if (f_v) {
		cout << "action::induced_action_on_sets "
				"allocating action_on_sets" << endl;
		}
	AOS = NEW_OBJECT(action_on_sets);
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
	base_len = 0;
	init_function_pointers_induced_action();
	if (f_v) {
		cout << "action::induced_action_on_sets "
				"calling allocate_base_data" << endl;
		}
	allocate_base_data(0);
	
	
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

action *action::create_induced_action_on_subgroups(sims *S, 
	int nb_subgroups, int group_order, subgroup **Subgroups,
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
	action *old_action, sims *S,
	int nb_subgroups, int group_order, subgroup **Subgroups, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = FALSE; // (verbose_level >= 2);
	action_on_subgroups *AOS;
	
	if (f_v) {
		cout << "action::induced_action_on_subgroups" << endl;
		cout << "action::induced_action_on_sets "
				"the old_action " << old_action->label
				<< " has base_length = " << old_action->base_len
			<< " and degree " << old_action->degree << endl;
		cout << "action::induced_action_on_subgroups "
				"verbose_level = " << verbose_level << endl;
		}
	sprintf(group_prefix, "%s_on_subgroups%d_%d",
			old_action->label, nb_subgroups, group_order);
	sprintf(label, "%s_on_subgroups%d_%d",
			old_action->label, nb_subgroups, group_order);
	sprintf(label_tex, "%s on_subgroups%d_%d",
			old_action->label_tex, nb_subgroups, group_order);
	f_has_subaction = TRUE;
	subaction = old_action;
	if (f_v) {
		cout << "action::induced_action_on_subgroups "
				"allocating action_on_subgroups" << endl;
		}
	AOS = NEW_OBJECT(action_on_subgroups);
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
	base_len = 0;
	init_function_pointers_induced_action();
	if (f_v) {
		cout << "action::induced_action_on_subgroups "
				"calling allocate_base_data" << endl;
		}
	allocate_base_data(0);
	
	
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
	int f_induce_action, sims *old_G, 
	schreier_vector *Schreier_vector,
	int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action_by_restriction *ABR;
	
	if (f_v) {
		cout << "action::induced_action_by_restriction_"
				"on_orbit_with_schreier_vector" << endl;
		cout << "old_action ";
		old_action.print_info();
		cout << "pt = " << pt << endl;
		cout << "f_induce_action = " << f_induce_action << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	sprintf(group_prefix, "%s_res_sv%d",
			old_action.label, pt);
	sprintf(label, "%s_res_sv%d",
			old_action.label, pt);
	sprintf(label_tex, "%s res_sv%d",
			old_action.label_tex, pt);
	f_has_subaction = TRUE;
	subaction = &old_action;
	ABR = NEW_OBJECT(action_by_restriction);
	
	if (f_v) {
		cout << "action::induced_action_by_restriction_"
				"on_orbit_with_schreier_vector "
				"before ABR->init_from_schreier_vector" << endl;
	}
	ABR->init_from_schreier_vector(Schreier_vector, pt, verbose_level - 1);
	if (f_v) {
		cout << "action::induced_action_by_restriction_"
				"on_orbit_with_schreier_vector "
				"after ABR->init_from_schreier_vector" << endl;
	}
	
	type_G = action_by_restriction_t;
	G.ABR = ABR;
	f_allocated = TRUE;
	make_element_size = old_action.make_element_size;
	low_level_point_size = old_action.low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = ABR->nb_points;
	base_len = 0;
	init_function_pointers_induced_action();
	if (FALSE) {
		cout << "action::induced_action_by_restriction_"
				"on_orbit_with_schreier_vector "
				"calling allocate_base_data" << endl;
		}
	allocate_base_data(0);
	
	
	elt_size_in_int = old_action.elt_size_in_int;
	coded_elt_size_in_char = old_action.coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_induce_action) {
		if (f_v) {
			cout << "action::induced_action_by_restriction_"
					"on_orbit_with_schreier_vector "
					"calling induced_action_override_sims" << endl;
			}
		induced_action_override_sims(old_action, old_G, verbose_level - 2);
		}
	if (f_v) {
		cout << "action::induced_action_by_restriction_"
				"on_orbit_with_schreier_vector "
				"finished, created action " << label << endl;
		//Sims->print_transversal_lengths();
		//cout << endl;
		print_info();
		}
}

action *action::restricted_action(
		int *points, int nb_points, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	action_by_restriction *ABR;
	
	if (f_v) {
		cout << "action::restricted_action" << endl;
		cout << "old_action ";
		print_info();
		cout << "nb_points = " << nb_points << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	A = NEW_OBJECT(action);
	sprintf(A->group_prefix, "%s_res%d",
			label, nb_points);
	sprintf(A->label, "%s_res%d",
			label, nb_points);
	sprintf(A->label_tex, "%s res%d",
			label_tex, nb_points);
	A->f_has_subaction = TRUE;
	A->subaction = this;
	ABR = NEW_OBJECT(action_by_restriction);
	ABR->init(nb_points, points, verbose_level);
	A->type_G = action_by_restriction_t;
	A->G.ABR = ABR;
	A->f_allocated = TRUE;
	A->make_element_size = make_element_size;
	A->low_level_point_size = low_level_point_size;
	
	A->f_has_strong_generators = FALSE;
	
	A->degree = nb_points;
	A->base_len = 0;
	A->init_function_pointers_induced_action();
	if (FALSE) {
		cout << "action::restricted_action "
				"calling allocate_base_data" << endl;
		}
	A->allocate_base_data(0);
	
	
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

void action::induced_action_by_restriction(
	action &old_action,
	int f_induce_action, sims *old_G, 
	int nb_points, int *points, int verbose_level)
// uses action_by_restriction data type
{
	int f_v = (verbose_level >= 1);
	action_by_restriction *ABR;
	
	if (f_v) {
		cout << "action::induced_action_by_restriction" << endl;
		cout << "old_action ";
		old_action.print_info();
		cout << "nb_points = " << nb_points << endl;
		cout << "f_induce_action = " << f_induce_action << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	sprintf(group_prefix, "%s_res%d",
			old_action.label, nb_points);
	sprintf(label, "%s_res%d",
			old_action.label, nb_points);
	sprintf(label_tex, "%s res%d",
			old_action.label_tex, nb_points);
	f_has_subaction = TRUE;
	subaction = &old_action;
	ABR = NEW_OBJECT(action_by_restriction);
	ABR->init(nb_points, points, verbose_level);
	type_G = action_by_restriction_t;
	G.ABR = ABR;
	f_allocated = TRUE;
	make_element_size = old_action.make_element_size;
	low_level_point_size = old_action.low_level_point_size;
	
	f_has_strong_generators = FALSE;
	
	degree = nb_points;
	base_len = 0;
	init_function_pointers_induced_action();
	if (FALSE) {
		cout << "action::induced_action_by_restriction "
				"calling allocate_base_data" << endl;
		}
	allocate_base_data(0);
	
	
	elt_size_in_int = old_action.elt_size_in_int;
	coded_elt_size_in_char = old_action.coded_elt_size_in_char;
	
	allocate_element_data();
	
	if (f_induce_action) {
		if (f_v) {
			cout << "action::induced_action_by_restriction "
					"calling induced_action_override_sims" << endl;
			}
		induced_action_override_sims(old_action,
				old_G, verbose_level - 2);
		}
	if (f_v) {
		cout << "action::induced_action_by_restriction "
				"finished, created action " << label << endl;
		//Sims->print_transversal_lengths();
		//cout << endl;
		print_info();
		}
}

void action::induced_action_on_pairs(
	action &old_action, sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action::induced_action_on_pairs" << endl;
		cout << "the old_action " << old_action.label
			<< " has base_length = " << old_action.base_len
			<< " and degree " << old_action.degree << endl;
		}
	sprintf(group_prefix, "%s_on_pairs",
			old_action.label);
	sprintf(label, "%s_on_pairs",
			old_action.label);
	sprintf(label_tex, "%s^{[2]}",
			old_action.label_tex);
	f_has_subaction = TRUE;
	subaction = &old_action;
	type_G = action_on_pairs_t;
	f_allocated = FALSE;
	
	f_has_strong_generators = FALSE;
	
	degree = int_n_choose_k(old_action.degree, 2);
	base_len = 0;
	init_function_pointers_induced_action();
	allocate_base_data(0);
	
	
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
	action &old_action, sims *old_G,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "action::induced_action_on_ordered_pairs" << endl;
		cout << "the old_action " << old_action.label
			<< " has base_length = " << old_action.base_len
			<< " and degree " << old_action.degree << endl;
		}
	sprintf(group_prefix, "%s_on_ordered_pairs",
			old_action.label);
	sprintf(label, "%s_on_ordered_pairs",
			old_action.label);
	sprintf(label_tex, "%s^{(2)}",
			old_action.label_tex);
	f_has_subaction = TRUE;
	subaction = &old_action;
	type_G = action_on_ordered_pairs_t;
	f_allocated = FALSE;
	
	f_has_strong_generators = FALSE;
	
	degree = old_action.degree * (old_action.degree - 1);
	base_len = 0;
	init_function_pointers_induced_action();
	allocate_base_data(0);
	
	
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
	action_on_k_subsets *On_k_subsets;
	
	if (f_v) {
		cout << "action::induced_action_on_k_subsets" << endl;
		cout << "the old_action " << old_action.label
			<< " has base_length = " << old_action.base_len
			<< " and degree " << old_action.degree << endl;
		}
	sprintf(group_prefix, "%s_on_%d_subsets",
			old_action.label, k);
	sprintf(label, "%s_on_%d_subsets",
			old_action.label, k);
	sprintf(label_tex, "%s^{[%d]}",
			old_action.label_tex, k);


	On_k_subsets = NEW_OBJECT(action_on_k_subsets);
	On_k_subsets->init(&old_action, k, verbose_level);


	f_has_subaction = TRUE;
	subaction = &old_action;
	type_G = action_on_k_subsets_t;
	G.on_k_subsets = On_k_subsets;
	f_allocated = TRUE;
	
	
	f_has_strong_generators = FALSE;
	
	degree = On_k_subsets->degree;
	base_len = 0;
	init_function_pointers_induced_action();
	allocate_base_data(0);
	
	
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
	schreier *Sch, int f_play_it_safe,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action_on_orbits *On_orbits;
	
	if (f_v) {
		cout << "action::induced_action_on_orbits" << endl;
		cout << "the old_action " << old_action->label
			<< " has base_length = " << old_action->base_len
			<< " and degree " << old_action->degree << endl;
		}
	sprintf(group_prefix, "%s_on_orbits",
			old_action->label);
	sprintf(label, "%s_on_orbits",
			old_action->label);
	sprintf(label_tex, "%s\\_on\\_orbits",
			old_action->label_tex);


	On_orbits = NEW_OBJECT(action_on_orbits);
	On_orbits->init(old_action, Sch,
			f_play_it_safe, verbose_level);


	f_has_subaction = TRUE;
	subaction = old_action;
	type_G = action_on_orbits_t;
	G.OnOrbits = On_orbits;
	f_allocated = TRUE;
	
	
	f_has_strong_generators = FALSE;
	
	degree = On_orbits->degree;
	base_len = 0;
	init_function_pointers_induced_action();
	allocate_base_data(0);
	
	
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
	action_on_flags *On_flags;
	
	if (f_v) {
		cout << "action::induced_action_on_flags" << endl;
		cout << "the old_action " << old_action->label
			<< " has base_length = " << old_action->base_len
			<< " and degree " << old_action->degree << endl;
		}
	sprintf(group_prefix, "%s_on_flags",
			old_action->label);
	sprintf(label, "%s_on_flags",
			old_action->label);
	sprintf(label_tex, "%s\\_on\\_flags",
			old_action->label_tex);


	On_flags = NEW_OBJECT(action_on_flags);
	On_flags->init(old_action, type,
			type_len, verbose_level);


	f_has_subaction = TRUE;
	subaction = old_action;
	type_G = action_on_flags_t;
	G.OnFlags = On_flags;
	f_allocated = TRUE;
	
	
	f_has_strong_generators = FALSE;
	
	degree = On_flags->degree;
	base_len = 0;
	init_function_pointers_induced_action();
	allocate_base_data(0);
	
	
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
	brick_domain *B, int f_linear_action,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action_on_bricks *On_bricks;
	
	if (f_v) {
		cout << "action::induced_action_on_bricks" << endl;
		cout << "the old_action " << old_action.label
			<< " has base_length = " << old_action.base_len
			<< " and degree " << old_action.degree << endl;
		}
	sprintf(group_prefix, "%s_on_bricks", old_action.label);
	sprintf(label, "%s_on_bricks", old_action.label);
	sprintf(label_tex, "%s on bricks", old_action.label_tex);


	On_bricks = NEW_OBJECT(action_on_bricks);
	On_bricks->init(&old_action, B, f_linear_action, verbose_level);


	f_has_subaction = TRUE;
	subaction = &old_action;
	type_G = action_on_bricks_t;
	G.OnBricks = On_bricks;
	f_allocated = TRUE;
	
	
	f_has_strong_generators = FALSE;
	
	degree = B->nb_bricks;
	base_len = 0;
	init_function_pointers_induced_action();
	allocate_base_data(0);
	
	
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
	action *An1, andre_construction *Andre,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action_on_andre *On_andre;
	
	if (f_v) {
		cout << "action::induced_action_on_andre" << endl;
		cout << "action An = " << An->label
				<< " has degree " << An->degree << endl;
		cout << "action An1 = " << An1->label
				<< " has degree " << An1->degree << endl;
		}
	sprintf(group_prefix, "%s_on_andre", An1->label);
	sprintf(label, "%s_on_andre", An1->label);
	sprintf(label_tex, "%s on andre", An1->label_tex);


	On_andre = NEW_OBJECT(action_on_andre);
	On_andre->init(An, An1, Andre, verbose_level);


	f_has_subaction = TRUE;
	subaction = An1;
	type_G = action_on_andre_t;
	G.OnAndre = On_andre;
	f_allocated = TRUE;
	
	
	f_has_strong_generators = FALSE;
	
	degree = On_andre->degree;
	base_len = 0;
	init_function_pointers_induced_action();
	allocate_base_data(0);
	
	
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
	product_action *PA;
	int i;
	
	if (f_v) {
		cout << "action::setup_product_action" << endl;
		}
	sprintf(group_prefix, "%s_x_%s", A1->label, A2->label);
	sprintf(label, "%s_x_%s", A1->label, A2->label);
	sprintf(label_tex, "%s \\times %s", A1->label_tex, A2->label_tex);
	PA = NEW_OBJECT(product_action);
	PA->init(A1, A2, f_use_projections, verbose_level);
	f_has_subaction = TRUE;
	subaction = NULL;
	type_G = product_action_t;
	G.product_action_data = PA;
	f_allocated = TRUE;
	
	f_has_strong_generators = FALSE;
	
	degree = PA->degree;
	
	base_len = 0;
	
	init_function_pointers_induced_action();
	
	allocate_base_data(0);
	
	
	elt_size_in_int = PA->elt_size_in_int;
	coded_elt_size_in_char = PA->coded_elt_size_in_char;
	
	make_element_size = A1->make_element_size + A2->make_element_size;
	
	allocate_element_data();

	base_len = A1->base_len + A2->base_len;
	allocate_base_data(base_len);
	if (f_use_projections) {
		for (i = 0; i < A1->base_len; i++) {
			base[i] = A1->base[i];
			}
		for (i = 0; i < A2->base_len; i++) {
			base[A1->base_len + i] = A1->degree + A2->base[i];
			}
		}
	else {
		for (i = 0; i < A1->base_len; i++) {
			base[i] = A1->base[i] * A2->degree;
			}
		for (i = 0; i < A2->base_len; i++) {
			base[A1->base_len + i] = A2->base[i];
			}
		}
	
	if (f_vv) {
		cout << "make_element_size=" << make_element_size << endl;
		cout << "base_len=" << base_len << endl;
		}
	if (f_v) {
		cout << "action::setup_product_action finished" << endl;
		print_info();
		}
}


void action::induced_action_on_homogeneous_polynomials(
	action *A_old,
	homogeneous_polynomial_domain *HPD, 
	int f_induce_action, sims *old_G, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	action_on_homogeneous_polynomials *OnHP;
	//matrix_group *M;
	
	if (f_v) {
		cout << "action::induced_action_on_homogeneous_polynomials "
				"f_induce_action=" << f_induce_action << endl;
		}
	A = A_old;
	OnHP = NEW_OBJECT(action_on_homogeneous_polynomials);
	sprintf(group_prefix, "%s_HP_%d_%d", A->label, HPD->n, HPD->degree);
	sprintf(label, "%s_HP_%d_%d", A->label, HPD->n, HPD->degree);
	sprintf(label_tex, "%s HP %d %d", A->label_tex, HPD->n, HPD->degree);
	if (f_v) {
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len
			<< " and degree " << A->degree << endl;
		}
	f_has_subaction = TRUE;
	subaction = A;
	if (A->type_G != matrix_group_t) {
		cout << "action::induced_action_on_homogeneous_polynomials "
				"action not of matrix group type" << endl;
		exit(1);
		}
	//M = A->G.matrix_grp;

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
	base_len = 0;
	init_function_pointers_induced_action();
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
	homogeneous_polynomial_domain *HPD, 
	int *Equations, int nb_equations, 
	int f_induce_action, sims *old_G, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	action_on_homogeneous_polynomials *OnHP;
	//matrix_group *M;
	
	if (f_v) {
		cout << "action::induced_action_on_homogeneous_"
				"polynomials_given_by_equations "
				"f_induce_action=" << f_induce_action << endl;
		}
	A = A_old;
	OnHP = NEW_OBJECT(action_on_homogeneous_polynomials);
	sprintf(group_prefix, "%s_HP_%d_%d_eqn%d",
			A->label, HPD->n, HPD->degree, nb_equations);
	sprintf(label, "%s_HP_%d_%d_eqn%d",
			A->label, HPD->n, HPD->degree, nb_equations);
	sprintf(label_tex, "%s HP %d %d %d",
			A->label_tex, HPD->n, HPD->degree, nb_equations);
	if (f_v) {
		cout << "the old_action " << A->label
			<< " has base_length = " << A->base_len
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
	//M = A->G.matrix_grp;

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
	base_len = 0;
	init_function_pointers_induced_action();
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
	sims *old_G;
	
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
	action &old_action, sims *old_G,
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

void action::induce(action *old_action, sims *old_G, 
	int base_of_choice_len, int *base_of_choice,
	int verbose_level)
// after this procedure, action will have
// a sims for the group and the kernel
// it will also have strong generators
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	action *subaction;
	sims *G, *K;
		// will become part of the action object
		// 'this' by the end of this procedure
	longinteger_object go, go1, go2, go3;
	longinteger_object G_order, K_order;
	longinteger_domain D;
	int b, i, old_base_len;
	action *fallback_action;
	
	if (f_v) {
		cout << "action::induce" << endl;
		}
	if (f_v) {
		cout << "inducing from action:" << endl;
		old_action->print_info();
		cout << "the old group is in action:" << endl;
		old_G->A->print_info();
		}
	//sprintf(label, "%s_ind", old_action->label);
	//sprintf(label_tex, "%s ind", old_action->label_tex);
	
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
		cout << "base_length = " << old_action->base_len << endl;
		cout << "subaction->base_len = " << subaction->base_len << endl;
		if (base_of_choice_len) {
			cout << "base of choice:" << endl;
			int_vec_print(cout, base_of_choice, base_of_choice_len);
			cout << endl;
			}
		else {
			cout << "no base of choice" << endl;
			}
		}
	
	G = NEW_OBJECT(sims);
	K = NEW_OBJECT(sims);
	G->init_without_base(this);
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
			old_base_len = base_len;
			reallocate_base(b);
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
	if (fallback_action->base_len == 0) {
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
	K->init(fallback_action);

	if (f_v) {
		cout << "action::induce before init_trivial_group" << endl;
		}
		
	G->init_trivial_group(verbose_level - 2);
	K->init_trivial_group(verbose_level - 2);
	if (f_v) {
		cout << "action::induce "
				"after init_trivial_group" << endl;
		cout << "action::induce "
				"calling build_up_group_random_process" << endl;
		}

	G->build_up_group_random_process(K, old_G, go, 
		FALSE /*f_override_chose_next_base_point*/,
		NULL /*choose_next_base_point_method*/, 
		verbose_level);
	if (f_v) {
		cout << "action::induce "
				"after build_up_group_random_process" << endl;
		}

	G->group_order(G_order);
	K->group_order(K_order);
	if (f_v) {
		cout << "action::induce: ";
		cout << "found a group in action " << G->A->label
				<< " of order " << G_order << " ";
		cout << "transversal lengths:" << endl;
		int_vec_print(cout, G->orbit_len, G->A->base_len);
		cout << endl;

		cout << "kernel in action " << K->A->label
				<< " of order " << K_order << " ";
		cout << "transversal lengths:" << endl;
		int_vec_print(cout, K->orbit_len, K->A->base_len);
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
		cout << "action::induce before init_sims" << endl;
		}
	init_sims(G, verbose_level - 2);
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
	int *set;
	int *old_base;
	int i, lmp, old_base_len;

	if (f_v) {
		cout << "action::lex_least_base_in_place action "
				<< label << " base=";
		int_vec_print(cout, base, base_len);
		cout << endl;
		print_info();
		//cout << "the generators are:" << endl;
		//Sims->print_generators();
		}

	set = NEW_int(degree);
	old_base = NEW_int(base_len);
	old_base_len = base_len;
	for (i = 0; i < base_len; i++) {
		old_base[i] = base[i];
		}
	
	
	
	for (i = 0; i < base_len; i++) {
		set[i] = base[i];
		if (f_v) {
			cout << "action::lex_least_base_in_place "
					"i=" << i << " computing the least moved point" << endl;
			}
		lmp = least_moved_point_at_level(i, verbose_level - 2);
		if (f_v) {
			cout << "action::lex_least_base_in_place "
					"i=" << i << " the least moved point is " << lmp << endl;
			}
		if (lmp >= 0 && lmp < base[i]) {
			if (f_v) {
				cout << "action::lex_least_base_in_place "
						"i=" << i << " least moved point = " << lmp
					<< " less than base point " << base[i] << endl;
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
		int_vec_print(cout, base, base_len);
		cout << endl;
		print_info();
		//cout << "the generators are:" << endl;
		//Sims->print_generators();
		int f_changed = FALSE;

		if (old_base_len != base_len) {
			f_changed = TRUE;
			}
		if (!f_changed) {
			for (i = 0; i < base_len; i++) {
				if (old_base[i] != base[i]) {
					f_changed = TRUE;
					break;
					}
				}
			}
		if (f_changed) {
			cout << "The base has changed !!!" << endl;
			cout << "old base: ";
			int_vec_print(cout, old_base, old_base_len);
			cout << endl;
			cout << "new base: ";
			int_vec_print(cout, base, base_len);
			cout << endl;
			}
		}
	FREE_int(old_base);
	FREE_int(set);
}

void action::lex_least_base(action *old_action, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *set;
	action *A, *old_A;
	int i, lmp;

	if (f_v) {
		cout << "action::lex_least_base action "
				<< old_action->label << " base=";
		int_vec_print(cout, old_action->base, old_action->base_len);
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

	set = NEW_int(old_action->degree);
	
	old_A = old_action;
	
	if (!old_action->f_has_sims) {
		cout << "action::lex_least_base does not have Sims" << endl;
		exit(1);
		}
	
	for (i = 0; i < old_A->base_len; i++) {
		set[i] = old_A->base[i];
		if (f_v) {
			cout << "action::lex_least_base "
					"calling least_moved_point_at_level " << i << endl;
			}
		lmp = old_A->least_moved_point_at_level(i, verbose_level - 2);
		if (lmp < old_A->base[i]) {
			if (f_v) {
				cout << "action::lex_least_base least moved point = " << lmp 
					<< " less than base point " << old_A->base[i] << endl;
				cout << "doing a base change:" << endl;
				}
			set[i] = lmp;
			A = NEW_OBJECT(action);
			A->base_change(old_A, i + 1, set, verbose_level - 2);
			old_A = A;
			}
		}
	base_change(old_A, old_A->base_len,
			old_A->base, verbose_level - 1);
	FREE_int(set);
	if (f_v) {
		cout << "action::lex_least_base action " << label << " base=";
		int_vec_print(cout, base, base_len);
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
	for (i = 0; i < base_len; i++) {
		if (AA->base_len >= i) {
			if (base[i] > AA->base[i]) {
				cout << "action::test_if_lex_least_base "
						"returns FALSE" << endl;
				cout << "base[i]=" << base[i] << endl;
				cout << "AA->base[i]=" << AA->base[i] << endl;
				FREE_OBJECT(AA);
				return FALSE;
				}
			}
		}
	FREE_OBJECT(AA);
	return TRUE;
}

void action::base_change_in_place(int size, int *set, int verbose_level)
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
	free_base_data();
	if (f_v5) {
		cout << "action::base_change_in_place after free_base_data" << endl;
		}
	allocate_base_data(A->base_len);
	if (f_v5) {
		cout << "action::base_change_in_place after allocate_base_data"
				<< endl;
		}
	base_len = A->base_len;
	for (i = 0; i < A->base_len; i++) {
		base[i] = A->base[i];
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
		cout << "action::base_change_in_place before init_sims" << endl;
		}
	init_sims(A->Sims, verbose_level);
	if (f_v5) {
		cout << "action::base_change_in_place after init_sims" << endl;
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
	int size, int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "action::base_change to the following set:" << endl;
		int_vec_print(cout, set, size);
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
	base_len = 0;
	init_function_pointers_induced_action();

	allocate_base_data(0);
	
	
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
	sprintf(group_prefix, "%s_base_change",
			old_action->group_prefix);
	sprintf(label, "%s_base_change",
			old_action->label);
	sprintf(label_tex, "%s_base_change",
			old_action->label_tex);
	
	if (f_v) {
		longinteger_object go, K_go;
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


