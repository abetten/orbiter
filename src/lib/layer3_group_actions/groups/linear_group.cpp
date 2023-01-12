// linear_group.cpp
//
// Anton Betten
// December 24, 2015

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;



namespace orbiter {
namespace layer3_group_actions {
namespace groups {



linear_group::linear_group()
{
	description = NULL;
	n = 0;
	input_q = 0;
	F = NULL;
	f_semilinear = FALSE;
	// label
	// label_tex
	initial_strong_gens = NULL;
	A_linear = NULL;
	Mtx = NULL;
	f_has_strong_generators = FALSE;
	Strong_gens = NULL;
	A2 = NULL;
	vector_space_dimension = 0;
	q = 0;
	f_has_nice_gens = FALSE;
	nice_gens = NULL;
}




linear_group::~linear_group()
{
}

void linear_group::linear_group_init(
		linear_group_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_group::linear_group_init" << endl;
	}
	linear_group::description = description;

	if (description->f_import_group_of_plane) {

		if (f_v) {
			cout << "linear_group::linear_group_init f_import_group_of_plane" << endl;
		}

		if (f_v) {
			cout << "linear_group::linear_group_init before linear_group_import" << endl;
		}
		linear_group_import(verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_init after linear_group_import" << endl;
		}


	}
	else {

		if (f_v) {
			cout << "linear_group::linear_group_init before linear_group_create" << endl;
		}
		linear_group_create(verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_init after linear_group_create" << endl;
		}

	}

	if (description->f_export_magma) {
		if (f_v) {
			cout << "linear_group::linear_group_init f_export_magma" << endl;
		}
		Strong_gens->export_magma(A_linear, cout, verbose_level);
	}


	if (f_v) {
		cout << "linear_group::linear_group_init finalized" << endl;
		cout << "linear_group::linear_group_init label=" << label << endl;
		cout << "linear_group::linear_group_init degree=" << A2->degree << endl;
		cout << "linear_group::linear_group_init go=";
		ring_theory::longinteger_object go;
		Strong_gens->group_order(go);
		cout << go << endl;
		cout << "linear_group::linear_group_init label=" << label << endl;
	}



	if (f_v) {
		cout << "linear_group::linear_group_init done" << endl;
	}
}

void linear_group::linear_group_import(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_group::linear_group_import" << endl;
	}

	if (description->f_import_group_of_plane) {

		if (f_v) {
			cout << "linear_group::linear_group_import before linear_group_import_group_of_plane" << endl;
		}

		linear_group_import_group_of_plane(verbose_level);

		if (f_v) {
			cout << "linear_group::linear_group_import after linear_group_import_group_of_plane" << endl;
		}
	}
	if (f_v) {
		cout << "linear_group::linear_group_import done" << endl;
	}
}

void linear_group::linear_group_import_group_of_plane(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_group::linear_group_import_group_of_plane" << endl;
	}

	if (f_v) {
		cout << "linear_group::linear_group_import_group_of_plane import_group_of_plane_label=" << description->import_group_of_plane_label << endl;
	}

	int idx;

	idx = orbiter_kernel_system::Orbiter->Orbiter_symbol_table->find_symbol(description->import_group_of_plane_label);

	data_structures_groups::translation_plane_via_andre_model *TP;

	TP = (data_structures_groups::translation_plane_via_andre_model *)
		orbiter_kernel_system::Orbiter->Orbiter_symbol_table->get_object(idx);

#if 0
	std::string label_txt;
	std::string label_tex;

	field_theory::finite_field *F;
	int q;
	int k;
	int n;
	int k1;
	int n1;
	int order_of_plane;

	geometry::andre_construction *Andre;
	int N; // number of points = number of lines
	int twoN; // 2 * N
	int f_semilinear;

	geometry::andre_construction_line_element *Line;
	int *Incma;
	int *pts_on_line;
	int *Line_through_two_points; // [N * N]
	int *Line_intersection; // [N * N]

	actions::action *An;
	actions::action *An1;

	actions::action *OnAndre;

	groups::strong_generators *strong_gens;
#endif

	n = TP->n1;
	F = TP->F;
	input_q = F->q;
	f_semilinear = TP->f_semilinear; //TP->An1->is_semilinear_matrix_group();

	A_linear = TP->An1;
	initial_strong_gens = TP->strong_gens;

	Mtx = A_linear->G.matrix_grp;
	vector_space_dimension = n;

	label.assign("group_of_plane_");
	label.append(TP->label_txt);
	label_tex.assign("group of plane ");
	label_tex.append(TP->label_tex);

	A2 = TP->OnAndre;
	vector_space_dimension = n;
	q = input_q;
	f_has_strong_generators = TRUE;
	Strong_gens = initial_strong_gens;

	if (f_v) {
		cout << "linear_group::linear_group_import_group_of_plane done" << endl;
	}
}


void linear_group::linear_group_create(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_group::linear_group_create" << endl;
	}

	data_structures::string_tools ST;

	if (ST.starts_with_a_number(description->input_q)) {
		int q;

		q = ST.strtoi(description->input_q);
		if (f_v) {
			cout << "linear_group::linear_group_create "
					"creating the finite field of order " << q << endl;
		}
		description->F = NEW_OBJECT(field_theory::finite_field);
		description->F->finite_field_init_small_order(q, FALSE /* f_without_tables */, verbose_level - 1);
		if (f_v) {
			cout << "linear_group::linear_group_create "
					"the finite field of order " << q << " has been created" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "linear_group::linear_group_create "
					"using existing finite field " << input_q << endl;
		}
		int idx;
		idx = orbiter_kernel_system::Orbiter->Orbiter_symbol_table->find_symbol(description->input_q);
		if (idx < 0) {
			cout << "linear_group::linear_group_create done cannot find finite field object" << endl;
			exit(1);
		}
		description->F = (field_theory::finite_field *) orbiter_kernel_system::Orbiter->Orbiter_symbol_table->get_object(idx);
	}

	n = description->n;
	F = description->F;
	input_q = F->q;
	f_semilinear = description->f_semilinear;
	if (f_v) {
		cout << "linear_group::linear_group_create n=" << n << endl;
		cout << "linear_group::linear_group_create q=" << input_q << endl;
		cout << "linear_group::linear_group_create f_semilinear=" << f_semilinear << endl;
	}


	if (f_v) {
		cout << "linear_group::linear_group_create initializing projective group" << endl;
	}



	initial_strong_gens = NEW_OBJECT(strong_generators);

	if (f_v) {
		cout << "linear_group::linear_group_create before "
				"initial_strong_gens->init_linear_group_from_scratch" << endl;
	}

	initial_strong_gens->init_linear_group_from_scratch(
		A_linear,
		F, n,
		description,
		nice_gens,
		label, label_tex,
		verbose_level - 3);

	if (f_v) {
		cout << "linear_group::linear_group_create after "
				"initial_strong_gens->init_linear_group_from_scratch" << endl;
		cout << "label = " << label << endl;
		cout << "group order = ";
		initial_strong_gens->print_group_order(cout);
		cout << endl;
	}


	if (f_v) {
		cout << "linear_group::linear_group_create initializing "
				"initial_strong_gens done" << endl;
	}

	//label.assign(A_linear->label);
	//label_tex.assign(A_linear->label_tex);

	if (f_v) {
		cout << "linear_group::linear_group_create label=" << label << endl;
		cout << "linear_group::linear_group_create degree=" << A_linear->degree << endl;
		cout << "linear_group::linear_group_create go=";
		ring_theory::longinteger_object go;
		initial_strong_gens->group_order(go);
		//A_linear->Strong_gens->group_order(go);
		cout << go << endl;
	}


	Mtx = A_linear->G.matrix_grp;
	vector_space_dimension = n;

	int f_OK = FALSE;

	if (f_v) {
		cout << "linear_group::linear_group_create before linear_group_apply_modification" << endl;
	}

	f_OK = linear_group_apply_modification(
			description,
			verbose_level);

	if (f_v) {
		cout << "linear_group::linear_group_create after linear_group_apply_modification" << endl;
	}


	if (!f_OK) {
		if (f_v) {
			cout << "linear_group::linear_group_create "
					"!f_OK, A2 = A_linear" << endl;
		}
		A2 = A_linear;
		vector_space_dimension = n;
		q = input_q;
		f_has_strong_generators = TRUE;
		Strong_gens = initial_strong_gens;
	}

	if (f_v) {
		cout << "linear_group::linear_group_create done" << endl;
	}
}

int linear_group::linear_group_apply_modification(
		linear_group_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_group::linear_group_apply_modification" << endl;
	}
	int f_OK = FALSE;

	if (description->f_PGL2OnConic) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_PGL2q_OnConic" << endl;
		}
		init_PGL2q_OnConic(verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_PGL2q_OnConic" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_wedge_action) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_wedge_action" << endl;
		}
		init_wedge_action(verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_wedge_action" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_wedge_action_detached) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_wedge_action_detached" << endl;
		}
		init_wedge_action_detached(verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_wedge_action_detached" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_monomial_group) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_monomial_group" << endl;
		}
		init_monomial_group(verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_monomial_group" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_diagonal_group) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_diagonal_group" << endl;
		}
		init_diagonal_group(verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_diagonal_group" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_null_polarity_group) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_null_polarity_group" << endl;
		}
		init_null_polarity_group(verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_null_polarity_group" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_symplectic_group) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_symplectic_group" << endl;
		}
		init_symplectic_group(verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_symplectic_group" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_borel_subgroup_upper) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_borel_subgroup_upper" << endl;
		}
		init_borel_subgroup_upper(verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_borel_subgroup_upper" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_borel_subgroup_lower) {
		cout << "linear_group::linear_group_apply_modification borel_subgroup_lower "
				"not yet implemented" << endl;
		exit(1);
		}
	if (description->f_singer_group) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_singer_group" << endl;
		}
		init_singer_group(
				description->singer_power, verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_singer_group" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_singer_group_and_frobenius) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_singer_group_and_frobenius" << endl;
		}
		init_singer_group_and_frobenius(
				description->singer_power, verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_singer_group_and_frobenius" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_identity_group) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_identity_subgroup" << endl;
		}
		init_identity_subgroup(verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_identity_subgroup" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_subfield_structure_action) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_subfield_structure_action" << endl;
		}
		init_subfield_structure_action(
				description->s, verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_subfield_structure_action" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_orthogonal_group) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_orthogonal_group" << endl;
		}
		init_orthogonal_group(
				description->orthogonal_group_epsilon,
				verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_orthogonal_group" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_subgroup_from_file) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_subgroup_from_file" << endl;
		}
		init_subgroup_from_file(
			description->subgroup_fname,
			description->subgroup_label,
			verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_subgroup_from_file" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_subgroup_by_generators) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"f_subgroup_by_generators" << endl;
		}

		int *gens;
		int sz;

		Get_int_vector_from_label(
				description->subgroup_generators_label, gens, sz,
				verbose_level);

		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"gens of size " << sz << ":" << endl;
			Int_vec_print(cout, gens, sz);
			cout << endl;
		}


		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_subgroup_by_generators" << endl;
		}
		init_subgroup_by_generators(
			description->subgroup_label,
			description->subgroup_order_text,
			description->nb_subgroup_generators,
			gens,
			verbose_level - 1);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_subgroup_by_generators" << endl;
		}
		FREE_int(gens);
		f_OK = TRUE;
	}
	if (description->f_Janko1) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before init_subgroup_Janko1" << endl;
		}
		init_subgroup_Janko1(verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after init_subgroup_Janko1" << endl;
		}
		f_OK = TRUE;
	}
	if (description->f_on_tensors) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"f_on_tensors" << endl;
		}
		wreath_product *W;
		long int *points;
		int nb_points;
		int i;

		W = A_linear->G.wreath_product_group;
		nb_points = W->degree_of_tensor_action;
		points = NEW_lint(nb_points);
		for (i = 0; i < nb_points; i++) {
			points[i] = W->perm_offset_i[W->nb_factors] + i;
		}

		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before A_wreath->restricted_action" << endl;
		}
		A2 = A_linear->restricted_action(points, nb_points,
				verbose_level);
		A2->f_is_linear = TRUE;
		A2->dimension = W->dimension_of_tensor_action;
		f_OK = TRUE;
		f_has_strong_generators = TRUE;
		Strong_gens = initial_strong_gens;
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after A_linear->restricted_action" << endl;
		}
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"f_on_tensors done" << endl;
		}

	}
	if (description->f_on_rank_one_tensors) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification f_on_rank_one_tensors" << endl;
		}
		wreath_product *W;
		long int *points;
		int nb_points;
		int i;

		W = A_linear->G.wreath_product_group;
		nb_points = W->nb_rank_one_tensors;
		points = NEW_lint(nb_points);
		for (i = 0; i < nb_points; i++) {
			points[i] = W->perm_offset_i[W->nb_factors] + W->rank_one_tensors_in_PG[i];
		}

		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before A_wreath->restricted_action" << endl;
		}
		A2 = A_linear->restricted_action(points, nb_points,
				verbose_level);
		A2->f_is_linear = TRUE;
		A2->dimension = W->dimension_of_tensor_action;
		f_OK = TRUE;
		f_has_strong_generators = TRUE;
		Strong_gens = initial_strong_gens;
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after A_linear->restricted_action" << endl;
		}

		if (f_v) {
			cout << "linear_group::linear_group_apply_modification f_on_rank_one_tensors done" << endl;
		}
	}
	if (f_v) {
		cout << "linear_group::linear_group_apply_modification done" << endl;
	}
	return f_OK;
}

void linear_group::init_PGL2q_OnConic(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_group::init_PGL2q_OnConic "
				"initializing action of PGL(2,q) on conic" << endl;
	}
	if (!A_linear->f_has_sims) {
		cout << "linear_group::init_PGL2q_OnConic "
				"A_linear does not have sims, so we create it" << endl;
		A_linear->create_sims(verbose_level);
	}
	if (!A_linear->f_has_strong_generators) {
		cout << "linear_group::init_PGL2q_OnConic "
				"A_linear does not have strong generators" << endl;
		//A_linear->create_sims(verbose_level);
		exit(1);
	}
	A2 = NEW_OBJECT(actions::action);
	A2->induced_action_by_representation_on_conic(A_linear, 
		FALSE /* f_induce_action */, NULL, 
		verbose_level);

	vector_space_dimension = A2->G.Rep->dimension;
	q = input_q;
	Strong_gens = initial_strong_gens; //A_linear->Strong_gens;
	f_has_strong_generators = FALSE;

	if (f_v) {
		cout << "linear_group::init_PGL2q_OnConic "
				"vector_space_dimension=" << vector_space_dimension << endl;
	}
	if (f_v) {
		cout << "linear_group::init_PGL2q_OnConic "
				"created action of PGL2_on conic:" << endl;
		A2->print_info();
	}
	char str1[1000];
	char str2[1000];

	snprintf(str1, sizeof(str1), "_OnConic_%d_%d", n, q);
	snprintf(str2, sizeof(str2), "{\\rm OnConic}(%d,%d)", n, q);
	label.append(str1);
	label_tex.append(str2);
	if (f_v) {
		cout << "linear_group::init_PGL2q_OnConic "
				"created group " << label << endl;
	}
}

void linear_group::init_wedge_action(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_group::init_wedge_action "
				"initializing wedge action" << endl;
	}
#if 0
	if (!A_linear->f_has_sims) {
		cout << "linear_group::init_wedge_action "
				"A_linear does not have sims, so we create it" << endl;
		A_linear->create_sims(verbose_level);
	}
#endif
	if (!A_linear->f_has_strong_generators) {
		cout << "linear_group::init_wedge_action "
				"A_linear does not have strong generators" << endl;
		exit(1);
	}
	A2 = NEW_OBJECT(actions::action);
	//action_on_wedge_product *AW;

	

#if 0

	if (f_v) {
		cout << "linear_group::init_wedge_action "
				"before induced_wedge_action:" << endl;
	}
	AW = NEW_OBJECT(action_on_wedge_product);

	AW->init(*A_linear, verbose_level);
#endif
	

	if (f_v) {
		cout << "linear_group::init_wedge_action "
				"vector_space_dimension="
				<< vector_space_dimension << endl;
	}
		
	A2 = A_linear->induced_action_on_wedge_product(verbose_level);

	vector_space_dimension = A2->G.AW->wedge_dimension;
	q = input_q;
	Strong_gens = initial_strong_gens; //A_linear->Strong_gens;
	f_has_strong_generators = TRUE;

	if (f_v) {
		cout << "linear_group::init_wedge_action "
				"created wedge action:" << endl;
		A2->print_info();
	}
	char str1[1000];
	char str2[1000];

	snprintf(str1, sizeof(str1), "_Wedge_%d_%d", n, q);
	snprintf(str2, sizeof(str2), "{\\rm Wedge}(%d,%d)", n, q);
	label.append(str1);
	label_tex.append(str2);
	if (f_v) {
		cout << "linear_group::init_wedge_action "
				"created group " << label << endl;
	}
}

void linear_group::init_wedge_action_detached(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_group::init_wedge_action_detached "
				"initializing wedge action" << endl;
	}
	if (!A_linear->f_has_strong_generators) {
		cout << "linear_group::init_wedge_action_detached "
				"A_linear does not have strong generators" << endl;
		exit(1);
	}
	A2 = NEW_OBJECT(actions::action);

	if (f_v) {
		cout << "linear_group::init_wedge_action_detached initializing projective group" << endl;
	}


	strong_generators *secondary_strong_gens;
	//strong_generators *exterior_square_strong_gens;
	data_structures_groups::vector_ge *secondary_nice_gens;
	int n2;

	combinatorics::combinatorics_domain Combi;

	n2 = Combi.binomial2(n);

	secondary_strong_gens = NEW_OBJECT(strong_generators);
	//exterior_square_strong_gens = NEW_OBJECT(strong_generators);

	if (f_v) {
		cout << "linear_group::init_wedge_action_detached before "
				"initial_strong_gens->init_linear_group_from_scratch" << endl;
	}

	secondary_strong_gens->init_linear_group_from_scratch(
			A2,
			F, n2,
			description,
			secondary_nice_gens,
			label, label_tex,
			verbose_level);

	if (f_v) {
		cout << "linear_group::init_wedge_action_detached after "
				"initial_strong_gens->init_linear_group_from_scratch" << endl;
	}



	q = F->q;

	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->init(A2, verbose_level);
	Strong_gens->exterior_square(
				A2,
				A_linear->Strong_gens,
				nice_gens,
				verbose_level);
	f_has_strong_generators = TRUE;

	A_linear = A2; // override the original action!

	f_has_nice_gens = TRUE;

	if (f_v) {
		cout << "linear_group::init_wedge_action_detached "
				"nice generators are:" << endl;
		nice_gens->print(cout);
	}


	if (f_v) {
		cout << "linear_group::init_wedge_action_detached "
				"created detached wedge action:" << endl;
		A2->print_info();
	}
	char str1[1000];
	char str2[1000];

	snprintf(str1, sizeof(str1), "_Wedge_%d_%d_detached", n, q);
	snprintf(str2, sizeof(str2), "{\\rm WedgeDetached}(%d,%d)", n, q);
	label.append(str1);
	label_tex.append(str2);
	if (f_v) {
		cout << "linear_group::init_wedge_action_detached "
				"created group " << label << endl;
	}
}

void linear_group::init_monomial_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_monomial_group "
				"initializing monomial group" << endl;
	}
		
	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->generators_for_the_monomial_group(A_linear, 
		Mtx, verbose_level - 1);
	f_has_strong_generators = TRUE;
	
	A2 = A_linear;


	char str1[1000];
	char str2[1000];

	snprintf(str1, sizeof(str1), "_Monomial_%d_%d", n, q);
	snprintf(str2, sizeof(str2), "{\\rm Monomial}(%d,%d)", n, q);
	label.append(str1);
	label_tex.append(str2);

	if (f_v) {
		cout << "linear_group::init_monomial_group "
				"created group " << label << endl;
	}
}

void linear_group::init_diagonal_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_diagonal_group "
				"initializing monomial group" << endl;
	}
		
	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->generators_for_the_diagonal_group(A_linear, 
		Mtx, verbose_level - 1);
	f_has_strong_generators = TRUE;
	
	A2 = A_linear;

	char str1[1000];
	char str2[1000];

	snprintf(str1, sizeof(str1), "_Diagonal_%d_%d", n, q);
	snprintf(str2, sizeof(str2), "{\\rm Diagonal}(%d,%d)", n, q);
	label.append(str1);
	label_tex.append(str2);

	if (f_v) {
		cout << "linear_group::init_diagonal_group "
				"created group " << label << endl;
	}
}

void linear_group::init_singer_group(int singer_power, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_singer_group "
				"initializing singer group" << endl;
	}

	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->generators_for_the_singer_cycle(
			A_linear, Mtx, singer_power, nice_gens,
			verbose_level - 1);
	f_has_strong_generators = TRUE;
	f_has_nice_gens = TRUE;
	

	A2 = A_linear;

	char str1[1000];
	char str2[1000];

	snprintf(str1, sizeof(str1), "_Singer_%d_%d_%d", n, q, singer_power);
	snprintf(str2, sizeof(str2), "{\\rm Singer}(%d,%d,%d)", n, q, singer_power);
	label.append(str1);
	label_tex.append(str2);

	if (f_v) {
		cout << "linear_group::init_singer_group "
				"created group " << label << endl;
	}
}

void linear_group::init_singer_group_and_frobenius(
		int singer_power, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_singer_group_and_frobenius "
				"initializing singer group" << endl;
	}

	vector_space_dimension = n;
	q = input_q;

	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->generators_for_the_singer_cycle_and_the_Frobenius(
			A_linear, Mtx, singer_power, nice_gens,
			verbose_level - 1);
	f_has_strong_generators = TRUE;
	f_has_nice_gens = TRUE;


	A2 = A_linear;

	char str1[1000];
	char str2[1000];

	snprintf(str1, sizeof(str1), "_Singer_and_Frob%d_%d_%d", n, q, singer_power);
	snprintf(str2, sizeof(str2), "{\\rm SingerFrob}(%d,%d,%d)", n, q, singer_power);
	label.append(str1);
	label_tex.append(str2);

	if (f_v) {
		cout << "linear_group::init_singer_group_and_frobenius "
				"created group " << label << endl;
	}
}

void linear_group::init_null_polarity_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_null_polarity_group "
				"initializing null polarity group" << endl;
	}

	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->generators_for_the_null_polarity_group(
			A_linear, Mtx, verbose_level - 1);
	f_has_strong_generators = TRUE;
	
	A2 = A_linear;


	char str1[1000];
	char str2[1000];

	snprintf(str1, sizeof(str1), "_NullPolarity_%d_%d", n, q);
	snprintf(str2, sizeof(str2), "{\\rm NullPolarity}(%d,%d)", n, q);
	label.append(str1);
	label_tex.append(str2);

	if (f_v) {
		cout << "linear_group::init_null_polarity_group "
				"created group " << label << endl;
	}
}

void linear_group::init_borel_subgroup_upper(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_borel_subgroup_upper "
				"initializing borel subgroup of upper "
				"triangular matrices" << endl;
	}

	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->generators_for_the_borel_subgroup_upper(
			A_linear, Mtx, verbose_level - 1);
	f_has_strong_generators = TRUE;
	
	A2 = A_linear;


	char str1[1000];
	char str2[1000];

	snprintf(str1, sizeof(str1), "_BorelUpper_%d_%d", n, q);
	snprintf(str2, sizeof(str2), "{\\rm BorelUpper}(%d,%d)", n, q);
	label.append(str1);
	label_tex.append(str2);

	if (f_v) {
		cout << "linear_group::init_borel_subgroup_upper "
				"created group " << label << endl;
	}
}

void linear_group::init_identity_subgroup(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_identity_subgroup "
				"initializing identify subgroup" << endl;
	}

	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->generators_for_the_identity_subgroup(
			A_linear, Mtx, verbose_level - 1);
	f_has_strong_generators = TRUE;
	
	A2 = A_linear;

	char str1[1000];
	char str2[1000];

	snprintf(str1, sizeof(str1), "_Identity_%d_%d", n, q);
	snprintf(str2, sizeof(str2), "{\\rm Identity}(%d,%d)", n, q);
	label.append(str1);
	label_tex.append(str2);

	if (f_v) {
		cout << "linear_group::init_identity_subgroup "
				"created group " << label << endl;
	}
}

void linear_group::init_symplectic_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_symplectic_group "
				"initializing symplectic group" << endl;
	}

	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(strong_generators);
	if (f_v) {
		cout << "linear_group::init_symplectic_group "
				"before Strong_gens->generators_for_symplectic_group" << endl;
	}
	Strong_gens->generators_for_symplectic_group(
			A_linear, Mtx, verbose_level - 1);
	if (f_v) {
		cout << "linear_group::init_symplectic_group "
				"after Strong_gens->generators_for_symplectic_group" << endl;
		cout << "linear_group::init_symplectic_group "
				"number of generators is " << Strong_gens->gens->len << endl;
	}
	f_has_strong_generators = TRUE;
	
	A2 = A_linear;


	char str1[1000];
	char str2[1000];

	snprintf(str1, sizeof(str1), "_Sp_%d_%d", n, q);
	snprintf(str2, sizeof(str2), "{\\rm Sp}(%d,%d)", n, q);
	label.append(str1);
	label_tex.append(str2);


	if (f_v) {
		cout << "linear_group::init_symplectic_group "
				"created group " << label << endl;
	}
}

void linear_group::init_subfield_structure_action(int s, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "linear_group::init_subfield_structure_action" << endl;
		cout << "s=" << s << endl;
	}

	if (f_v) {
		cout << "linear_group::init_subfield_structure_action "
				"before field_reduction" << endl;
	}

	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->field_reduction(A_linear,
			n, s, F, verbose_level - 1);
	//lift_generators_to_subfield_structure(A_linear,
	//P->n + 1, s, P->F, SGens, verbose_level - 1);
	f_has_strong_generators = TRUE;

	A2 = A_linear;

	char str1[1000];
	char str2[1000];

	snprintf(str1, sizeof(str1), "_Subfield_%d_%d_%d", n, q, s);
	snprintf(str2, sizeof(str2), "{\\rm SubfieldAction}(%d,%d,%d)", n, q, s);
	label.append(str1);
	label_tex.append(str2);

	if (f_v) {
		cout << "linear_group::init_subfield_structure_action "
				"created group " << label << endl;
	}
	
	if (f_v) {
		cout << "linear_group::init_subfield_structure_action "
				"done" << endl;
	}
}

void linear_group::init_orthogonal_group(int epsilon, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_orthogonal_group "
				"initializing orthogonal group" << endl;
		cout << "epsilon=" << epsilon << endl;
	}
		
	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->generators_for_the_orthogonal_group(A_linear, 
		F, n, 
		epsilon, 
		f_semilinear, 
		verbose_level - 1);
	f_has_strong_generators = TRUE;
	
	A2 = A_linear;

	char str1[1000];
	char str2[1000];

	if (EVEN(n)) {
		if (epsilon == 1) {
			snprintf(str1, sizeof(str1), "_Orthogonal_plus_%d_%d", n, q);
			snprintf(str2, sizeof(str2), "{\\rm O}^+(%d,%d)", n, q);
			label.append(str1);
			label_tex.append(str2);
		}
		else {
			snprintf(str1, sizeof(str1), "_Orthogonal_minus_%d_%d", n, q);
			snprintf(str2, sizeof(str2), "{\\rm O}^-(%d,%d)", n, q);
			label.append(str1);
			label_tex.append(str2);
		}
	}
	else {
		snprintf(str1, sizeof(str1), "_Orthogonal_%d_%d", n, q);
		snprintf(str2, sizeof(str2), "{\\rm O}(%d,%d)", n, q);
		label.append(str1);
		label_tex.append(str2);
	}
	if (f_v) {
		cout << "linear_group::init_orthogonal_group "
				"created group " << label << endl;
	}
}


void linear_group::init_subgroup_from_file(
	std::string &subgroup_fname, std::string &subgroup_label,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "linear_group::init_subgroup_from_file" << endl;
		cout << "fname=" << subgroup_fname << endl;
		cout << "label=" << subgroup_label << endl;
	}


	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(strong_generators);
	if (f_v) {
		cout << "linear_group::init_subgroup_from_file "
				"reading generators from file " << subgroup_fname << endl;
	}

	Strong_gens->read_file(A_linear,
			subgroup_fname, verbose_level - 1);

	if (f_v) {
		cout << "linear_group::init_subgroup_from_file "
				"read generators from file" << endl;
	}

	f_has_strong_generators = TRUE;

	A2 = A_linear;


	char str1[1000];
	char str2[1000];

	snprintf(str1, sizeof(str1), "_SubgroupFile_%d_%d", n, q);
	snprintf(str2, sizeof(str2), "{\\rm SubgroupFile}(%d,%d)", n, q);
	label.append(str1);
	label_tex.append(str2);


	if (f_v) {
		cout << "linear_group::init_subgroup_from_file "
				"created group " << label << endl;
	}
}

void linear_group::init_subgroup_by_generators(
		std::string &subgroup_label,
		std::string &subgroup_order_text,
		int nb_subgroup_generators,
		int *subgroup_generators_data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "linear_group::init_subgroup_by_generators" << endl;
		cout << "label=" << subgroup_label << endl;
	}

	Strong_gens = NEW_OBJECT(strong_generators);
	if (f_v) {
		cout << "linear_group::init_subgroup_by_generators before "
				"Strong_gens->init_subgroup_by_generators" << endl;
	}

	Strong_gens->init_subgroup_by_generators(A_linear,
			nb_subgroup_generators, subgroup_generators_data,
			subgroup_order_text,
			nice_gens,
			verbose_level - 1);

	f_has_nice_gens = TRUE;

	if (f_v) {
		cout << "linear_group::init_subgroup_by_generators after "
				"Strong_gens->init_subgroup_by_generators" << endl;
	}

	ring_theory::longinteger_object go;


	f_has_strong_generators = TRUE;

	Strong_gens->group_order(go);

	if (f_v) {
		cout << "linear_group::init_subgroup_by_generators go = " << go << endl;
	}



	A2 = A_linear;

	stringstream str;
	orbiter_kernel_system::latex_interface L;
	int max_len = 80;
	int line_skip = 0;


	L.latexable_string(str, subgroup_label.c_str(), max_len, line_skip);



	label.append("_Subgroup_");
	label.append(subgroup_label);
	label.append("_");
	label.append(subgroup_order_text);


	label_tex.append("{\\rm Subgroup ");
	label_tex.append(str.str());
	label_tex.append(" order ");
	label_tex.append(subgroup_order_text);
	label_tex.append("}");

	if (f_v) {
		cout << "linear_group::init_subgroup_by_generators "
				"created group " << label << endl;
	}
}

void linear_group::init_subgroup_Janko1(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "linear_group::init_subgroup_Janko1" << endl;
	}

	Strong_gens = NEW_OBJECT(strong_generators);
	if (f_v) {
		cout << "linear_group::init_subgroup_Janko1 before "
				"Strong_gens->init_subgroup_by_generators" << endl;
	}

	matrix_group *M;

	M = A_linear->get_matrix_group();

	Strong_gens->Janko1(
			A_linear,
			M->GFq,
			verbose_level);

	if (f_v) {
		cout << "linear_group::init_subgroup_Janko1 after "
				"Strong_gens->init_subgroup_by_generators" << endl;
	}

	f_has_nice_gens = FALSE;
	f_has_strong_generators = TRUE;

	A2 = A_linear;


	char str1[1000];
	char str2[1000];

	snprintf(str1, sizeof(str1), "_Subgroup_Janko1");
	snprintf(str2, sizeof(str2), "{\\rm Subgroup Janko1}");
	label.append(str1);
	label_tex.append(str2);

	if (f_v) {
		cout << "linear_group::init_subgroup_Janko1 "
				"created group " << label << endl;
	}
}

void linear_group::report(std::ostream &ost,
		int f_sylow, int f_group_table,
		int f_conjugacy_classes_and_normalizers,
		graphics::layered_graph_draw_options *LG_Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sims *H;
	actions::action *A;

	A = A2;
	if (f_v) {
		cout << "linear_group::report creating report for group " << label << endl;
	}

	//G = initial_strong_gens->create_sims(verbose_level);
	if (f_v) {
		cout << "linear_group::report before Strong_gens->create_sims" << endl;
	}
	H = Strong_gens->create_sims(0 /*verbose_level*/);

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;

	int *Elt;
	ring_theory::longinteger_object go;

	Elt = NEW_int(A->elt_size_in_int);
	H->group_order(go);


	{

		//H->print_all_group_elements_tex(fp);

		ring_theory::longinteger_object go;
		//sims *G;
		//sims *H;

		//G = initial_strong_gens->create_sims(verbose_level);
		//H = Strong_gens->create_sims(verbose_level);



		ost << "\\section*{The Group $" << label_tex << "$}" << endl;


		H->group_order(go);

		ost << "\\noindent The order of the group $"
				<< label_tex
				<< "$ is " << go << "\\\\" << endl;


#if 0
		fp << "\\noindent The field ${\\mathbb F}_{"
				<< F->q
				<< "}$ :\\\\" << endl;
		if (f_v) {
			cout << "linear_group::report before F->cheat_sheet" << endl;
		}
		F->cheat_sheet(fp, verbose_level);
		if (f_v) {
			cout << "linear_group::report after F->cheat_sheet" << endl;
		}
#endif


#if 0
		ost << "\\noindent The group acts on a set of size "
				<< A2->degree << "\\\\" << endl;
#endif
		A2->report_what_we_act_on(ost,
				LG_Draw_options,
				verbose_level);


#if 0
		if (A->degree < 1000) {

			A->print_points(fp);
		}
#endif

		//cout << "Order H = " << H->group_order_int() << "\\\\" << endl;

		if (f_has_nice_gens) {
			ost << "Nice generators:\\\\" << endl;
			nice_gens->print_tex(ost);
		}
		else {
		}

		cout << "Strong generators:\\\\" << endl;
		Strong_gens->print_generators_tex(ost);


		if (f_v) {
			cout << "linear_group::report before A2->report" << endl;
		}

		A2->report(ost, TRUE /*f_sims*/, H,
				TRUE /* f_strong_gens */, Strong_gens,
				LG_Draw_options,
				verbose_level);

		if (f_v) {
			cout << "linear_group::report after A2->report" << endl;
		}

		if (f_v) {
			cout << "linear_group::report before A2->report_basic_orbits" << endl;
		}

		A2->report_basic_orbits(ost);

		if (f_v) {
			cout << "linear_group::report after A2->report_basic_orbits" << endl;
		}

		if (f_group_table) {
			if (f_v) {
				cout << "linear_group::report f_group_table is true" << endl;
			}

			int *Table;
			long int n;
			orbiter_kernel_system::file_io Fio;
			string fname_group_table;
			H->create_group_table(Table, n, verbose_level);

			cout << "linear_group::report The group table is:" << endl;
			Int_matrix_print(Table, n, n);

			fname_group_table.assign(label);
			fname_group_table.append("_group_table.csv");
			Fio.int_matrix_write_csv(fname_group_table, Table, n, n);
			cout << "Written file " << fname_group_table << " of size " << Fio.file_size(fname_group_table) << endl;

			{
				orbiter_kernel_system::latex_interface L;

				ost << "\\begin{sidewaystable}" << endl;
				ost << "$$" << endl;
				L.int_matrix_print_tex(ost, Table, n, n);
				ost << "$$" << endl;
				ost << "\\end{sidewaystable}" << endl;

				int f_with_permutation = FALSE;
				int f_override_action = FALSE;
				actions::action *A_special = NULL;

				H->print_all_group_elements_tex(ost, f_with_permutation, f_override_action, A_special);

			}

			{
				string fname2;
				//int x_min = 0, y_min = 0;
				//int xmax = ONE_MILLION;
				//int ymax = ONE_MILLION;

				//int f_embedded = TRUE;
				//int f_sideways = FALSE;
				int *labels;

				char str[1000];

				int i;

				labels = NEW_int(2 * n);

				for (i = 0; i < n; i++) {
					labels[i] = i;
				}
				if (n > 100) {
					for (i = 0; i < n; i++) {
						labels[n + i] = n + i % 100;
					}
				}
				else {
					for (i = 0; i < n; i++) {
						labels[n + i] = n + i;
					}
				}

				fname2.assign(label);
				snprintf(str, sizeof(str), "_group_table_order_%ld", n);
				fname2.append(str);

				{
					graphics::mp_graphics G;

					G.init(fname2, LG_Draw_options, verbose_level);

#if 0
					mp_graphics G(fname2, x_min, y_min, xmax, ymax, f_embedded, f_sideways, verbose_level - 1);
					//G.setup(fname2, 0, 0, ONE_MILLION, ONE_MILLION, xmax, ymax, f_embedded, scale, line_width);
					G.out_xmin() = 0;
					G.out_ymin() = 0;
					G.out_xmax() = xmax;
					G.out_ymax() = ymax;
					//cout << "xmax/ymax = " << xmax << " / " << ymax << endl;

					//G.tikz_global_scale = LG_Draw_options->scale;
					//G.tikz_global_line_width = LG_Draw_options->line_width;
#endif

					G.header();
					G.begin_figure(1000 /* factor_1000*/);

					int color_scale[] = {8,5,6,4,3,2,18,19, 7,9,10,11,12,13,14,15,16,17,20,21,22,23,24,25,1};
					int nb_colors = sizeof(color_scale) / sizeof(int);

					G.draw_matrix_in_color(
						FALSE /* f_row_grid */, FALSE /* f_col_grid */,
						Table  /* Table */, n /* nb_colors */,
						n, n, //xmax, ymax,
						color_scale, nb_colors,
						TRUE /* f_has_labels */, labels);

					G.finish(cout, TRUE);
				}
				FREE_int(labels);

			}


			FREE_int(Table);


		}

		if (f_sylow) {

			if (f_v) {
				cout << "linear_group::report f_sylow is true" << endl;
			}

			sylow_structure *Syl;

			Syl = NEW_OBJECT(sylow_structure);
			Syl->init(H, verbose_level);
			Syl->report(ost);

		}
		else {

			if (f_v) {
				cout << "linear_group::report f_sylow is false" << endl;
			}

		}

		if (f_conjugacy_classes_and_normalizers) {


			groups::magma_interface M;


			if (f_v) {
				cout << "linear_group::report f_conjugacy_classes_and_normalizers is true" << endl;
			}

			M.report_conjugacy_classes_and_normalizers(A2, ost, H,
					verbose_level);

			if (f_v) {
				cout << "linear_group::report A2->report_conjugacy_classes_and_normalizers" << endl;
			}
		}

		//L.foot(fp);
	}

	FREE_int(Elt);

}


void linear_group::create_latex_report(
		graphics::layered_graph_draw_options *O,
		int f_sylow, int f_group_table, int f_classes,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::create_latex_report" << endl;
	}

	{
		string fname;
		string title;
		string author, extra_praeamble;

		fname.assign(label);
		fname.append("_report.tex");
		title.assign("The group $");
		title.append(label_tex);
		title.append("$");

		author.assign("");


		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "linear_group::create_latex_report before report" << endl;
			}
			report(ost, f_sylow, f_group_table,
					f_classes,
					O,
					verbose_level);
			if (f_v) {
				cout << "linear_group::create_latex_report after report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}




	if (f_v) {
		cout << "linear_group::create_latex_report done" << endl;
	}
}

}}}


