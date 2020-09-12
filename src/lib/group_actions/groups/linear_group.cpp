// linear_group.cpp
//
// Anton Betten
// December 24, 2015

#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;



namespace orbiter {
namespace group_actions {



linear_group::linear_group()
{
	null();
}

linear_group::~linear_group()
{
	freeself();
}

void linear_group::null()
{
	description = NULL;
	initial_strong_gens = NULL;
	A_linear = NULL;
	A2 = NULL;
	Mtx = NULL;
	f_has_strong_generators = FALSE;
	Strong_gens = NULL;
	f_has_nice_gens = FALSE;
	nice_gens = NULL;
}

void linear_group::freeself()
{
	null();
}

void linear_group::init(
		linear_group_description *description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_group::init" << endl;
		}
	linear_group::description = description;
	n = description->n;
	F = description->F;
	input_q = F->q;
	f_semilinear = description->f_semilinear;


	if (f_v) {
		cout << "linear_group::init initializing projective group" << endl;
		}


	
	initial_strong_gens = NEW_OBJECT(strong_generators);

	if (f_v) {
		cout << "linear_group::init before "
				"initial_strong_gens->init_linear_group_from_scratch" << endl;
	}
	
	initial_strong_gens->init_linear_group_from_scratch(
		A_linear,
		F, n, 
		description->f_projective,
		description->f_general,
		description->f_affine,
		description->f_semilinear,
		description->f_special,
		description->f_GL_d_q_wr_Sym_n,
		description->GL_wreath_Sym_d, description->GL_wreath_Sym_n,
		nice_gens,
		verbose_level);

	if (f_v) {
		cout << "linear_group::init after "
				"initial_strong_gens->init_linear_group_from_scratch" << endl;
	}


	if (f_v) {
		cout << "linear_group::init initializing "
				"initial_strong_gens done" << endl;
	}
	if (f_v) {
		cout << "linear_group::init degreee=" << A_linear->degree << endl;
		cout << "linear_group::init go=";
		longinteger_object go;
		A_linear->Strong_gens->group_order(go);
		cout << go << endl;
	}

	label.assign(A_linear->label);
	label_tex.assign(A_linear->label_tex);
	//strcpy(prefix, A_linear->label);
	//strcpy(label_latex, A_linear->label_tex);

	Mtx = A_linear->G.matrix_grp;
	vector_space_dimension = n;

	int f_OK = FALSE;

	if (description->f_PGL2OnConic) {
		if (f_v) {
			cout << "linear_group::init "
					"before init_PGL2q_OnConic" << endl;
		}
		init_PGL2q_OnConic(verbose_level);
		if (f_v) {
			cout << "linear_group::init "
					"after init_PGL2q_OnConic" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_wedge_action) {
		if (f_v) {
			cout << "linear_group::init "
					"before init_wedge_action" << endl;
		}
		init_wedge_action(verbose_level);
		if (f_v) {
			cout << "linear_group::init "
					"after init_wedge_action" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_monomial_group) {
		if (f_v) {
			cout << "linear_group::init "
					"before init_monomial_group" << endl;
		}
		init_monomial_group(verbose_level);
		if (f_v) {
			cout << "linear_group::init "
					"after init_monomial_group" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_diagonal_group) {
		if (f_v) {
			cout << "linear_group::init "
					"before init_diagonal_group" << endl;
		}
		init_diagonal_group(verbose_level);
		if (f_v) {
			cout << "linear_group::init "
					"after init_diagonal_group" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_null_polarity_group) {
		if (f_v) {
			cout << "linear_group::init "
					"before init_null_polarity_group" << endl;
		}
		init_null_polarity_group(verbose_level);
		if (f_v) {
			cout << "linear_group::init "
					"after init_null_polarity_group" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_symplectic_group) {
		if (f_v) {
			cout << "linear_group::init "
					"before init_symplectic_group" << endl;
		}
		init_symplectic_group(verbose_level);
		if (f_v) {
			cout << "linear_group::init "
					"after init_symplectic_group" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_borel_subgroup_upper) {
		if (f_v) {
			cout << "linear_group::init "
					"before init_borel_subgroup_upper" << endl;
		}
		init_borel_subgroup_upper(verbose_level);
		if (f_v) {
			cout << "linear_group::init "
					"after init_borel_subgroup_upper" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_borel_subgroup_lower) {
		cout << "linear_group::init borel_subgroup_lower "
				"not yet implemented" << endl;
		exit(1);
		}
	if (description->f_singer_group) {
		if (f_v) {
			cout << "linear_group::init "
					"before init_singer_group" << endl;
		}
		init_singer_group(
				description->singer_power, verbose_level);
		if (f_v) {
			cout << "linear_group::init "
					"after init_singer_group" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_singer_group_and_frobenius) {
		if (f_v) {
			cout << "linear_group::init "
					"before init_singer_group_and_frobenius" << endl;
		}
		init_singer_group_and_frobenius(
				description->singer_power, verbose_level);
		if (f_v) {
			cout << "linear_group::init "
					"after init_singer_group_and_frobenius" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_identity_group) {
		if (f_v) {
			cout << "linear_group::init "
					"before init_identity_subgroup" << endl;
		}
		init_identity_subgroup(verbose_level);
		if (f_v) {
			cout << "linear_group::init "
					"after init_identity_subgroup" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_subfield_structure_action) {
		if (f_v) {
			cout << "linear_group::init "
					"before init_subfield_structure_action" << endl;
		}
		init_subfield_structure_action(
				description->s, verbose_level);
		if (f_v) {
			cout << "linear_group::init "
					"after init_subfield_structure_action" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_orthogonal_group) {
		if (f_v) {
			cout << "linear_group::init "
					"before init_orthogonal_group" << endl;
		}
		init_orthogonal_group(
				description->orthogonal_group_epsilon,
				verbose_level);
		if (f_v) {
			cout << "linear_group::init "
					"after init_orthogonal_group" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_subgroup_from_file) {
		if (f_v) {
			cout << "linear_group::init "
					"before init_subgroup_from_file" << endl;
		}
		init_subgroup_from_file(
			description->subgroup_fname,
			description->subgroup_label,
			verbose_level);
		if (f_v) {
			cout << "linear_group::init "
					"after init_subgroup_from_file" << endl;
		}
		f_OK = TRUE;
		}
	if (description->f_subgroup_by_generators) {
		if (f_v) {
			cout << "linear_group::init "
					"before init_subgroup_by_generators" << endl;
		}
		init_subgroup_by_generators(
			description->subgroup_label,
			description->subgroup_order_text,
			description->nb_subgroup_generators,
			description->subgroup_generators_as_string,
			verbose_level);
		if (f_v) {
			cout << "linear_group::init "
					"after init_subgroup_by_generators" << endl;
		}
		f_OK = TRUE;
	}
	if (description->f_Janko1) {
		if (f_v) {
			cout << "linear_group::init "
					"before init_subgroup_Janko1" << endl;
		}
		init_subgroup_Janko1(verbose_level);
		if (f_v) {
			cout << "linear_group::init "
					"after init_subgroup_Janko1" << endl;
		}
		f_OK = TRUE;
	}
	if (description->f_on_tensors) {
		if (f_v) {
			cout << "linear_group::init "
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
			cout << "action::init_wreath_product_group_and_restrict "
					"before A_wreath->restricted_action" << endl;
		}
		A2 = A_linear->restricted_action(points, nb_points,
				verbose_level);
		A2->f_is_linear = TRUE;
		A2->dimension = W->dimension_of_tensor_action;
		f_OK = TRUE;
		Strong_gens = initial_strong_gens;
		if (f_v) {
			cout << "action::init_wreath_product_group_and_restrict "
					"after A_linear->restricted_action" << endl;
		}
		if (f_v) {
			cout << "linear_group::init "
					"f_on_tensors done" << endl;
		}

	}
	if (description->f_on_rank_one_tensors) {
		if (f_v) {
			cout << "linear_group::init f_on_rank_one_tensors" << endl;
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
			cout << "action::init_wreath_product_group_and_restrict "
					"before A_wreath->restricted_action" << endl;
		}
		A2 = A_linear->restricted_action(points, nb_points,
				verbose_level);
		A2->f_is_linear = TRUE;
		A2->dimension = W->dimension_of_tensor_action;
		f_OK = TRUE;
		Strong_gens = initial_strong_gens;
		if (f_v) {
			cout << "action::init_wreath_product_group_and_restrict "
					"after A_linear->restricted_action" << endl;
		}

		if (f_v) {
			cout << "linear_group::init f_on_rank_one_tensors done" << endl;
		}
	}

	if (description->f_on_k_subspaces) {
		action_on_grassmannian *AG;
		grassmann *Grass;
		action *A3;
		
		cout << "linear_group::init creating induced action "
				"on k-subspaces for k="
				<< description->on_k_subspaces_k << endl;
		AG = NEW_OBJECT(action_on_grassmannian);
		
		Grass = NEW_OBJECT(grassmann);

		A3 = NEW_OBJECT(action);


		Grass->init(n,
				description->on_k_subspaces_k,
				F, 0 /* verbose_level */);
		
		AG->init(*A2, Grass, verbose_level - 2);
	
		A3->induced_action_on_grassmannian(A2, AG, 
			FALSE /* f_induce_action */, NULL /*sims *old_G */, 
			MINIMUM(verbose_level - 2, 2));
		A3->f_is_linear = TRUE;
	
		if (f_v) {
			cout << "linear_group::init action A3 created: ";
			A3->print_info();
			}

		A2 = A3;
		f_OK = TRUE;

		char str1[1000];
		char str2[1000];

		sprintf(str1, "_OnGr_%d", description->on_k_subspaces_k);
		sprintf(str2, " {\\rm Gr}_{%d,%d}(%d)",
				n, description->on_k_subspaces_k, F->q);
		label.append(str1);
		label_tex.append(str2);


		cout << "linear_group::init creating induced "
				"action on k-subspaces done" << endl;
		
	}
	if (description->f_restricted_action) {
		if (f_v) {
			cout << "linear_group::init "
					"restricted_action" << endl;
		}
		long int *points;
		int nb_points;
		action *A3;

		lint_vec_scan(description->restricted_action_text, points, nb_points);
		A3 = A2->restricted_action(points, nb_points,
				verbose_level);
		A3->f_is_linear = TRUE;
		f_OK = TRUE;
		//sprintf(A2->prefix + strlen(A2->prefix), "_restr_%d", nb_points);
		//sprintf(A2->label_latex + strlen(A2->label_latex),
		//		"{\\rm restr}(%d)", nb_points);
		//Strong_gens = initial_strong_gens;

		A2 = A3;
	if (f_v) {
			cout << "linear_group::init "
					"after restricted_action" << endl;
		}
		f_OK = TRUE;
	}

	if (!f_OK) {
		A2 = A_linear;
		vector_space_dimension = n;
		q = input_q;
		Strong_gens = initial_strong_gens;
		//sprintf(prefix, "PGL_%d_%d", n, input_q);
		//sprintf(label_latex, "\\PGL(%d,%d)", n, input_q);
		}


	if (description->f_export_magma) {
		if (f_v) {
			cout << "linear_group::init f_export_magma" << endl;
		}
		Strong_gens->export_magma(A_linear, cout);
	}

	if (f_v) {
		cout << "linear_group::init done" << endl;
	}
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
	A2 = NEW_OBJECT(action);
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

	sprintf(str1, "_OnConic_%d_%d", n, q);
	sprintf(str2, "{\\rm OnConic}(%d,%d)", n, q);
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
		//>create_sims(verbose_level);
		exit(1);
		}
	A2 = NEW_OBJECT(action);
	action_on_wedge_product *AW;

	


	if (f_v) {
		cout << "linear_group::init_wedge_action "
				"before induced_wedge_action:" << endl;
		}
	AW = NEW_OBJECT(action_on_wedge_product);

	AW->init(*A_linear, verbose_level);
	
	vector_space_dimension = AW->wedge_dimension;
	q = input_q;
	Strong_gens = initial_strong_gens; //A_linear->Strong_gens;
	f_has_strong_generators = FALSE;


	if (f_v) {
		cout << "linear_group::init_wedge_action "
				"vector_space_dimension="
				<< vector_space_dimension << endl;
		}
		
	A2->induced_action_on_wedge_product(A_linear, 
		AW, 
		FALSE /* f_induce_action */, NULL, 
		verbose_level);
	if (f_v) {
		cout << "linear_group::init_wedge_action "
				"created wedge action:" << endl;
		A2->print_info();
		}
	char str1[1000];
	char str2[1000];

	sprintf(str1, "_Wedge_%d_%d", n, q);
	sprintf(str2, "{\\rm Wedge}(%d,%d)", n, q);
	label.append(str1);
	label_tex.append(str2);
	if (f_v) {
		cout << "linear_group::init_wedge_action "
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

	sprintf(str1, "_Monomial_%d_%d", n, q);
	sprintf(str2, "{\\rm Monomial}(%d,%d)", n, q);
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

	sprintf(str1, "_Diagonal_%d_%d", n, q);
	sprintf(str2, "{\\rm Diagonal}(%d,%d)", n, q);
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

	sprintf(str1, "_Singer_%d_%d_%d", n, q, singer_power);
	sprintf(str2, "{\\rm Singer}(%d,%d,%d)", n, q, singer_power);
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

	sprintf(str1, "_Singer_and_Frob%d_%d_%d", n, q, singer_power);
	sprintf(str2, "{\\rm SingerFrob}(%d,%d,%d)", n, q, singer_power);
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

	sprintf(str1, "_NullPolarity_%d_%d", n, q);
	sprintf(str2, "{\\rm NullPolarity}(%d,%d)", n, q);
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

	sprintf(str1, "_BorelUpper_%d_%d", n, q);
	sprintf(str2, "{\\rm BorelUpper}(%d,%d)", n, q);
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

	sprintf(str1, "_Identity_%d_%d", n, q);
	sprintf(str2, "{\\rm Identity}(%d,%d)", n, q);
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
	Strong_gens->generators_for_symplectic_group(
			A_linear, Mtx, verbose_level - 1);
	f_has_strong_generators = TRUE;
	
	A2 = A_linear;


	char str1[1000];
	char str2[1000];

	sprintf(str1, "_Sp_%d_%d", n, q);
	sprintf(str2, "{\\rm Sp}(%d,%d)", n, q);
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

	sprintf(str1, "_Subfield_%d_%d_%d", n, q, s);
	sprintf(str2, "{\\rm SubfieldAction}(%d,%d,%d)", n, q, s);
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
			sprintf(str1, "_Orthogonal_plus_%d_%d", n, q);
			sprintf(str2, "{\\rm O}^+(%d,%d)", n, q);
			label.append(str1);
			label_tex.append(str2);
			}
		else {
			sprintf(str1, "_Orthogonal_minus_%d_%d", n, q);
			sprintf(str2, "{\\rm O}^-(%d,%d)", n, q);
			label.append(str1);
			label_tex.append(str2);
			}
		}
	else {
		sprintf(str1, "_Orthogonal_%d_%d", n, q);
		sprintf(str2, "{\\rm O}(%d,%d)", n, q);
		label.append(str1);
		label_tex.append(str2);
		}
	if (f_v) {
		cout << "linear_group::init_orthogonal_group "
				"created group " << label << endl;
		}
}


void linear_group::init_subgroup_from_file(
	const char *subgroup_fname, const char *subgroup_label, 
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

	sprintf(str1, "_SubgroupFile_%d_%d", n, q);
	sprintf(str2, "{\\rm SubgroupFile}(%d,%d)", n, q);
	label.append(str1);
	label_tex.append(str2);


	if (f_v) {
		cout << "linear_group::init_subgroup_from_file "
				"created group " << label << endl;
		}
}

void linear_group::init_subgroup_by_generators(
	const char *subgroup_label,
	const char *subgroup_order_text,
	int nb_subgroup_generators,
	std::string *subgroup_generators_as_string,
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
			nb_subgroup_generators, subgroup_generators_as_string,
			subgroup_order_text,
			nice_gens,
			verbose_level);

	f_has_nice_gens = TRUE;

	if (f_v) {
		cout << "linear_group::init_subgroup_by_generators after "
				"Strong_gens->init_subgroup_by_generators" << endl;
		}

	f_has_strong_generators = TRUE;

	A2 = A_linear;

	stringstream str;
	latex_interface L;
	int max_len = 80;
	int line_skip = 0;


	L.latexable_string(str, subgroup_label, max_len, line_skip);



	char str1[1000];
	char str2[1000];

	sprintf(str1, "_Subgroup_%s_%s", subgroup_label, subgroup_order_text);
	sprintf(str2, "{\\rm Subgroup %s order %s}", str.str().c_str(), subgroup_order_text);
	label.append(str1);
	label_tex.append(str2);
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

	sprintf(str1, "_Subgroup_Janko1");
	sprintf(str2, "{\\rm Subgroup Janko1}");
	label.append(str1);
	label_tex.append(str2);

	if (f_v) {
		cout << "linear_group::init_subgroup_Janko1 "
				"created group " << label << endl;
		}
}

void linear_group::report(std::ostream &fp, int f_sylow, int f_group_table,
		int f_conjugacy_classes_and_normalizers,
		layered_graph_draw_options *LG_Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sims *H;
	action *A;

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
	longinteger_object go;

	Elt = NEW_int(A->elt_size_in_int);
	H->group_order(go);


	{

		//H->print_all_group_elements_tex(fp);

		longinteger_object go;
		//sims *G;
		//sims *H;

		//G = initial_strong_gens->create_sims(verbose_level);
		//H = Strong_gens->create_sims(verbose_level);



		fp << "\\section{The Group $" << label_tex << "$}" << endl;


		H->group_order(go);

		fp << "\\noindent The order of the group $"
				<< label_tex
				<< "$ is " << go << "\\\\" << endl;

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


		fp << "\\noindent The group acts on a set of size "
				<< A->degree << "\\\\" << endl;

#if 0
		if (A->degree < 1000) {

			A->print_points(fp);
		}
#endif

		//cout << "Order H = " << H->group_order_int() << "\\\\" << endl;

		if (f_has_nice_gens) {
			fp << "Nice generators:\\\\" << endl;
			nice_gens->print_tex(fp);
		}
		else {
			cout << "Strong generators:\\\\" << endl;
			Strong_gens->print_generators_tex(fp);
		}

		if (f_v) {
			cout << "linear_group::report before A2->report" << endl;
		}

		A2->report(fp, TRUE /*f_sims*/, H,
				TRUE /* f_strong_gens */, Strong_gens,
				LG_Draw_options,
				verbose_level);

		if (f_v) {
			cout << "linear_group::report after A2->report" << endl;
		}

		if (f_v) {
			cout << "linear_group::report before A2->report_basic_orbits" << endl;
		}

		A2->report_basic_orbits(fp);

		if (f_v) {
			cout << "linear_group::report after A2->report_basic_orbits" << endl;
		}

		if (f_group_table) {
			if (f_v) {
				cout << "linear_group::report f_group_table is true" << endl;
			}

			int *Table;
			long int n;
			H->create_group_table(Table, n, verbose_level);
			cout << "linear_group::report The group table is:" << endl;
			int_matrix_print(Table, n, n, 2);
			{
				latex_interface L;

				fp << "\\begin{sidewaystable}" << endl;
				fp << "$$" << endl;
				L.int_matrix_print_tex(fp, Table, n, n);
				fp << "$$" << endl;
				fp << "\\end{sidewaystable}" << endl;

				H->print_all_group_elements_tex(fp);

			}

			{
				string fname2;
				int x_min = 0, y_min = 0;
				int xmax = ONE_MILLION;
				int ymax = ONE_MILLION;

				int f_embedded = TRUE;
				int f_sideways = FALSE;
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
				sprintf(str, "_group_table_order_%ld", n);
				fname2.append(str);

				{
				mp_graphics G(fname2, x_min, y_min, xmax, ymax, f_embedded, f_sideways, verbose_level - 1);
				//G.setup(fname2, 0, 0, ONE_MILLION, ONE_MILLION, xmax, ymax, f_embedded, scale, line_width);
				G.out_xmin() = 0;
				G.out_ymin() = 0;
				G.out_xmax() = xmax;
				G.out_ymax() = ymax;
				//cout << "xmax/ymax = " << xmax << " / " << ymax << endl;

				G.tikz_global_scale = LG_Draw_options->scale;
				G.tikz_global_line_width = LG_Draw_options->line_width;

				G.header();
				G.begin_figure(1000 /* factor_1000*/);

				int color_scale[] = {8,5,6,4,3,2,18,19, 7,9,10,11,12,13,14,15,16,17,20,21,22,23,24,25,1};
				int nb_colors = sizeof(color_scale) / sizeof(int);

				G.draw_matrix_in_color(
					FALSE /* f_row_grid */, FALSE /* f_col_grid */,
					Table  /* Table */, n /* nb_colors */,
					n, n, xmax, ymax,
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
			Syl->report(fp);

		}
		else {

			if (f_v) {
				cout << "linear_group::report f_sylow is false" << endl;
			}

		}

		if (f_conjugacy_classes_and_normalizers) {

			if (f_v) {
				cout << "linear_group::report f_conjugacy_classes_and_normalizers is true" << endl;
			}

			A2->report_conjugacy_classes_and_normalizers(fp, H,
					verbose_level);

			if (f_v) {
				cout << "linear_group::report A2->report_conjugacy_classes_and_normalizers" << endl;
			}
		}

		//L.foot(fp);
	}

	FREE_int(Elt);

}



}}


