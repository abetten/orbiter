// linear_group.C
//
// Anton Betten
// December 24, 2015

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"



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
		cout << "linear_group::init initializing "
				"projective group" << endl;
		}


	
	initial_strong_gens = NEW_OBJECT(strong_generators);

	if (f_v) {
		cout << "linear_group::init before "
				"initial_strong_gens->init_linear_group_"
				"from_scratch" << endl;
		}
	
	initial_strong_gens->init_linear_group_from_scratch(
		A_linear,
		F, n, 
		description->f_projective,
		description->f_general,
		description->f_affine,
		description->f_semilinear,
		description->f_special,
		verbose_level);


	if (f_v) {
		cout << "linear_group::init initializing "
				"initial_strong_gens done" << endl;
		}


	Mtx = A_linear->G.matrix_grp;

	if (description->f_PGL2OnConic) {
		init_PGL2q_OnConic(prefix, verbose_level);
		}
	else if (description->f_wedge_action) {
		init_wedge_action(prefix, verbose_level);
		}
	else if (description->f_monomial_group) {
		init_monomial_group(prefix, verbose_level);
		}
	else if (description->f_diagonal_group) {
		init_diagonal_group(prefix, verbose_level);
		}
	else if (description->f_null_polarity_group) {
		init_null_polarity_group(prefix, verbose_level);
		}
	else if (description->f_symplectic_group) {
		init_symplectic_group(prefix, verbose_level);
		}
	else if (description->f_borel_subgroup_upper) {
		init_borel_subgroup_upper(prefix, verbose_level);
		}
	else if (description->f_borel_subgroup_lower) {
		cout << "linear_group::init borel_subgroup_lower "
				"not yet implemented" << endl;
		exit(1);
		}
	else if (description->f_singer_group) {
		init_singer_group(prefix,
				description->singer_power, verbose_level);
		}
	else if (description->f_identity_group) {
		init_identity_subgroup(prefix, verbose_level);
		}
	else if (description->f_subfield_structure_action) {
		init_subfield_structure_action(prefix,
				description->s, verbose_level);
		}
	else if (description->f_orthogonal_group) {
		init_orthogonal_group(prefix,
				description->orthogonal_group_epsilon,
				verbose_level);
		}
	else if (description->f_subgroup_from_file) {
		init_subgroup_from_file(prefix, 
			description->subgroup_fname,
			description->subgroup_label,
			verbose_level);
		}
	else {
		A2 = A_linear;
		vector_space_dimension = n;
		q = input_q;
		Strong_gens = initial_strong_gens;
		sprintf(prefix, "PGL_%d_%d", n, input_q);
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
	
		if (f_v) {
			cout << "linear_group::init action A3 created: ";
			A3->print_info();
			}

		A2 = A3;


		cout << "linear_group::init creating induced "
				"action on k-subspaces done" << endl;
		
		}
	if (f_v) {
		cout << "linear_group::init done" << endl;
		}
}

void linear_group::init_PGL2q_OnConic(char *prefix,
		int verbose_level)
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
	sprintf(prefix, "PGL2_OnConic_%d_%d", n, q);
	if (f_v) {
		cout << "linear_group::init_PGL2q_OnConic "
				"created group " << prefix << endl;
		}
}

void linear_group::init_wedge_action(char *prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_group::init_wedge_action "
				"initializing wedge action" << endl;
		}
	if (!A_linear->f_has_sims) {
		cout << "linear_group::init_wedge_action "
				"A_linear does not have sims, so we create it" << endl;
		A_linear->create_sims(verbose_level);
		}
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
	sprintf(prefix, "Wedge_%d_%d", n, q);
	if (f_v) {
		cout << "linear_group::init_wedge_action "
				"created group " << prefix << endl;
		}
}

void linear_group::init_monomial_group(char *prefix,
		int verbose_level)
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

	sprintf(prefix, "Monomial_%d_%d", n, q);
	if (f_v) {
		cout << "linear_group::init_monomial_group "
				"created group " << prefix << endl;
		}

	if (f_v) {
		cout << "linear_group::init_monomial_group "
				"done, prefix = " << prefix << endl;
		}
}

void linear_group::init_diagonal_group(char *prefix,
		int verbose_level)
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

	sprintf(prefix, "Diagonal_%d_%d", n, q);
	if (f_v) {
		cout << "linear_group::init_diagonal_group "
				"created group " << prefix << endl;
		}

	if (f_v) {
		cout << "linear_group::init_diagonal_group "
				"done, prefix = " << prefix << endl;
		}
}

void linear_group::init_singer_group(char *prefix,
		int singer_power, int verbose_level)
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
			A_linear, Mtx, singer_power, verbose_level - 1);
	f_has_strong_generators = TRUE;
	

	A2 = A_linear;

	sprintf(prefix, "Singer_%d_%d_power%d",
			n, q, singer_power);
	if (f_v) {
		cout << "linear_group::init_singer_group "
				"created group " << prefix << endl;
		}

	if (f_v) {
		cout << "linear_group::init_singer_group "
				"done, prefix = " << prefix << endl;
		}
}

void linear_group::init_null_polarity_group(char *prefix,
		int verbose_level)
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

	sprintf(prefix, "NullPolarity_%d_%d", n, q);
	if (f_v) {
		cout << "linear_group::init_null_polarity_group "
				"created group " << prefix << endl;
		}

	if (f_v) {
		cout << "linear_group::init_null_polarity_group "
				"done, prefix = " << prefix << endl;
		}
}

void linear_group::init_borel_subgroup_upper(char *prefix,
		int verbose_level)
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

	sprintf(prefix, "BorelUpper_%d_%d", n, q);
	if (f_v) {
		cout << "linear_group::init_borel_subgroup_upper "
				"created group " << prefix << endl;
		}

	if (f_v) {
		cout << "linear_group::init_borel_subgroup_upper "
				"done" << endl;
		}
}

void linear_group::init_identity_subgroup(char *prefix,
		int verbose_level)
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

	sprintf(prefix, "Identity_%d_%d", n, q);
	if (f_v) {
		cout << "linear_group::init_identity_subgroup "
				"created group " << prefix << endl;
		}

	if (f_v) {
		cout << "linear_group::init_identity_subgroup "
				"done" << endl;
		}
}

void linear_group::init_symplectic_group(char *prefix,
		int verbose_level)
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

	sprintf(prefix, "Sp_%d_%d", n, q);
	if (f_v) {
		cout << "linear_group::init_symplectic_group "
				"created group " << prefix << endl;
		}

	if (f_v) {
		cout << "linear_group::init_symplectic_group "
				"done, prefix = " << prefix << endl;
		}
}

void linear_group::init_subfield_structure_action(
		char *prefix, int s, int verbose_level)
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

	sprintf(prefix, "Subfield_%d_%d_%d", n, q, s);
	if (f_v) {
		cout << "linear_group::init_subfield_structure_action "
				"created group " << prefix << endl;
		}
	
	if (f_v) {
		cout << "linear_group::init_subfield_structure_action "
				"done" << endl;
		}
}

void linear_group::init_orthogonal_group(char *prefix, 
	int epsilon, int verbose_level)
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

	if (EVEN(n)) {
		if (epsilon == 1) {
			sprintf(prefix, "Orthogonal_plus_%d_%d", n, q);
			}
		else {
			sprintf(prefix, "Orthogonal_minus_%d_%d", n, q);
			}
		}
	else {
		sprintf(prefix, "Orthogonal_%d_%d", n, q);
		}
	if (f_v) {
		cout << "linear_group::init_orthogonal_group "
				"created group " << prefix << endl;
		}

	if (f_v) {
		cout << "linear_group::init_orthogonal_group "
				"done, prefix = " << prefix << endl;
		}
}


void linear_group::init_subgroup_from_file(char *prefix, 
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

	if (f_v) {
		cout << "linear_group::init_subgroup_from_file "
				"before field_reduction" << endl;
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

	sprintf(prefix, "Subgroup_%s_%d_%d",
			subgroup_label, n, q);
	if (f_v) {
		cout << "linear_group::init_subgroup_from_file "
				"created group " << prefix << endl;
		}
	
	if (f_v) {
		cout << "linear_group::init_subgroup_from_file done" << endl;
		}
}



