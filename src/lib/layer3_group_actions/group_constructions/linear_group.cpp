// linear_group.cpp
//
// Anton Betten
// December 24, 2015

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;



namespace orbiter {
namespace layer3_group_actions {
namespace group_constructions {



linear_group::linear_group()
{
	Record_birth();
	description = NULL;
	n = 0;
	input_q = 0;
	F = NULL;
	f_semilinear = false;
	// label
	// label_tex
	initial_strong_gens = NULL;
	A_linear = NULL;
	Mtx = NULL;
	f_has_strong_generators = false;
	Strong_gens = NULL;
	A2 = NULL;
	vector_space_dimension = 0;
	q = 0;
	f_has_nice_gens = false;
	nice_gens = NULL;
}




linear_group::~linear_group()
{
	Record_death();
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
			cout << "linear_group::linear_group_init "
					"f_import_group_of_plane" << endl;
		}

		if (f_v) {
			cout << "linear_group::linear_group_init "
					"before linear_group_import" << endl;
		}
		linear_group_import(verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_init "
					"after linear_group_import" << endl;
		}


	}
#if 0
	else if (description->f_stabilizer_of_variety) {

		if (f_v) {
			cout << "linear_group::linear_group_init "
					"f_stabilizer_of_variety" << endl;
		}

		if (f_v) {
			cout << "linear_group::linear_group_init "
					"before linear_group_import" << endl;
		}
		do_stabilizer_of_variety(stabilizer_of_variety_label, verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_init "
					"after linear_group_import" << endl;
		}


	}
#endif
	else {

		if (f_v) {
			cout << "linear_group::linear_group_init "
					"before linear_group_create" << endl;
		}
		linear_group_create(verbose_level);
		if (f_v) {
			cout << "linear_group::linear_group_init "
					"after linear_group_create" << endl;
		}

	}

	if (description->f_export_magma) {
		if (f_v) {
			cout << "linear_group::linear_group_init "
					"f_export_magma" << endl;
		}
		Strong_gens->export_magma(A_linear, cout, verbose_level);
	}


	if (f_v) {
		cout << "linear_group::linear_group_init finalized" << endl;
		cout << "linear_group::linear_group_init label=" << label << endl;
		cout << "linear_group::linear_group_init degree=" << A2->degree << endl;
		cout << "linear_group::linear_group_init go=";
		algebra::ring_theory::longinteger_object go;
		Strong_gens->group_order(go);
		cout << go << endl;
		cout << "linear_group::linear_group_init label=" << label << endl;
	}



	if (f_v) {
		cout << "linear_group::linear_group_init done" << endl;
	}
}

void linear_group::linear_group_import(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_group::linear_group_import" << endl;
	}

	if (description->f_import_group_of_plane) {

		if (f_v) {
			cout << "linear_group::linear_group_import "
					"before linear_group_import_group_of_plane" << endl;
		}

		linear_group_import_group_of_plane(verbose_level);

		if (f_v) {
			cout << "linear_group::linear_group_import "
					"after linear_group_import_group_of_plane" << endl;
		}
	}
	if (f_v) {
		cout << "linear_group::linear_group_import done" << endl;
	}
}

void linear_group::linear_group_import_group_of_plane(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_group::linear_group_import_group_of_plane" << endl;
	}

	if (f_v) {
		cout << "linear_group::linear_group_import_group_of_plane "
				"import_group_of_plane_label=" << description->import_group_of_plane_label << endl;
	}

	int idx;

	idx = other::orbiter_kernel_system::Orbiter->Orbiter_symbol_table->find_symbol(
			description->import_group_of_plane_label);

	combinatorics_with_groups::translation_plane_via_andre_model *TP;

	TP = (combinatorics_with_groups::translation_plane_via_andre_model *)
		other::orbiter_kernel_system::Orbiter->Orbiter_symbol_table->get_object(idx);

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

	label = "group_of_plane_" + TP->label_txt;
	label_tex = "group of plane " + TP->label_tex;

	A2 = TP->OnAndre;
	if (TP->OnAndre->degree == 0) {
		cout << "linear_group::linear_group_import_group_of_plane "
				"TP->OnAndre->degree == 0" << endl;
		exit(1);
	}
	vector_space_dimension = n;
	q = input_q;
	f_has_strong_generators = true;
	Strong_gens = initial_strong_gens;

	if (f_v) {
		cout << "linear_group::linear_group_import_group_of_plane done" << endl;
	}
}

#if 0
void linear_group::do_stabilizer_of_variety(
		std::string &variety_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_group::do_stabilizer_of_variety" << endl;
	}

	if (f_v) {
		cout << "linear_group::do_stabilizer_of_variety "
				"variety_label=" << variety_label << endl;
	}



	if (f_v) {
		cout << "linear_group::do_stabilizer_of_variety done" << endl;
	}
}
#endif


void linear_group::linear_group_create(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_group::linear_group_create" << endl;
	}
	if (description->F == NULL) {
		cout << "linear_group::linear_group_create "
				"please specify a finite field" << endl;
		exit(1);
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
		cout << "linear_group::linear_group_create "
				"initializing projective group" << endl;
	}



	initial_strong_gens = NEW_OBJECT(groups::strong_generators);

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
		algebra::ring_theory::longinteger_object go;
		initial_strong_gens->group_order(go);
		//A_linear->Strong_gens->group_order(go);
		cout << go << endl;
	}





	Mtx = A_linear->G.matrix_grp;
	vector_space_dimension = n;


	int f_OK = false;

	if (f_v) {
		cout << "linear_group::linear_group_create before linear_group_apply_modification" << endl;
	}

	f_OK = linear_group_apply_modification(
			description,
			verbose_level);

	if (f_v) {
		cout << "linear_group::linear_group_create after linear_group_apply_modification" << endl;
		cout << "linear_group::linear_group_create label = " << label << endl;
		cout << "linear_group::linear_group_create label_tex = " << label_tex << endl;
	}


	if (!f_OK) {
		if (f_v) {
			cout << "linear_group::linear_group_create "
					"!f_OK, A2 = A_linear" << endl;
		}
		A2 = A_linear;
		vector_space_dimension = n;
		q = input_q;
		f_has_strong_generators = true;
		Strong_gens = initial_strong_gens;
	}

	if (f_v) {
		cout << "linear_group::linear_group_create label=" << label << endl;
		cout << "linear_group::linear_group_create degree=" << A2->degree << endl;
		cout << "linear_group::linear_group_create go=";
		algebra::ring_theory::longinteger_object go;
		Strong_gens->group_order(go);
		//A_linear->Strong_gens->group_order(go);
		cout << go << endl;
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
	int f_OK = false;


	if (description->f_lex_least_base) {
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"computing lex least base" << endl;
		}

		groups::sims *Sims;

		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before initial_strong_gens->create_sims" << endl;
		}
		Sims = initial_strong_gens->create_sims(0 /*verbose_level*/);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"after initial_strong_gens->create_sims" << endl;
			//cout << "linear_group::linear_group_apply_modification "
			//		"Sims = " << Sims << endl;
			//Sims->print(verbose_level);

		}

		long int *old_base;
		int old_base_len;

		old_base_len = A_linear->base_len();
		old_base = NEW_lint(old_base_len);
		Lint_vec_copy(A_linear->get_base(), old_base, old_base_len);

		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before A_linear->lex_least_base_in_place" << endl;
		}
		A_linear->lex_least_base_in_place(Sims, verbose_level - 2);
		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"computing lex least base done" << endl;
			cout << "base: ";
			Lint_vec_print(cout, A_linear->get_base(), A_linear->base_len());
			cout << endl;
		}

		long int *new_base;
		int new_base_len;

		new_base_len = A_linear->base_len();
		new_base = NEW_lint(new_base_len);
		Lint_vec_copy(A_linear->get_base(), new_base, new_base_len);

		int f_base_has_changed = false;

		if (new_base_len != old_base_len) {
			cout << "linear_group::linear_group_apply_modification new_base_len != old_base_len" << endl;
			cout << "linear_group::linear_group_apply_modification new_base_len = " << new_base_len << endl;
			cout << "linear_group::linear_group_apply_modification old_base_len = " << old_base_len << endl;
			f_base_has_changed = true;
		}
		if (!f_base_has_changed) {
			if (Lint_vec_compare(old_base, new_base, old_base_len)) {
				f_base_has_changed = true;
			}
		}

		if (f_base_has_changed) {
			cout << "linear_group::linear_group_apply_modification "
					"The base has changed" << endl;
			cout << "old base: ";
			Lint_vec_print(cout, old_base, old_base_len);
			cout << endl;
			cout << "new base: ";
			Lint_vec_print(cout, new_base, new_base_len);
			cout << endl;

			groups::sims *Sims2;

			if (f_v) {
				cout << "linear_group::linear_group_apply_modification "
						"before A_linear->create_sims_from_generators_with_target_group_order_factorized" << endl;
			}
			Sims2 = A_linear->create_sims_from_generators_with_target_group_order_factorized(
					initial_strong_gens->gens, initial_strong_gens->tl, old_base_len,
					0 /* verbose_level */);
			if (f_v) {
				cout << "linear_group::linear_group_apply_modification "
						"after A_linear->create_sims_from_generators_with_target_group_order_factorized" << endl;
			}

			groups::strong_generators *new_strong_generators;

			new_strong_generators = NEW_OBJECT(groups::strong_generators);
			if (f_v) {
				cout << "linear_group::linear_group_apply_modification "
						"before new_strong_generators->init_from_sims" << endl;
			}
			new_strong_generators->init_from_sims(
					Sims2, verbose_level);
			if (f_v) {
				cout << "linear_group::linear_group_apply_modification "
						"after new_strong_generators->init_from_sims" << endl;
			}

			FREE_OBJECT(initial_strong_gens);
			FREE_OBJECT(Sims2);
			initial_strong_gens = new_strong_generators;

		}
		else {
			cout << "linear_group::linear_group_apply_modification "
					"The base did not change." << endl;
		}

		if (f_v) {
			A_linear->print_base();
		}

		FREE_lint(old_base);
		FREE_lint(new_base);

		FREE_OBJECT(Sims);
	}

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
		f_OK = true;
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
		f_OK = true;
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
		f_OK = true;
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
		f_OK = true;
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
		f_OK = true;
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
		f_OK = true;
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
		f_OK = true;
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
		f_OK = true;
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
		f_OK = true;
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
		f_OK = true;
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
		f_OK = true;
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
		f_OK = true;
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
		f_OK = true;
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
		f_OK = true;
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
		f_OK = true;
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
		f_OK = true;
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
		std::string label_of_set;
		std::string label_of_set_tex;

		W = A_linear->G.wreath_product_group;
		nb_points = W->degree_of_tensor_action;
		points = NEW_lint(nb_points);
		for (i = 0; i < nb_points; i++) {
			points[i] = W->perm_offset_i[W->nb_factors] + i;
		}

		label_of_set.assign("_on_tensors");
		label_of_set_tex.assign("\\_on\\_tensors");

		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before A_wreath->restricted_action" << endl;
		}
		A2 = A_linear->Induced_action->restricted_action(
				points, nb_points,
				label_of_set,
				label_of_set_tex,
				verbose_level);
		A2->f_is_linear = true;
		A2->dimension = W->dimension_of_tensor_action;
		f_OK = true;
		f_has_strong_generators = true;
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
		std::string label_of_set;
		std::string label_of_set_tex;

		W = A_linear->G.wreath_product_group;
		nb_points = W->nb_rank_one_tensors;
		points = NEW_lint(nb_points);
		for (i = 0; i < nb_points; i++) {
			points[i] = W->perm_offset_i[W->nb_factors] + W->rank_one_tensors_in_PG[i];
		}

		label_of_set.assign("_on_rank_one_tensors");
		label_of_set_tex.assign("\\_on\\_rank\\_one\\_tensors");

		if (f_v) {
			cout << "linear_group::linear_group_apply_modification "
					"before A_wreath->restricted_action" << endl;
		}
		A2 = A_linear->Induced_action->restricted_action(
				points, nb_points,
				label_of_set, label_of_set_tex,
				verbose_level);
		A2->f_is_linear = true;
		A2->dimension = W->dimension_of_tensor_action;
		f_OK = true;
		f_has_strong_generators = true;
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

void linear_group::init_PGL2q_OnConic(
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
		A_linear->Known_groups->create_sims(verbose_level);
	}
	if (!A_linear->f_has_strong_generators) {
		cout << "linear_group::init_PGL2q_OnConic "
				"A_linear does not have strong generators" << endl;
		//A_linear->create_sims(verbose_level);
		exit(1);
	}
	//A2 = NEW_OBJECT(actions::action);
	A2 = A_linear->Induced_action->induced_action_by_representation_on_conic(
			false /* f_induce_action */, NULL,
		verbose_level);

	vector_space_dimension = A2->G.Rep->dimension;
	q = input_q;
	Strong_gens = initial_strong_gens; //A_linear->Strong_gens;
	f_has_strong_generators = false;

	if (f_v) {
		cout << "linear_group::init_PGL2q_OnConic "
				"vector_space_dimension=" << vector_space_dimension << endl;
	}
	if (f_v) {
		cout << "linear_group::init_PGL2q_OnConic "
				"created action of PGL2_on conic:" << endl;
		A2->print_info();
	}

	label += "_OnConic_" + std::to_string(n) + "_" + std::to_string(q);
	label_tex += "\\_OnConic\\_" + std::to_string(n) + "\\_" + std::to_string(q);

	if (f_v) {
		cout << "linear_group::init_PGL2q_OnConic "
				"created group " << label << endl;
	}
}

void linear_group::init_wedge_action(
		int verbose_level)
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
	//A2 = NEW_OBJECT(actions::action);
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
		
	if (f_v) {
		cout << "linear_group::init_wedge_action "
				"before A_linear->Induced_action->induced_action_on_wedge_product" << endl;
	}
	A2 = A_linear->Induced_action->induced_action_on_wedge_product(verbose_level);
	if (f_v) {
		cout << "linear_group::init_wedge_action "
				"after A_linear->Induced_action->induced_action_on_wedge_product" << endl;
	}

	vector_space_dimension = A2->G.AW->wedge_dimension;
	q = input_q;
	Strong_gens = initial_strong_gens; //A_linear->Strong_gens;
	f_has_strong_generators = true;

	if (f_v) {
		cout << "linear_group::init_wedge_action "
				"created wedge action:" << endl;
		A2->print_info();
	}

	label += "_Wedge_" + std::to_string(n) + "_" + std::to_string(q);
	label_tex += "\\_Wedge\\_" + std::to_string(n) + "\\_" + std::to_string(q);

	if (f_v) {
		cout << "linear_group::init_wedge_action "
				"created group " << label << endl;
	}
}

void linear_group::init_wedge_action_detached(
		int verbose_level)
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


	groups::strong_generators *secondary_strong_gens;
	//strong_generators *exterior_square_strong_gens;
	data_structures_groups::vector_ge *secondary_nice_gens;
	int n2;

	combinatorics::other_combinatorics::combinatorics_domain Combi;

	n2 = Combi.binomial2(n);

	secondary_strong_gens = NEW_OBJECT(groups::strong_generators);
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

	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->init(A2, verbose_level);
	Strong_gens->exterior_square(
				A2,
				A_linear->Strong_gens,
				nice_gens,
				verbose_level);
	f_has_strong_generators = true;

	A_linear = A2; // override the original action!

	f_has_nice_gens = true;

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

	label += "_Wedge_" + std::to_string(n) + "_" + std::to_string(q) + "_detached";
	label_tex += "{\\rm WedgeDetached}(" + std::to_string(n) + "," + std::to_string(q) + ")";

	if (f_v) {
		cout << "linear_group::init_wedge_action_detached "
				"created group " << label << endl;
	}
}

void linear_group::init_monomial_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_monomial_group "
				"initializing monomial group" << endl;
	}
		
	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->generators_for_the_monomial_group(
			A_linear,
		Mtx, verbose_level - 1);
	f_has_strong_generators = true;
	
	A2 = A_linear;


	label += "_Monomial_" + std::to_string(n) + "_" + std::to_string(q);
	label_tex += "{\\rm Monomial}(" + std::to_string(n) + "," + std::to_string(q) + ")";

	if (f_v) {
		cout << "linear_group::init_monomial_group "
				"created group " << label << endl;
	}
}

void linear_group::init_diagonal_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_diagonal_group "
				"initializing monomial group" << endl;
	}
		
	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->generators_for_the_diagonal_group(A_linear, 
		Mtx, verbose_level - 1);
	f_has_strong_generators = true;
	
	A2 = A_linear;

	label += "_Diagonal_" + std::to_string(n) + "_" + std::to_string(q);
	label_tex += "{\\rm Diagonal}(" + std::to_string(n) + "," + std::to_string(q) + ")";

	if (f_v) {
		cout << "linear_group::init_diagonal_group "
				"created group " << label << endl;
	}
}

void linear_group::init_singer_group(
		int singer_power, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_singer_group "
				"initializing singer group" << endl;
	}

	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->generators_for_the_singer_cycle(
			A_linear, Mtx, singer_power, nice_gens,
			verbose_level - 1);
	f_has_strong_generators = true;
	f_has_nice_gens = true;
	

	A2 = A_linear;

	label += "_Singer_" + std::to_string(n) + "_" + std::to_string(q) + "_" + std::to_string(singer_power);
	label_tex += "{\\rm Singer}(" + std::to_string(n) + "," + std::to_string(q) + "," + std::to_string(singer_power) + ")";

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

	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->generators_for_the_singer_cycle_and_the_Frobenius(
			A_linear, Mtx, singer_power, nice_gens,
			verbose_level - 1);
	f_has_strong_generators = true;
	f_has_nice_gens = true;


	A2 = A_linear;

	label += "_Singer_and_Frob" + std::to_string(n) + "_" + std::to_string(q) + "_" + std::to_string(singer_power);
	label_tex += "{\\rm SingerFrob}(" + std::to_string(n) + "," + std::to_string(q) + "," + std::to_string(singer_power) + ")";


	if (f_v) {
		cout << "linear_group::init_singer_group_and_frobenius "
				"created group " << label << endl;
	}
}

void linear_group::init_null_polarity_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_null_polarity_group "
				"initializing null polarity group" << endl;
	}

	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->generators_for_the_null_polarity_group(
			A_linear, Mtx, verbose_level - 1);
	f_has_strong_generators = true;
	
	A2 = A_linear;

	label += "_NullPolarity_" + std::to_string(n) + "_" + std::to_string(q);
	label_tex += "{\\rm NullPolarity}(" + std::to_string(n) + "," + std::to_string(q) + ")";

	if (f_v) {
		cout << "linear_group::init_null_polarity_group "
				"created group " << label << endl;
	}
}

void linear_group::init_borel_subgroup_upper(
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
	
	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->generators_for_the_borel_subgroup_upper(
			A_linear, Mtx, verbose_level - 1);
	f_has_strong_generators = true;
	
	A2 = A_linear;


	label += "_BorelUpper_" + std::to_string(n) + "_" + std::to_string(q);
	label_tex += "{\\rm BorelUpper}(" + std::to_string(n) + "," + std::to_string(q) + ")";

	if (f_v) {
		cout << "linear_group::init_borel_subgroup_upper "
				"created group " << label << endl;
	}
}

void linear_group::init_identity_subgroup(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_identity_subgroup "
				"initializing identify subgroup" << endl;
	}

	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->generators_for_the_identity_subgroup(
			A_linear, Mtx, verbose_level - 1);
	f_has_strong_generators = true;
	
	A2 = A_linear;

	label += "_Identity_" + std::to_string(n) + "_" + std::to_string(q);
	label_tex += "{\\rm Identity}(" + std::to_string(n) + "," + std::to_string(q) + ")";

	if (f_v) {
		cout << "linear_group::init_identity_subgroup "
				"created group " << label << endl;
	}
}

void linear_group::init_symplectic_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_symplectic_group "
				"initializing symplectic group" << endl;
	}

	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(groups::strong_generators);
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
	f_has_strong_generators = true;
	
	A2 = A_linear;


	if (Mtx->f_projective) {
		label = "PSp_" + std::to_string(n) + "_" + std::to_string(q);
		label_tex = "{\\rm PSp}(" + std::to_string(n) + "," + std::to_string(q) + ")";
	}
	else if (Mtx->f_affine) {
		label = "ASp_" + std::to_string(n) + "_" + std::to_string(q);
		label_tex = "{\\rm ASp}(" + std::to_string(n) + "," + std::to_string(q) + ")";
	}
	else {
		label = "Sp_" + std::to_string(n) + "_" + std::to_string(q);
		label_tex = "{\\rm Sp}(" + std::to_string(n) + "," + std::to_string(q) + ")";
	}


	if (f_v) {
		cout << "linear_group::init_symplectic_group "
				"created group " << label << endl;
	}
}

void linear_group::init_subfield_structure_action(
		int s, int verbose_level)
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
	
	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->field_reduction(
			A_linear,
			n, s, F, verbose_level - 1);
	//lift_generators_to_subfield_structure(A_linear,
	//P->n + 1, s, P->F, SGens, verbose_level - 1);
	f_has_strong_generators = true;

	A2 = A_linear;


	label += "_Subfield_" + std::to_string(n) + "_" + std::to_string(q) + "_" + std::to_string(s);
	label_tex += "{\\rm SubfieldAction}(" + std::to_string(n) + "," + std::to_string(q)  + "," + std::to_string(s) + ")";

	if (f_v) {
		cout << "linear_group::init_subfield_structure_action "
				"created group " << label << endl;
	}
	
	if (f_v) {
		cout << "linear_group::init_subfield_structure_action "
				"done" << endl;
	}
}

void linear_group::init_orthogonal_group(
		int epsilon, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "linear_group::init_orthogonal_group "
				"initializing orthogonal group" << endl;
		cout << "epsilon=" << epsilon << endl;
	}
		
	geometry::orthogonal_geometry::orthogonal *O;

	if (f_v) {
		cout << "linear_group::init_orthogonal_group "
				"verbose_level=" << verbose_level << endl;
	}
	O = NEW_OBJECT(geometry::orthogonal_geometry::orthogonal);
	if (f_v) {
		cout << "linear_group::init_orthogonal_group "
				"before O->init" << endl;
	}
	O->init(epsilon, n, F, verbose_level);
	if (f_v) {
		cout << "linear_group::init_orthogonal_group "
				"after O->init" << endl;
	}

	vector_space_dimension = n;
	q = input_q;
	
	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->generators_for_the_orthogonal_group(
			A_linear,
			O,
			f_semilinear,
			verbose_level - 1);
	f_has_strong_generators = true;
	
	A2 = A_linear;

	label += "G" + O->label_txt;
	label_tex += "G" + O->label_tex;

#if 0
	if (EVEN(n)) {
		if (epsilon == 1) {
			label += "_Orthogonal_plus_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex += "{\\rm O}^+(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
		else {
			label += "_Orthogonal_minus_" + std::to_string(n) + "_" + std::to_string(q);
			label_tex += "{\\rm O}^-(" + std::to_string(n) + "," + std::to_string(q) + ")";
		}
	}
	else {
		label += "_Orthogonal_" + std::to_string(n) + "_" + std::to_string(q);
		label_tex += "{\\rm O}(" + std::to_string(n) + "," + std::to_string(q) + ")";
	}
#endif

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
	
	Strong_gens = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "linear_group::init_subgroup_from_file "
				"reading generators from file " << subgroup_fname << endl;
	}

	Strong_gens->read_file(
			A_linear,
			subgroup_fname, verbose_level - 1);

	if (f_v) {
		cout << "linear_group::init_subgroup_from_file "
				"read generators from file" << endl;
	}

	f_has_strong_generators = true;

	A2 = A_linear;


	label += "_SubgroupFile_" + std::to_string(n) + "_" + std::to_string(q);
	label_tex += "{\\rm SubgroupFile}(" + std::to_string(n) + "," + std::to_string(q) + ")";


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

	Strong_gens = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "linear_group::init_subgroup_by_generators before "
				"Strong_gens->init_subgroup_by_generators" << endl;
	}

	Strong_gens->init_subgroup_by_generators(
			A_linear,
			nb_subgroup_generators, subgroup_generators_data,
			subgroup_order_text,
			nice_gens,
			verbose_level - 1);

	f_has_nice_gens = true;

	if (f_v) {
		cout << "linear_group::init_subgroup_by_generators after "
				"Strong_gens->init_subgroup_by_generators" << endl;
	}

	algebra::ring_theory::longinteger_object go;


	f_has_strong_generators = true;

	Strong_gens->group_order(go);

	if (f_v) {
		cout << "linear_group::init_subgroup_by_generators go = " << go << endl;
	}



	A2 = A_linear;

	stringstream str;
	other::l1_interfaces::latex_interface L;
	int max_len = 80;
	int line_skip = 0;


	L.latexable_string(str, subgroup_label.c_str(), max_len, line_skip);



	label += "_Subgroup_" + subgroup_label + "_" + subgroup_order_text;

	label_tex += "{\\rm Subgroup ";
	label_tex += str.str();
	label_tex += " order " + subgroup_order_text + "}";

	if (f_v) {
		cout << "linear_group::init_subgroup_by_generators "
				"created group " << label << endl;
	}
}

void linear_group::init_subgroup_Janko1(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "linear_group::init_subgroup_Janko1" << endl;
	}

	Strong_gens = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "linear_group::init_subgroup_Janko1 before "
				"Strong_gens->init_subgroup_by_generators" << endl;
	}

	algebra::basic_algebra::matrix_group *M;

	M = A_linear->get_matrix_group();

	Strong_gens->Janko1(
			A_linear,
			M->GFq,
			verbose_level);

	if (f_v) {
		cout << "linear_group::init_subgroup_Janko1 after "
				"Strong_gens->init_subgroup_by_generators" << endl;
	}

	f_has_nice_gens = false;
	f_has_strong_generators = true;

	A2 = A_linear;


	label = "Janko1";
	label_tex = "{\\rm Janko1}";

	if (f_v) {
		cout << "linear_group::init_subgroup_Janko1 "
				"created group " << label << endl;
	}
}


}}}


