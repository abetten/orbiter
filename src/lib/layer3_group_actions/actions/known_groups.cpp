/*
 * known_groups.cpp
 *
 *  Created on: Jan 28, 2023
 *      Author: betten
 */

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {



known_groups::known_groups()
{
	A = NULL;
}

known_groups::~known_groups()
{

}



void known_groups::init(
		action *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "known_groups::init" << endl;
	}
	known_groups::A = A;
}



void known_groups::init_linear_group(
		field_theory::finite_field *F, int m,
	int f_projective, int f_general, int f_affine,
	int f_semilinear, int f_special,
	data_structures_groups::vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_basis = true;
	int f_init_sims = false;

	if (f_v) {
		cout << "known_groups::init_linear_group "
				"m=" << m << " q=" << F->q << endl;
	}

	if (f_projective) {
		if (f_v) {
			cout << "known_groups::init_linear_group "
					"before init_projective_group "
					"m=" << m << " q=" << F->q
					<< " f_semilinear=" << f_semilinear << endl;
		}
		init_projective_group(
				m, F, f_semilinear,
			f_basis, f_init_sims,
			nice_gens,
			verbose_level);
		if (f_v) {
			cout << "known_groups::init_linear_group "
					"after init_projective_group "
					"m=" << m << " q=" << F->q
					<< " f_semilinear=" << f_semilinear << endl;
		}
	}
	else if (f_general) {
		if (f_v) {
			cout << "known_groups::init_linear_group "
					"before init_general_linear_group "
					"m=" << m << " q=" << F->q
					<< " f_semilinear=" << f_semilinear << endl;
		}
		init_general_linear_group(
				m, F, f_semilinear,
			f_basis, f_init_sims,
			nice_gens,
			verbose_level);
		if (f_v) {
			cout << "known_groups::init_linear_group "
					"after init_general_linear_group "
					"m=" << m << " q=" << F->q
					<< " f_semilinear=" << f_semilinear << endl;
		}
	}
	else if (f_affine) {
		if (f_v) {
			cout << "known_groups::init_linear_group "
					"before init_affine_group "
					"m=" << m << " q=" << F->q
					<< " f_semilinear=" << f_semilinear << endl;
		}
		init_affine_group(
				m, F, f_semilinear,
			f_basis, f_init_sims,
			nice_gens,
			verbose_level);
		if (f_v) {
			cout << "known_groups::init_linear_group "
					"after init_affine_group "
					"m=" << m << " q=" << F->q
					<< " f_semilinear=" << f_semilinear << endl;
		}
	}
	else {
		cout << "known_groups::init_linear_group "
				"the type of group is not specified" << endl;
		exit(1);
	}


	if (!A->f_has_strong_generators) {
		cout << "known_groups::init_linear_group "
				"fatal: !f_has_strong_generators" << endl;
	}



	if (f_special) {

		if (f_v) {
			cout << "known_groups::init_linear_group "
					"before compute_special_subgroup" << endl;
		}
		compute_special_subgroup(
				verbose_level);
		if (f_v) {
			cout << "known_groups::init_linear_group "
					"after compute_special_subgroup" << endl;
		}

	}

#if 0
	else {
		if (f_v) {
			cout << "known_groups::init_linear_group "
					"before Strong_gens->create_sims" << endl;
		}

		S = Strong_gens->create_sims(verbose_level - 2);
		}

	if (f_v) {
		cout << "known_groups::init_linear_group "
				"sims object has been created" << endl;
		}
#endif






	if (f_v) {
		A->print_base();
#if 0
		if (f_projective) {
			display_all_PG_elements(m - 1, *F);
		}
#endif
	}



	if (f_v) {
		cout << "known_groups::init_linear_group finished" << endl;
	}
}


void known_groups::init_projective_group(
		int n, field_theory::finite_field *F,
	int f_semilinear,
	int f_basis, int f_init_sims,
	data_structures_groups::vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::matrix_group *M;

	if (f_v) {
		cout << "known_groups::init_projective_group" << endl;
		cout << "n=" << n << " q=" << F->q << endl;
		cout << "f_semilinear=" << f_semilinear << endl;
		cout << "f_basis=" << f_basis << endl;
	}



	M = NEW_OBJECT(algebra::matrix_group);



	A->type_G = matrix_group_t;
	A->G.matrix_grp = M;
	A->f_allocated = true;

	A->f_is_linear = true;
	A->dimension = n;

	if (f_v) {
		cout << "known_groups::init_projective_group "
				"before M->init_projective_group" << endl;
	}
	M->init_projective_group(
			n, F,
			f_semilinear,
			verbose_level - 3);
	if (f_v) {
		cout << "known_groups::init_projective_group "
				"after M->init_projective_group" << endl;
	}

	action_global AG;

	if (f_v) {
		cout << "known_groups::init_projective_group "
				"before AG.init_base" << endl;
	}
	AG.init_base(
			A, M, verbose_level - 1);
	if (f_v) {
		cout << "known_groups::init_projective_group "
				"after AG.init_base" << endl;
	}


	A->low_level_point_size = M->low_level_point_size;
	if (f_v) {
		cout << "known_groups::init_projective_group "
				"low_level_point_size="
			<< A->low_level_point_size << endl;
	}
	A->label.assign(M->label);
	A->label_tex.assign(M->label_tex);
	if (f_v) {
		cout << "known_groups::init_projective_group "
				"label=" << A->label << endl;
		cout << "known_groups::init_projective_group "
				"label_tex=" << A->label_tex << endl;
	}

	A->degree = M->degree;
	A->make_element_size = M->elt_size_int_half;

	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_matrix_group();

	A->elt_size_in_int = M->elt_size_int;
	A->coded_elt_size_in_char = M->char_per_elt;
	A->Group_element->allocate_element_data();


	if (f_basis) {
		if (f_v) {
			cout << "known_groups::init_projective_group "
					"before setup_linear_group_from_strong_generators"
					<< endl;
		}
		setup_linear_group_from_strong_generators(
				M,
				nice_gens, f_init_sims,
				verbose_level - 3);
		if (f_v) {
			cout << "known_groups::init_projective_group "
					"after setup_linear_group_from_strong_generators"
					<< endl;
		}
	}
	if (f_v) {
		cout << "known_groups::init_projective_group, "
				"finished setting up "
				<< A->label;
		cout << ", a permutation group of degree " << A->degree << " ";
		cout << "and of order ";
		A->print_group_order(cout);
		cout << endl;
		//cout << "make_element_size=" << make_element_size << endl;
		//cout << "base_len=" << base_len << endl;
		//cout << "f_semilinear=" << f_semilinear << endl;
	}
}

void known_groups::init_affine_group(
		int n, field_theory::finite_field *F,
	int f_semilinear,
	int f_basis, int f_init_sims,
	data_structures_groups::vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::matrix_group *M;

	if (f_v) {
		cout << "known_groups::init_affine_group" << endl;
		cout << "n=" << n << " q=" << F->q << endl;
		cout << "f_semilinear=" << f_semilinear << endl;
		cout << "f_basis=" << f_basis << endl;
	}

	M = NEW_OBJECT(algebra::matrix_group);



	A->type_G = matrix_group_t;
	A->G.matrix_grp = M;
	A->f_allocated = true;

	A->f_is_linear = true;
	A->dimension = n;

	if (f_v) {
		cout << "known_groups::init_affine_group "
				"before M->init_affine_group" << endl;
	}
	M->init_affine_group(
			n, F, f_semilinear, /*A,*/
			verbose_level - 1);
	if (f_v) {
		cout << "known_groups::init_affine_group "
				"after M->init_affine_group" << endl;
	}
	action_global AG;

	if (f_v) {
		cout << "known_groups::init_affine_group "
				"before AG.init_base" << endl;
	}
	AG.init_base(
			A, M, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "known_groups::init_affine_group "
				"after AG.init_base" << endl;
	}


	A->low_level_point_size = M->low_level_point_size;
	if (f_v) {
		cout << "known_groups::init_affine_group "
				"low_level_point_size="
		<< A->low_level_point_size<< endl;
	}
	A->label.assign(M->label);
	A->label_tex.assign(M->label_tex);
	if (f_v) {
		cout << "known_groups::init_affine_group "
				"label=" << A->label << endl;
	}

	A->degree = M->degree;
	A->make_element_size = M->elt_size_int_half;

	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_matrix_group();

	A->elt_size_in_int = M->elt_size_int;
	A->coded_elt_size_in_char = M->char_per_elt;
	A->Group_element->allocate_element_data();

	if (f_basis) {
		setup_linear_group_from_strong_generators(
				M,
				nice_gens, f_init_sims,
				verbose_level);
	}
	if (f_v) {
		cout << "known_groups::init_affine_group, "
				"finished setting up "
				<< A->label;
		cout << ", a permutation group of degree " << A->degree << " ";
		cout << "and of order ";
		A->print_group_order(cout);
		cout << endl;
		//cout << "make_element_size=" << make_element_size << endl;
		//cout << "base_len=" << base_len << endl;
		//cout << "f_semilinear=" << f_semilinear << endl;
	}
}

void known_groups::init_general_linear_group(
		int n, field_theory::finite_field *F,
	int f_semilinear,
	int f_basis, int f_init_sims,
	data_structures_groups::vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::matrix_group *M;

	if (f_v) {
		cout << "known_groups::init_general_linear_group" << endl;
		cout << "n=" << n << " q=" << F->q << endl;
		cout << "f_semilinear=" << f_semilinear << endl;
		cout << "f_basis=" << f_basis << endl;
	}

	M = NEW_OBJECT(algebra::matrix_group);



	A->type_G = matrix_group_t;
	A->G.matrix_grp = M;
	A->f_allocated = true;

	A->f_is_linear = true;
	A->dimension = n;

	M->init_general_linear_group(
			n, F,
		f_semilinear, /*A,*/ verbose_level - 1);

	action_global AG;

	if (f_v) {
		cout << "known_groups::init_general_linear_group "
				"before AG.init_base" << endl;
	}
	AG.init_base(A, M, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "known_groups::init_general_linear_group "
				"after AG.init_base" << endl;
	}



	A->low_level_point_size = M->low_level_point_size;
	if (f_v) {
		cout << "known_groups::init_general_linear_group "
			"low_level_point_size="
			<< A->low_level_point_size << endl;
	}
	A->label.assign(M->label);
	A->label_tex.assign(M->label_tex);
	if (f_v) {
		cout << "known_groups::init_general_linear_group "
				"label=" << A->label << endl;
	}

	A->degree = M->degree;
	A->make_element_size = M->elt_size_int_half;

	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_matrix_group();

	A->elt_size_in_int = M->elt_size_int;
	A->coded_elt_size_in_char = M->char_per_elt;
	A->Group_element->allocate_element_data();


	if (f_basis) {
		setup_linear_group_from_strong_generators(
				M,
				nice_gens, f_init_sims,
				verbose_level);
	}
	if (f_v) {
		cout << "known_groups::init_general_linear_group, "
				"finished setting up " << A->label;
		cout << ", a permutation group of degree " << A->degree << " ";
		cout << "and of order ";
		A->print_group_order(cout);
		cout << endl;
		//cout << "make_element_size=" << make_element_size << endl;
		//cout << "base_len=" << base_len << endl;
		//cout << "f_semilinear=" << f_semilinear << endl;
	}
}

void known_groups::compute_special_subgroup(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "known_groups::compute_special_subgroup" << endl;
	}

	if (f_v) {
		cout << "known_groups::compute_special_subgroup "
				"computing intersection with "
				"special linear group" << endl;
	}


	action *A_on_det;
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "known_groups::compute_special_subgroup "
				"before A->Induced_action->induced_action_on_determinant" << endl;
	}
	A_on_det = A->Induced_action->induced_action_on_determinant(
			A->Sims, verbose_level);
	if (f_v) {
		cout << "known_groups::compute_special_subgroup "
				"after A->Induced_action->induced_action_on_determinant" << endl;
	}
	A_on_det->Kernel->group_order(go);
	if (f_v) {
		cout << "known_groups::compute_special_subgroup "
				"kernel has order " << go << endl;
	}


	groups::strong_generators *SG;

	SG = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "known_groups::compute_special_subgroup "
				"creating strong generators "
				"for the kernel in the action "
				"on the determinant" << endl;
	}

	SG->init_from_sims(
			A_on_det->Kernel, 0 /* verbose_level */);
	//S = SG->create_sims(0 /* verbose_level */);
	FREE_OBJECT(SG);

	FREE_OBJECT(A_on_det);
	if (f_v) {
		cout << "known_groups::compute_special_subgroup done" << endl;
	}

}


void known_groups::setup_linear_group_from_strong_generators(
		algebra::matrix_group *M,
		data_structures_groups::vector_ge *&nice_gens,
		int f_init_sims,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "known_groups::setup_linear_group_from_strong_generators "
				"setting up a basis" << endl;
		cout << "known_groups::setup_linear_group_from_strong_generators "
				"before init_matrix_group_strong_generators_builtin" << endl;
	}
	init_matrix_group_strong_generators_builtin(
			M,
			nice_gens,
			verbose_level - 2);
		// see below
	if (f_v) {
		cout << "known_groups::setup_linear_group_from_strong_generators "
				"after init_matrix_group_strong_generators_builtin" << endl;
	}



	if (f_init_sims) {

		if (f_v) {
			cout << "known_groups::setup_linear_group_from_strong_generators "
					"before init_sims_from_generators" << endl;
		}
		init_sims_from_generators(verbose_level);
		if (f_v) {
			cout << "known_groups::setup_linear_group_from_strong_generators "
					"after init_sims_from_generators" << endl;
		}

	}

	if (f_v) {
		cout << "known_groups::setup_linear_group_from_strong_generators done" << endl;
	}
}

void known_groups::init_sims_from_generators(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::sims *S;


	if (f_v) {
		cout << "known_groups::init_sims_from_generators" << endl;
	}
	S = NEW_OBJECT(groups::sims);

	S->init(A, verbose_level - 2);
	if (f_v) {
		cout << "known_groups::init_sims_from_generators "
				"before S->init_generators" << endl;
	}
	S->init_generators(
			*A->Strong_gens->gens, 0/*verbose_level*/);
	if (f_v) {
		cout << "known_groups::init_sims_from_generators "
				"after S->init_generators" << endl;
	}
	if (f_v) {
		cout << "known_groups::init_sims_from_generators "
				"before S->compute_base_orbits_known_length" << endl;
	}
	S->compute_base_orbits_known_length(
			A->get_transversal_length(), verbose_level);
	if (f_v) {
		cout << "known_groups::init_sims_from_generators "
				"after S->compute_base_orbits_known_length" << endl;
	}


	if (f_v) {
		cout << "known_groups::init_sims_from_generators "
				"before init_sims" << endl;
	}
	A->init_sims_only(S, verbose_level);
	if (f_v) {
		cout << "known_groups::init_sims_from_generators "
				"after init_sims" << endl;
	}
	if (f_v) {
		cout << "known_groups::init_sims_from_generators done" << endl;
	}

}

void known_groups::init_projective_special_group(
	int n, field_theory::finite_field *F,
	int f_semilinear, int f_basis, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "known_groups::init_projective_special_group" << endl;
		cout << "n=" << n << " q=" << F->q << endl;
		cout << "f_semilinear=" << f_semilinear << endl;
		cout << "f_basis=" << f_basis << endl;
	}


	data_structures_groups::vector_ge *nice_gens;
	int f_init_sims = true;

	if (f_v) {
		cout << "known_groups::init_projective_special_group "
				"before init_projective_group" << endl;
	}
	init_projective_group(
			n, F,
			f_semilinear, f_basis, f_init_sims,
			nice_gens,
			verbose_level);
	if (f_v) {
		cout << "known_groups::init_projective_special_group "
				"after init_projective_group" << endl;
	}
	FREE_OBJECT(nice_gens);

	{
		action *A_on_det;
		ring_theory::longinteger_object go;
		groups::strong_generators *gens;
		groups::sims *Sims2;

		gens = NEW_OBJECT(groups::strong_generators);

		if (f_v) {
			cout << "known_groups::init_projective_special_group "
					"computing intersection with special linear group" << endl;
		}
		A_on_det = A->Induced_action->induced_action_on_determinant(
				A->Sims, verbose_level);
		if (f_v) {
			cout << "known_groups::init_projective_special_group "
					"induced_action_on_determinant finished" << endl;
			A_on_det->Kernel->group_order(go);
			cout << "known_groups::init_projective_special_group "
					"intersection has order " << go << endl;
		}
		gens->init_from_sims(
				A_on_det->Kernel, verbose_level - 1);


		Sims2 = gens->create_sims(verbose_level - 1);

		FREE_OBJECT(gens);
		A->init_sims_only(Sims2, verbose_level);

		A->compute_strong_generators_from_sims(0/*verbose_level - 2*/);
		FREE_OBJECT(A_on_det);
	}

	if (f_v) {
		cout << "known_groups::init_projective_special_group done" << endl;
	}
}

void known_groups::init_matrix_group_strong_generators_builtin(
		algebra::matrix_group *M,
		data_structures_groups::vector_ge *&nice_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int n, q;
	field_theory::finite_field *F;
	int *data;
	int size, nb_gens;

	if (f_v) {
		cout << "known_groups::init_matrix_group_strong_generators_builtin" << endl;
	}
	F = M->GFq;
	n = M->n;
	q = F->q;
	if (f_v) {
		cout << "known_groups::init_matrix_group_strong_generators_builtin "
				"computing strong generators builtin group" << endl;
		cout << "n=" << n << endl;
		cout << "q=" << q << endl;
		cout << "p=" << F->p << endl;
		cout << "e=" << F->e << endl;
		cout << "f_semilinear=" << M->f_semilinear << endl;
	}

	if (f_v) {
		cout << "known_groups::init_matrix_group_strong_generators_builtin "
				"getting strong generators" << endl;
	}
	if (M->f_projective) {

		algebra::group_generators_domain GGD;

		if (f_v) {
			cout << "known_groups::init_matrix_group_strong_generators_builtin "
					"before GGD.strong_generators_for_projective_linear_group" << endl;
		}
		GGD.strong_generators_for_projective_linear_group(
				n, F,
			M->f_semilinear,
			data, size, nb_gens,
			0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "known_groups::init_matrix_group_strong_generators_builtin "
					"after GGD.strong_generators_for_projective_linear_group" << endl;
		}
	}
	else if (M->f_affine) {

		algebra::group_generators_domain GGD;

		if (f_v) {
			cout << "known_groups::init_matrix_group_strong_generators_builtin "
					"before GGD.strong_generators_for_affine_linear_group" << endl;
		}
		GGD.strong_generators_for_affine_linear_group(
				n, F,
			M->f_semilinear,
			data, size, nb_gens,
			0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "known_groups::init_matrix_group_strong_generators_builtin "
					"after GGD.strong_generators_for_affine_linear_group" << endl;
		}
	}
	else if (M->f_general_linear) {

		algebra::group_generators_domain GGD;

		if (f_v) {
			cout << "known_groups::init_matrix_group_strong_generators_builtin "
					"before GGD.strong_generators_for_general_linear_group" << endl;
		}
		GGD.strong_generators_for_general_linear_group(
				n, F,
			M->f_semilinear,
			data, size, nb_gens,
			0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "known_groups::init_matrix_group_strong_generators_builtin "
					"after GGD.strong_generators_for_general_linear_group" << endl;
		}
	}
	else {
		cout << "known_groups::init_matrix_group_strong_generators_builtin "
				"unknown group type" << endl;
		exit(1);
	}


	A->f_has_strong_generators = true;
	A->Strong_gens = NEW_OBJECT(groups::strong_generators);

	if (f_v) {
		cout << "known_groups::init_matrix_group_strong_generators_builtin "
				"before Strong_gens->init_from_data" << endl;
	}
	A->Strong_gens->init_from_data(
			A, data, nb_gens, size,
			A->get_transversal_length(),
			nice_gens,
			verbose_level - 1);
	if (f_v) {
		cout << "known_groups::init_matrix_group_strong_generators_builtin "
				"after Strong_gens->init_from_data" << endl;
	}
	if (f_v) {
		cout << "known_groups::init_matrix_group_strong_generators_builtin "
				"generators:" << endl;
		A->Strong_gens->print_generators_tex(cout);
	}

	FREE_int(data);

	if (f_v) {
		cout << "known_groups::init_matrix_group_strong_generators_builtin done" << endl;
	}
}

void known_groups::init_permutation_group(
		int degree, int f_no_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "known_groups::init_permutation_group, "
				"degree=" << degree
				<< " f_no_base=" << f_no_base
				<< " verbose_level=" << verbose_level << endl;
	}


	int page_length_log = PAGE_LENGTH_LOG;
	group_constructions::permutation_representation_domain *P;


	A->label = "Perm_" + std::to_string(degree);
	A->label_tex = "Perm\\_" + std::to_string(degree);

	P = NEW_OBJECT(group_constructions::permutation_representation_domain);
	A->type_G = perm_group_t;
	A->G.perm_grp = P;
	A->f_allocated = true;

	if (f_v) {
		cout << "known_groups::init_permutation_group "
				"before P->init" << endl;
	}

	P->init(
			degree, page_length_log,
			verbose_level - 2);

	if (f_v) {
		cout << "known_groups::init_permutation_group "
				"after P->init" << endl;
	}

	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_permutation_group();


	A->elt_size_in_int = P->elt_size_int;
	A->coded_elt_size_in_char = P->char_per_elt;
	if (f_vv) {
		cout << "elt_size_in_int = " << A->elt_size_in_int << endl;
		cout << "coded_elt_size_in_char = " << A->coded_elt_size_in_char << endl;
	}
	A->Group_element->allocate_element_data();
	A->degree = degree;
	A->make_element_size = degree;


	// ToDo


	if (f_no_base) {
		if (f_vv) {
			cout << "known_groups::init_permutation_group "
					"no base" << endl;
		}
	}
	else {
		if (degree > 20000) {
			cout << "known_groups::init_permutation_group "
					"the degree is too large" << endl;
			cout << "known_groups::init_permutation_group "
					"degree = " << degree << endl;
			exit(1);
		}
		if (f_vv) {
			cout << "known_groups::init_permutation_group "
					"calling allocate_base_data" << endl;
		}
		A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
		A->Stabilizer_chain->allocate_base_data(A, degree, verbose_level);

		// init trivial base:
		int i;
		for (i = 0; i < A->base_len(); i++) {
			A->base_i(i) = i;
		}
	}

	// ToDo
	if (f_v) {
		cout << "known_groups::init_permutation_group finished" << endl;
		cout << "a permutation group of degree " << A->degree << endl;
		cout << "and of order ";
		A->print_group_order(cout);
		cout << endl;
		A->print_info();
	}
	if (f_v) {
		cout << "known_groups::init_permutation_group done" << endl;
	}

}

void known_groups::init_permutation_group_from_nauty_output(
		l1_interfaces::nauty_output *NO,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "known_groups::init_permutation_group_from_nauty_output" << endl;
	}


	if (NO->invariant_set_size != NO->N) {
		if (f_v) {
			cout << "known_groups::init_permutation_group_from_nauty_output"
					"using invariant set" << endl;
		}

		// restrict the permutation action to the invariant set:
		// The idea is that the invariant set is small
		// and hence it is easier to establish the permutation group
		// on it.

		int i;
		long int b;

		for (i = 0; i < NO->Base_length; i++) {
			b = NO->Base_lint[i];
			if (b >= NO->invariant_set_start + NO->invariant_set_size) {
				cout << "known_groups::init_permutation_group_from_nauty_output"
					"using invariant set base point is out of range" << endl;
				exit(1);
			}
		}

		int *gens;
		long int *fresh_base;
		int j;
		int image;

		fresh_base = NEW_lint(NO->Base_length);
		for (i = 0; i < NO->Base_length; i++) {
			b = NO->Base_lint[i] - NO->invariant_set_start;
			if (b >= NO->invariant_set_size) {
				cout << "known_groups::init_permutation_group_from_nauty_output "
						"the base does not belong to the invariant set" << endl;
				exit(1);
			}
			fresh_base[i] = b;
		}

		gens = NEW_int(NO->Aut_counter * NO->invariant_set_size);
		for (i = 0; i < NO->Aut_counter; i++) {
			for (j = 0; j < NO->invariant_set_size; j++) {
				image = NO->Aut[i * NO->N + NO->invariant_set_start + j];
				image -= NO->invariant_set_start;
				if (image >= NO->invariant_set_size) {
					cout << "known_groups::init_permutation_group_from_nauty_output "
							"the set if not invariant" << endl;
					exit(1);
				}
				gens[i * NO->invariant_set_size + j] = image;
			}
		}
		if (f_v) {
			cout << "known_groups::init_permutation_group_from_nauty_output "
					"before init_permutation_group_from_generators" << endl;
		}
		init_permutation_group_from_generators(
				NO->invariant_set_size,
			true, *NO->Ago,
			NO->Aut_counter, gens,
			NO->Base_length, fresh_base,
			true /* f_given_base */,
			verbose_level - 2);
		if (f_v) {
			cout << "known_groups::init_permutation_group_from_nauty_output "
					"after init_permutation_group_from_generators" << endl;
		}
		FREE_lint(fresh_base);
		FREE_int(gens);

	}
	else {
		if (f_v) {
			cout << "known_groups::init_permutation_group_from_nauty_output "
					"before init_permutation_group_from_generators" << endl;
		}
		init_permutation_group_from_generators(
				NO->N,
			true, *NO->Ago,
			NO->Aut_counter, NO->Aut,
			NO->Base_length, NO->Base_lint,
			true /* f_given_base */,
			verbose_level - 2);
		if (f_v) {
			cout << "known_groups::init_permutation_group_from_nauty_output "
					"after init_permutation_group_from_generators" << endl;
		}
	}

	if (f_v) {
		cout << "known_groups::init_permutation_group_from_nauty_output done" << endl;
	}
}


void known_groups::init_permutation_group_from_generators(
		int degree,
	int f_target_go, ring_theory::longinteger_object &target_go,
	int nb_gens, int *gens,
	int given_base_length, long int *given_base,
	int f_given_base,
	int verbose_level)
// calls init_base_and_generators is f_given_base is true, otherwise does not initialize group
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "known_groups::init_permutation_group_from_generators "
				"degree=" << degree << " nb_gens=" << nb_gens
				<< " given_base_length=" << given_base_length << endl;
		if (f_target_go) {
			cout << "known_groups::init_permutation_group_from_generators "
					"target group order is " << target_go << endl;
		}
		else {
			cout << "known_groups::init_permutation_group_from_generators "
					"no target group order is given" << endl;
		}
	}
	if (f_v) {
		cout << "known_groups::init_permutation_group_from_generators "
				"given base: ";
		Lint_vec_print(cout, given_base, given_base_length);
		cout << endl;
	}

	A->label = "Perm_" + std::to_string(degree);
	A->label_tex = "Perm\\_" + std::to_string(degree);

	if (f_vv) {
		cout << "known_groups::init_permutation_group_from_generators "
				"the " << nb_gens << " generators are" << endl;
		if (nb_gens > 20) {
			cout << "known_groups::init_permutation_group_from_generators "
					"too many to print" << endl;
		}
		else {
			for (i = 0; i < nb_gens; i++) {
				cout << i << " : ";
				if (degree < 20) {
					Combi.perm_print(cout, gens + i * degree, degree);
				}
				else {
					cout << "known_groups::init_permutation_group_from_generators "
							"too large to print";
				}
				cout << endl;
			}
		}
	}

	if (f_vv) {
		cout << "known_groups::init_permutation_group_from_generators "
				"calling init_permutation_group" << endl;
	}
	init_permutation_group(
			degree,
			f_given_base /*f_no_base*/,
			verbose_level - 2);
	if (f_vv) {
		cout << "known_groups::init_permutation_group_from_generators "
				"after init_permutation_group" << endl;
	}

	if (A->Stabilizer_chain) {
		FREE_OBJECT(A->Stabilizer_chain);
		A->Stabilizer_chain = NULL;
	}

#if 0
	if (!f_given_base) {
		if (f_vv) {
			cout << "known_groups::init_permutation_group_from_generators "
					"!f_given_base" << endl;
		}
	}
	else {
#endif

		if (f_vv) {
			cout << "known_groups::init_permutation_group_from_generators "
					"before init_base_and_generators" << endl;
		}
		init_base_and_generators(
				f_target_go, target_go,
				nb_gens, gens,
				given_base_length, given_base,
				f_given_base,
				verbose_level - 2);
		if (f_vv) {
			cout << "known_groups::init_permutation_group_from_generators "
					"after init_base_and_generators" << endl;
		}

	//}

	if (f_v) {
		A->print_info();
	}
	if (f_v) {
		cout << "known_groups::init_permutation_group_from_generators done" << endl;
	}
}

void known_groups::init_base_and_generators(
		int f_target_go, ring_theory::longinteger_object &target_go,
		int nb_gens, int *gens,
		int given_base_length, long int *given_base,
		int f_given_base,
		int verbose_level)
// calls A->generators_to_strong_generators
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "known_groups::init_base_and_generators "
				"degree=" << A->degree << " nb_gens=" << nb_gens
				<< " given_base_length=" << given_base_length << endl;
		if (f_target_go) {
			cout << "known_groups::init_base_and_generators "
					"target group order is " << target_go << endl;
		}
		else {
			cout << "known_groups::init_base_and_generators "
					"no target group order is given" << endl;
		}
	}

	if (f_vv) {
		cout << "known_groups::init_base_and_generators "
				"calling allocate_base_data" << endl;
		cout << "given_base:";
		Lint_vec_print(cout, given_base, given_base_length);
		cout << " of length " << given_base_length << endl;
	}
	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(
			A, given_base_length, verbose_level - 10);

	// init base:
	for (i = 0; i < A->base_len(); i++) {
		A->base_i(i) = given_base[i];
	}



	if (f_vv) {
		cout << "known_groups::init_base_and_generators, "
				"now trying to set up the group from the given generators"
				<< endl;
	}

	data_structures_groups::vector_ge *generators;
	groups::strong_generators *Strong_gens;

	generators = NEW_OBJECT(data_structures_groups::vector_ge);
	generators->init(A, verbose_level - 2);
	generators->allocate(nb_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		A->Group_element->make_element(
				generators->ith(i), gens + i * A->degree,
			0 /*verbose_level*/);
	}


	if (f_vv) {
		cout << "known_groups::init_base_and_generators "
				"before generators_to_strong_generators" << endl;
	}
	A->generators_to_strong_generators(
		f_target_go, target_go,
		generators, Strong_gens,
		0 /*verbose_level - 5*/);
	if (f_vv) {
		cout << "known_groups::init_base_and_generators "
				"after generators_to_strong_generators" << endl;
	}

	groups::sims *G;

	if (f_vv) {
		cout << "known_groups::init_base_and_generators "
				"before Strong_gens->create_sims" << endl;
	}
	G = Strong_gens->create_sims(verbose_level - 10);
	if (f_vv) {
		cout << "known_groups::init_base_and_generators "
				"after Strong_gens->create_sims" << endl;
	}


	if (f_vv) {
		cout << "known_groups::init_base_and_generators "
				"before init_sims_only" << endl;
	}
	A->init_sims_only(G, verbose_level - 10);
	FREE_OBJECT(generators);
	FREE_OBJECT(Strong_gens);


	if (f_vv) {
		cout << "known_groups::init_base_and_generators "
				"after init_sims_only" << endl;
	}



	if (f_vv) {
		cout << "known_groups::init_base_and_generators "
				"before compute_strong_generators_from_sims" << endl;
	}
	A->compute_strong_generators_from_sims(verbose_level - 10);
	if (f_vv) {
		cout << "known_groups::init_base_and_generators "
				"after_strong_generators_from_sims" << endl;
	}
	if (f_v) {
		cout << "known_groups::init_base_and_generators done" << endl;
	}
}


void known_groups::init_symmetric_group(
		int degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_gens, *gens;
	int given_base_length;
	long int *given_base;
	int i, j;
	ring_theory::longinteger_object go;
	ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "known_groups::init_symmetric_group" << endl;
	}

	D.factorial(go, degree);

	A->make_element_size = degree;
	nb_gens = degree - 1;
	given_base_length = degree - 1;
	gens = NEW_int(nb_gens * degree);
	given_base = NEW_lint(given_base_length);

	// Coxeter generators for the symmetric group:
	for (i = 0; i < nb_gens; i++) {
		for (j = 0; j < degree; j++) {
			gens[i * degree + j] = j;
		}
		gens[i * degree + i] = i + 1;
		gens[i * degree + i + 1] = i;
	}

	for (i = 0; i < given_base_length; i++) {
		given_base[i] = i;
	}
	if (f_v) {
		cout << "known_groups::init_symmetric_group "
				"before init_permutation_group_from_generators" << endl;
	}
	init_permutation_group_from_generators(
		degree,
		true, go,
		nb_gens, gens,
		given_base_length, given_base,
		true /* f_given_base */,
		verbose_level);
	if (f_v) {
		cout << "known_groups::init_symmetric_group "
				"after init_permutation_group_from_generators" << endl;
	}
	FREE_int(gens);
	FREE_lint(given_base);


	A->label = "Sym_" + std::to_string(degree);
	A->label_tex = "{\\rm Sym}_{" + std::to_string(degree) + "}";

	if (f_v) {
		cout << "known_groups::init_symmetric_group done" << endl;
	}
}

void known_groups::init_cyclic_group(
		int degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_gens, *gens;
	int given_base_length;
	long int *given_base;
	int i; //, j;
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "known_groups::init_cyclic_group" << endl;
	}

	go.create(degree);

	A->make_element_size = degree;
	nb_gens = 1;
	given_base_length = 1;
	gens = NEW_int(nb_gens * degree);
	given_base = NEW_lint(given_base_length);


	combinatorics::combinatorics_domain Combi;

	// create the cycle of degree 'degree':
	Combi.perm_cycle(
			gens, degree);

#if 0
	for (j = 0; j < degree; j++) {
		if (j < degree - 1) {
			gens[0 * degree + j] = j + 1;
		}
		else {
			gens[0 * degree + j] = 0;
		}
	}
#endif

	for (i = 0; i < given_base_length; i++) {
		given_base[i] = i;
	}
	if (f_v) {
		cout << "known_groups::init_cyclic_group "
				"before init_permutation_group_from_generators" << endl;
	}
	init_permutation_group_from_generators(
			degree,
		true, go,
		nb_gens, gens,
		given_base_length, given_base,
		true /* f_given_base */,
		verbose_level);
	if (f_v) {
		cout << "known_groups::init_cyclic_group "
				"after init_permutation_group_from_generators" << endl;
	}
	FREE_int(gens);
	FREE_lint(given_base);

	A->label = "C_" + std::to_string(degree);
	A->label_tex = "{\\rm C}_{" + std::to_string(degree) + "}";

	if (f_v) {
		cout << "known_groups::init_cyclic_group done" << endl;
	}
}





void known_groups::init_elementary_abelian_group(
		int order, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_gens, *gens;
	int given_base_length;
	long int *given_base;
	int i, j, k;
	int p, h;
	int offset;
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "known_groups::init_elementary_abelian_group" << endl;
	}

	number_theory::number_theory_domain NT;

	if (!NT.is_prime_power(
			order, p, h)) {
		cout << "known_groups::init_elementary_abelian_group "
				"group order needs to be a prime power" << endl;
		exit(1);
	}

	go.create(order);

	int degree;

	degree = h * p;

	A->make_element_size = degree;


	combinatorics::combinatorics_domain Combi;

	// create the generators for the elementary group

	nb_gens = h;
	given_base_length = h;
	gens = NEW_int(nb_gens * degree);
	given_base = NEW_lint(given_base_length);

	for (i = 0; i < nb_gens; i++) {

		for (j = 0; j < h; j++) {
			if (j == i) {
				Combi.perm_cycle(
						gens + i * degree + j * p, p);
			}
			else {
				Combi.perm_identity(
						gens + i * degree + j * p, p);
			}
			offset = j * p;
			for (k = 0; k < p; k++) {
				gens[i * degree + j * p + k] += offset;
			}
		}
	}


	for (i = 0; i < given_base_length; i++) {
		given_base[i] = i * p;
	}
	if (f_v) {
		cout << "known_groups::init_elementary_abelian_group "
				"before init_permutation_group_from_generators" << endl;
	}
	init_permutation_group_from_generators(
			degree,
		true, go,
		nb_gens, gens,
		given_base_length, given_base,
		true /* f_given_base */,
		verbose_level);
	if (f_v) {
		cout << "known_groups::init_elementary_abelian_group "
				"after init_permutation_group_from_generators" << endl;
	}
	FREE_int(gens);
	FREE_lint(given_base);

	A->label = "E_" + std::to_string(order);
	A->label_tex = "{\\rm E}_{" + std::to_string(order) + "}";

	if (f_v) {
		cout << "known_groups::init_elementary_abelian_group done" << endl;
	}
}




void known_groups::init_identity_group(
		int degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_gens, *gens;
	int given_base_length;
	long int *given_base;
	int i, j;
	ring_theory::longinteger_object go;
	//ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "known_groups::init_identity_group" << endl;
	}
	//D.factorial(go, degree);
	go.create(1);

	A->make_element_size = degree;
	nb_gens = 1;
	given_base_length = 1;
	gens = NEW_int(nb_gens * degree);
	given_base = NEW_lint(given_base_length);

	// create the identity map:
	for (j = 0; j < degree; j++) {
		gens[0 * degree + j] = j;
	}

	for (i = 0; i < given_base_length; i++) {
		given_base[i] = i;
	}
	if (f_v) {
		cout << "known_groups::init_identity_group "
				"before init_permutation_group_from_generators" << endl;
	}
	init_permutation_group_from_generators(
			degree,
		true, go,
		nb_gens, gens,
		given_base_length, given_base,
		true /* f_given_base */,
		verbose_level);
	if (f_v) {
		cout << "known_groups::init_identity_group "
				"after init_permutation_group_from_generators" << endl;
	}
	FREE_int(gens);
	FREE_lint(given_base);

	A->label = "Id_" + std::to_string(degree);
	A->label_tex = "{\\rm Id}_{" + std::to_string(degree) + "}";


	if (f_v) {
		cout << "known_groups::init_identity_group done" << endl;
	}
}

void known_groups::create_sims(
		int verbose_level)
// Creates a sims object from the strong_generators
{
	int f_v = (verbose_level >= 1);
	groups::sims *S;

	if (f_v) {
		cout << "known_groups::create_sims" << endl;
		}
	if (!A->f_has_strong_generators) {
		cout << "known_groups::create_sims "
				"we need strong generators" << endl;
		exit(1);
	}

	S = A->Strong_gens->create_sims(verbose_level - 1);

	A->init_sims_only(S, verbose_level);
	A->compute_strong_generators_from_sims(0/*verbose_level - 2*/);
	if (f_v) {
		cout << "known_groups::create_sims done" << endl;
	}
}

#if 0
void known_groups::init_BLT(
		field_theory::finite_field *F, int f_basis,
		int f_init_hash_table, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int p, hh, epsilon, n;
	int f_semilinear = false;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "known_groups::init_BLT q=" << F->q
				<< " f_init_hash_table=" << f_init_hash_table << endl;
		cout << "f_basis=" << f_basis << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}
	NT.is_prime_power(F->q, p, hh);
	if (hh > 1)
		f_semilinear = true;
	else
		f_semilinear = false;
	epsilon = 0;
	n = 5;


	if (f_v) {
		cout << "known_groups::init_BLT "
				"before init_orthogonal_group" << endl;
	}
	init_orthogonal_group(
			epsilon, n, F,
		true /* f_on_points */,
		false /* f_on_lines */,
		false /* f_on_points_and_lines */,
		f_semilinear,
		f_basis,
		verbose_level - 2);
	if (f_v) {
		cout << "known_groups::init_BLT "
				"after init_orthogonal_group" << endl;
	}

	if (!A->f_has_sims) {
		cout << "known_groups::init_BLT "
				"we need a Sims" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "known_groups::init_BLT "
				"computing lex least base" << endl;
	}
	A->lex_least_base_in_place(A->Sims, verbose_level - 2);
	if (f_v) {
		cout << "known_groups::init_BLT "
				"computing lex least base done" << endl;
		cout << "base: ";
		Lint_vec_print(cout, A->get_base(), A->base_len());
		cout << endl;
	}

	if (f_v) {
		A->print_base();
	}


	if (A->f_has_strong_generators) {
		if (f_v) {
			cout << "known_groups::init_BLT strong "
					"generators have been computed" << endl;
		}
		if (f_vv) {
			A->Strong_gens->print_generators(cout);
		}
	}
	else {
		cout << "known_groups::init_BLT we don't have strong generators" << endl;
		exit(1);
	}

#if 0
	if (f_init_hash_table) {
		matrix_group *M;
		orthogonal *O;

		M = subaction->G.matrix_grp;
		O = M->O;

		if (f_v) {
			cout << "calling init_hash_table_parabolic" << endl;
		}
		init_hash_table_parabolic(*O->F, 4, 0 /* verbose_level */);
	}
#endif

	if (f_v) {
		A->print_info();
	}
	if (f_v) {
		cout << "known_groups::init_BLT done" << endl;
	}
}
#endif


#if 0
void known_groups::init_orthogonal_group(
		int epsilon,
	int n, field_theory::finite_field *F,
	int f_on_points, int f_on_lines, int f_on_points_and_lines,
	int f_semilinear,
	int f_basis, int verbose_level)
// creates an object of type orthogonal
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	orthogonal_geometry::orthogonal *O;

	if (f_v) {
		cout << "known_groups::init_orthogonal_group "
				"verbose_level=" << verbose_level << endl;
	}
	O = NEW_OBJECT(orthogonal_geometry::orthogonal);
	if (f_v) {
		cout << "known_groups::init_orthogonal_group "
				"before O->init" << endl;
	}
	O->init(epsilon, n, F, verbose_level);
	if (f_v) {
		cout << "known_groups::init_orthogonal_group "
				"after O->init" << endl;
	}

	if (f_v) {
		cout << "known_groups::init_orthogonal_group "
				"before init_orthogonal_group_with_O" << endl;
	}
	init_orthogonal_group_with_O(
			O,
			f_on_points, f_on_lines, f_on_points_and_lines,
			f_semilinear,
			f_basis, verbose_level);
	if (f_v) {
		cout << "known_groups::init_orthogonal_group "
				"after init_orthogonal_group_with_O" << endl;
	}


	if (f_v) {
		cout << "known_groups::init_orthogonal_group done" << endl;
	}
}
#endif

void known_groups::init_orthogonal_group_with_O(
	orthogonal_geometry::orthogonal *O,
	int f_on_points, int f_on_lines, int f_on_points_and_lines,
	int f_semilinear,
	int f_basis, int verbose_level)
// sets up a projective group first.
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	action *A_PGL;
	induced_actions::action_on_orthogonal *AO;
	int q = O->F->q;
	algebra::group_generators_domain GG;

	if (f_v) {
		cout << "known_groups::init_orthogonal_group_with_O "
				"verbose_level=" << verbose_level << endl;
	}
	A_PGL = NEW_OBJECT(action);
	if (f_v) {
		cout << "known_groups::init_orthogonal_group_with_O "
				"before A_PGL->Known_groups->init_projective_group" << endl;
	}
	data_structures_groups::vector_ge *nice_gens = NULL;
	A_PGL->Known_groups->init_projective_group(
			O->Quadratic_form->n,
			O->F,
			f_semilinear,
			false /* f_basis */, // we don't need a basis
			true /* f_init_sims */,
			nice_gens,
			verbose_level - 2);
	if (f_v) {
		cout << "known_groups::init_orthogonal_group_with_O "
				"after A_PGL->Known_groups->init_projective_group" << endl;
	}

	if (nice_gens) {
		FREE_OBJECT(nice_gens);
	}

	AO = NEW_OBJECT(induced_actions::action_on_orthogonal);
	if (f_v) {
		cout << "known_groups::init_orthogonal_group_with_O "
				"before AO->init" << endl;
	}
	AO->init(
			A_PGL, O, f_on_points, f_on_lines,
			f_on_points_and_lines,
			verbose_level);
	if (f_v) {
		cout << "known_groups::init_orthogonal_group_with_O "
				"after AO->init" << endl;
	}

	A->type_G = action_on_orthogonal_t;
	A->G.AO = AO;

	A->f_has_subaction = true;
	A->subaction = A_PGL;
	A->degree = AO->degree;
	A->low_level_point_size = A_PGL->low_level_point_size;
	A->elt_size_in_int = A_PGL->elt_size_in_int;
	A->coded_elt_size_in_char = A_PGL->coded_elt_size_in_char;
	A->Group_element->allocate_element_data();

	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_induced_action();
	A->make_element_size = A_PGL->make_element_size;


	data_structures::string_tools String;

	String.name_of_orthogonal_group(
			A->label,
			A->label_tex,
			O->Quadratic_form->epsilon, O->Quadratic_form->n, q,
			f_semilinear, verbose_level - 1);


	if (f_basis) {
		ring_theory::longinteger_object target_go;

		if (f_v) {
			cout << "known_groups::init_orthogonal_group_with_O "
					"we will create the orthogonal group now" << endl;
		}

		action_global AG;

		if (AG.get_orthogonal_group_type_f_reflection()) {
			if (f_v) {
				cout << "known_groups::init_orthogonal_group_with_O "
						"with reflections, before order_PO_epsilon" << endl;
			}
			GG.order_PO_epsilon(
					f_semilinear,
					O->Quadratic_form->epsilon,
					O->Quadratic_form->n - 1,
					O->F->q,
					target_go, verbose_level - 2);
			if (f_v) {
				cout << "known_groups::init_orthogonal_group_with_O "
						"with reflections, after order_PO_epsilon" << endl;
			}
		}
		else {
			if (f_v) {
				cout << "known_groups::init_orthogonal_group_with_O "
						"without reflections, before order_POmega_epsilon"
						<< endl;
			}
			GG.order_POmega_epsilon(
					O->Quadratic_form->epsilon, O->Quadratic_form->n - 1,
					O->F->q, target_go, verbose_level);
		}

		if (f_v) {
			cout << "known_groups::init_orthogonal_group_with_O "
					"the target group order is " << target_go << endl;
		}

		if (f_v) {
			cout << "known_groups::init_orthogonal_group_with_O "
					"before create_orthogonal_group" << endl;
		}
		create_orthogonal_group(
				A_PGL /*subaction*/,
			true /* f_has_target_go */, target_go,
			callback_choose_random_generator_orthogonal,
			verbose_level - 2);
		if (f_v) {
			cout << "known_groups::init_orthogonal_group_with_O "
					"after create_orthogonal_group" << endl;
		}
	}

	if (f_v) {
		cout << "known_groups::init_orthogonal_group_with_O done" << endl;
	}
}




void known_groups::init_wreath_product_group_and_restrict(
		int nb_factors, int n,
		field_theory::finite_field *F,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A_wreath;
	action *Awr;
	group_constructions::wreath_product *W;
	long int *points;
	int nb_points;
	int i;

	if (f_v) {
		cout << "known_groups::init_wreath_product_group_and_restrict" << endl;
		cout << "nb_factors=" << nb_factors
				<< " n=" << n << " q=" << F->q << endl;
	}
	A_wreath = NEW_OBJECT(action);
	if (f_v) {
		cout << "known_groups::init_wreath_product_group_and_restrict "
				"before A_wreath->init_wreath_product_group" << endl;
	}
	A_wreath->Known_groups->init_wreath_product_group(
			nb_factors, n, F, nice_gens,
			verbose_level);
	if (f_v) {
		cout << "known_groups::init_wreath_product_group_and_restrict "
				"after A_wreath->init_wreath_product_group" << endl;
	}

	W = A_wreath->G.wreath_product_group;
	nb_points = W->degree_of_tensor_action;
	points = NEW_lint(nb_points);
	for (i = 0; i < nb_points; i++) {
		points[i] = W->perm_offset_i[nb_factors] + i;
	}

	std::string label_of_set;
	std::string label_of_set_tex;

	label_of_set.assign("_wreath_product");
	label_of_set_tex.assign("\\_wreath\\_product");


	if (f_v) {
		cout << "known_groups::init_wreath_product_group_and_restrict "
				"before A_wreath->Induced_action->restricted_action" << endl;
	}
	Awr = A_wreath->Induced_action->restricted_action(
			points, nb_points,
			label_of_set, label_of_set_tex,
			verbose_level);
	Awr->f_is_linear = true;
	if (f_v) {
		cout << "known_groups::init_wreath_product_group_and_restrict "
				"after A_wreath->Induced_action->restricted_action" << endl;
	}

	memcpy(this, Awr, sizeof(action)); // ToDo !!!! this is wrong
	Awr->null();
	FREE_OBJECT(Awr);
}


void known_groups::init_wreath_product_group(
		int nb_factors, int n,
		field_theory::finite_field *F,
		data_structures_groups::vector_ge *&nice_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A_mtx;
	group_constructions::wreath_product *W;
	algebra::matrix_group *M;

	if (f_v) {
		cout << "known_groups::init_wreath_product_group" << endl;
		cout << "nb_factors=" << nb_factors
				<< " n=" << n << " q=" << F->q << endl;
	}

	A_mtx = NEW_OBJECT(action);
	M = NEW_OBJECT(algebra::matrix_group);
	W = NEW_OBJECT(group_constructions::wreath_product);



	A->type_G = wreath_product_t;
	A->G.wreath_product_group = W;
	A->f_allocated = true;

	if (f_v) {
		cout << "known_groups::init_wreath_product_group "
				"before M->init_general_linear_group" << endl;
	}
	M->init_general_linear_group(
			n, F,
			false /* f_semilinear */,
			verbose_level - 1);
	if (f_v) {
		cout << "known_groups::init_wreath_product_group "
				"after M->init_general_linear_group" << endl;
	}
	action_global AG;

	if (f_v) {
		cout << "known_groups::init_wreath_product_group "
				"before AG.init_base" << endl;
	}
	AG.init_base(A_mtx, M, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "known_groups::init_wreath_product_group "
				"after AG.init_base" << endl;
	}


	if (f_v) {
		cout << "known_groups::init_wreath_product_group "
				"before W->init_tensor_wreath_product" << endl;
	}
	W->init_tensor_wreath_product(
			M, A_mtx, nb_factors,
			verbose_level);
	if (f_v) {
		cout << "known_groups::init_wreath_product_group "
				"after W->init_tensor_wreath_product" << endl;
	}

	A->f_is_linear = true;
	A->dimension = W->dimension_of_tensor_action;


	A->low_level_point_size = W->low_level_point_size;
	if (f_v) {
		cout << "known_groups::init_wreath_product_group "
			"low_level_point_size="
			<< A->low_level_point_size << endl;
	}

	A->label.assign(W->label);
	A->label_tex.assign(W->label_tex);


	if (f_v) {
		cout << "known_groups::init_wreath_product_group "
				"label=" << A->label << endl;
	}

	A->degree = W->degree_overall;
	A->make_element_size = W->make_element_size;

	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_wreath_product_group();

	A->elt_size_in_int = W->elt_size_int;
	A->coded_elt_size_in_char = W->char_per_elt;
	A->Group_element->allocate_element_data();




	if (f_v) {
		cout << "known_groups::init_wreath_product_group "
				"degree=" << A->degree << endl;
	}

	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	if (f_v) {
		cout << "known_groups::init_wreath_product_group "
				"before Stabilizer_chain->allocate_base_data" << endl;
	}
	A->Stabilizer_chain->allocate_base_data(
			A, W->base_length, verbose_level);

	if (f_v) {
		cout << "known_groups::init_wreath_product_group "
				"base_len=" << A->base_len() << endl;
	}

	Lint_vec_copy(W->the_base, A->get_base(), A->base_len());
	Int_vec_copy(W->the_transversal_length,
			A->get_transversal_length(), A->base_len());

	int *gens_data;
	int gens_size;
	int gens_nb;

	if (f_v) {
		cout << "known_groups::init_wreath_product_group "
				"before W->make_strong_generators_data" << endl;
	}
	W->make_strong_generators_data(
			gens_data,
			gens_size, gens_nb,
			verbose_level - 10);
	if (f_v) {
		cout << "known_groups::init_wreath_product_group "
				"after W->make_strong_generators_data" << endl;
	}
	A->Strong_gens = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "known_groups::init_wreath_product_group "
				"before Strong_gens->init_from_data" << endl;
	}


	A->Strong_gens->init_from_data(
			A, gens_data, gens_nb, gens_size,
			A->get_transversal_length(),
			nice_gens,
			verbose_level - 10);
	if (f_v) {
		cout << "known_groups::init_wreath_product_group "
				"after Strong_gens->init_from_data" << endl;
	}
	A->f_has_strong_generators = true;
	FREE_int(gens_data);


	if (A->degree < STABILIZER_CHAIN_DATA_MAX_DEGREE) {
		groups::sims *S;

		S = NEW_OBJECT(groups::sims);

		S->init(A, verbose_level - 2);
		if (f_v) {
			cout << "known_groups::init_wreath_product_group "
					"before S->init_generators" << endl;
		}
		S->init_generators(*A->Strong_gens->gens, verbose_level);
		if (f_v) {
			cout << "known_groups::init_wreath_product_group "
					"after S->init_generators" << endl;
		}
		if (f_v) {
			cout << "known_groups::init_wreath_product_group "
					"before S->compute_base_orbits_known_length" << endl;
		}
		S->compute_base_orbits_known_length(
				A->get_transversal_length(), verbose_level);
		if (f_v) {
			cout << "known_groups::init_wreath_product_group "
					"after S->compute_base_orbits_known_length" << endl;
		}


		if (f_v) {
			cout << "known_groups::init_wreath_product_group "
					"before init_sims_only" << endl;
		}

		A->init_sims_only(S, verbose_level);

		if (f_v) {
			cout << "known_groups::init_wreath_product_group "
					"after init_sims_only" << endl;
		}

		A->compute_strong_generators_from_sims(0/*verbose_level - 2*/);
	}
	else {
		cout << "known_groups::init_wreath_product_group "
				"because the degree is very large, "
				"we are not creating a sims object" << endl;
	}

	if (f_v) {
		cout << "known_groups::init_wreath_product_group, finished setting up "
				<< A->label;
		cout << ", a permutation group of degree " << A->degree << " ";
		cout << "and of order ";
		A->print_group_order(cout);
		cout << endl;
		//cout << "make_element_size=" << make_element_size << endl;
		//cout << "base_len=" << base_len << endl;
		//cout << "f_semilinear=" << f_semilinear << endl;
	}
}

void known_groups::init_permutation_representation(
		action *A_original,
		int f_stay_in_the_old_action,
		data_structures_groups::vector_ge *gens,
		int *Perms, int degree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::permutation_representation *P;

	if (f_v) {
		cout << "known_groups::init_permutation_representation" << endl;
		cout << "original action=" << A_original->label
				<< " restricted to degree " << degree << endl;
		cout << "f_stay_in_the_old_action=" << f_stay_in_the_old_action << endl;
	}

	P = NEW_OBJECT(group_constructions::permutation_representation);

	if (f_v) {
		cout << "known_groups::init_permutation_representation "
				"before P->init" << endl;
	}
	P->init(
			A_original,
			f_stay_in_the_old_action,
			gens,
			Perms, degree,
			verbose_level - 2);
	if (f_v) {
		cout << "known_groups::init_permutation_representation "
				"after P->init" << endl;
	}

	A->type_G = permutation_representation_t;
	A->G.Permutation_representation = P;
	A->f_allocated = true;

	if (f_stay_in_the_old_action) {
		A->f_is_linear = A_original->f_is_linear;
		A->dimension = A_original->f_is_linear;
		A->low_level_point_size = A_original->low_level_point_size;
		A->degree = A_original->degree;

		group_constructions::wreath_product *W;
		if (A_original->type_G != wreath_product_t) {
			cout << "known_groups::init_permutation_representation "
					"A_original->type_G != wreath_product_t" << endl;
			exit(1);
		}
		W = A_original->G.wreath_product_group;

		if (f_v) {
			cout << "known_groups::init_permutation_representation "
					"degree=" << degree << endl;
		}

		A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
		if (f_v) {
			cout << "known_groups::init_permutation_representation "
					"before Stabilizer_chain->allocate_base_data" << endl;
		}
		A->Stabilizer_chain->allocate_base_data(
				A, W->base_length, verbose_level);
		//allocate_base_data(base_len);
		//Stabilizer_chain->base_len = W->base_length;
		if (f_v) {
			cout << "known_groups::init_permutation_representation "
					"base_len=" << A->base_len() << endl;
		}

		Lint_vec_copy(W->the_base, A->get_base(), A->base_len());
		Int_vec_copy(W->the_transversal_length,
				A->get_transversal_length(), A->base_len());


		A->label = P->label + "_induced" + std::to_string(degree) + "_prev";
		A->label_tex = P->label_tex + "\\_induced" + std::to_string(degree) + "\\_prev";

	}
	else {
		A->f_is_linear = false;
		A->dimension = 0;
		A->low_level_point_size = 0;
		A->degree = degree;

		A->label = P->label + "_induced" + std::to_string(degree);
		A->label_tex = P->label_tex + "\\_induced" + std::to_string(degree);

	}

	A->make_element_size = P->make_element_size;



	if (f_v) {
		cout << "known_groups::init_permutation_representation "
				"label=" << A->label << endl;
	}

	A->ptr = NEW_OBJECT(action_pointer_table);
	A->ptr->init_function_pointers_permutation_representation_group();

	A->elt_size_in_int = P->elt_size_int;
	A->coded_elt_size_in_char = P->char_per_elt;
	A->Group_element->allocate_element_data();

	//group_prefix.assign(label);



	if (f_v) {
		cout << "known_groups::init_permutation_representation "
				"degree=" << degree << endl;
	}

	if (f_v) {
		cout << "known_groups::init_permutation_representation, "
				"finished setting up " << A->label;
		cout << ", a permutation group of degree " << degree << " ";
		cout << "and of order ";
		A->print_group_order(cout);
		cout << endl;
		//cout << "make_element_size=" << make_element_size << endl;
		//cout << "base_len=" << base_len << endl;
		//cout << "f_semilinear=" << f_semilinear << endl;
	}
}

void known_groups::init_group_from_strong_generators(
		data_structures_groups::vector_ge *gens,
		groups::sims *K,
	int given_base_length, int *given_base,
	int verbose_level)
// calls sims::build_up_group_from_generators
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	groups::sims *G;
	ring_theory::longinteger_object G_order;
	int i;


	if (f_v) {
		cout << "known_groups::init_group_from_strong_generators" << endl;
	}
	A->label.assign("from sgs");
	A->label_tex.assign("from sgs");

	if (f_vv) {
		cout << "generators are" << endl;
		gens->print(cout);
		cout << endl;
	}


	if (f_vv) {
		cout << "known_groups::init_group_from_strong_generators "
				"calling allocate_base_data, initial base:";
		Int_vec_print(cout, given_base, given_base_length);
		cout << " of length " << given_base_length << endl;
	}
	A->Stabilizer_chain = NEW_OBJECT(stabilizer_chain_base_data);
	A->Stabilizer_chain->allocate_base_data(
			A, given_base_length, verbose_level);
	//allocate_base_data(given_base_length);
	//Stabilizer_chain->base_len = given_base_length;

	for (i = 0; i < A->base_len(); i++) {
		A->base_i(i) = given_base[i];
	}



	if (f_vv) {
		cout << "known_groups::init_group_from_strong_generators, "
				"now trying to set up the group "
				"from the given generators" << endl;
	}

	G = NEW_OBJECT(groups::sims);

	G->init(A, verbose_level - 2);
	G->init_trivial_group(verbose_level - 1);
	G->group_order(G_order);

	G->build_up_group_from_generators(
			K, gens,
		false, NULL, /* target_go */
		false /* f_override_choose_next_base_point */,
		NULL,
		verbose_level);

	G->group_order(G_order);


	if (f_vvv) {
		//G.print(true);
		//G.print_generator_depth_and_perm();
	}

	if (f_v) {
		cout << "init_group_from_strong_generators: "
				"found a group of order " << G_order << endl;
		if (f_vv) {
			cout << "transversal lengths:" << endl;
			//int_vec_print(cout, G->orbit_len, base_len());
			for (int t = 0; t < G->A->base_len(); t++) {
				cout << G->get_orbit_length(t) << ", ";
			}
			cout << endl;
		}
	}

	if (f_vv) {
		cout << "known_groups::init_group_from_strong_generators "
				"before init_sims_only" << endl;
	}
	A->init_sims_only(G, 0/*verbose_level - 1*/);
	if (f_vv) {
		cout << "action::init_group_from_strong_generators "
				"after init_sims_only" << endl;
	}
	A->compute_strong_generators_from_sims(0/*verbose_level - 2*/);

	if (f_v) {
		A->print_info();
	}
	if (f_v) {
		cout << "known_groups::init_group_from_strong_generators done" << endl;
	}
}



void known_groups::create_orthogonal_group(
		action *subaction,
	int f_has_target_group_order,
	ring_theory::longinteger_object &target_go,
	void (* callback_choose_random_generator)(int iteration,
		int *Elt, void *data, int verbose_level),
	int verbose_level)
// uses groups::schreier_sims
{
	//verbose_level = 10;

	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "known_groups::create_orthogonal_group" << endl;
	}

	action_global AG;
	algebra::matrix_group *Mtx;
	int degree_save;

	Mtx = subaction->get_matrix_group();

	degree_save = A->degree;
	if (f_v) {
		cout << "known_groups::create_orthogonal_group "
				"before AG.init_base_projective" << endl;
	}
	AG.init_base_projective(
			A, Mtx, verbose_level);
	// initializes base, base_len, degree,
	// transversal_length, orbit, orbit_inv
	if (f_v) {
		cout << "known_groups::create_orthogonal_group "
				"after AG.init_base_projective" << endl;
	}
	A->degree = degree_save;


	if (f_v) {
		cout << "known_groups::create_orthogonal_group "
				"before allocating a schreier_sims object" << endl;
	}

	{
		groups::schreier_sims ss;

		if (f_v) {
			cout << "known_groups::create_orthogonal_group "
					"before ss.init" << endl;
		}
		ss.init(A, verbose_level - 1);
		if (f_v) {
			cout << "known_groups::create_orthogonal_group "
					"after ss.init" << endl;
		}

		ss.interested_in_kernel(subaction, verbose_level - 1);

		if (f_has_target_group_order) {
			ss.init_target_group_order(target_go, verbose_level - 1);
		}

		if (f_v) {
			cout << "known_groups::create_orthogonal_group "
					"before ss.init_random_process" << endl;
		}
		ss.init_random_process(
			callback_choose_random_generator,
			&ss,
			verbose_level - 1);
		if (f_v) {
			cout << "known_groups::create_orthogonal_group "
					"after ss.init_random_process" << endl;
		}

		if (f_v) {
			cout << "known_groups::create_orthogonal_group "
					"before ss.create_group" << endl;
		}
		ss.create_group(verbose_level - 1);
		if (f_v) {
			cout << "known_groups::create_orthogonal_group "
					"after ss.create_group" << endl;
		}

		A->init_sims_only(ss.G, verbose_level);
		A->compute_strong_generators_from_sims(0/*verbose_level - 2*/);

		A->f_has_kernel = true;
		A->Kernel = ss.K;

		ss.K = NULL;
		ss.G = NULL;

		//init_transversal_reps_from_stabilizer_chain(G, verbose_level - 2);
		if (f_v) {
			cout << "known_groups::create_orthogonal_group after init_sims, "
					"calling compute_strong_generators_from_sims" << endl;
		}
		A->compute_strong_generators_from_sims(verbose_level - 2);
		if (f_v) {
			cout << "known_groups::create_orthogonal_group done, "
					"freeing schreier_sims object" << endl;
		}
	}

	if (f_v) {
		cout << "known_groups::create_orthogonal_group "
				"done" << endl;
	}
}





}}}



