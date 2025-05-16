/*
 * algebra_global_with_action.cpp
 *
 *  Created on: Dec 15, 2019
 *      Author: betten
 */



#include "orbiter.h"


using namespace std;


namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {

algebra_global_with_action::algebra_global_with_action()
{
	Record_birth();

}

algebra_global_with_action::~algebra_global_with_action()
{
	Record_death();

}


void algebra_global_with_action::element_processing(
		groups::any_group *Any_group,
		element_processing_description *element_processing_descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::element_processing" << endl;
	}


	apps_algebra::vector_ge_builder *Elements_builder;
	data_structures_groups::vector_ge *Elements;


	if (!element_processing_descr->f_input) {

		cout << "please use -input <label> to define input elements" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "algebra_global_with_action::element_processing "
				"getting input" << endl;
	}

	Elements_builder = Get_object_of_type_vector_ge(
			element_processing_descr->input_label);

	Elements = Elements_builder->V;




	if (element_processing_descr->f_print) {
		if (f_v) {
			cout << "algebra_global_with_action::element_processing f_print" << endl;
		}

		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"before Any_group->print_given_elements_tex" << endl;
		}

		Any_group->print_given_elements_tex(
				element_processing_descr->input_label,
				Elements,
				element_processing_descr->f_with_permutation,
				element_processing_descr->f_with_fix_structure,
				verbose_level);

		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"after Any_group->print_given_elements_tex" << endl;
		}

	}
	else if (element_processing_descr->f_apply_isomorphism_wedge_product_4to6) {
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"f_apply_isomorphism_wedge_product_4to6" << endl;
		}

		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"before Any_group->apply_isomorphism_wedge_product_4to6" << endl;
		}
		Any_group->apply_isomorphism_wedge_product_4to6(
				element_processing_descr->input_label,
				Elements,
				verbose_level);
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"after Any_group->apply_isomorphism_wedge_product_4to6" << endl;
		}


	}
	else if (element_processing_descr->f_order_of_products_of_pairs) {
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"f_order_of_products_of_pairs" << endl;
		}

		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"before Any_group->order_of_products_of_pairs" << endl;
		}
		Any_group->order_of_products_of_pairs(
				element_processing_descr->input_label,
				Elements,
				verbose_level);
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"after Any_group->order_of_products_of_pairs" << endl;
		}


	}
#if 0
	else if (element_processing_descr->f_products_of_pairs) {
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"f_class_of_products_of_pairs" << endl;
		}

		data_structures_groups::vector_ge *Products;


		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"before Any_group->products_of_pairs" << endl;
		}
		Any_group->products_of_pairs(
				Elements,
				Products,
				verbose_level);
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"after Any_group->products_of_pairs" << endl;
		}

		other::orbiter_kernel_system::file_io Fio;
		string fname;

		fname = element_processing_descr->input_label + "_pairs.csv";

		Products->save_csv(
				fname, verbose_level);

		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"Written file " << fname << " of size "
						<< Fio.file_size(fname) << endl;
		}

	}
#endif
	else if (element_processing_descr->f_conjugate) {
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"f_conjugate" << endl;
		}

		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"before Any_group->conjugate" << endl;
		}
		Any_group->conjugate(
				element_processing_descr->input_label,
				element_processing_descr->conjugate_data,
				Elements,
				verbose_level);
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"after Any_group->conjugate" << endl;
		}


	}

	else if (element_processing_descr->f_print_action_on_surface) {
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"f_print_action_on_surface" << endl;
		}


		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"before print_action_on_surface" << endl;
		}
		print_action_on_surface(
				Any_group,
				element_processing_descr->print_action_on_surface_label,
				element_processing_descr->input_label,
				Elements,
				//element_data, nb_elements,
				verbose_level);
		if (f_v) {
			cout << "algebra_global_with_action::element_processing "
					"after print_action_on_surface" << endl;
		}


	}



	if (f_v) {
		cout << "algebra_global_with_action::element_processing done" << endl;
	}
}















void algebra_global_with_action::young_symmetrizer(
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::young_symmetrizer" << endl;
	}

	young *Y;

	Y = NEW_OBJECT(young);

	Y->init(n, verbose_level);



	int *elt1, *elt2, *h_alpha, *elt4, *elt5, *elt6, *elt7;

	Y->group_ring_element_create(Y->A, Y->S, elt1);
	Y->group_ring_element_create(Y->A, Y->S, elt2);
	Y->group_ring_element_create(Y->A, Y->S, h_alpha);
	Y->group_ring_element_create(Y->A, Y->S, elt4);
	Y->group_ring_element_create(Y->A, Y->S, elt5);
	Y->group_ring_element_create(Y->A, Y->S, elt6);
	Y->group_ring_element_create(Y->A, Y->S, elt7);



	int *part;
	int *parts;

	int *Base;
	int *Base_inv;
	int *Fst;
	int *Len;
	int cnt, s, i, j;
	combinatorics::other_combinatorics::combinatorics_domain Combi;


	part = NEW_int(n);
	parts = NEW_int(n);
	Fst = NEW_int(Y->goi);
	Len = NEW_int(Y->goi);
	Base = NEW_int(Y->goi * Y->goi * Y->D->size_of_instance_in_int);
	Base_inv = NEW_int(Y->goi * Y->goi * Y->D->size_of_instance_in_int);
	s = 0;
	Fst[0] = 0;

		// create the first partition in exponential notation:
	Combi.partition_first(part, n);
	cnt = 0;


	while (true) {
		int nb_parts;

		// turn the partition from exponential notation into the list of parts:
		// the large parts come first.
		nb_parts = 0;
		for (i = n - 1; i >= 0; i--) {
			for (j = 0; j < part[i]; j++) {
				parts[nb_parts++] = i + 1;
			}
		}

		if (f_v) {
			cout << "partition ";
			Int_vec_print(cout, parts, nb_parts);
			cout << endl;
		}


			// Create the young symmetrizer based on the partition.
			// We do the very first tableau for this partition.

		int *tableau;

		tableau = NEW_int(n);
		for (i = 0; i < n; i++) {
			tableau[i] = i;
		}
		Y->young_symmetrizer(parts, nb_parts, tableau, elt1, elt2, h_alpha, verbose_level);
		FREE_int(tableau);


		if (f_v) {
			cout << "h_alpha =" << endl;
			Y->group_ring_element_print(Y->A, Y->S, h_alpha);
			cout << endl;
		}


		Y->group_ring_element_copy(Y->A, Y->S, h_alpha, elt4);
		Y->group_ring_element_mult(Y->A, Y->S, elt4, elt4, elt5);

		if (f_v) {
			cout << "h_alpha * h_alpha=" << endl;
			Y->group_ring_element_print(Y->A, Y->S, elt5);
			cout << endl;
		}

		int *Module_Base;
		int *base_cols;
		int rk;


		Y->create_module(h_alpha,
			Module_Base, base_cols, rk,
			verbose_level);

		if (f_v) {
			cout << "Module_Basis=" << endl;
			Y->D->print_matrix(Module_Base, rk, Y->goi);
		}


		for (i = 0; i < rk; i++) {
			for (j = 0; j < Y->goi; j++) {
				Y->D->copy(Y->D->offset(Module_Base, i * Y->goi + j),
						Y->D->offset(Base, s * Y->goi + j), 0);
			}
			s++;
		}
		Len[cnt] = s - Fst[cnt];
		Fst[cnt + 1] = s;

		Y->create_representations(Module_Base, base_cols, rk, verbose_level);


		FREE_int(Module_Base);
		FREE_int(base_cols);


			// create the next partition in exponential notation:
		if (!Combi.partition_next(part, n)) {
			break;
		}
		cnt++;
	}

	if (f_v) {
		cout << "Basis of submodule=" << endl;
		Y->D->print_matrix(Base, s, Y->goi);
	}


	FREE_int(part);
	FREE_int(parts);
	FREE_int(Fst);
	FREE_int(Len);
	if (f_v) {
		cout << "before freeing Base" << endl;
	}
	FREE_int(Base);
	FREE_int(Base_inv);
	if (f_v) {
		cout << "before freeing Y" << endl;
	}
	FREE_OBJECT(Y);
	if (f_v) {
		cout << "before freeing elt1" << endl;
	}
	FREE_int(elt1);
	FREE_int(elt2);
	FREE_int(h_alpha);
	FREE_int(elt4);
	FREE_int(elt5);
	FREE_int(elt6);
	FREE_int(elt7);
	if (f_v) {
		cout << "algebra_global_with_action::young_symmetrizer done" << endl;
	}
}

void algebra_global_with_action::young_symmetrizer_sym_4(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::young_symmetrizer_sym_4" << endl;
	}
	young *Y;
	int n = 4;

	Y = NEW_OBJECT(young);

	Y->init(n, verbose_level);



	int *elt1, *elt2, *h_alpha, *elt4, *elt5, *elt6, *elt7;

	Y->group_ring_element_create(Y->A, Y->S, elt1);
	Y->group_ring_element_create(Y->A, Y->S, elt2);
	Y->group_ring_element_create(Y->A, Y->S, h_alpha);
	Y->group_ring_element_create(Y->A, Y->S, elt4);
	Y->group_ring_element_create(Y->A, Y->S, elt5);
	Y->group_ring_element_create(Y->A, Y->S, elt6);
	Y->group_ring_element_create(Y->A, Y->S, elt7);



	int *part;
	int *parts;

	int *Base;
	int *Base_inv;
	int *Fst;
	int *Len;
	int cnt, s, i, j;

	part = NEW_int(n);
	parts = NEW_int(n);
	Fst = NEW_int(Y->goi);
	Len = NEW_int(Y->goi);
	Base = NEW_int(Y->goi * Y->goi * Y->D->size_of_instance_in_int);
	Base_inv = NEW_int(Y->goi * Y->goi * Y->D->size_of_instance_in_int);
	s = 0;
	Fst[0] = 0;

		// create the first partition in exponential notation:
	//partition_first(part, n);
	cnt = 0;

	int Part[10][5] = {
		{4, -1, 0, 0, 0},
		{3, 1, -1, 0, 0},
		{3, 1, -1, 0, 0},
		{3, 1, -1, 0, 0},
		{2, 2, -1, 0, 0},
		{2, 2, -1, 0, 0},
		{2, 1, 1, -1, 0},
		{2, 1, 1, -1, 0},
		{2, 1, 1, -1, 0},
		{1, 1, 1, 1, -1},
			};
	int Tableau[10][4] = {
		{0,1,2,3},
		{0,1,2,3}, {0,1,3,2}, {0,2,3,1},
		{0,1,2,3}, {0,2,1,3},
		{0,1,2,3}, {0,2,1,3}, {0,3,1,2},
		{0,1,2,3}
		};

	for (cnt = 0; cnt < 10; cnt++) {
		int nb_parts;

		// turn the partition from exponential notation into the list of parts:
		// the large parts come first.
		nb_parts = 0;
		for (i = 0; i < 4; i++) {
			parts[nb_parts] = Part[cnt][i];
			if (parts[nb_parts] == -1) {
				break;
			}
			nb_parts++;
		}

		if (f_v) {
			cout << "partition ";
			Int_vec_print(cout, parts, nb_parts);
			cout << endl;
		}


			// Create the young symmetrizer based on the partition.
			// We do the very first tableau for this partition.

		Y->young_symmetrizer(parts, nb_parts,
				Tableau[cnt], elt1, elt2, h_alpha,
				verbose_level);


		if (f_v) {
			cout << "h_alpha =" << endl;
			Y->group_ring_element_print(Y->A, Y->S, h_alpha);
			cout << endl;
		}


		Y->group_ring_element_copy(Y->A, Y->S, h_alpha, elt4);
		Y->group_ring_element_mult(Y->A, Y->S, elt4, elt4, elt5);

		if (f_v) {
			cout << "h_alpha * h_alpha=" << endl;
			Y->group_ring_element_print(Y->A, Y->S, elt5);
			cout << endl;
		}

		int *Module_Base;
		int *base_cols;
		int rk;


		Y->create_module(h_alpha,
			Module_Base, base_cols, rk,
			verbose_level);

		if (f_v) {
			cout << "Module_Basis=" << endl;
			Y->D->print_matrix(Module_Base, rk, Y->goi);
		}


		for (i = 0; i < rk; i++) {
			for (j = 0; j < Y->goi; j++) {
				Y->D->copy(
						Y->D->offset(Module_Base, i * Y->goi + j),
						Y->D->offset(Base, s * Y->goi + j), 0);
			}
			s++;
		}
		Len[cnt] = s - Fst[cnt];
		Fst[cnt + 1] = s;

		Y->create_representations(Module_Base, base_cols, rk, verbose_level);


		FREE_int(Module_Base);
		FREE_int(base_cols);


	}

	if (f_v) {
		cout << "Basis of submodule=" << endl;
		//Y->D->print_matrix(Base, s, Y->goi);
		Y->D->print_matrix_for_maple(Base, s, Y->goi);
	}

	FREE_int(part);
	FREE_int(parts);
	FREE_int(Fst);
	FREE_int(Len);
	if (f_v) {
		cout << "before freeing Base" << endl;
	}
	FREE_int(Base);
	FREE_int(Base_inv);
	if (f_v) {
		cout << "before freeing Y" << endl;
	}
	FREE_OBJECT(Y);
	if (f_v) {
		cout << "before freeing elt1" << endl;
	}
	FREE_int(elt1);
	FREE_int(elt2);
	FREE_int(h_alpha);
	FREE_int(elt4);
	FREE_int(elt5);
	FREE_int(elt6);
	FREE_int(elt7);
	if (f_v) {
		cout << "algebra_global_with_action::young_symmetrizer_sym_4 done" << endl;
	}
}










void algebra_global_with_action::do_character_table_symmetric_group(
		int deg, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::do_character_table_symmetric_group" << endl;
		cout << "deg=" << deg << endl;
	}

	apps_algebra::character_table_burnside *CTB;

	CTB = NEW_OBJECT(apps_algebra::character_table_burnside);

	if (f_v) {
		cout << "algebra_global_with_action::do_character_table_symmetric_group "
				"before CTB->do_it" << endl;
	}
	CTB->do_it(deg, verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::do_character_table_symmetric_group "
				"after CTB->do_it" << endl;
	}

	FREE_OBJECT(CTB);

	if (f_v) {
		cout << "algebra_global_with_action::do_character_table_symmetric_group done" << endl;
	}
}

void algebra_global_with_action::group_of_automorphisms_by_images_of_generators(
		data_structures_groups::vector_ge *Elements_ge,
		int *Images, int m, int n,
		groups::any_group *AG,
		std::string &label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators" << endl;
	}

	int *Perms;
	long int go;



	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"before automorphism_by_generator_images" << endl;
	}

	automorphism_by_generator_images(
			label,
			AG->A,
			AG->Subgroup_gens,
			AG->Subgroup_sims,
			Elements_ge,
			Images, m, n,
			Perms, go,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"after automorphism_by_generator_images" << endl;
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"we found " << m << " permutations of degree " << go << endl;
	}


	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"before AG->automorphism_by_generator_images_save" << endl;
	}

	AG->automorphism_by_generator_images_save(
			Images, m, n,
			Perms, go,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"after AG->automorphism_by_generator_images_save" << endl;
	}



	actions::action *A_perm;

	A_perm = NEW_OBJECT(actions::action);


	algebra::ring_theory::longinteger_object target_go;
	int f_target_go = true;

	target_go.create(m);


	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"before A_perm->Known_groups->init_permutation_group" << endl;
	}
	A_perm->Known_groups->init_permutation_group(
			go /* degree */, false /* f_no_base */,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"after A_perm->Known_groups->init_permutation_group" << endl;
	}

	data_structures_groups::vector_ge *gens;

	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"before gens->init_from_data" << endl;
	}
	gens->init_from_data(
			A_perm, Perms,
			m /*nb_elements*/, A_perm->make_element_size,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"after gens->init_from_data" << endl;
	}



#if 0
	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"before A_perm->Known_groups->init_permutation_group_from_generators" << endl;
	}
	A_perm->Known_groups->init_permutation_group_from_generators(
			go /* degree */,
		true /* f_target_go */, target_go,
		m /* nb_gens */, Perms,
		0 /* given_base_length */, NULL /* long int *given_base */,
		false /* f_given_base */,
		verbose_level - 2);
	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"after A_perm->Known_groups->init_permutation_group_from_generators" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"create_automorphism_group_of_incidence_structure: created action ";
		A_perm->print_info();
		cout << endl;
	}
#endif

	groups::sims *Sims;

	{
		groups::schreier_sims *ss;

		ss = NEW_OBJECT(groups::schreier_sims);

		if (f_v) {
			cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
					"before ss->init" << endl;
		}
		ss->init(A_perm, verbose_level - 1);
		if (f_v) {
			cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
					"after ss->init" << endl;
		}

		//ss->interested_in_kernel(A_subaction, verbose_level - 1);

		if (f_target_go) {
			if (f_v) {
				cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
						"before ss->init_target_group_order" << endl;
			}
			ss->init_target_group_order(target_go, verbose_level - 1);
			if (f_v) {
				cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
						"after ss->init_target_group_order" << endl;
			}
		}

		if (f_v) {
			cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
					"before ss->init_generators" << endl;
		}
		ss->init_generators(gens, verbose_level - 2);
		if (f_v) {
			cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
					"after ss->init_generators" << endl;
		}

		if (f_v) {
			cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
					"before ss->create_group" << endl;
		}
		ss->create_group(verbose_level - 2);
		if (f_v) {
			cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
					"after ss->create_group" << endl;
		}
		Sims = ss->G;
		ss->G = NULL;
		//*this = *ss->G;

		//ss->G->null();

		//cout << "create_sims_from_generators_randomized
		// before FREE_OBJECT ss" << endl;
		FREE_OBJECT(ss);
	}


	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"transversal lengths:" << endl;
		Sims->print_transversal_lengths();
	}

	groups::strong_generators *Strong_gens;

	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->init_from_sims(Sims, verbose_level - 1);

	if (f_v) {
		cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
				"Strong_gens:" << endl;
		Strong_gens->print_generators_for_make_element(cout);

		std::string fname;


		fname = label + ".gap";

		{
			ofstream ost(fname);

			Strong_gens->print_generators_gap(
					ost, verbose_level);
		}
		other::orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "algebra_global_with_action::group_of_automorphisms_by_images_of_generators "
					"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}
	}

}

void algebra_global_with_action::automorphism_by_generator_images(
		std::string &label,
		actions::action *A,
		groups::strong_generators *Subgroup_gens,
		groups::sims *Subgroup_sims,
		data_structures_groups::vector_ge *Elements_ge,
		int *Images, int m, int n,
		int *&Perms, long int &go,
		int verbose_level)
// An automorphism of a group is determined by the images of the generators.
// Here, we assume that we have a certain set of standard generators, and that
// the images of these generators are known.
// Using the right regular representation and a Schreier tree,
// we can then compute the automorphisms associated to the Images.
// Any automorphism is computed as a permutation of the elements
// in the ordering defined by the sims object Subgroup_sims
// The images in Images[] and the generators
// in Subgroup_gens->gens must correspond.
// This means that n must equal Subgroup_gens->gens->len
//
// We use orbits_schreier::orbit_of_sets for the Schreier tree.
// We need Subgroup_sims to set up action by right multiplication
// output: Perms[m * go]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images" << endl;
	}

	go = Subgroup_sims->group_order_lint();
	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images go = " << go << endl;
	}


	actions::action *A_rm;
	// action by right multiplication

	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"before A->Induced_action->induced_action_by_right_multiplication" << endl;
	}
	A_rm = A->Induced_action->induced_action_by_right_multiplication(
			false /* f_basis */, NULL /* old_G */,
			Subgroup_sims /*Base_group*/, false /* f_ownership */,
			verbose_level - 2);

	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"after A->Induced_action->induced_action_by_right_multiplication" << endl;
	}


	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"generators:" << endl;
		Subgroup_gens->gens->print_quick(cout);
		cout << endl;
	}

	if (Subgroup_gens->gens->len != n) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"Subgroup_gens->gens->len != n" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"before Orb.init" << endl;
	}


	orbits_schreier::orbit_of_sets Orb;
	long int set[1];
	//orbiter_kernel_system::file_io Fio;

	set[0] = 0;

	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"before Orb.init" << endl;
	}
	Orb.init(A, A_rm,
			set, 1 /* sz */,
			Subgroup_gens->gens,
			verbose_level - 2);
	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"Found an orbit of size " << Orb.used_length << endl;
	}
	if (Orb.used_length != go) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"orbit length != go" << endl;
		exit(1);
	}


	string fname;

	fname = label + "_tree.layered_graph";

	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"before Orb.export_tree_as_layered_graph_to_file" << endl;
	}

	Orb.export_tree_as_layered_graph_to_file(
			fname,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images "
				"after Orb.export_tree_as_layered_graph_to_file" << endl;
	}



	int h;
	int *Elt;
	int *perm;

	Elt = NEW_int(A->elt_size_in_int);
	perm = NEW_int(go);


	Perms = NEW_int(m * go);


	for (h = 0; h < m; h++) {

		if (f_vv) {
			cout << "algebra_global_with_action::automorphism_by_generator_images "
					"h=" << h << " : ";
			Int_vec_print(cout, Images + h * n, n);
			cout << endl;
		}

		int verbose_level1;

		if (h == 0) {
			verbose_level1 = verbose_level;
		}
		else {
			verbose_level1 = 0;
		}
		create_permutation(
				A,
				Subgroup_gens,
				Subgroup_sims,
				&Orb,
				Elements_ge,
				Images + h * n, n, h,
				Elt,
				perm, go,
				verbose_level1);

		Int_vec_copy(perm, Perms + h * go, go);

	}


	FREE_int(Elt);
	FREE_int(perm);

	FREE_OBJECT(A_rm);



	if (f_v) {
		cout << "algebra_global_with_action::automorphism_by_generator_images done" << endl;
	}
}


void algebra_global_with_action::create_permutation(
		actions::action *A,
		groups::strong_generators *Subgroup_gens,
		groups::sims *Subgroup_sims,
		orbits_schreier::orbit_of_sets *Orb,
		data_structures_groups::vector_ge *Elements_ge,
		int *Images, int n, int h,
		int *Elt,
		int *perm, long int go,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "algebra_global_with_action::create_permutation" << endl;
	}

	if (f_vv) {
		cout << "algebra_global_with_action::create_permutation h=" << h << " : ";
		Int_vec_print(cout, Images, n);
		cout << endl;
	}

	int in, out;
	int i, a, b, c;
	long int new_set[1];
	int pos;

	int nb_gens;

	nb_gens = Subgroup_gens->gens->len;

	if (f_vv) {
		for (i = 0; i < n; i++) {
			cout << "generator image " << i << " : " << Images[i] << " is: ";
			A->Group_element->print(cout, Elements_ge->ith(Images[i]));
			cout << endl;
		}
	}

	for (in = 0; in < go; in++) {

		uint32_t hash;
		std::vector<int> path;

		new_set[0] = in;

		if (!Orb->find_set(
				new_set, pos, hash)) {
			cout << "algebra_global_with_action::create_permutation !find_set" << endl;
			exit(1);
		}
		Orb->get_path(
				path,
				pos);

		if (f_vvv) {
			cout << "algebra_global_with_action::create_permutation "
					"in=" << in << " pos=" << pos << " path=";
			Int_vec_stl_print(cout, path);
			cout << " of length " << path.size();
			cout << endl;
		}


		int *path1;
		int *word;

		path1 = NEW_int(path.size());
		word = NEW_int(path.size());

		for (i = 0; i < path.size(); i++) {
			a = path[i];
			b = nb_gens - 1 - a; // reverse ordering because the Coxeter generators are listed in reverse
			path1[i] = b;
			c = Images[b];

			word[i] = c;
		}

		if (f_vvv) {
			cout << "algebra_global_with_action::create_permutation in=" << in << " path=";
			Int_vec_stl_print(cout, path);
			cout << " -> ";
			Int_vec_print(cout, path1, path.size());
			cout << " -> ";
			Int_vec_print(cout, word, path.size());
			cout << endl;
		}


		A->Group_element->evaluate_word(
				Elt, word, path.size(),
				Elements_ge,
				verbose_level - 3);

		if (f_vvv) {
			cout << "The word evaluates to" << endl;
			A->Group_element->element_print_quick(Elt, cout);
			cout << endl;
			cout << "in latex:" << endl;
			A->Group_element->element_print_latex(Elt, cout);
			cout << endl;
		}

		algebra::ring_theory::longinteger_object rk_out;

		Subgroup_sims->element_rank(rk_out, Elt);

		out = rk_out.as_int();

		perm[in] = out;

		if (f_vvv) {
			cout << "algebra_global_with_action::create_permutation "
					"in=" << in << " -> " << out << endl;
		}


		FREE_int(path1);
		FREE_int(word);

	}

	combinatorics::other_combinatorics::combinatorics_domain Combi;


	int f_is_perm;

	f_is_perm = Combi.Permutations->is_permutation(
			perm, go);
	if (f_is_perm) {
		if (f_vv) {
			cout << "algebra_global_with_action::create_permutation "
					"h = " << h << " output is a permutation" << endl;
		}
	}
	else {
		cout << "algebra_global_with_action::create_permutation "
				"h = " << h << " output is not a permutation" << endl;
		exit(1);
	}

	if (f_vv) {
		cout << "algebra_global_with_action::create_permutation "
				"h = " << h << ", perm = ";
		Combi.Permutations->perm_print_list(
					cout, perm, go);
		cout << endl;
		cout << "in cycle notation: ";
		Combi.Permutations->perm_print(
				cout, perm, go);
		cout << endl;

		int *cycles;
		int nb_cycles;

		cycles = NEW_int(go);

		Combi.Permutations->perm_cycle_type(
				perm, go, cycles, nb_cycles);
		cout << "The cycle type is: ";

		other::data_structures::tally T;

		T.init(cycles,
				nb_cycles, false /* f_second */, 0 /* verbose_level*/);

		T.print(true /* f_backwards*/);

		FREE_int(cycles);
		cout << endl;

	}
	if (f_v) {
		cout << "algebra_global_with_action::create_permutation done" << endl;
	}

}





void algebra_global_with_action::print_action_on_surface(
		groups::any_group *Any_group,
		std::string &surface_label,
		std::string &label_of_elements,
		data_structures_groups::vector_ge *Elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::print_action_on_surface" << endl;
	}

	applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_create *SC;

	SC = Get_object_of_cubic_surface(surface_label);

	if (f_v) {
		cout << "algebra_global_with_action::print_action_on_surface "
				"before SC->SOG->print_action_on_surface" << endl;
	}
	SC->SOG->print_action_on_surface(
			label_of_elements,
			Elements,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::print_action_on_surface "
				"after SC->SOG->print_action_on_surface" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::print_action_on_surface done" << endl;
	}

}


void algebra_global_with_action::subgroup_lattice_identify_subgroup(
		groups::any_group *Any_group,
		std::string &group_label,
		int &go, int &layer_idx,
		int &orb_idx, int &group_idx,
		int verbose_level)
// could go to level 3
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::subgroup_lattice_identify_subgroup" << endl;
	}

	if (!Any_group->f_has_subgroup_lattice) {
		cout << "algebra_global_with_action::subgroup_lattice_identify_subgroup "
				"subgroup lattice is not available" << endl;
		exit(1);
	}

	groups::any_group *Subgroup;

	groups::strong_generators *Subgroup_strong_gens;

	Subgroup = Get_any_group(group_label);

	Subgroup_strong_gens = Subgroup->Subgroup_gens;

	if (f_v) {
		cout << "algebra_global_with_action::subgroup_lattice_identify_subgroup "
				"before Any_group->Subgroup_lattice->identify_subgroup" << endl;
	}
	Any_group->Subgroup_lattice->identify_subgroup(
			Subgroup_strong_gens,
			go, layer_idx, orb_idx, group_idx,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::subgroup_lattice_identify_subgroup "
				"after Any_group->Subgroup_lattice->identify_subgroup" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::identify_subgroup "
				"found subgroup of order " << go << " in layer " << layer_idx
				<< " in orbit " << orb_idx << " at position " << group_idx << endl;
	}


	if (f_v) {
		cout << "algebra_global_with_action::subgroup_lattice_identify_subgroup done" << endl;
	}
}

void algebra_global_with_action::create_flag_transitive_incidence_structure(
		groups::any_group *Any_group,
		groups::any_group *P,
		groups::any_group *Q,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure" << endl;
	}

	groups::strong_generators *G_gens;
	groups::sims *G_sims;

	G_gens = Any_group->Subgroup_gens;
	G_sims = Any_group->Subgroup_sims;

	long int group_order, go_G, go_P, go_Q;

	group_order = G_sims->group_order_lint();
	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"group_order=" << group_order << endl;
	}

	if (P->Subgroup_sims == NULL) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"P->Subgroup_sims == NULL" << endl;
		exit(1);
	}
	if (Q->Subgroup_sims == NULL) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"Q->Subgroup_sims == NULL" << endl;
		exit(1);
	}
	go_G = Any_group->Subgroup_sims->group_order_lint();
	go_P = P->Subgroup_sims->group_order_lint();
	go_Q = Q->Subgroup_sims->group_order_lint();


	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"go_G = " << go_G << endl;
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"go_P = " << go_P << endl;
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"go_Q = " << go_Q << endl;
	}

	other::data_structures::sorting Sorting;

	long int *Elements_P;
	long int *Elements_Q;
	long int i, rk;
	int *Elt;

	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure" << endl;
	}
	Elt = NEW_int(Any_group->A->elt_size_in_int);

	Elements_P = NEW_lint(go_P);
	for (i = 0; i < go_P; i++) {
		P->Subgroup_sims->element_unrank_lint(i, Elt);
		rk = G_sims->element_rank_lint(Elt);
		Elements_P[i] = rk;
	}
	Sorting.lint_vec_heapsort(Elements_P, go_P);

	Elements_Q = NEW_lint(go_Q);
	for (i = 0; i < go_Q; i++) {
		Q->Subgroup_sims->element_unrank_lint(i, Elt);
		rk = G_sims->element_rank_lint(Elt);
		Elements_Q[i] = rk;
	}
	Sorting.lint_vec_heapsort(Elements_Q, go_Q);


	actions::action *A_conj;

	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"before Any_group->A->create_induced_action_by_conjugation" << endl;
	}
	A_conj = Any_group->A->Induced_action->create_induced_action_by_conjugation(
			G_sims /*Base_group*/, false /* f_ownership */,
			false /* f_basis */, NULL /* old_G */,
			verbose_level - 2);
	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"after Any_group->A->create_induced_action_by_conjugation" << endl;
	}

	orbits_schreier::orbit_of_sets *Orbits_P;
	orbits_schreier::orbit_of_sets *Orbits_Q;

	Orbits_P = NEW_OBJECT(orbits_schreier::orbit_of_sets);
	Orbits_Q = NEW_OBJECT(orbits_schreier::orbit_of_sets);


	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"before Orbits_P->init" << endl;
	}
	Orbits_P->init(
			Any_group->A,
			A_conj,
			Elements_P, go_P,
			G_gens->gens,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"after Orbits_P->init" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"before Orbits_Q->init" << endl;
	}
	Orbits_Q->init(
			Any_group->A,
			A_conj,
			Elements_Q, go_Q,
			G_gens->gens,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"after Orbits_Q->init" << endl;
	}


	int sz1, sz2;

	sz1 = Orbits_P->used_length;
	sz2 = Orbits_Q->used_length;
	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"sz1 = " << sz1 << endl;
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"sz2 = " << sz2 << endl;
	}

	other::data_structures::set_of_sets *SoS;


	SoS = NEW_OBJECT(other::data_structures::set_of_sets);

	int *Intersection;
	int j;
	long int *v3;
	int nb_sets = sz1 * sz2;

	v3 = NEW_lint(group_order);


	Intersection = NEW_int(sz1 * sz2);
	Int_vec_zero(Intersection, sz1 * sz2);

	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"computing Intersection" << endl;
	}

	for (i = 0; i < sz1; i++) {
		for (j = 0; j < sz2; j++) {

			int len3;

			Sorting.lint_vec_intersect_sorted_vectors(
					Orbits_P->Sets[i], go_P,
					Orbits_Q->Sets[j], go_Q,
					v3, len3);

			Intersection[i * sz2 + j] = len3;

		}
	}

	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"Intersection = " << endl;
		Int_matrix_print(Intersection, sz1, sz2);
	}

	SoS->init_basic_with_Sz_in_int(
			go_G,
			nb_sets, Intersection,
			0 /* verbose_level */);

	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"computing SoS" << endl;
	}

	for (i = 0; i < sz1; i++) {
		for (j = 0; j < sz2; j++) {

			int len3;

			Sorting.lint_vec_intersect_sorted_vectors(
					Orbits_P->Sets[i], go_P,
					Orbits_Q->Sets[j], go_Q,
					v3, len3);

			Lint_vec_copy(v3, SoS->Sets[i * sz2 + j], len3);

		}
	}



	FREE_lint(v3);

	other::orbiter_kernel_system::file_io Fio;

	string fname;

	fname = Any_group->label + "_" + P->label + "_" + Q->label + "_intersection_size.csv";

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Intersection, sz1, sz2);
	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	fname = Any_group->label + "_" + P->label + "_" + Q->label + "_intersection_sets.csv";


	SoS->save_csv(
			fname,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure "
				"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}


	FREE_OBJECT(SoS);
	FREE_int(Intersection);

	FREE_OBJECT(Orbits_P);
	FREE_OBJECT(Orbits_Q);
	FREE_OBJECT(A_conj);
	FREE_int(Elt);
	FREE_lint(Elements_P);
	FREE_lint(Elements_Q);

	if (f_v) {
		cout << "algebra_global_with_action::create_flag_transitive_incidence_structure done" << endl;
	}
}


void algebra_global_with_action::find_overgroup(
		groups::any_group *AG,
		groups::any_group *Subgroup,
		int overgroup_order,
		classes_of_subgroups_expanded *&Classes_of_subgroups_expanded,
		std::vector<int> &Class_idx, std::vector<int> &Class_idx_subgroup_idx,
		int verbose_level)
// uses AG->Subgroup_sims to define the ranks of group elements
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::find_overgroup" << endl;
		cout << "algebra_global_with_action::find_overgroup "
				"overgroup_order = " << overgroup_order << endl;
	}


	groups::sims *Sims_G;

	if (AG->Subgroup_sims == NULL) {
		cout << "algebra_global_with_action::find_overgroup "
				"Subgroup_sims == NULL" << endl;
		exit(1);
	}

	Sims_G = AG->Subgroup_sims;


	interfaces::conjugacy_classes_of_subgroups *class_data;

	if (f_v) {
		cout << "algebra_global_with_action::find_overgroup "
				"before AG->get_subgroup_lattice" << endl;
	}
	AG->get_subgroup_lattice(
			Sims_G,
			class_data,
			verbose_level - 2);

	if (f_v) {
		cout << "algebra_global_with_action::find_overgroup "
				"after AG->get_subgroup_lattice" << endl;
	}



	if (f_v) {
		cout << "algebra_global_with_action::find_overgroup "
				"before get_classes_of_subgroups_expanded" << endl;
	}

	Classes_of_subgroups_expanded = get_classes_of_subgroups_expanded(
			class_data,
			Sims_G,
			AG,
			overgroup_order,
			verbose_level - 2);

	if (f_v) {
		cout << "algebra_global_with_action::find_overgroup "
				"after get_classes_of_subgroups_expanded" << endl;
	}

	groups::sims *Sims_P;
	long int *Elements_P;
	long int go_P;

	groups::strong_generators *subgroup_gens;

	subgroup_gens = Subgroup->Subgroup_gens;

	if (f_v) {
		cout << "algebra_global_with_action::find_overgroup "
				"before element_ranks_in_overgroup" << endl;
	}
	element_ranks_in_overgroup(
			Sims_G,
			subgroup_gens,
			Sims_P, Elements_P, go_P,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::find_overgroup "
				"after element_ranks_in_overgroup" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::find_overgroup "
				"before Classes_of_subgroups_expanded->find_overgroups" << endl;
	}
	Classes_of_subgroups_expanded->find_overgroups(
			Elements_P,
			go_P,
			overgroup_order,
			Class_idx, Class_idx_subgroup_idx,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::find_overgroup "
				"after Classes_of_subgroups_expanded->find_overgroups" << endl;
	}





	FREE_lint(Elements_P);
	FREE_OBJECT(Sims_P);

	if (f_v) {
		cout << "algebra_global_with_action::find_overgroup done" << endl;
	}
}

void algebra_global_with_action::identify_subgroups_from_file(
		groups::any_group *AG,
		std::string &fname,
		std::string &col_label,
		int expand_go,
		int verbose_level)
// uses AG->Subgroup_sims to define the ranks of group elements
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::identify_subgroups_from_file" << endl;
		cout << "algebra_global_with_action::identify_subgroups_from_file "
				"expand_go = " << expand_go << endl;
	}

	groups::sims *Sims;
	interfaces::conjugacy_classes_of_subgroups *class_data;

	if (AG->Subgroup_sims == NULL) {
		cout << "algebra_global_with_action::identify_subgroups_from_file Subgroup_sims == NULL" << endl;
		exit(1);
	}

	Sims = AG->Subgroup_sims;


	if (f_v) {
		cout << "algebra_global_with_action::identify_subgroups_from_file "
				"before AG->get_subgroup_lattice" << endl;
	}

	AG->get_subgroup_lattice(
			Sims,
			class_data,
			verbose_level - 2);

	if (f_v) {
		cout << "algebra_global_with_action::identify_subgroups_from_file "
				"after AG->get_subgroup_lattice" << endl;
	}



	if (f_v) {
		cout << "algebra_global_with_action::identify_subgroups_from_file "
				"before identify_groups_from_csv_file" << endl;
	}
	identify_groups_from_csv_file(
			class_data,
			Sims /* override_sims */,
			AG,
			expand_go,
			fname,
			col_label,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::identify_subgroups_from_file "
				"after identify_groups_from_csv_file" << endl;
	}


	if (f_v) {
		cout << "algebra_global_with_action::identify_subgroups_from_file done" << endl;
	}
}

void algebra_global_with_action::all_elements_by_class(
		groups::sims *Sims,
		groups::any_group *Any_group,
		int class_order,
		int class_id,
		data_structures_groups::vector_ge *&vec,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::all_elements_by_class, "
				"class_order = " << class_order << " class_id = " << class_id << endl;
	}


	algebra::ring_theory::longinteger_object go;
	long int goi;

	Sims->group_order(go);
	goi = go.as_int();


	classes_of_elements_expanded *Classes_of_elements_expanded;
	data_structures_groups::vector_ge *Reps;

	if (f_v) {
		cout << "algebra_global_with_action::all_elements_by_class "
				"before get_classses_expanded" << endl;
	}
	get_classses_expanded(
			Sims,
			Any_group,
			goi,
			Classes_of_elements_expanded,
			Reps,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::all_elements_by_class "
				"after get_classses_expanded" << endl;
	}
#if 0
	interfaces::conjugacy_classes_and_normalizers *Classes;
	groups::sims *sims_G;
	groups::any_group *Any_group;
	int expand_by_go;
	std::string label;
	std::string label_latex;

	int *Idx;
	int nb_idx;

	actions::action *A_conj;

	orbit_of_elements **Orbit_of_elements; // [nb_idx]
#endif

	int nb_classes;
	int h, cnt;


	nb_classes = Classes_of_elements_expanded->Classes->nb_classes;

	cout << "nb_classes = " << nb_classes << endl;
	cnt = 0;
	for (h = 0; h < nb_classes; h++) {
		if (Classes_of_elements_expanded->Classes->class_order_of_element[h] == class_order) {
			cout << "class " << h << " consists of elements of order " << class_order << endl;
			if (cnt == class_id) {
				cout << "found class, h=" << h << endl;
				break;
			}
			cnt++;
		}
	}
	if (h == nb_classes) {
		cout << "did not find class class_order =" << class_order << " class_id=" << class_id << endl;
		exit(1);
	}
	cout << "found class, h=" << h << endl;

	int idx;

	for (idx = 0; idx < Classes_of_elements_expanded->nb_idx; idx++) {
		if (Classes_of_elements_expanded->Idx[idx] == h) {
			cout << "found class, idx=" << idx << endl;
			break;
		}
	}
	if (idx == Classes_of_elements_expanded->nb_idx) {
		cout << "did not find class" << endl;
		exit(1);
	}



	//int go_P;

	//go_P = Classes_of_elements_expanded->Orbit_of_elements[h]->go_P;

#if 0
	int idx;


	long int go_P;
	int *Element;
	long int Element_rk;
	long int *Elements_P;
	orbits_schreier::orbit_of_sets *Orbits_P;

	int orbit_length;
	long int *Table_of_elements; // sorted

#endif

	int class_size;
	long int *Table;

	class_size = Classes_of_elements_expanded->Orbit_of_elements[idx]->orbit_length;
	Table = Classes_of_elements_expanded->Orbit_of_elements[idx]->Table_of_elements;


	cout << "class_size = " << class_size << endl;
	cout << "Table_of_elements = ";
	Lint_vec_print_fully(cout, Table, class_size);
	cout << endl;



	vec = NEW_OBJECT(data_structures_groups::vector_ge);
	vec->init(Sims->A, 0 /*verbose_level*/);
	vec->allocate(class_size, verbose_level);

	int i;

	for (i = 0; i < class_size; i++) {
		Sims->element_unrank_lint(Table[i], vec->ith(i));
	}


	if (f_v) {
		cout << "algebra_global_with_action::all_elements_by_class done" << endl;
	}
}




classes_of_subgroups_expanded *algebra_global_with_action::get_classes_of_subgroups_expanded(
		interfaces::conjugacy_classes_of_subgroups *Classes,
		groups::sims *sims_G,
		groups::any_group *Any_group,
		int expand_by_go,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::get_classes_of_subgroups_expanded" << endl;
		cout << "algebra_global_with_action::get_classes_of_subgroups_expanded expand_by_go = " << expand_by_go << endl;
	}


	classes_of_subgroups_expanded *Classes_of_subgroups_expanded;

	Classes_of_subgroups_expanded = NEW_OBJECT(classes_of_subgroups_expanded);

	if (f_v) {
		cout << "algebra_global_with_action::get_classes_of_subgroups_expanded "
				"before Classes_of_subgroups_expanded->init" << endl;
	}
	Classes_of_subgroups_expanded->init(
			Classes,
			sims_G,
			Any_group,
			expand_by_go,
			Any_group->label,
			Any_group->label_tex,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::get_classes_of_subgroups_expanded "
				"after Classes_of_subgroups_expanded->init" << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::get_classes_of_subgroups_expanded "
				"done" << endl;
	}
	return Classes_of_subgroups_expanded;
}


void algebra_global_with_action::identify_groups_from_csv_file(
		interfaces::conjugacy_classes_of_subgroups *Classes,
		groups::sims *sims_G,
		groups::any_group *Any_group,
		int expand_by_go,
		std::string &fname,
		std::string &col_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::identify_groups_from_csv_file" << endl;
		cout << "algebra_global_with_action::identify_groups_from_csv_file expand_by_go = " << expand_by_go << endl;
		cout << "algebra_global_with_action::identify_groups_from_csv_file fname = " << fname << endl;
	}


	classes_of_subgroups_expanded *Classes_of_subgroups_expanded;

	if (f_v) {
		cout << "algebra_global_with_action::identify_groups_from_csv_file "
				"before get_classes_of_subgroups_expanded" << endl;
	}

	Classes_of_subgroups_expanded = get_classes_of_subgroups_expanded(
			Classes,
			sims_G,
			Any_group,
			expand_by_go,
			verbose_level - 2);

	if (f_v) {
		cout << "algebra_global_with_action::identify_groups_from_csv_file "
				"after get_classes_of_subgroups_expanded" << endl;
	}

#if 0
	Classes_of_subgroups_expanded = NEW_OBJECT(classes_of_subgroups_expanded);

	if (f_v) {
		cout << "algebra_global_with_action::identify_groups_from_csv_file "
				"before Classes_of_subgroups_expanded->init" << endl;
	}
	Classes_of_subgroups_expanded->init(
			Classes,
			sims_G,
			Any_group,
			expand_by_go,
			Any_group->label,
			Any_group->label_tex,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::identify_groups_from_csv_file "
				"after Classes_of_subgroups_expanded->init" << endl;
	}
#endif


	if (f_v) {
		cout << "algebra_global_with_action::identify_groups_from_csv_file "
				"before Classes_of_subgroups_expanded->report" << endl;
	}
	Classes_of_subgroups_expanded->report(
			Any_group->label,
			Any_group->label_tex,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::identify_groups_from_csv_file "
				"after Classes_of_subgroups_expanded->report" << endl;
	}


	other::orbiter_kernel_system::file_io Fio;
	other::data_structures::set_of_sets *SoS;

	//SoS = NEW_OBJECT(other::data_structures::set_of_sets);


	//int underlying_set_size = 0;

	if (f_v) {
		cout << "algebra_global_with_action::identify_groups_from_csv_file "
				"before Fio.Csv_file_support->read_column_as_set_of_sets" << endl;
	}
#if 0
	SoS->init_from_csv_file(
			underlying_set_size,
			fname,
			verbose_level);
#endif
	Fio.Csv_file_support->read_column_as_set_of_sets(
			fname, col_label,
			SoS,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::identify_groups_from_csv_file "
				"after Fio.Csv_file_support->read_column_as_set_of_sets" << endl;
	}

	other::data_structures::sorting Sorting;


	std::string *Table;
	int nb_rows, nb_cols;

	nb_rows = SoS->nb_sets;
	nb_cols = 6;

	Table = new std::string [nb_rows * nb_cols];

	int i, h, idx;
	int nb_sets = 0;
	int nb_found = 0;

	for (i = 0; i < SoS->nb_sets; i++) {

		if (f_v) {
			cout << "algebra_global_with_action::identify_groups_from_csv_file "
				"identifying set " << i << " / " << SoS->nb_sets << endl;
		}

		if (SoS->Set_size[i] != expand_by_go) {
			if (f_v) {
				cout << "algebra_global_with_action::identify_groups_from_csv_file "
					"skipping" << endl;
			}
			continue;
		}


		if (f_v) {
			cout << "algebra_global_with_action::identify_groups_from_csv_file "
					"The set " << i << " / " << SoS->nb_sets << " with number " << nb_sets << " is : ";
			Lint_vec_print(cout, SoS->Sets[i], SoS->Set_size[i]);
			cout << endl;
		}



		int j;
		int f_found;
		int found_at_class_h, found_at_class_idx, found_at_group_idx;

		f_found = false;
		found_at_class_h = -1;
		found_at_class_idx = -1;
		found_at_group_idx = -1;

		for (h = 0; h < Classes_of_subgroups_expanded->nb_idx; h++) {

			idx = Classes_of_subgroups_expanded->Idx[h];

			long int *Elements;
			int class_sz;


			//Elements = Classes_of_subgroups_expanded->Orbit_of_subgroups[h]->Elements_P;

			class_sz = Classes_of_subgroups_expanded->Orbit_of_subgroups[h]->Orbits_P->used_length;
			if (f_v) {
				cout << "algebra_global_with_action::identify_groups_from_csv_file "
						"The set " << i << " / " << SoS->nb_sets
						<< " with number " << nb_sets << " is : ";
				Lint_vec_print(cout, SoS->Sets[i], SoS->Set_size[i]);
				cout << " checking class " << h << " / " << Classes_of_subgroups_expanded->nb_idx << " = " << idx
						<< " of size " << class_sz << endl;
			}

			for (j = 0; j < class_sz; j++) {

				Elements = Classes_of_subgroups_expanded->Orbit_of_subgroups[h]->Orbits_P->Sets[j];

				//int orbit_of_sets::find_set(
				//		long int *new_set, int &pos, uint32_t &hash)

				if (false) {
					cout << "algebra_global_with_action::identify_groups_from_csv_file "
							"The conjugate group " << j << " / " << class_sz << " is : ";
					Lint_vec_print(cout, Elements, expand_by_go);
					cout << endl;
				}

				if (Sorting.compare_sets_lint(
						Elements, SoS->Sets[i], expand_by_go, expand_by_go) == 0) {
					if (f_v) {
						cout << "algebra_global_with_action::identify_groups_from_csv_file "
								"The set " << i << " / " << SoS->nb_sets << " with number " << nb_sets << " is : ";
						Lint_vec_print(cout, SoS->Sets[i], SoS->Set_size[i]);
						cout << " found in class " << h << " / " << Classes_of_subgroups_expanded->nb_idx << " = " << idx
								<< " at position " << j << endl;
					}
					f_found = true;
					found_at_class_h = h;
					found_at_class_idx = idx;
					found_at_group_idx = j;
					break;
				}
			}
			if (f_found) {
				// no need to look at any further classes
				break;
			}

		}

		if (f_found) {
			cout << "algebra_global_with_action::identify_groups_from_csv_file "
					"The set " << i << " / " << SoS->nb_sets << " with number " << nb_sets << " is : ";
			Lint_vec_print(cout, SoS->Sets[i], SoS->Set_size[i]);
			cout << " has been found at class " << found_at_class_idx
					<< " and group index " << found_at_group_idx << endl;
			nb_found++;
		}
		else {
			cout << "algebra_global_with_action::identify_groups_from_csv_file "
					"The set " << i << " / " << SoS->nb_sets << " with number " << nb_sets << " is : ";
			Lint_vec_print(cout, SoS->Sets[i], SoS->Set_size[i]);
			cout << " has *not* been found" << endl;
		}

		Table[nb_sets * nb_cols + 0] = std::to_string(i);
		Table[nb_sets * nb_cols + 1] = "\"" + Lint_vec_stringify(SoS->Sets[i], SoS->Set_size[i]) + "\"";
		Table[nb_sets * nb_cols + 2] = std::to_string(f_found);
		Table[nb_sets * nb_cols + 3] = std::to_string(found_at_class_h);
		Table[nb_sets * nb_cols + 4] = std::to_string(found_at_class_idx);
		Table[nb_sets * nb_cols + 5] = std::to_string(found_at_group_idx);


		nb_sets++;
	}

	if (f_v) {
		cout << "algebra_global_with_action::identify_groups_from_csv_file nb_sets in file = " << SoS->nb_sets << endl;
		cout << "algebra_global_with_action::identify_groups_from_csv_file nb_sets of size " << expand_by_go << " = " << nb_sets << endl;
		cout << "algebra_global_with_action::identify_groups_from_csv_file of these, nb_found = " << nb_found << endl;
	}

	std::string fname_identify;

	fname_identify = Any_group->label + "_identify_order_" + std::to_string(expand_by_go) + ".csv";

	std::string *Col_headings;

	Col_headings = new string [nb_cols];

	Col_headings[0] = "incidence";
	Col_headings[1] = "subgroup";
	Col_headings[2] = "f_found";
	Col_headings[3] = "class_local";
	Col_headings[4] = "class_global";
	Col_headings[5] = "group_idx";


	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname_identify,
			nb_sets, nb_cols, Table,
			Col_headings,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::identify_groups_from_csv_file "
				"written file " << fname_identify << " of size "
				<< Fio.file_size(fname_identify) << endl;
	}

	delete [] Col_headings;
	delete [] Table;


	FREE_OBJECT(SoS);


	if (f_v) {
		cout << "algebra_global_with_action::identify_groups_from_csv_file done" << endl;
	}
}

void algebra_global_with_action::get_classses_expanded(
		groups::sims *Sims,
		groups::any_group *Any_group,
		int expand_by_go,
		classes_of_elements_expanded *&Classes_of_elements_expanded,
		data_structures_groups::vector_ge *&Reps,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::get_classses_expanded" << endl;
	}

	interfaces::conjugacy_classes_and_normalizers *class_data;


	if (f_v) {
		cout << "algebra_global_with_action::get_classses_expanded "
				"before AG->get_conjugacy_classes_of_elements" << endl;
	}
	Any_group->get_conjugacy_classes_of_elements(
			Sims, class_data, verbose_level - 2);
	if (f_v) {
		cout << "algebra_global_with_action::get_classses_expanded "
				"after AG->get_conjugacy_classes_of_elements" << endl;
	}


	if (f_v) {
		cout << "algebra_global_with_action::get_classses_expanded "
				"before class_data->get_representatives" << endl;
	}
	class_data->get_representatives(
			Sims,
			Reps,
			verbose_level - 2);
	if (f_v) {
		cout << "algebra_global_with_action::get_classses_expanded "
				"after class_data->get_representatives" << endl;
	}



	//classes_of_elements_expanded *Classes_of_elements_expanded;

	Classes_of_elements_expanded = NEW_OBJECT(classes_of_elements_expanded);


	if (f_v) {
		cout << "algebra_global_with_action::get_classses_expanded "
				"before Classes_of_elements_expanded->init" << endl;
	}
	Classes_of_elements_expanded->init(
			class_data,
			Sims,
			Any_group,
			expand_by_go,
			Any_group->label,
			Any_group->label_tex,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::get_classses_expanded "
				"after Classes_of_elements_expanded->init" << endl;
	}


	//FREE_OBJECT(class_data);
	//FREE_OBJECT(Classes_of_elements_expanded);

	if (f_v) {
		cout << "algebra_global_with_action::get_classses_expanded done" << endl;
	}

}

void algebra_global_with_action::split_by_classes(
		groups::sims *Sims,
		groups::any_group *Any_group,
		int expand_by_go,
		std::string &fname,
		std::string &col_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::split_by_classes" << endl;
	}

	classes_of_elements_expanded *Classes_of_elements_expanded;
	data_structures_groups::vector_ge *Reps;



	if (f_v) {
		cout << "algebra_global_with_action::split_by_classes "
				"before get_classses_expanded" << endl;
	}
	get_classses_expanded(
			Sims,
			Any_group,
			expand_by_go,
			Classes_of_elements_expanded,
			Reps,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::split_by_classes "
				"after get_classses_expanded" << endl;
	}



	int nb_classes;

	nb_classes = Classes_of_elements_expanded->Classes->nb_classes;

	other::orbiter_kernel_system::file_io Fio;
	other::data_structures::set_of_sets *SoS;

	if (f_v) {
		cout << "algebra_global_with_action::split_by_classes "
				"before read_column_as_set_of_sets" << endl;
	}
	Fio.Csv_file_support->read_column_as_set_of_sets(
			fname, col_label,
			SoS,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::split_by_classes "
				"after read_column_as_set_of_sets" << endl;
	}

	other::data_structures::sorting Sorting;


	std::string *Table;
	int nb_rows, nb_cols;

	nb_rows = SoS->nb_sets;
	nb_cols = 3 + nb_classes;

	Table = new std::string [nb_rows * nb_cols];

	int i;

	int *First;

	First = NEW_int(nb_classes);


	// we assume that all classes have been expanded:
	int j;
	for (j = 0; j < nb_classes; j++) {
		if (j == 0) {
			First[j] = 0;
		}
		else {
			First[j] = First[j - 1] + Classes_of_elements_expanded->Orbit_of_elements[j - 1]->orbit_length;
		}
	}


	std::string *Table2;
	int nb_rows2 = nb_rows + 1;
	int nb_cols2 = nb_cols;


	Table2 = new std::string [nb_rows2 * nb_cols2];

	for (j = 0; j < nb_classes; j++) {

		long int class_size;


		class_size = Classes_of_elements_expanded->Orbit_of_elements[j]->orbit_length;

		Table2[0 * nb_cols2 + 3 + j] = std::to_string(class_size);

	}


	for (i = 0; i < SoS->nb_sets; i++) {


		if (f_v) {
			cout << "algebra_global_with_action::split_by_classes "
					"The set " << i << " / " << SoS->nb_sets << " is : ";
			Lint_vec_print(cout, SoS->Sets[i], SoS->Set_size[i]);
			cout << endl;
		}

		long int *vec_combined;

		vec_combined = NEW_lint(SoS->Set_size[i]);

		std::vector<std::vector<long int>> V;

		// prepare V to be a vector of empty vectors:
		for (j = 0; j < nb_classes; j++) {
			std::vector<long int> v;

			V.push_back(v);
		}

		int h, idx;

		for (h = 0; h < SoS->Set_size[i]; h++) {

			long int a;
			int f_found;

			a = SoS->Sets[i][h];
			f_found = false;

			for (j = 0; j < nb_classes; j++) {
				if (Sorting.lint_vec_search(
						Classes_of_elements_expanded->Orbit_of_elements[j]->Table_of_elements,
						Classes_of_elements_expanded->Orbit_of_elements[j]->orbit_length,
						a, idx, 0 /*verbose_level*/)) {
					//V[j].push_back(idx);
					V[j].push_back(a);
					f_found = true;
					break;
				}
			}
			if (!f_found) {
				cout << "algebra_global_with_action::split_by_classes did not find element " << endl;
			}

		}

		Table[i * nb_cols + 0] = std::to_string(i);
		Table[i * nb_cols + 1] = "\"" + Lint_vec_stringify(SoS->Sets[i], SoS->Set_size[i]) + "\"";


		Table2[(i + 1) * nb_cols2 + 0] = std::to_string(i);
		Table2[(i + 1) * nb_cols2 + 1] = std::to_string(SoS->Set_size[i]);


#if 0
		int cur;

		cur = 0;
		for (j = 0; j < nb_classes; j++) {
			long int *vec;
			int len;

			len = V[j].size();
			vec = NEW_lint(len);
			for (h = 0; h < len; h++) {
				vec[h] = First[j] + V[j][h];
			}
			Lint_vec_copy(vec, vec_combined + cur, len);
			cur += len;
			FREE_lint(vec);
		}
		Table[i * nb_cols + 2] = "\"" + Lint_vec_stringify(vec_combined, SoS->Set_size[i]) + "\"";
#endif
		for (j = 0; j < nb_classes; j++) {

			long int *vec;
			int len;

			len = V[j].size();
			vec = NEW_lint(len);
			for (h = 0; h < len; h++) {
				vec[h] = V[j][h];
			}



			Table[i * nb_cols + 3 + j] = "\"" + Lint_vec_stringify(vec, len) + "\"";

			Table2[(i + 1) * nb_cols2 + 3 + j] = std::to_string(len);

			FREE_lint(vec);

		}

		FREE_lint(vec_combined);


	}




	FREE_int(First);


	std::string fname_identify;
	std::string fname_identify_size;

	fname_identify = Any_group->label + "_split_by_classes_" + std::to_string(expand_by_go) + ".csv";
	fname_identify_size = Any_group->label + "_split_by_classes_" + std::to_string(expand_by_go) + "_size.csv";

	std::string *Col_headings;

	Col_headings = new string [nb_cols];

	Col_headings[0] = "line";
	Col_headings[1] = "set";
	Col_headings[2] = "set_out";

	for (j = 0; j < nb_classes; j++) {
		Col_headings[3 + j] = "C" + std::to_string(j);
	}

	if (f_v) {
		cout << "algebra_global_with_action::split_by_classes "
				"nb_rows = " << nb_rows << endl;
		cout << "algebra_global_with_action::split_by_classes "
				"nb_cols = " << nb_cols << endl;
	}

	if (f_v) {
		cout << "algebra_global_with_action::split_by_classes "
				"writing file " << fname_identify << endl;
	}

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname_identify,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::split_by_classes "
				"written file " << fname_identify << " of size "
				<< Fio.file_size(fname_identify) << endl;
	}




	if (f_v) {
		cout << "algebra_global_with_action::split_by_classes "
				"writing file " << fname_identify_size << endl;
	}

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname_identify_size,
			nb_rows2, nb_cols2, Table2,
			Col_headings,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::split_by_classes "
				"written file " << fname_identify_size << " of size "
				<< Fio.file_size(fname_identify_size) << endl;
	}




	delete [] Col_headings;
	delete [] Table;
	delete [] Table2;


	FREE_OBJECT(SoS);
	FREE_OBJECT(Classes_of_elements_expanded);
	FREE_OBJECT(Reps);


}


void algebra_global_with_action::identify_elements_by_classes(
		groups::sims *Sims,
		groups::any_group *Any_group_H,
		groups::any_group *Any_group_G,
		int expand_by_go,
		std::string &fname, std::string &col_label,
		int *&Class_index,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::identify_elements_by_classes" << endl;
	}


	classes_of_elements_expanded *Classes_of_elements_expanded;
	data_structures_groups::vector_ge *Reps;


	if (f_v) {
		cout << "algebra_global_with_action::identify_elements_by_classes "
				"before get_classses_expanded" << endl;
	}
	get_classses_expanded(
			Sims,
			Any_group_H,
			expand_by_go,
			Classes_of_elements_expanded,
			Reps,
			verbose_level);
	if (f_v) {
		cout << "algebra_global_with_action::identify_elements_by_classes "
				"after get_classses_expanded" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	other::data_structures::set_of_sets *SoS;

	if (f_v) {
		cout << "algebra_global_with_action::identify_elements_by_classes "
				"before read_column_as_set_of_sets" << endl;
	}
	Fio.Csv_file_support->read_column_as_set_of_sets(
			fname, col_label,
			SoS,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::identify_elements_by_classes "
				"after read_column_as_set_of_sets" << endl;
	}

	int nb_elements;

	nb_elements = SoS->nb_sets;

	if (f_v) {
		cout << "algebra_global_with_action::identify_elements_by_classes "
				"nb_elements = " << nb_elements << endl;
	}

	other::data_structures::sorting Sorting;

	Class_index = NEW_int(nb_elements);

	int nb_classes;

	nb_classes = Classes_of_elements_expanded->Classes->nb_classes;
	if (f_v) {
		cout << "algebra_global_with_action::identify_elements_by_classes "
				"nb_classes = " << nb_classes << endl;
		cout << "algebra_global_with_action::identify_elements_by_classes "
				"nb_expanded_classes = " << Classes_of_elements_expanded->nb_idx << endl;
	}

	int *Elt;
	int *data;
	int i, j, idx;
	long int a;
	int f_found;
	int f_is_member;

	Elt = NEW_int(Any_group_G->A->elt_size_in_int);
	data = NEW_int(Any_group_G->A->make_element_size);


	for (i = 0; i < nb_elements; i++) {

		Lint_vec_copy_to_int(SoS->Sets[i], data, Any_group_G->A->make_element_size);

		Any_group_G->A->Group_element->make_element(Elt, data, 0 /*verbose_level */);


		algebra::ring_theory::longinteger_object rk;

		if (f_v) {
			cout << "algebra_global_with_action::identify_elements_by_classes "
					"before Sims->test_membership_and_rank_element" << endl;
		}
		f_is_member = Sims->test_membership_and_rank_element(
				rk, Elt, 0 /*verbose_level */);
		if (f_v) {
			cout << "algebra_global_with_action::identify_elements_by_classes "
					"after Sims->test_membership_and_rank_element" << endl;
		}


		if (f_is_member) {

			a = rk.as_lint();

			f_found = false;

			for (j = 0; j < Classes_of_elements_expanded->nb_idx; j++) {
				if (Sorting.lint_vec_search(
						Classes_of_elements_expanded->Orbit_of_elements[j]->Table_of_elements,
						Classes_of_elements_expanded->Orbit_of_elements[j]->orbit_length,
						a, idx, 0 /*verbose_level*/)) {
					Class_index[i] = j;
					f_found = true;
					break;
				}
			}
			if (!f_found) {
				if (f_v) {
					cout << "algebra_global_with_action::identify_elements_by_classes "
							"did not find element " << endl;
				}
				Class_index[i] = -2;
				//exit(1);
			}
		}
		else {
			Class_index[i] = -1;
		}

	}

	//other::orbiter_kernel_system::file_io Fio;
	std::string new_col_label;

	new_col_label= "class_idx";

	string fname_out;

	Fio.Csv_file_support->append_column_of_int(
			fname, fname_out,
			Class_index, nb_elements,
			new_col_label,
			verbose_level);

	if (f_v) {
		cout << "algebra_global_with_action::identify_elements_by_classes written file "
				<< fname_out << " of size " << Fio.file_size(fname_out) << endl;
	}


	FREE_OBJECT(Classes_of_elements_expanded);
	FREE_OBJECT(Reps);
	FREE_OBJECT(SoS);
	FREE_int(Elt);
	FREE_int(data);

	if (f_v) {
		cout << "algebra_global_with_action::identify_elements_by_classes done" << endl;
	}
}

void algebra_global_with_action::element_ranks_in_overgroup(
		groups::sims *Sims_G,
		groups::strong_generators *subgroup_gens,
		groups::sims *&Sims_P, long int *&Elements_P, long int &go_P,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebra_global_with_action::element_ranks_in_overgroup" << endl;
	}

	int *Elt;

	Elt = NEW_int(Sims_G->A->elt_size_in_int);

	long int rk;

	Sims_P = subgroup_gens->create_sims(verbose_level);
	go_P = Sims_P->group_order_lint();

	Elements_P = NEW_lint(go_P);

	int i;
	for (i = 0; i < go_P; i++) {
		Sims_P->element_unrank_lint(i, Elt);
		rk = Sims_G->element_rank_lint(Elt);
		Elements_P[i] = rk;
	}

	other::data_structures::sorting Sorting;

	Sorting.lint_vec_heapsort(Elements_P, go_P);




	FREE_int(Elt);

	if (f_v) {
		cout << "algebra_global_with_action::element_ranks_in_overgroup done" << endl;
	}


}

}}}

