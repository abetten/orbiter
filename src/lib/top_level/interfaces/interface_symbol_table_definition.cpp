/*
 * interface_symbol_table_definition.cpp
 *
 *  Created on: Apr 12, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {

void interface_symbol_table::definition_of_finite_field(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_finite_field" << endl;
	}
	Finite_field_description->print();
	finite_field *F;

	F = NEW_OBJECT(finite_field);
	F->init(Finite_field_description, verbose_level);

	orbiter_symbol_table_entry Symb;
	Symb.init_finite_field(define_label, F, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_finite_field before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_finite_field done" << endl;
	}
}

void interface_symbol_table::definition_of_projective_space(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_projective_space" << endl;
	}
	finite_field *F;

	if (string_starts_with_a_number(Projective_space_with_action_description->input_q)) {
		int q;

		q = strtoi(Projective_space_with_action_description->input_q);
		if (f_v) {
			cout << "interface_symbol_table::definition_of_projective_space "
					"creating finite field of order " << q << endl;
		}
		F = NEW_OBJECT(finite_field);
		F->finite_field_init(q, 0);
	}
	else {
		if (f_v) {
			cout << "interface_symbol_table::definition_of_projective_space "
					"using existing finite field " << Projective_space_with_action_description->input_q << endl;
		}
		int idx;
		idx = Orbiter_top_level_session->find_symbol(Projective_space_with_action_description->input_q);
		F = (finite_field *) Orbiter_top_level_session->get_object(idx);
	}

	Projective_space_with_action_description->F = F;

	int f_semilinear;
	number_theory_domain NT;


	if (NT.is_prime(F->q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}

	projective_space_with_action *PA;

	PA = NEW_OBJECT(projective_space_with_action);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_projective_space before PA->init" << endl;
	}
	PA->init(Projective_space_with_action_description->F, Projective_space_with_action_description->n,
		f_semilinear,
		TRUE /*f_init_incidence_structure*/,
		0 /* verbose_level */);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_projective_space after PA->init" << endl;
	}

	orbiter_symbol_table_entry Symb;
	Symb.init_projective_space(define_label, PA, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_projective_space before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_projective_space done" << endl;
	}
}

void interface_symbol_table::definition_of_orthogonal_space(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_orthogonal_space" << endl;
	}
	finite_field *F;

	if (string_starts_with_a_number(Orthogonal_space_with_action_description->input_q)) {
		int q;

		q = strtoi(Orthogonal_space_with_action_description->input_q);
		if (f_v) {
			cout << "interface_symbol_table::definition_of_orthogonal_space "
					"creating finite field of order " << q << endl;
		}
		F = NEW_OBJECT(finite_field);
		F->finite_field_init(q, 0);
	}
	else {
		if (f_v) {
			cout << "interface_symbol_table::definition_of_orthogonal_space "
					"using existing finite field " << Orthogonal_space_with_action_description->input_q << endl;
		}
		int idx;
		idx = Orbiter_top_level_session->find_symbol(Orthogonal_space_with_action_description->input_q);
		F = (finite_field *) Orbiter_top_level_session->get_object(idx);
	}

	Orthogonal_space_with_action_description->F = F;

	int f_semilinear;
	number_theory_domain NT;


	if (NT.is_prime(F->q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}

	orthogonal_space_with_action *OA;

	OA = NEW_OBJECT(orthogonal_space_with_action);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_orthogonal_space before OA->init" << endl;
	}
	OA->init(Orthogonal_space_with_action_description,
		verbose_level - 2);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_orthogonal_space after OA->init" << endl;
	}

	orbiter_symbol_table_entry Symb;
	Symb.init_orthogonal_space(define_label, OA, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_orthogonal_space before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_orthogonal_space done" << endl;
	}
}

void interface_symbol_table::definition_of_linear_group(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_linear_group" << endl;
	}

	finite_field *F;

	if (string_starts_with_a_number(Linear_group_description->input_q)) {
		int q;

		q = strtoi(Linear_group_description->input_q);
		if (f_v) {
			cout << "interface_symbol_table::definition "
					"creating finite field of order " << q << endl;
		}
		F = NEW_OBJECT(finite_field);
		F->finite_field_init(q, 0);
	}
	else {
		if (f_v) {
			cout << "interface_symbol_table::definition "
					"using existing finite field " << Linear_group_description->input_q << endl;
		}
		int idx;
		idx = Orbiter_top_level_session->find_symbol(Linear_group_description->input_q);
		F = (finite_field *) Orbiter_top_level_session->get_object(idx);
	}



	Linear_group_description->F = F;
	//q = Descr->input_q;

	linear_group *LG;

	LG = NEW_OBJECT(linear_group);
	if (f_v) {
		cout << "interface_symbol_table::definition before LG->init, "
				"creating the group" << endl;
	}

	LG->linear_group_init(Linear_group_description, verbose_level - 5);

	orbiter_symbol_table_entry Symb;
	Symb.init_linear_group(define_label, LG, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_linear_group done" << endl;
	}
}

void interface_symbol_table::definition_of_formula(orbiter_top_level_session *Orbiter_top_level_session,
		formula *F,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_formula" << endl;
	}

	orbiter_symbol_table_entry Symb;
	Symb.init_formula(define_label, F, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_formula before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_formula done" << endl;
	}
}

void interface_symbol_table::definition_of_collection(orbiter_top_level_session *Orbiter_top_level_session,
		std::string &list_of_objects,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_collection" << endl;
	}

	orbiter_symbol_table_entry Symb;
	Symb.init_collection(define_label, list_of_objects, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_formula before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_collection done" << endl;
	}
}

void interface_symbol_table::definition_of_combinatorial_object(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_combinatorial_object" << endl;
	}

	combinatorial_object_create *COC;

	COC = NEW_OBJECT(combinatorial_object_create);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_combinatorial_object before COC->init" << endl;
	}
	COC->init(Combinatorial_object_description, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_combinatorial_object after COC->init" << endl;
	}



	if (f_v) {
		cout << "interface_symbol_table::definition_of_combinatorial_object we created a set of " << COC->nb_pts
				<< " points, called " << COC->fname << endl;

#if 0
		cout << "list of points:" << endl;

		cout << COC->nb_pts << endl;
		for (i = 0; i < COC->nb_pts; i++) {
			cout << COC->Pts[i] << " ";
			}
		cout << endl;
#endif
	}


	orbiter_symbol_table_entry Symb;
	Symb.init_combinatorial_object(define_label, COC, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_combinatorial_object before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



	if (f_v) {
		cout << "interface_symbol_table::definition_of_combinatorial_object done" << endl;
	}
}


void interface_symbol_table::definition_of_graph(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_graph" << endl;
	}

	create_graph *Gr;

	Gr = NEW_OBJECT(create_graph);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_graph before Gr->init" << endl;
	}
	Gr->init(Create_graph_description, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_graph after Gr->init" << endl;
	}
	if (f_v) {
		cout << "Gr->N=" << Gr->N << endl;
		cout << "Gr->label=" << Gr->label << endl;
		//cout << "Adj:" << endl;
		//int_matrix_print(Gr->Adj, Gr->N, Gr->N);
	}



	if (f_v) {
		cout << "interface_symbol_table::definition_of_graph we created a graph on " << Gr->N
				<< " points, called " << Gr->label << endl;

#if 0
		cout << "list of points:" << endl;

		cout << COC->nb_pts << endl;
		for (i = 0; i < COC->nb_pts; i++) {
			cout << COC->Pts[i] << " ";
			}
		cout << endl;
#endif
	}


	orbiter_symbol_table_entry Symb;
	Symb.init_graph(define_label, Gr, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_graph before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



	if (f_v) {
		cout << "interface_symbol_table::definition_of_graph done" << endl;
	}
}


void interface_symbol_table::definition_of_spread_table(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_spread_table" << endl;
	}

	if (f_v) {
		cout << "interface_symbol_table::definition_of_spread_table "
				"using existing PA " << spread_table_label_PA << endl;
	}
	int idx;
	projective_space_with_action *PA;

	idx = Orbiter_top_level_session->find_symbol(spread_table_label_PA);
	PA = (projective_space_with_action *) Orbiter_top_level_session->get_object(idx);




	packing_classify *P;

	if (f_v) {
		cout << "interface_symbol_table::definition_of_spread_table before P->spread_table_init" << endl;
	}

	P = NEW_OBJECT(packing_classify);

	P->spread_table_init(
			PA,
			dimension_of_spread_elements,
			TRUE /* f_select_spread */, spread_selection_text,
			spread_tables_prefix,
			verbose_level);


	if (f_v) {
		cout << "interface_symbol_table::definition_of_spread_table after do_spread_table_init" << endl;
	}




	orbiter_symbol_table_entry Symb;
	Symb.init_spread_table(define_label, P, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_spread_table before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



	if (f_v) {
		cout << "interface_symbol_table::definition_of_spread_table done" << endl;
	}
}


void interface_symbol_table::definition_of_packing_was(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_was" << endl;
	}

	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_was "
				"using existing spread table " << packing_was_label_spread_table << endl;
	}
	int idx;
	packing_classify *P;

	idx = Orbiter_top_level_session->find_symbol(packing_was_label_spread_table);
	P = (packing_classify *) Orbiter_top_level_session->get_object(idx);






	packing_was *PW;

	PW = NEW_OBJECT(packing_was);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_was before PW->init" << endl;
	}

	PW->init(packing_was_descr, P, verbose_level);

	if (f_v) {
		cout << "spread_table_activity::perform_activity after PW->init" << endl;
	}




	orbiter_symbol_table_entry Symb;
	Symb.init_packing_was(define_label, PW, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_was before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_was done" << endl;
	}
}



void interface_symbol_table::definition_of_packing_was_choose_fixed_points(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_was_choose_fixed_points" << endl;
	}

	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_was_choose_fixed_points "
				"using existing object " << packing_with_assumed_symmetry_label << endl;
	}
	int idx;
	packing_was *PW;

	idx = Orbiter_top_level_session->find_symbol(packing_with_assumed_symmetry_label);
	PW = (packing_was *) Orbiter_top_level_session->get_object(idx);


	packing_was_fixpoints *PWF;

	PWF = NEW_OBJECT(packing_was_fixpoints);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_was_choose_fixed_points before PWF->init" << endl;
	}

	PWF->init(PW,
			packing_with_assumed_symmetry_choose_fixed_points_clique_size,
			packing_with_assumed_symmetry_choose_fixed_points_control,
			verbose_level);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_was_choose_fixed_points after PWF->init" << endl;
	}

	if (packing_with_assumed_symmetry_choose_fixed_points_clique_size > 0) {
		PWF->compute_cliques_on_fixpoint_graph(
				packing_with_assumed_symmetry_choose_fixed_points_clique_size,
				packing_with_assumed_symmetry_choose_fixed_points_control,
				verbose_level);
	}
	else {
		cout << "interface_symbol_table::definition_of_packing_was_choose_fixed_points clique size on fixed spreads is zero, so nothing to do" << endl;
	}




	orbiter_symbol_table_entry Symb;
	Symb.init_packing_was_choose_fixed_points(define_label, PWF, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_was_choose_fixed_points before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_was_choose_fixed_points done" << endl;
	}
}





void interface_symbol_table::definition_of_packing_long_orbits(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_long_orbits" << endl;
	}

	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_long_orbits "
				"using existing object " << packing_long_orbits_choose_fixed_points_label << endl;
	}
	int idx;

	packing_was_fixpoints *PWF;

	idx = Orbiter_top_level_session->find_symbol(packing_long_orbits_choose_fixed_points_label);
	PWF = (packing_was_fixpoints *) Orbiter_top_level_session->get_object(idx);


	packing_long_orbits *PL;

	PL = NEW_OBJECT(packing_long_orbits);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_long_orbits before PL->init" << endl;
	}

	PL->init(PWF, Packing_long_orbits_description, verbose_level);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_long_orbits after PL->init" << endl;
	}




	orbiter_symbol_table_entry Symb;

	Symb.init_packing_long_orbits(define_label, PL, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_long_orbits before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



	if (f_v) {
		cout << "interface_symbol_table::definition_of_packing_long_orbits done" << endl;
	}
}


void interface_symbol_table::definition_of_graph_classification(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_graph_classification" << endl;
	}

	if (f_v) {
		cout << "interface_symbol_table::definition_of_graph_classification "
				"using existing object " << packing_long_orbits_choose_fixed_points_label << endl;
	}


	graph_classify *GC;


	GC = NEW_OBJECT(graph_classify);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_graph_classification before GC->init" << endl;
	}

	GC->init(Graph_classify_description, verbose_level);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_graph_classification after GC->init" << endl;
	}




	orbiter_symbol_table_entry Symb;

	Symb.init_graph_classify(define_label, GC, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_graph_classification before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



	if (f_v) {
		cout << "interface_symbol_table::definition_of_graph_classification done" << endl;
	}
}

void interface_symbol_table::definition_of_diophant(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_diophant" << endl;
	}

	if (f_v) {
		cout << "interface_symbol_table::definition_of_diophant "
				"using existing object " << packing_long_orbits_choose_fixed_points_label << endl;
	}


	diophant_create *Dio;


	Dio = NEW_OBJECT(diophant_create);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_diophant before Dio->init" << endl;
	}

	Dio->init(Diophant_description, verbose_level);


	if (f_v) {
		cout << "interface_symbol_table::definition_of_diophant after Dio->init" << endl;
	}




	orbiter_symbol_table_entry Symb;

	Symb.init_diophant(define_label, Dio, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_diophant before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



	if (f_v) {
		cout << "interface_symbol_table::definition_of_diophant done" << endl;
	}
}



void interface_symbol_table::definition_of_design(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_design" << endl;
	}


	design_create *DC;


	DC = NEW_OBJECT(design_create);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_design before DC->init" << endl;
	}

	DC->init(Design_create_description, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_design after DC->init" << endl;
	}




	orbiter_symbol_table_entry Symb;

	Symb.init_design(define_label, DC, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_design before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



	if (f_v) {
		cout << "interface_symbol_table::definition_of_design done" << endl;
	}
}



void interface_symbol_table::definition_of_design_table(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_design_table" << endl;
	}

	if (f_v) {
		cout << "interface_symbol_table::definition_of_design_table "
				"using existing design " << design_table_label_design << endl;
	}
	int idx;
	design_create *DC;

	idx = Orbiter_top_level_session->find_symbol(design_table_label_design);
	DC = (design_create *) Orbiter_top_level_session->get_object(idx);






	strong_generators *Gens;
	Gens = NEW_OBJECT(strong_generators);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_design_table before Gens->init_from_data_with_go" << endl;
	}
	Gens->init_from_data_with_go(
			DC->A, design_table_generators_data,
			design_table_go_text,
			verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_design_table after Gens->init_from_data_with_go" << endl;
	}


	combinatorics_global Combi;
	design_tables *T;


	if (f_v) {
		cout << "interface_symbol_table::definition_of_design_table before Combi.create_design_table" << endl;
	}

	Combi.create_design_table(DC,
			design_table_label,
			T,
			Gens,
			verbose_level);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_design_table after Combi.create_design_table" << endl;
	}



	large_set_classify *LS;

	LS = NEW_OBJECT(large_set_classify);

	LS->init(DC,
			T,
			verbose_level);



	orbiter_symbol_table_entry Symb;
	Symb.init_design_table(define_label, LS, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_design_table before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



	if (f_v) {
		cout << "interface_symbol_table::definition_of_design_table done" << endl;
	}
}


void interface_symbol_table::definition_of_large_set_was(orbiter_top_level_session *Orbiter_top_level_session,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_large_set_was" << endl;
	}

	if (f_v) {
		cout << "interface_symbol_table::definition_of_large_set_was "
				"using existing spread table " << packing_was_label_spread_table << endl;
	}
	int idx;
	large_set_classify *LS;

	idx = Orbiter_top_level_session->find_symbol(large_set_was_label_design_table);
	LS = (large_set_classify *) Orbiter_top_level_session->get_object(idx);






	large_set_was *LSW;

	LSW = NEW_OBJECT(large_set_was);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_large_set_was before LSW->init" << endl;
	}

	LSW->init(large_set_was_descr, LS, verbose_level);

	if (f_v) {
		cout << "interface_symbol_table::definition_of_large_set_was after LSW->init" << endl;
	}




	orbiter_symbol_table_entry Symb;
	Symb.init_large_set_was(define_label, LSW, verbose_level);
	if (f_v) {
		cout << "interface_symbol_table::definition_of_large_set_was before add_symbol_table_entry" << endl;
	}
	Orbiter_top_level_session->add_symbol_table_entry(
			define_label, &Symb, verbose_level);



	if (f_v) {
		cout << "interface_symbol_table::definition_of_large_set_was done" << endl;
	}
}




}}

