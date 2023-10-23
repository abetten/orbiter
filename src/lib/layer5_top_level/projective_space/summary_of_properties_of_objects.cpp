/*
 * summary_of_properties_of_objects.cpp
 *
 *  Created on: Oct 20, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace projective_geometry {



summary_of_properties_of_objects::summary_of_properties_of_objects()
{
	field_orders = NULL;
	nb_fields = 0;

	Nb_objects = NULL;
	nb_E = NULL;
	Ago = NULL;

	Table = NULL;
	E_freq_total = NULL;
	E_type_idx = NULL;
	nb_E_max = 0;
	E = NULL;
	nb_E_types = 0;

}

summary_of_properties_of_objects::~summary_of_properties_of_objects()
{
	if (Nb_objects) {
		FREE_int(Nb_objects);
	}
	if (nb_E) {
		int i;

		for (i = 0; i < nb_fields; i++) {
			FREE_int(nb_E[i]);
		}
		FREE_pint(nb_E);
	}
	if (Ago) {
		int i;

		for (i = 0; i < nb_fields; i++) {
			FREE_lint(Ago[i]);
		}
		FREE_plint(Ago);
	}
	if (Table) {
		FREE_lint(Table);
	}
	if (E_freq_total) {
		FREE_int(E_freq_total);
	}
	if (E_type_idx) {
		FREE_int(E_type_idx);
	}
}

void summary_of_properties_of_objects::init_surfaces(
		int *field_orders, int nb_fields,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "summary_of_properties_of_objects::init_surfaces" << endl;
	}
	int i, j, q, idx, nb_e;
	knowledge_base::knowledge_base K;
	data_structures::string_tools String;


	nb_E = NEW_pint(nb_fields);
	Ago = NEW_plint(nb_fields);

	Nb_objects = NEW_int(nb_fields);


	nb_E_max = 0;
	for (i = 0; i < nb_fields; i++) {
		q = field_orders[i];
		Nb_objects[i] = K.cubic_surface_nb_reps(q);

		nb_E[i] = NEW_int(Nb_objects[i]);

		Ago[i] = NEW_lint(Nb_objects[i]);

		for (j = 0; j < Nb_objects[i]; j++) {
			nb_e = K.cubic_surface_nb_Eckardt_points(q, j);
			nb_E[i][j] = nb_e;

			int *data;
			int nb_gens;
			int data_size;
			string stab_order_str;

			K.cubic_surface_stab_gens(q, j,
					data, nb_gens, data_size, stab_order_str);

			Ago[i][j] = String.strtoi(stab_order_str);

			nb_E_max = MAXIMUM(nb_E_max, nb_e);
		}
	}

	if (f_v) {
		cout << "nb_E_max=" << nb_E_max << endl;
	}

	E_freq_total = NEW_int(nb_E_max + 1);
	Int_vec_zero(E_freq_total, nb_E_max + 1);
	for (i = 0; i < nb_fields; i++) {
		q = field_orders[i];
		for (j = 0; j < Nb_objects[i]; j++) {
			nb_e = K.cubic_surface_nb_Eckardt_points(q, j);
			E_freq_total[nb_e]++;
		}
	}



	if (f_v) {
		cout << "E_freq_total=";
		Int_vec_print(cout, E_freq_total, nb_E_max + 1);
		cout << endl;
	}

	E = NEW_int(nb_E_max + 1);
	nb_E_types = 0;

	E_type_idx = NEW_int(nb_E_max + 1);
	for (j = 0; j <= nb_E_max; j++) {
		if (E_freq_total[j]) {
			E[nb_E_types] = j;
			E_type_idx[j] = nb_E_types;
			nb_E_types++;
		}
		else {
			E_type_idx[j] = -1;
		}
	}


	Table = NEW_lint(nb_fields * nb_E_types);
	Lint_vec_zero(Table, nb_fields * nb_E_types);
	for (i = 0; i < nb_fields; i++) {
		q = field_orders[i];
		for (j = 0; j < Nb_objects[i]; j++) {
			nb_e = nb_E[i][j];
			idx = E_type_idx[nb_e];
			Table[i * nb_E_types + idx]++;
		}
	}
	if (f_v) {
		cout << "Table:" << endl;
		Lint_matrix_print(Table, nb_fields, nb_E_types);
	}

	if (f_v) {
		cout << "summary_of_properties_of_objects::init_surfaces done" << endl;
	}
}

void summary_of_properties_of_objects::init_quartic_curves(
		int *field_orders, int nb_fields,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "summary_of_properties_of_objects::init_quartic_curves" << endl;
	}

	int i, j;
	int q;

	projective_geometry::projective_space_with_action **PA;


	PA = (projective_geometry::projective_space_with_action **) NEW_pvoid(nb_fields);

	nb_E = NEW_pint(nb_fields);
	Ago = NEW_plint(nb_fields);

	Nb_objects = NEW_int(nb_fields);

	for (i = 0; i < nb_fields; i++) {
		q = field_orders[i];

		projective_geometry::projective_space_with_action_description
			Projective_space_with_action_description;

		Projective_space_with_action_description.f_n = true;
		Projective_space_with_action_description.n = 2;

		Projective_space_with_action_description.f_q = true;
		Projective_space_with_action_description.q = q;

#if 0
		int f_n;
		int n;

		int f_q;
		int q;

		int f_field_label;
		std::string field_label;

		int f_field_pointer;
		field_theory::finite_field *F;

		int f_use_projectivity_subgroup;

		int f_override_verbose_level;
		int override_verbose_level;
#endif

		PA[i] = NEW_OBJECT(projective_geometry::projective_space_with_action);

		if (f_v) {
			cout << "symbol_definition::definition_of_projective_space "
					"before PA->init_from_description" << endl;
		}
		PA[i]->init_from_description(&Projective_space_with_action_description, 0 /*verbose_level*/);
		if (f_v) {
			cout << "symbol_definition::definition_of_projective_space "
					"after PA->init_from_description" << endl;
		}


		applications_in_algebraic_geometry::quartic_curves::quartic_curve_create **QC;



		if (f_v) {
			cout << "summary_of_properties_of_objects::init_quartic_curves "
					"before PA->QCDA->create_all_quartic_curves_over_a_given_field" << endl;
		}

		PA[i]->QCDA->create_all_quartic_curves_over_a_given_field(
					QC,
					Nb_objects[i],
					verbose_level - 2);

		if (f_v) {
			cout << "summary_of_properties_of_objects::init_quartic_curves "
					"after PA->QCDA->create_all_quartic_curves_over_a_given_field" << endl;
		}

		nb_E[i] = NEW_int(Nb_objects[i]);

		Ago[i] = NEW_lint(Nb_objects[i]);

		for (j = 0; j < Nb_objects[i]; j++) {

			nb_E[i][j] = QC[j]->QO->QP->Kovalevski->nb_Kovalevski;

			Ago[i][j] = QC[j]->QOG->Aut_gens->group_order_as_lint();
		}

		for (j = 0; j < Nb_objects[i]; j++) {
			FREE_OBJECT(QC[j]);
		}
		FREE_pvoid((void **) QC);


	}





	int nb_e;


	nb_E_max = 0;
	for (i = 0; i < nb_fields; i++) {
		q = field_orders[i];
		for (j = 0; j < Nb_objects[i]; j++) {
			nb_E_max = MAXIMUM(nb_E_max, nb_E[i][j]);
		}
	}

	if (f_v) {
		cout << "nb_E_max=" << nb_E_max << endl;
	}

	E_freq_total = NEW_int(nb_E_max + 1);
	Int_vec_zero(E_freq_total, nb_E_max + 1);
	for (i = 0; i < nb_fields; i++) {
		q = field_orders[i];
		for (j = 0; j < Nb_objects[i]; j++) {
			nb_e = nb_E[i][j];
			E_freq_total[nb_e]++;
		}
	}



	if (f_v) {
		cout << "E_freq_total=";
		Int_vec_print(cout, E_freq_total, nb_E_max + 1);
		cout << endl;
	}

	E = NEW_int(nb_E_max + 1);
	nb_E_types = 0;

	E_type_idx = NEW_int(nb_E_max + 1);
	for (j = 0; j <= nb_E_max; j++) {
		if (E_freq_total[j]) {
			E[nb_E_types] = j;
			E_type_idx[j] = nb_E_types;
			nb_E_types++;
		}
		else {
			E_type_idx[j] = -1;
		}
	}


	int idx;

	Table = NEW_lint(nb_fields * nb_E_types);
	Lint_vec_zero(Table, nb_fields * nb_E_types);
	for (i = 0; i < nb_fields; i++) {
		q = field_orders[i];
		for (j = 0; j < Nb_objects[i]; j++) {
			nb_e = nb_E[i][j];
			idx = E_type_idx[nb_e];
			Table[i * nb_E_types + idx]++;
		}
	}
	if (f_v) {
		cout << "Table:" << endl;
		Lint_matrix_print(Table, nb_fields, nb_E_types);
	}

	for (i = 0; i < nb_fields; i++) {
		FREE_OBJECT(PA[i]);
	}
	FREE_pvoid((void **) PA);

	if (f_v) {
		cout << "summary_of_properties_of_objects::init_quartic_curves done" << endl;
	}
}

}}}


