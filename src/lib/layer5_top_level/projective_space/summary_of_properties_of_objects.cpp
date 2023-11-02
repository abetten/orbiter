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

	//std::string label_EK;

	f_quartic_curves = false;

	Nb_objects = NULL;
	nb_E = NULL;
	Ago = NULL;

	Table = NULL;
	E_freq_total = NULL;
	E_type_idx = NULL;
	nb_E_max = 0;
	E = NULL;
	nb_E_types = 0;
	Nb_total = 0;

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

	summary_of_properties_of_objects::field_orders = field_orders;
	summary_of_properties_of_objects::nb_fields = nb_fields;


	label_EK = "E";
	f_quartic_curves = false;

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

	compute_Nb_total();

	if (f_v) {
		cout << "nb_E_max=" << nb_E_max << endl;
		cout << "Nb_total=" << Nb_total << endl;
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

	summary_of_properties_of_objects::field_orders = field_orders;
	summary_of_properties_of_objects::nb_fields = nb_fields;


	label_EK = "K";
	f_quartic_curves = true;


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

	compute_Nb_total();




	nb_E_max = 0;
	for (i = 0; i < nb_fields; i++) {
		q = field_orders[i];
		for (j = 0; j < Nb_objects[i]; j++) {
			nb_E_max = MAXIMUM(nb_E_max, nb_E[i][j]);
		}
	}

	if (f_v) {
		cout << "nb_E_max=" << nb_E_max << endl;
		cout << "Nb_total=" << Nb_total << endl;
	}

	E_freq_total = NEW_int(nb_E_max + 1);
	Int_vec_zero(E_freq_total, nb_E_max + 1);
	for (i = 0; i < nb_fields; i++) {
		q = field_orders[i];
		for (j = 0; j < Nb_objects[i]; j++) {

			int nb_e;

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
			int nb_e;

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

void summary_of_properties_of_objects::compute_Nb_total()
{
	int i, nb_reps;

	Nb_total = 0;
	for (i = 0; i < nb_fields; i++) {
		nb_reps = Nb_objects[i];

		Nb_total += nb_reps;
	}

}

void summary_of_properties_of_objects::export_table_csv(
		std::string &prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "summary_of_properties_of_objects::export_table_csv" << endl;
	}
	long int *Table2;

	int nb_cols;
	int i, j, k;

	nb_cols = nb_E_types + 1;
	Table2 = NEW_lint(nb_fields * nb_cols);
	for (i = 0; i < nb_fields; i++) {
		Table2[i * nb_cols + 0] = field_orders[i];
		for (j = 0; j < nb_E_types; j++) {
			k = E[j];
			Table2[i * nb_cols + 1 + j] =
					Table[i * nb_E_types + k];
		}
	}

	std::string fname;

	if (f_quartic_curves) {
		fname = "table_of_quartic_curves_" + prefix + ".csv";
	}
	else {
		fname = "table_of_cubic_surfaces_" + prefix + ".csv";

	}

	if (f_v) {
		cout << "summary_of_properties_of_objects::export_table_csv "
				"preparing headers" << endl;
	}
	std::string *headers;

	headers = new string[nb_E_types + 1];



	headers[0].assign("Q");
	for (j = 0; j < nb_E_types; j++) {
		headers[1 + j] = label_EK + std::to_string(E[j]);
	}
	if (f_v) {
		cout << "summary_of_properties_of_objects::export_table_csv "
				"preparing headers done" << endl;
	}

	orbiter_kernel_system::file_io Fio;

	Fio.Csv_file_support->lint_matrix_write_csv_override_headers(
			fname, headers, Table2, nb_fields, nb_cols);

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "summary_of_properties_of_objects::export_table_csv "
				"before FREE_lint(Table2)" << endl;
	}
	FREE_lint(Table2);
	if (f_v) {
		cout << "summary_of_properties_of_objects::export_table_csv "
				"after FREE_lint(Table2)" << endl;
	}


	if (f_v) {
		cout << "summary_of_properties_of_objects::export_table_csv done" << endl;
	}

}

void summary_of_properties_of_objects::table_latex(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "summary_of_properties_of_objects::table_latex" << endl;
	}
	if (f_v) {
		cout << "surface_domain_high_level::make_table_of_objects "
				"before writing table 1" << endl;
	}
	int i, j, q, nb_reps;

	ost << "$$" << endl;
	ost << "\\begin{array}{|r||r||*{" << nb_E_types << "}{r|}}" << endl;
	ost << "\\hline" << endl;
	ost << "q  & \\mbox{total} ";
	for (j = 0; j < nb_E_types; j++) {
		ost << " & " << E[j];
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_fields; i++) {
		q = field_orders[i];
		ost << q;

		nb_reps = Nb_objects[i];

		//Nb_total += nb_reps;
		ost << " & ";
		ost << nb_reps;
		for (j = 0; j < nb_E_types; j++) {
			ost << " & " << Table[i * nb_E_types + j];
		}
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
	}
	//cout << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;

	ost << "Total: " << Nb_total << endl;


	if (f_v) {
		cout << "summary_of_properties_of_objects::table_latex done" << endl;
	}

}

void summary_of_properties_of_objects::table_ago(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "summary_of_properties_of_objects::table_ago" << endl;
	}

	int i, j;

	ost << "\\bigskip" << endl;

	for (j = 0; j < nb_E_types; j++) {
		ost << "\\subsection*{" << E[j] << " " << label_EK << " Points}" << endl;


		ost << "$$" << endl;
		ost << "\\begin{array}{|r|r|p{8cm}|}" << endl;
		ost << "\\hline" << endl;
		ost << "q & \\mbox{total} & \\mbox{Ago} \\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;

		for (i = 0; i < nb_fields; i++) {

			int nb_reps;
			int q;

			q = field_orders[i];
			nb_reps = Table[i * nb_E_types + j];
			if (nb_reps) {

				int nb_e;
				int *Ago_q;
				int h, u, nb_total;
				data_structures::string_tools ST;

				nb_total = Nb_objects[i]; //K.cubic_surface_nb_reps(q);
				Ago_q = NEW_int(nb_reps);
				u = 0;
				for (h = 0; h < nb_total; h++) {
					nb_e = nb_E[i][h];
					if (nb_e != E[j]) {
						continue;
					}

					Ago_q[u++] = Ago[i][h];
				}

				if (u != nb_reps) {
					cout << "u != nb_reps" << endl;
					exit(1);
				}
				data_structures::tally C;

				C.init(Ago_q, nb_reps, false, 0);
				ost << q << " & " << nb_reps << " & ";
				ost << "$";
				C.print_bare_tex(ost, true /* f_backwards*/);
				ost << "$\\\\" << endl;

				FREE_int(Ago_q);



				ost << "\\hline" << endl;
			}
		}


		ost << "\\end{array}" << endl;
		ost << "$$" << endl;

		ost << "Total: " << Nb_total << endl;


		ost << "\\bigskip" << endl;


	} // next j

}


void summary_of_properties_of_objects::make_detailed_table_of_objects(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "summary_of_properties_of_objects::make_detailed_table_of_objects" << endl;
	}
	int i, j, q, cur;
	knowledge_base::knowledge_base K;
	long int *Big_table;
	orbiter_kernel_system::file_io Fio;

	Big_table = NEW_lint(Nb_total * 4);

	cur = 0;
	for (i = 0; i < nb_fields; i++) {
		q = field_orders[i];

		for (j = 0; j < Nb_objects[i]; j++, cur++) {


			Big_table[cur * 4 + 0] = q;
			Big_table[cur * 4 + 1] = nb_E[i][j];
			Big_table[cur * 4 + 3] = Ago[i][j];
			Big_table[cur * 4 + 2] = j;

		}
	}

	if (cur != Nb_total) {
		cout << "summary_of_properties_of_objects::make_detailed_table_of_objects "
				"cur != Nb_total" << endl;
		exit(1);
	}
	std::string fname;

	fname.assign("table_of_objects_QECA.csv");

	std::string *headers;

	headers = new string[4];


	headers[0].assign("Q");
	headers[1].assign("E");
	headers[2].assign("OCN");
	headers[3].assign("AUT");


	Fio.Csv_file_support->lint_matrix_write_csv_override_headers(
			fname, headers, Big_table, Nb_total, 4);

	FREE_lint(Big_table);
	if (f_v) {
		cout << "summary_of_properties_of_objects::make_detailed_table_of_objects "
				"done" << endl;
	}
}


}}}


