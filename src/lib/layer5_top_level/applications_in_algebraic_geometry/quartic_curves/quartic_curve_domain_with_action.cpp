/*
 * quartic_curve_domain_with_action.cpp
 *
 *  Created on: May 21, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace quartic_curves {



quartic_curve_domain_with_action::quartic_curve_domain_with_action()
{
	Record_birth();

	PA = NULL;
	f_semilinear = false;
	Dom = NULL;
	A = NULL;
	A_on_lines = NULL;
	Elt1 = NULL;
	AonHPD_4_3 = NULL;
}


quartic_curve_domain_with_action::~quartic_curve_domain_with_action()
{
	Record_death();
}

void quartic_curve_domain_with_action::init(
		geometry::algebraic_geometry::quartic_curve_domain *Dom,
		projective_geometry::projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_domain_with_action::init" << endl;
	}
	quartic_curve_domain_with_action::Dom = Dom;
	quartic_curve_domain_with_action::PA = PA;



	A = PA->A;

	if (f_v) {
		cout << "quartic_curve_domain_with_action::init action A:" << endl;
		A->print_info();
	}



	A_on_lines = PA->A_on_lines;
	if (f_v) {
		cout << "quartic_curve_domain_with_action::init "
				"action A_on_lines:" << endl;
		A_on_lines->print_info();
	}
	f_semilinear = A->is_semilinear_matrix_group();
	if (f_v) {
		cout << "quartic_curve_domain_with_action::init "
				"f_semilinear=" << f_semilinear << endl;
	}


	Elt1 = NEW_int(A->elt_size_in_int);

	AonHPD_4_3 = NEW_OBJECT(induced_actions::action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "quartic_curve_domain_with_action::init "
				"before AonHPD_4_3->init" << endl;
	}
	AonHPD_4_3->init(A, Dom->Poly4_3, verbose_level);

	if (f_v) {
		cout << "quartic_curve_domain_with_action::init done" << endl;
	}
}

void quartic_curve_domain_with_action::table_of_quartic_curves(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "quartic_curve_domain_with_action::table_of_quartic_curves" << endl;
	}

	applications_in_algebraic_geometry::quartic_curves::quartic_curve_create **QC;

	int nb_quartic_curves;


	if (f_v) {
		cout << "quartic_curve_domain_with_action::table_of_quartic_curves "
				"before create_all_quartic_curves_over_a_given_field" << endl;
	}

	create_all_quartic_curves_over_a_given_field(
				QC,
				nb_quartic_curves,
				verbose_level);

	if (f_v) {
		cout << "quartic_curve_domain_with_action::table_of_quartic_curves "
				"after create_all_quartic_curves_over_a_given_field" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_domain_with_action::table_of_quartic_curves "
				"before create_table_of_strings" << endl;
	}


	string headings;
	std::string *Table;
	int nb_cols;

	create_table_of_strings(
				QC,
				nb_quartic_curves,
				Table, nb_cols,
				headings,
				verbose_level);

	if (f_v) {
		cout << "quartic_curve_domain_with_action::table_of_quartic_curves "
				"after create_table_of_strings" << endl;
	}

	if (f_v) {
		cout << "quartic_curve_domain_with_action::table_of_quartic_curves "
				"writing file" << endl;
	}


	other::orbiter_kernel_system::file_io Fio;

	string fname;


	fname = "quartic_curves_q" + std::to_string(PA->F->q) + "_info.csv";

	Fio.Csv_file_support->write_table_of_strings(
			fname,
			nb_quartic_curves, nb_cols, Table,
			headings,
			verbose_level);



	delete [] Table;


	if (f_v) {
		cout << "quartic_curve_domain_with_action::table_of_quartic_curves done" << endl;
	}

}

void quartic_curve_domain_with_action::create_all_quartic_curves_over_a_given_field(
		applications_in_algebraic_geometry::quartic_curves::quartic_curve_create **&QC,
		int &nb_quartic_curves,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "quartic_curve_domain_with_action::create_all_quartic_curves_over_a_given_field" << endl;
	}

	combinatorics::knowledge_base::knowledge_base K;

	nb_quartic_curves = K.quartic_curves_nb_reps(PA->F->q);

	int h;

	QC = (applications_in_algebraic_geometry::quartic_curves::quartic_curve_create **)
			NEW_pvoid(nb_quartic_curves);

	for (h = 0; h < nb_quartic_curves; h++) {

		if (f_vv) {
			cout << "quartic_curve_domain_with_action::create_all_quartic_curves_over_a_given_field "
					<< h << " / " << nb_quartic_curves << endl;
		}
		applications_in_algebraic_geometry::quartic_curves::quartic_curve_create_description
			Quartic_curve_descr;


		Quartic_curve_descr.f_space_pointer = true;
		Quartic_curve_descr.space_pointer = PA;

		Quartic_curve_descr.f_catalogue = true;
		Quartic_curve_descr.iso = h;

		QC[h] = NEW_OBJECT(applications_in_algebraic_geometry::quartic_curves::quartic_curve_create);


		if (f_vv) {
			cout << "quartic_curve_domain_with_action::create_all_quartic_curves_over_a_given_field "
					<< h << " / " << nb_quartic_curves
					<< " before create_quartic_curve" << endl;
		}
		QC[h]->create_quartic_curve(
					&Quartic_curve_descr,
					verbose_level);



		if (f_v) {
			cout << "quartic_curve_domain_with_action::create_all_quartic_curves_over_a_given_field "
					<< h << " / " << nb_quartic_curves << " done" << endl;
		}

	}


	if (f_v) {
		cout << "quartic_curve_domain_with_action::create_all_quartic_curves_over_a_given_field done" << endl;
	}
}

void quartic_curve_domain_with_action::create_table_of_strings(
		applications_in_algebraic_geometry::quartic_curves::quartic_curve_create **QC,
		int nb_quartic_curves,
		std::string *&Table, int &nb_cols,
		std::string &headings,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "quartic_curve_domain_with_action::create_table_of_strings" << endl;
	}


	int h;

	headings.assign("OCN,K,Ago,NbPts,BisecantType,Eqn15,Eqn,Pts,Bitangents28");
	nb_cols = 9;

	Table = new string [nb_quartic_curves * nb_cols];

	for (h = 0; h < nb_quartic_curves; h++) {

		if (f_vv) {
			cout << "quartic_curve_domain_with_action::table_of_quartic_curves "
					<< h << " / " << nb_quartic_curves << endl;
		}
		Table[h * nb_cols + 0] = std::to_string(h);


		if (QC[h]->QOG) {


			string s_Bisecant_line_type;

			s_Bisecant_line_type = Int_vec_stringify(QC[h]->QO->QP->Kovalevski->line_type_distribution, 3);


			Table[h * nb_cols + 1] = std::to_string(QC[h]->QO->QP->Kovalevski->nb_Kovalevski);
			//Table[h * nb_cols + 2] = std::to_string(QC[h]->QO->QP->Kovalevski->nb_Kovalevski_on);
			//Table[h * nb_cols + 3] = std::to_string(QC[h]->QO->QP->Kovalevski->nb_Kovalevski_off);
			Table[h * nb_cols + 2] = QC[h]->QOG->Aut_gens->group_order_stringify();
			Table[h * nb_cols + 3] = std::to_string(QC[h]->QO->nb_pts);
			Table[h * nb_cols + 4] = "\"" + s_Bisecant_line_type + "\"";
		}

		string s_Eqn;
		string s_Eqn_maple;
		string s_Pts;
		string s_Bitangents;

		QC[h]->QO->stringify(
				s_Eqn, s_Pts, s_Bitangents);

		s_Eqn_maple = QC[h]->QCDA->Dom->stringify_equation_maple(
				QC[h]->QO->eqn15);

		Table[h * nb_cols + 5] = "\"" + s_Eqn + "\"";
		Table[h * nb_cols + 6] = "\"$" + s_Eqn_maple + "$\"";
		Table[h * nb_cols + 7] = "\"" + s_Pts + "\"";
		Table[h * nb_cols + 8] = "\"" + s_Bitangents + "\"";
	}
	if (f_v) {
		cout << "quartic_curve_domain_with_action::create_table_of_strings done" << endl;
	}

}






}}}}

