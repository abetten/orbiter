/*
 * projective_space_activity.cpp
 *
 *  Created on: Jan 5, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


projective_space_activity::projective_space_activity()
{
	Descr = NULL;
	PA = NULL;
}

projective_space_activity::~projective_space_activity()
{

}

void projective_space_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::perform_activity" << endl;
	}

	if (Descr->f_canonical_form_PG) {

		PA->canonical_form(
				Descr->Canonical_form_PG_Descr,
				verbose_level);
	}

	else if (Descr->f_table_of_cubic_surfaces_compute_properties) {

		do_cubic_surface_properties(
				PA,
				Descr->table_of_cubic_surfaces_compute_fname_csv,
				Descr->table_of_cubic_surfaces_compute_defining_q,
				Descr->table_of_cubic_surfaces_compute_column_offset,
				verbose_level);
	}
	else if (Descr->f_cubic_surface_properties_analyze) {


		do_cubic_surface_properties_analyze(
				PA,
				Descr->cubic_surface_properties_fname_csv,
				Descr->cubic_surface_properties_defining_q,
				verbose_level);
	}
	else if (Descr->f_canonical_form_of_code) {

		canonical_form_of_code(
				PA,
				Descr->canonical_form_of_code_label,
				Descr->canonical_form_of_code_m, Descr->canonical_form_of_code_n,
				Descr->canonical_form_of_code_text,
				verbose_level);

	}
	else if (Descr->f_analyze_del_Pezzo_surface) {

		analyze_del_Pezzo_surface(
				PA,
				Descr->analyze_del_Pezzo_surface_label,
				Descr->analyze_del_Pezzo_surface_parameters,
				verbose_level);

	}




	if (f_v) {
		cout << "projective_space_activity::perform_activity done" << endl;
	}

}

void projective_space_activity::analyze_del_Pezzo_surface(
		projective_space_with_action *PA,
		std::string &label,
		std::string &evaluate_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::analyze_del_Pezzo_surface" << endl;
	}



	int idx;
	idx = The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->find_symbol(label);

	if (idx < 0) {
		cout << "could not find symbol " << label << endl;
		exit(1);
	}
	The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->get_object(idx);

	if (The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].type != t_object) {
		cout << "symbol table entry must be of type t_object" << endl;
		exit(1);
	}
	if (The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].object_type == t_collection) {
		cout << "symbol table entry is a collection" << endl;

		vector<string> *List;

		List = (vector<string> *) The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].ptr;
		int i;

		for (i = 0; i < List->size(); i++) {
			int idx1;

			idx1 = The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->find_symbol((*List)[i]);
			if (idx1 < 0) {
				cout << "could not find symbol " << (*List)[i] << endl;
				exit(1);
			}
			formula *F;
			F = (formula *) The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx1].ptr;

			analyze_del_Pezzo_surface_formula_given(
					PA,
					F,
					evaluate_text,
					verbose_level);
		}
	}
	else if (The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].object_type == t_formula) {
		cout << "symbol table entry is a formula" << endl;

		formula *F;
		F = (formula *) The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].ptr;

		analyze_del_Pezzo_surface_formula_given(
				PA,
				F,
				evaluate_text,
				verbose_level);
	}
	else {
		cout << "symbol table entry must be either a formula or a collection" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_activity::analyze_del_Pezzo_surface done" << endl;
	}
}

void projective_space_activity::analyze_del_Pezzo_surface_formula_given(
		projective_space_with_action *PA,
		formula *F,
		std::string &evaluate_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::analyze_del_Pezzo_surface_formula_given" << endl;
	}

	if (f_v) {
		cout << "projective_space_activity::analyze_del_Pezzo_surface_formula_given before PA->analyze_del_Pezzo_surface" << endl;
	}

	PA->analyze_del_Pezzo_surface(F, evaluate_text, verbose_level);

	if (f_v) {
		cout << "projective_space_activity::analyze_del_Pezzo_surface_formula_given after PA->analyze_del_Pezzo_surface" << endl;
	}

	if (f_v) {
		cout << "projective_space_activity::analyze_del_Pezzo_surface_formula_given done" << endl;
	}
}




void projective_space_activity::canonical_form_of_code(
		projective_space_with_action *PA,
		std::string &label, int m, int n,
		std::string &data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code" << endl;
		cout << "m=" << m << endl;
		cout << "n=" << n << endl;
		cout << "data=" << data << endl;
	}

	int *genma;
	int sz;
	int i, j;
	int *v;
	long int *set;

	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code before int_vec_scan" << endl;
	}
	int_vec_scan(data, genma, sz);
	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code after int_vec_scan, sz=" << sz << endl;
	}

	if (sz != m * n) {
		cout << "projective_space_activity::canonical_form_of_code sz != m * n" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "genma: " << endl;
		int_vec_print(cout, genma, sz);
		cout << endl;
	}
	v = NEW_int(m);
	set = NEW_lint(n);
	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
			v[i] = genma[i * n + j];
		}
		if (f_v) {
			cout << "projective_space_activity::canonical_form_of_code before PA->P->rank_point" << endl;
			int_vec_print(cout, v, m);
			cout << endl;
		}
		if (PA->P == NULL) {
			cout << "PA->P == NULL" << endl;
			exit(1);
		}
		set[j] = PA->P->rank_point(v);
	}
	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code set=";
		lint_vec_print(cout, set, n);
		cout << endl;
	}

	projective_space_object_classifier_description Descr;
	data_input_stream Data;
	string points_as_string;
	char str[1000];

	sprintf(str, "%ld", set[0]);
	points_as_string.assign(str);
	for (i = 1; i < n; i++) {
		points_as_string.append(",");
		sprintf(str, "%ld", set[i]);
		points_as_string.append(str);
	}
	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code points_as_string=" << points_as_string << endl;
	}

	Descr.f_input = TRUE;
	Descr.Data = &Data;

	Descr.f_save_classification = TRUE;
	Descr.save_prefix.assign("code_");

	Descr.f_report = TRUE;
	Descr.report_prefix.assign("code_");
	Descr.report_prefix.append(label);

	Descr.f_classification_prefix = TRUE;
	Descr.classification_prefix.assign("classify_code_");
	Descr.classification_prefix.append(label);

	Data.nb_inputs = 0;
	Data.input_type[Data.nb_inputs] = INPUT_TYPE_SET_OF_POINTS;
	Data.input_string[Data.nb_inputs] = points_as_string;
	Data.nb_inputs++;


	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code before PA->canonical_form" << endl;
	}

	PA->canonical_form(&Descr, verbose_level);

	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code after PA->canonical_form" << endl;
	}

#if 0
	int f_input;
	data_input_stream *Data;


	int f_save_classification;
	std::string save_prefix;

	int f_report;
	std::string report_prefix;

	int fixed_structure_order_list_sz;
	int fixed_structure_order_list[1000];

	int f_max_TDO_depth;
	int max_TDO_depth;

	int f_classification_prefix;
	std::string classification_prefix;

#if 0
	int f_save_incma_in_and_out;
	std::string save_incma_in_and_out_prefix;
#endif

	int f_save_canonical_labeling;

	int f_save_ago;

	int f_load_canonical_labeling;

	int f_load_ago;

	int f_save_cumulative_canonical_labeling;
	std::string cumulative_canonical_labeling_fname;

	int f_save_cumulative_ago;
	std::string cumulative_ago_fname;

	int f_save_cumulative_data;
	std::string cumulative_data_fname;

	int f_save_fibration;
	std::string fibration_fname;
#endif


	FREE_int(v);
	FREE_lint(set);

	if (f_v) {
		cout << "projective_space_activity::canonical_form_of_code done" << endl;
	}
}

void projective_space_activity::do_cubic_surface_properties(
		projective_space_with_action *PA,
		std::string fname_csv, int defining_q,
		int column_offset,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::do_cubic_surface_properties" << endl;
	}

	int i;
	finite_field *F0;
	finite_field *F;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	number_theory_domain NT;
	sorting Sorting;
	file_io Fio;




	F0 = NEW_OBJECT(finite_field);
	F0->finite_field_init(defining_q, 0);

	F = PA->P->F;


	Surf = NEW_OBJECT(surface_domain);
	Surf->init(F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_activity::do_cubic_surface_properties "
				"after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "projective_space_activity::do_cubic_surface_properties "
				"before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA->A, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_activity::do_cubic_surface_properties "
				"after Surf_A->init" << endl;
	}






	long int *M;
	int nb_orbits, n;

	Fio.lint_matrix_read_csv(fname_csv, M, nb_orbits, n, verbose_level);

	if (n != 3 + column_offset) {
		cout << "projective_space_activity::do_cubic_surface_properties "
				"n != 3 + column_offset" << endl;
		exit(1);
	}

	int orbit_idx;

	long int *Orbit;
	long int *Rep;
	long int *Stab_order;
	long int *Orbit_length;
	long int *Nb_pts;
	long int *Nb_lines;
	long int *Nb_Eckardt_points;
	long int *Nb_singular_pts;
	long int *Nb_Double_points;
	long int *Ago;

	Orbit = NEW_lint(nb_orbits);
	Rep = NEW_lint(nb_orbits);
	Stab_order = NEW_lint(nb_orbits);
	Orbit_length = NEW_lint(nb_orbits);
	Nb_pts = NEW_lint(nb_orbits);
	Nb_lines = NEW_lint(nb_orbits);
	Nb_Eckardt_points = NEW_lint(nb_orbits);
	Nb_singular_pts = NEW_lint(nb_orbits);
	Nb_Double_points = NEW_lint(nb_orbits);
	Ago = NEW_lint(nb_orbits);

	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		if (f_v) {
			cout << "projective_space_activity::do_cubic_surface_properties "
					"orbit_idx = " << orbit_idx << " / " << nb_orbits << endl;
		}
		int coeff20[20];
		char str[1000];


		Orbit[orbit_idx] = M[orbit_idx * n + 0];
		Rep[orbit_idx] = M[orbit_idx * n + column_offset + 0];
		Stab_order[orbit_idx] = M[orbit_idx * n + column_offset + 1];
		Orbit_length[orbit_idx] = M[orbit_idx * n + column_offset + 2];

		cout << "Rep=" << Rep[orbit_idx] << endl;
		F0->PG_element_unrank_modified_lint(coeff20, 1, 20, Rep[orbit_idx]);
		cout << "coeff20=";
		int_vec_print(cout, coeff20, 20);
		cout << endl;

		surface_create_description *Descr;

		Descr = NEW_OBJECT(surface_create_description);
		Descr->f_q = TRUE;
		Descr->q = F->q;
		Descr->f_by_coefficients = TRUE;
		sprintf(str, "%d,0", coeff20[0]);
		Descr->coefficients_text.assign(str);
		for (i = 1; i < 20; i++) {
			sprintf(str, ",%d,%d", coeff20[i], i);
			Descr->coefficients_text.append(str);
		}
		cout << "Descr->coefficients_text = " << Descr->coefficients_text << endl;


		surface_create *SC;
		SC = NEW_OBJECT(surface_create);

		if (f_v) {
			cout << "projective_space_activity::do_cubic_surface_properties "
					"before SC->init" << endl;
		}
		SC->init(Descr, Surf_A, 0 /*verbose_level*/);
		if (f_v) {
			cout << "projective_space_activity::do_cubic_surface_properties "
					"after SC->init" << endl;
		}


		if (SC->F->e == 1) {
			SC->F->f_print_as_exponentials = FALSE;
		}

		SC->F->PG_element_normalize(SC->SO->eqn, 1, 20);

		if (f_v) {
			cout << "projective_space_activity::do_cubic_surface_properties "
					"We have created the following surface:" << endl;
			cout << "$$" << endl;
			SC->Surf->print_equation_tex(cout, SC->SO->eqn);
			cout << endl;
			cout << "$$" << endl;

			cout << "$$" << endl;
			int_vec_print(cout, SC->SO->eqn, 20);
			cout << endl;
			cout << "$$" << endl;
		}


		// compute the group of the surface if we are over a small field.
		// Otherwise we don't, because it would take too long.


		if (F->q <= 8) {
			if (f_v) {
				cout << "projective_space_activity::do_cubic_surface_properties "
						"before SC->compute_group" << endl;
			}
			SC->compute_group(PA, verbose_level);
			if (f_v) {
				cout << "projective_space_activity::do_cubic_surface_properties "
						"after SC->compute_group" << endl;
			}
			Ago[orbit_idx] = SC->Sg->group_order_as_lint();
		}
		else {
			cout << "F->q = " << F->q << " we are not computing the automorphism group" << endl;
			Ago[orbit_idx] = 0;
		}


		Nb_pts[orbit_idx] = SC->SO->nb_pts;
		Nb_lines[orbit_idx] = SC->SO->nb_lines;
		Nb_Eckardt_points[orbit_idx] = SC->SO->SOP->nb_Eckardt_points;
		Nb_singular_pts[orbit_idx] = SC->SO->SOP->nb_singular_pts;
		Nb_Double_points[orbit_idx] = SC->SO->SOP->nb_Double_points;

		//SC->SO->SOP->print_everything(ost, verbose_level);






		FREE_OBJECT(SC);
		FREE_OBJECT(Descr);


	}


	string fname_data;

	fname_data.assign(fname_csv);
	chop_off_extension(fname_data);

	char str[1000];
	sprintf(str, "_F%d.csv", F->q);
	fname_data.append(str);

	long int *Vec[10];
	char str_A[1000];
	char str_P[1000];
	char str_L[1000];
	char str_E[1000];
	char str_S[1000];
	char str_D[1000];
	sprintf(str_A, "Ago-%d", F->q);
	sprintf(str_P, "Nb_P-%d", F->q);
	sprintf(str_L, "Nb_L-%d", F->q);
	sprintf(str_E, "Nb_E-%d", F->q);
	sprintf(str_S, "Nb_S-%d", F->q);
	sprintf(str_D, "Nb_D-%d", F->q);
	const char *column_label[] = {
			"Orbit_idx",
			"Rep",
			"StabOrder",
			"OrbitLength",
			str_A,
			str_P,
			str_L,
			str_E,
			str_S,
			str_D,
	};

	Vec[0] = Orbit;
	Vec[1] = Rep;
	Vec[2] = Stab_order;
	Vec[3] = Orbit_length;
	Vec[4] = Ago;
	Vec[5] = Nb_pts;
	Vec[6] = Nb_lines;
	Vec[7] = Nb_Eckardt_points;
	Vec[8] = Nb_singular_pts;
	Vec[9] = Nb_Double_points;

	Fio.lint_vec_array_write_csv(10 /* nb_vecs */, Vec, nb_orbits,
			fname_data, column_label);

	if (f_v) {
		cout << "Written file " << fname_data << " of size "
				<< Fio.file_size(fname_data) << endl;
	}



	FREE_lint(M);
	//FREE_OBJECT(PA);
	FREE_OBJECT(F0);
	FREE_OBJECT(Surf);
	FREE_OBJECT(Surf_A);

	if (f_v) {
		cout << "projective_space_activity::do_cubic_surface_properties done" << endl;
	}
}


struct cubic_surface_data_set {

	int orbit_idx;
	long int Orbit_idx;
	long int Rep;
	long int Stab_order;
	long int Orbit_length;
	long int Ago;
	long int Nb_pts;
	long int Nb_lines;
	long int Nb_Eckardt_points;
	long int Nb_singular_pts;
	long int Nb_Double_points;

};

void projective_space_activity::do_cubic_surface_properties_analyze(
		//linear_group *LG,
		projective_space_with_action *PA,
		std::string fname_csv, int defining_q,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::do_cubic_surface_properties_analyze" << endl;
	}

	finite_field *F0;
	finite_field *F;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	number_theory_domain NT;
	sorting Sorting;
	file_io Fio;



	F0 = NEW_OBJECT(finite_field);
	F0->finite_field_init(defining_q, 0);

	F = PA->P->F;


	Surf = NEW_OBJECT(surface_domain);
	Surf->init(F, 0 /* verbose_level - 1 */);
	if (f_v) {
		cout << "projective_space_activity::do_cubic_surface_properties_analyze "
				"after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "projective_space_activity::do_cubic_surface_properties_analyze "
				"before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA->A, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_activity::do_cubic_surface_properties_analyze "
				"after Surf_A->init" << endl;
	}


	int nb_orbits, n;
	int orbit_idx;
	struct cubic_surface_data_set *Data;

	{
		long int *M;

		Fio.lint_matrix_read_csv(fname_csv, M, nb_orbits, n, verbose_level);

		if (n != 10) {
			cout << "projective_space_activity::do_cubic_surface_properties_analyze n != 10" << endl;
			exit(1);
		}





		Data = new struct cubic_surface_data_set [nb_orbits];

		for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
			Data[orbit_idx].orbit_idx = orbit_idx;
			Data[orbit_idx].Orbit_idx = M[orbit_idx * n + 0];
			Data[orbit_idx].Rep = M[orbit_idx * n + 1];
			Data[orbit_idx].Stab_order = M[orbit_idx * n + 2];
			Data[orbit_idx].Orbit_length = M[orbit_idx * n + 3];
			Data[orbit_idx].Ago = M[orbit_idx * n + 4];
			Data[orbit_idx].Nb_pts = M[orbit_idx * n + 5];
			Data[orbit_idx].Nb_lines = M[orbit_idx * n + 6];
			Data[orbit_idx].Nb_Eckardt_points = M[orbit_idx * n + 7];
			Data[orbit_idx].Nb_singular_pts = M[orbit_idx * n + 8];
			Data[orbit_idx].Nb_Double_points = M[orbit_idx * n + 9];
		}
		FREE_lint(M);
	}
	long int *Nb_singular_pts;

	Nb_singular_pts = NEW_lint(nb_orbits);
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		Nb_singular_pts[orbit_idx] = Data[orbit_idx].Nb_singular_pts;
	}


	tally T_S;

	T_S.init_lint(Nb_singular_pts, nb_orbits, FALSE, 0);

	cout << "Classification by the number of singular points:" << endl;
	T_S.print(TRUE /* f_backwards */);

	{
		string fname_report;
		fname_report.assign(fname_csv);
		chop_off_extension(fname_report);
		fname_report.append("_report.tex");
		latex_interface L;
		file_io Fio;

		{
			ofstream ost(fname_report);
			L.head_easy(ost);

#if 0
			if (f_v) {
				cout << "projective_space_activity::do_cubic_surface_properties_analyze "
						"before get_A()->report" << endl;
			}

			if (!Descr->f_draw_options) {
				cout << "please use -draw_options" << endl;
				exit(1);
			}
			PA->A->report(ost,
					FALSE /* f_sims */,
					NULL, //A1/*LG->A_linear*/->Sims,
					FALSE /* f_strong_gens */,
					NULL,
					Descr->draw_options,
					verbose_level - 1);

			if (f_v) {
				cout << "projective_space_activity::do_cubic_surface_properties_analyze "
						"after LG->A_linear->report" << endl;
			}
#endif

			if (f_v) {
				cout << "algebra_global_with_action::do_cubic_surface_properties_analyze "
						"before report" << endl;
			}


			ost << "\\section{Surfaces over ${\\mathbb F}_{" << F->q << "}$}" << endl;


			ost << "Number of surfaces: " << nb_orbits << "\\\\" << endl;
			ost << "Classification by the number of singular points:" << endl;
			ost << "$$" << endl;
			T_S.print_file_tex_we_are_in_math_mode(ost, TRUE /* f_backwards */);
			ost << "$$" << endl;


			ost << "\\section{Singular Surfaces}" << endl;

			report_singular_surfaces(ost, Data, nb_orbits, verbose_level);

			ost << "\\section{Nonsingular Surfaces}" << endl;

			report_non_singular_surfaces(ost, Data, nb_orbits, verbose_level);



			if (f_v) {
				cout << "projective_space_activity::do_cubic_surface_properties_analyze "
						"after report" << endl;
			}

			L.foot(ost);
		}
		cout << "Written file " << fname_report << " of size "
				<< Fio.file_size(fname_report) << endl;
	}





	//FREE_OBJECT(PA);
	FREE_OBJECT(F0);
	FREE_OBJECT(Surf_A);
	FREE_OBJECT(Surf);

	if (f_v) {
		cout << "projective_space_activity::do_cubic_surface_properties_analyze done" << endl;
	}
}

void projective_space_activity::report_singular_surfaces(std::ostream &ost,
		struct cubic_surface_data_set *Data, int nb_orbits, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::report_singular_surfaces" << endl;
	}

	struct cubic_surface_data_set *Data_S;
	int nb_S, h, orbit_idx;


	nb_S = 0;
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		if (Data[orbit_idx].Nb_singular_pts) {
			nb_S++;
		}
	}


	Data_S = new struct cubic_surface_data_set [nb_S];

	h = 0;
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		if (Data[orbit_idx].Nb_singular_pts) {
			Data_S[h] = Data[orbit_idx];
			h++;
		}
	}
	if (h != nb_S) {
		cout << "h != nb_S" << endl;
		exit(1);
	}

	long int *Selected_Nb_lines;


	Selected_Nb_lines = NEW_lint(nb_S);


	for (h = 0; h < nb_S; h++) {
		Selected_Nb_lines[h] = Data_S[h].Nb_lines;
	}

	tally T_L;

	T_L.init_lint(Selected_Nb_lines, nb_S, FALSE, 0);

	ost << "Number of surfaces: " << nb_S << "\\\\" << endl;
	ost << "Classification by the number of lines:" << endl;
	ost << "$$" << endl;
	T_L.print_file_tex_we_are_in_math_mode(ost, TRUE /* f_backwards */);
	ost << "$$" << endl;

	report_surfaces_by_lines(ost, Data_S, T_L, verbose_level);



	FREE_lint(Selected_Nb_lines);
	delete [] Data_S;

	if (f_v) {
		cout << "projective_space_activity::report_singular_surfaces done" << endl;
	}
}


void projective_space_activity::report_non_singular_surfaces(std::ostream &ost,
		struct cubic_surface_data_set *Data, int nb_orbits, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::report_non_singular_surfaces" << endl;
	}

	struct cubic_surface_data_set *Data_NS;
	int nb_NS, h, orbit_idx;


	nb_NS = 0;
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		if (Data[orbit_idx].Nb_singular_pts == 0) {
			nb_NS++;
		}
	}


	Data_NS = new struct cubic_surface_data_set [nb_NS];

	h = 0;
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
		if (Data[orbit_idx].Nb_singular_pts == 0) {
			Data_NS[h] = Data[orbit_idx];
			h++;
		}
	}
	if (h != nb_NS) {
		cout << "h != nb_NS" << endl;
		exit(1);
	}

	long int *Selected_Nb_lines;


	Selected_Nb_lines = NEW_lint(nb_NS);


	for (h = 0; h < nb_NS; h++) {
		Selected_Nb_lines[h] = Data_NS[h].Nb_lines;
	}

	for (h = 0; h < nb_NS; h++) {
		cout << h << " : " << Data_NS[h].orbit_idx << " : " << Data_NS[h].Nb_lines << endl;
	}

	tally T_L;

	T_L.init_lint(Selected_Nb_lines, nb_NS, FALSE, 0);

	ost << "Number of surfaces: " << nb_NS << "\\\\" << endl;
	ost << "Classification by the number of lines:" << endl;
	ost << "$$" << endl;
	T_L.print_file_tex_we_are_in_math_mode(ost, TRUE /* f_backwards */);
	ost << "$$" << endl;


	report_surfaces_by_lines(ost, Data_NS, T_L, verbose_level);


	FREE_lint(Selected_Nb_lines);
	delete [] Data_NS;

	if (f_v) {
		cout << "projective_space_activity::report_non_singular_surfaces done" << endl;
	}
}

void projective_space_activity::report_surfaces_by_lines(std::ostream &ost,
		struct cubic_surface_data_set *Data, tally &T, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_activity::report_surfaces_by_lines" << endl;
	}

	int i, j, f, l, a, idx;

	for (i = T.nb_types - 1; i >= 0; i--) {
		f = T.type_first[i];
		l = T.type_len[i];
		a = T.data_sorted[f];

		int nb_L;
		struct cubic_surface_data_set *Data_L;

		nb_L = l;

		Data_L = new struct cubic_surface_data_set [nb_L];

		ost << "The number of surfaces with exactly " << a << " lines is " << nb_L << ": \\\\" << endl;

		for (j = 0; j < l; j++) {
			idx = T.sorting_perm_inv[f + j];
			Data_L[j] = Data[idx];

		}


		for (j = 0; j < l; j++) {
			ost << j
					<< " : i=" << Data_L[j].orbit_idx
					<< " : id=" << Data_L[j].Orbit_idx
					<< " : P=" << Data_L[j].Nb_pts
					<< " : S=" << Data_L[j].Nb_singular_pts
					<< " : E=" << Data_L[j].Nb_Eckardt_points
					<< " : D=" << Data_L[j].Nb_Double_points
					<< " : ago=" << Data_L[j].Ago
					<< " : Rep=" << Data_L[j].Rep
				<< "\\\\" << endl;
		}

		delete [] Data_L;
	}
	if (f_v) {
		cout << "projective_space_activity::report_surfaces_by_lines done" << endl;
	}

}



}}
