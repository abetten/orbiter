/*
 * interface_projective.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace interfaces {




interface_projective::interface_projective()
{

	f_create_points_on_quartic = FALSE;
	desired_distance = 0;

	f_create_points_on_parabola = FALSE;
	parabola_N = 0;
	parabola_a = 0;
	parabola_b = 0.;
	parabola_c = 0.;

	f_smooth_curve = FALSE;
	//smooth_curve_label;
	smooth_curve_N = 0;
	smooth_curve_t_min = 0;
	smooth_curve_t_max = 0;
	smooth_curve_boundary = 0;
	//smooth_curve_Polish = NULL;
	FP_descr = NULL;

	f_create_BLT_set = FALSE;
	BLT_set_descr = NULL;
	//nb_transform = 0;
	//const char *transform_coeffs[1000];
	//int f_inverse_transform[1000];

	f_create_spread = FALSE;
	Spread_create_description = FALSE;




	f_make_table_of_surfaces = FALSE;


	f_create_surface_reports = FALSE;
	f_create_surface_atlas = FALSE;
	create_surface_atlas_q_max = 0;


}


void interface_projective::print_help(int argc,
		std::string *argv, int i, int verbose_level)
{
	if (stringcmp(argv[i], "-classify_cubic_curves") == 0) {
		cout << "-classify_cubic_curves" << endl;
	}
	else if (stringcmp(argv[i], "-control_arcs") == 0) {
		cout << "-control_arcs <description>" << endl;
	}
	else if (stringcmp(argv[i], "-create_points_on_quartic") == 0) {
		cout << "-create_points_on_quartic <double : desired_distance>" << endl;
	}
	else if (stringcmp(argv[i], "-create_points_on_parabola") == 0) {
		cout << "-create_points_on_parabola <double : desired_distance> <double : a> <double : b> <double : c>" << endl;
	}
	else if (stringcmp(argv[i], "-smooth_curve") == 0) {
		cout << "-smooth_curve <string : label> <double : desired_distance> <int : N> <double : boundary> <double : t_min> <double : t_max> <function>" << endl;
	}
	else if (stringcmp(argv[i], "-create_BLT_set") == 0) {
		cout << "-create_BLT_set <description>" << endl;
	}
	else if (stringcmp(argv[i], "-create_spread") == 0) {
		cout << "-create_spread <description>" << endl;
	}
	else if (stringcmp(argv[i], "-make_table_of_surfaces") == 0) {
		cout << "-make_table_of_surfaces " << endl;
	}
	else if (stringcmp(argv[i], "-create_surface_reports") == 0) {
		cout << "-create_surface_reports <int : q_max>" << endl;
	}
	else if (stringcmp(argv[i], "-create_surface_atlas") == 0) {
		cout << "-create_surface_atlas <int : q_max>" << endl;
	}
}



int interface_projective::recognize_keyword(int argc,
		std::string *argv, int i, int verbose_level)
{
	if (i >= argc) {
		return false;
	}
	if (stringcmp(argv[i], "-classify_cubic_curves") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-control_arcs") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-create_points_on_quartic") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-create_points_on_parabola") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-smooth_curve") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-create_BLT_set") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-create_spread") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-make_table_of_surfaces") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-create_surface_reports") == 0) {
		return true;
	}
	else if (stringcmp(argv[i], "-create_surface_atlas") == 0) {
		return true;
	}
	return false;
}

int interface_projective::read_arguments(int argc,
		std::string *argv, int i0, int verbose_level)
{
	int i;

	cout << "interface_projective::read_arguments" << endl;

	for (i = i0; i < argc; i++) {
		if (stringcmp(argv[i], "-create_points_on_quartic") == 0) {
			f_create_points_on_quartic = TRUE;
			desired_distance = strtof(argv[++i]);
			cout << "-create_points_on_quartic " << desired_distance << endl;
			//i++;
		}
		else if (stringcmp(argv[i], "-create_points_on_parabola") == 0) {
			f_create_points_on_parabola = TRUE;
			desired_distance = strtof(argv[++i]);
			parabola_N = strtoi(argv[++i]);
			parabola_a = strtof(argv[++i]);
			parabola_b = strtof(argv[++i]);
			parabola_c = strtof(argv[++i]);
			cout << "-create_points_on_parabola " << desired_distance << " "
					<< parabola_N << " " << parabola_a << " "
					<< parabola_b << " " << parabola_c << endl;
			//i++;
		}
		else if (stringcmp(argv[i], "-smooth_curve") == 0) {
			f_smooth_curve = TRUE;
			smooth_curve_label.assign(argv[++i]);
			desired_distance = strtof(argv[++i]);
			smooth_curve_N = strtoi(argv[++i]);
			smooth_curve_boundary = strtof(argv[++i]);
			smooth_curve_t_min = strtof(argv[++i]);
			smooth_curve_t_max = strtof(argv[++i]);

			FP_descr = NEW_OBJECT(function_polish_description);

			i += FP_descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-smooth_curve "
					<< smooth_curve_label << " "
					<< desired_distance << " "
					<< smooth_curve_N << " "
					<< smooth_curve_boundary << " "
					<< smooth_curve_t_min << " "
					<< smooth_curve_t_max << " "
					<< endl;
			//i++;
		}
		else if (stringcmp(argv[i], "-create_BLT_set") == 0) {
			f_create_BLT_set = TRUE;
			cout << "-create_BLT_set" << endl;
			BLT_set_descr = NEW_OBJECT(BLT_set_create_description);
			i += BLT_set_descr->read_arguments(
					argc - (i - 1),
					argv + i + 1, verbose_level);
			cout << "interface_combinatorics::read_arguments finished "
					"reading -create_BLT_set" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (stringcmp(argv[i], "-create_spread") == 0) {
			f_create_spread = TRUE;
			cout << "-create_spread" << endl;
			Spread_create_description = NEW_OBJECT(spread_create_description);
			i += Spread_create_description->read_arguments(
					argc - (i - 1),
					argv + i + 1, verbose_level);
			cout << "interface_combinatorics::read_arguments finished "
					"reading -create_spread" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (stringcmp(argv[i], "-transform") == 0) {

			string s;

			s.assign(argv[++i]);
			transform_coeffs.push_back(s);
			f_inverse_transform.push_back(FALSE);
			cout << "-transform " << transform_coeffs[transform_coeffs.size() - 1] << endl;
		}
		else if (stringcmp(argv[i], "-transform_inverse") == 0) {
			string s;

			s.assign(argv[++i]);
			transform_coeffs.push_back(s);
			f_inverse_transform.push_back(TRUE);
			cout << "-transform_inverse " << transform_coeffs[transform_coeffs.size() - 1] << endl;
		}
		else if (stringcmp(argv[i], "-make_table_of_surfaces") == 0) {
			f_make_table_of_surfaces = TRUE;
			cout << "-make_table_of_surfaces" << endl;
		}
		else if (stringcmp(argv[i], "-create_surface_atlas") == 0) {
			f_create_surface_atlas = TRUE;
			create_surface_atlas_q_max = strtoi(argv[++i]);
			cout << "-create_surface_atlas " << create_surface_atlas_q_max << endl;
		}
		else if (stringcmp(argv[i], "-create_surface_reports") == 0) {
			f_create_surface_reports = TRUE;
			create_surface_atlas_q_max = strtoi(argv[++i]);
			cout << "-create_surface_reports " << create_surface_atlas_q_max << endl;
		}
		else {
			break;
		}
	}
	cout << "interface_projective::read_arguments done" << endl;
	return i;
}


void interface_projective::worker(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::worker" << endl;
	}

	if (f_create_points_on_quartic) {

		graphical_output GO;

		GO.do_create_points_on_quartic(desired_distance, verbose_level);
	}
	else if (f_create_points_on_parabola) {

		graphical_output GO;

		GO.do_create_points_on_parabola(desired_distance,
				parabola_N, parabola_a, parabola_b, parabola_c, verbose_level);
	}
	else if (f_smooth_curve) {

		graphical_output GO;

		GO.do_smooth_curve(smooth_curve_label,
				desired_distance, smooth_curve_N,
				smooth_curve_t_min, smooth_curve_t_max, smooth_curve_boundary,
				FP_descr, verbose_level);
	}
	else if (f_create_BLT_set) {

		do_create_BLT_set(BLT_set_descr, verbose_level);

	}
	else if (f_create_spread) {

		do_create_spread(Spread_create_description, verbose_level);

	}


	else if (f_make_table_of_surfaces) {

		surface_domain Surf;

		Surf.make_table_of_surfaces(verbose_level);
	}

	else if (f_create_surface_reports) {

		do_create_surface_reports(create_surface_atlas_q_max, verbose_level);

	}

	else if (f_create_surface_atlas) {

		do_create_surface_atlas(create_surface_atlas_q_max, verbose_level);

	}




	if (f_v) {
		cout << "interface_projective::worker done" << endl;
	}
}





void interface_projective::do_create_BLT_set(BLT_set_create_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::do_create_BLT_set" << endl;
	}


	BLT_set_create *BC;
	//int j;

	BC = NEW_OBJECT(BLT_set_create);

	cout << "before BC->init" << endl;
	BC->init(Descr, verbose_level);
	cout << "after BC->init" << endl;


	cout << "before BC->apply_transformations" << endl;
	BC->apply_transformations(transform_coeffs,
			f_inverse_transform, verbose_level);
	cout << "after BC->apply_transformations" << endl;

	action *A;
	//int *Elt1;
	int *Elt2;

	A = BC->A;

	Elt2 = NEW_int(A->elt_size_in_int);



#if 0
	if (BC->f_has_group) {
		for (i = 0; i < BC->Sg->gens->len; i++) {
			cout << "Testing generator " << i << " / "
					<< BC->Sg->gens->len << endl;
			A->element_invert(BC->Sg->gens->ith(i),
					Elt2, 0 /*verbose_level*/);


			cout << "Generator " << i << " / " << SC->Sg->gens->len
					<< " is good" << endl;
			}
		}
	else {
		cout << "We do not have information about the "
				"automorphism group" << endl;
		}
#endif

	latex_interface L;


	cout << "We have created the following BLT-set:" << endl;
	cout << "$$" << endl;
	L.int_set_print_tex(cout, BC->set, BC->q + 1);
	cout << endl;
	cout << "$$" << endl;

	if (BC->f_has_group) {
		cout << "The stabilizer is generated by:" << endl;
		BC->Sg->print_generators_tex(cout);
	}





	FREE_int(Elt2);

	FREE_OBJECT(BC);

	if (f_v) {
		cout << "interface_projective::do_create_BLT_set done" << endl;
	}

}

void interface_projective::do_create_spread(spread_create_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::do_create_spread" << endl;
	}


	spread_create *SC;

	SC = NEW_OBJECT(spread_create);

	cout << "before SC->init" << endl;
	SC->init(Descr, verbose_level);
	cout << "after SC->init" << endl;


	cout << "before SC->apply_transformations" << endl;
	SC->apply_transformations(transform_coeffs,
			f_inverse_transform, verbose_level);
	cout << "after SC->apply_transformations" << endl;


	action *A;
	//int *Elt1;
	int *Elt2;

	A = SC->A;

	Elt2 = NEW_int(A->elt_size_in_int);

	latex_interface L;

	cout << "We have created the following spread set:" << endl;
	cout << "$$" << endl;
	L.lint_set_print_tex(cout, SC->set, SC->sz);
	cout << endl;
	cout << "$$" << endl;

	if (SC->f_has_group) {
		cout << "The stabilizer is generated by:" << endl;
		SC->Sg->print_generators_tex(cout);
		}





	FREE_int(Elt2);

	FREE_OBJECT(SC);

	if (f_v) {
		cout << "interface_projective::do_create_spread done" << endl;
	}

}







struct table_surfaces_field_order {
	int q;
	int p;
	int h;

	linear_group_description *Descr;

	finite_field *F;

	linear_group *LG;

	surface_domain *Surf;
	surface_with_action *Surf_A;

	int nb_total;
	int *nb_E;

	tally *T_nb_E;



};

void interface_projective::do_create_surface_reports(int q_max, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "do_create_surface_reports" << endl;
		cout << "do_create_surface_reports verbose_level=" << verbose_level << endl;
	}

	knowledge_base K;
	number_theory_domain NT;
	file_io Fio;

	int q;
	int cur;

	cur = 0;
	for (q = 2; q <= q_max; q++) {

		int p;
		int h;

		if (!NT.is_prime_power(q, p, h)) {
			continue;
		}


		if (q == 2) {
			continue;
		}
		if (q == 3) {
			continue;
		}
		if (q == 5) {
			continue;
		}

		cout << "considering q=" << q << endl;

		int nb_total;
		int ocn;

		nb_total = K.cubic_surface_nb_reps(q);

		for (ocn = 0; ocn < nb_total; ocn++) {

			string cmd;
			string fname;
			char str[1000];

			make_fname_surface_report_tex(fname, q, ocn);

			cmd.assign(foundations::The_Orbiter_session->orbiter_path);
			cmd.append("/orbiter.out -v 2  ");

			if (h > 1) {
				sprintf(str, " -linear_group -PGGL 4 %d -wedge -end", q);
			}
			else {
				sprintf(str, " -linear_group -PGL 4 %d -wedge -end", q);
			}
			cmd.append(str);
			cmd.append(" -group_theoretic_activities ");
			cmd.append(" -control_six_arcs  -end ");

			sprintf(str, " -create_surface -q %d -catalogue %d -end ", q, ocn);
			cmd.append(str);
			cmd.append(" -draw_options -end ");

			cmd.append(" -end");


			if (f_v) {
				cout << "executing command: " << cmd << endl;
			}
			system(cmd.c_str());

			std::string fname_report_tex;

			make_fname_surface_report_tex(fname_report_tex, q, ocn);

			cmd.assign("pdflatex ");
			cmd.append(fname_report_tex);
			system(cmd.c_str());


		}


	}

	if (f_v) {
		cout << "do_create_surface_reports done" << endl;
	}
}

void interface_projective::do_create_surface_atlas(int q_max, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "do_create_surface_atlas" << endl;
		cout << "do_create_surface_atlas verbose_level=" << verbose_level << endl;
	}

	knowledge_base K;


	number_theory_domain NT;
	sorting Sorting;
	file_io Fio;


	struct table_surfaces_field_order *T;

	T = new struct table_surfaces_field_order[q_max];

	int q;
	int cur;
	int j;

	cur = 0;
	for (q = 2; q <= q_max; q++) {

		int p;
		int h;

		if (!NT.is_prime_power(q, p, h)) {
			continue;
		}

		cout << "considering q=" << q << endl;


		T[cur].q = q;
		T[cur].p = p;
		T[cur].h = h;


		T[cur].Descr = NEW_OBJECT(linear_group_description);

		T[cur].Descr->n = 4;
		T[cur].Descr->input_q = q;
		T[cur].Descr->f_projective = TRUE;
		T[cur].Descr->f_general = FALSE;
		T[cur].Descr->f_affine = FALSE;
		T[cur].Descr->f_semilinear = FALSE;

		if (h > 1) {
			T[cur].Descr->f_semilinear = TRUE;
		}
		T[cur].Descr->f_special = FALSE;

		T[cur].F = NEW_OBJECT(finite_field);
		T[cur].F->finite_field_init(q, 0);

		T[cur].Descr->F = T[cur].F;


		T[cur].LG = NEW_OBJECT(linear_group);

		cout << "before LG->init" << endl;
		T[cur].LG->init(T[cur].Descr, verbose_level);








		if (f_v) {
			cout << "do_create_surface_atlas before Surf->init" << endl;
		}

		T[cur].Surf = NEW_OBJECT(surface_domain);
		T[cur].Surf->init(T[cur].F, 0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "do_create_surface_atlas after Surf->init" << endl;
		}

		T[cur].Surf_A = NEW_OBJECT(surface_with_action);

		if (f_v) {
			cout << "do_create_surface_atlas before Surf_A->init" << endl;
		}
		T[cur].Surf_A->init(T[cur].Surf, T[cur].LG, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
		if (f_v) {
			cout << "do_create_surface_atlas after Surf_A->init" << endl;
		}


		if (T[cur].q == 2) {
			cur++;
			continue;
		}
		if (T[cur].q == 3) {
			cur++;
			continue;
		}
		if (T[cur].q == 5) {
			cur++;
			continue;
		}


		T[cur].nb_total = K.cubic_surface_nb_reps(T[cur].q);


		T[cur].nb_E = NEW_int(T[cur].nb_total);

		for (j = 0; j < T[cur].nb_total; j++) {
			T[cur].nb_E[j] = K.cubic_surface_nb_Eckardt_points(T[cur].q, j);
		}

		T[cur].T_nb_E = NEW_OBJECT(tally);

		T[cur].T_nb_E->init(T[cur].nb_E, T[cur].nb_total, FALSE, 0);


		cur++;
	}

	cout << "we found the following field orders:" << endl;

	int nb_fields;
	int c;


	nb_fields = cur;

	for (c = 0; c < nb_fields; c++) {
		cout << c << " : " << T[c].q << endl;
	}



	{
		string fname_report;

		fname_report.assign("surface");
		fname_report.append("_atlas.tex");

		{
			ofstream ost(fname_report);


			const char *title = "ATLAS of Cubic Surfaces";
			const char *author = "Anton Betten and Fatma Karaoglu";

			latex_interface L;

			//latex_head_easy(fp);
			L.head(ost,
				FALSE /* f_book */,
				TRUE /* f_title */,
				title, author,
				FALSE /*f_toc */,
				FALSE /* f_landscape */,
				FALSE /* f_12pt */,
				TRUE /*f_enlarged_page */,
				TRUE /* f_pagenumbers*/,
				NULL /* extra_praeamble */);


			int E[] = {0,1,2,3,4,5,6,9,10,13,18,45};
			int nb_possible_E = sizeof(E) / sizeof(int);
			int j;

			ost << "$$" << endl;
			ost << "\\begin{array}{|c|c|c|}" << endl;
			ost << "\\hline" << endl;
			ost << "\\ \\ q \\ \\ ";
			ost << "& \\ \\ \\mbox{Total} \\ \\ ";
			for (j = 0; j < nb_possible_E; j++) {
				ost << "&\\ \\ " << E[j] << "\\ \\ ";
			}
			ost << "\\\\" << endl;
			ost << "\\hline" << endl;
			for (c = 0; c < nb_fields; c++) {

				if (T[c].q == 2) {
					continue;
				}
				if (T[c].q == 3) {
					continue;
				}
				if (T[c].q == 5) {
					continue;
				}
				//ost << c << " & ";
				ost << T[c].q << " " << endl;

				ost << " & " << T[c].nb_total << " " << endl;

				for (j = 0; j < nb_possible_E; j++) {

					int *Idx;
					int nb;

					T[c].T_nb_E->get_class_by_value(Idx, nb, E[j], 0);

					if (nb) {

						int nb_e = E[j];
						string fname_report_tex;
						string fname_report_html;

						do_create_surface_atlas_q_e(q_max,
								T + c, nb_e, Idx, nb,
								fname_report_tex,
								verbose_level);

						fname_report_html.assign(fname_report_tex);
						chop_off_extension(fname_report_html);
						fname_report_html.append(".html");


						ost << " & ";
						ost << "%%tth: \\begin{html} <a href=\"" << fname_report_html << "\"> " << nb << " </a> \\end{html}" << endl;


						string cmd;

						cmd.assign("~/bin/tth ");
						cmd.append(fname_report_tex);
						system(cmd.c_str());
					}
					else {
						ost << " & ";
					}

					FREE_int(Idx);
				}
				ost << "\\\\" << endl;
				ost << "\\hline" << endl;
			}

			//

			ost << "\\end{array}" << endl;

			ost << "$$" << endl;

#if 0
			ost << "\\subsection*{The surface $" << SC->label_tex << "$}" << endl;


			if (SC->SO->SOP == NULL) {
				cout << "group_theoretic_activity::do_create_surface SC->SO->SOP == NULL" << endl;
				exit(1);
			}

			if (f_v) {
				cout << "group_theoretic_activity::do_create_surface "
						"before SC->SO->SOP->print_everything" << endl;
			}
			SC->SO->SOP->print_everything(ost, verbose_level);
			if (f_v) {
				cout << "group_theoretic_activity::do_create_surface "
						"after SC->SO->SOP->print_everything" << endl;
			}
#endif

			L.foot(ost);
		}
		file_io Fio;

		cout << "Written file " << fname_report << " of size "
			<< Fio.file_size(fname_report) << endl;


	}





	if (f_v) {
		cout << "do_create_surface_atlas done" << endl;
	}
}



void interface_projective::do_create_surface_atlas_q_e(int q_max,
		struct table_surfaces_field_order *T, int nb_e, int *Idx, int nb,
		std::string &fname_report_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "do_create_surface_atlas_q_e" << endl;
		cout << "do_create_surface_atlas q=" << T->q << " " << nb_e << endl;
	}

	knowledge_base K;


	number_theory_domain NT;
	sorting Sorting;
	file_io Fio;




	{
		char str[1000];

		sprintf(str, "_q%d_e%d", T->q, nb_e);
		fname_report_tex.assign("surface_atlas");
		fname_report_tex.append(str);
		fname_report_tex.append(".tex");

		{
			ofstream ost(fname_report_tex);


			string title;

			title.assign("ATLAS of Cubic Surfaces");
			sprintf(str, ", q=%d, \\#E=%d", T->q, nb_e);
			title.append(str);

			const char *author = "Anton Betten and Fatma Karaoglu";

			latex_interface L;

			//latex_head_easy(fp);
			L.head(ost,
				FALSE /* f_book */,
				TRUE /* f_title */,
				title.c_str(), author,
				FALSE /*f_toc */,
				FALSE /* f_landscape */,
				FALSE /* f_12pt */,
				TRUE /*f_enlarged_page */,
				TRUE /* f_pagenumbers*/,
				NULL /* extra_praeamble */);


			int i;

			ost << "$$" << endl;
			ost << "\\begin{array}{|c|c|c|}" << endl;
			ost << "\\hline" << endl;
			ost << "\\ \\ i \\ \\ ";
			ost << "& \\ \\ \\mbox{Orbiter Number} \\ \\ ";
			ost << "& \\ \\ \\mbox{Report} \\ \\ ";
			ost << "\\\\" << endl;
			ost << "\\hline" << endl;
			for (i = 0; i < nb; i++) {

				//ost << c << " & ";
				ost << i << " " << endl;

				ost << " & " << Idx[i] << " " << endl;


				std::string fname;

				make_fname_surface_report_pdf(fname, T->q, Idx[i]);

				ost << " & ";
				ost << "%%tth: \\begin{html} <a href=\"" << fname << "\"> report </a> \\end{html}" << endl;

				ost << "\\\\" << endl;
				ost << "\\hline" << endl;
			}

			//

			ost << "\\end{array}" << endl;

			ost << "$$" << endl;


			L.foot(ost);
		}
		file_io Fio;

		cout << "Written file " << fname_report_tex << " of size "
			<< Fio.file_size(fname_report_tex) << endl;


	}

}

void interface_projective::make_fname_surface_report_tex(std::string &fname, int q, int ocn)
{
	char str[1000];

	sprintf(str, "_q%d_iso%d_with_group", q, ocn);
	fname.assign("surface_catalogue");
	fname.append(str);
	fname.append(".tex");
}

void interface_projective::make_fname_surface_report_pdf(std::string &fname, int q, int ocn)
{
	char str[1000];

	sprintf(str, "_q%d_iso%d_with_group", q, ocn);
	fname.assign("surface_catalogue");
	fname.append(str);
	fname.append(".pdf");
}


}}

