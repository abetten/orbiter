// BLT_set_create.cpp
// 
// Anton Betten
//
// March 17, 2018
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


BLT_set_create::BLT_set_create()
{
	Descr = NULL;

	//std::string prefix;
	//std::string label_txt;
	//std::string label_tex;

	OA = NULL;
	set = NULL;
	ABC = NULL;
	f_has_group = FALSE;
	Sg = NULL;
	Blt_set_domain = NULL;
	BA = NULL;
}

BLT_set_create::~BLT_set_create()
{
	if (set) {
		FREE_lint(set);
	}
	if (ABC) {
		FREE_int(ABC);
	}
	if (Sg) {
		FREE_OBJECT(Sg);
	}
	if (Blt_set_domain) {
		FREE_OBJECT(Blt_set_domain);
	}
	if (BA) {
		FREE_OBJECT(BA);
	}
}

void BLT_set_create::init(
		orthogonal_geometry::blt_set_domain *Blt_set_domain,
		BLT_set_create_description *Descr,
		orthogonal_space_with_action *OA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;
	data_structures::string_tools ST;
	
	if (f_v) {
		cout << "BLT_set_create::init" << endl;
		}
	BLT_set_create::Descr = Descr;
	BLT_set_create::OA = OA;

	if (OA->Descr->n != 5) {
		cout << "BLT_set_create::init OA->Descr->n != 5" << endl;
		exit(1);
	}


	
	if (Descr->f_family) {
		if (f_v) {
			cout << "BLT_set_create::init family_name=" << Descr->family_name << endl;
		}

		char str[1000];
		char str_q[1000];
		f_has_group = FALSE;
		orthogonal_geometry::orthogonal_global OG;

		if (ST.stringcmp(Descr->family_name, "Linear") == 0) {
			if (f_v) {
				cout << "BLT_set_create::init creating object of family Linear" << endl;
			}
			set = NEW_lint(OA->Descr->F->q + 1);
			ABC = NEW_int(3 * (OA->Descr->F->q + 1));

			geometry::geometry_global GG;
			field_theory::finite_field *FQ;
			int q, Q;

			FQ = NEW_OBJECT(field_theory::finite_field);
			q = OA->Descr->F->q;
			Q = q * q;
			FQ->finite_field_init(Q, FALSE /* f_without_tables */, 0 /* verbose_level */);
			GG.create_Linear_BLT_set(set, ABC,
							FQ, OA->Descr->F, verbose_level);

			FREE_OBJECT(FQ);
			snprintf(str, sizeof(str), "Linear");
			snprintf(str_q, sizeof(str_q), "q%d", OA->Descr->F->q);

		}

		else if (ST.stringcmp(Descr->family_name, "Fisher") == 0) {
			if (f_v) {
				cout << "BLT_set_create::init creating object of family Fisher" << endl;
			}
			set = NEW_lint(OA->Descr->F->q + 1);
			ABC = NEW_int(3 * (OA->Descr->F->q + 1));

			geometry::geometry_global GG;
			field_theory::finite_field *FQ;
			int q, Q;

			FQ = NEW_OBJECT(field_theory::finite_field);
			q = OA->Descr->F->q;
			Q = q * q;
			FQ->finite_field_init(Q, FALSE /* f_without_tables */, 0 /* verbose_level */);
			GG.create_Fisher_BLT_set(set, ABC,
							FQ, OA->Descr->F, verbose_level);

			FREE_OBJECT(FQ);
			snprintf(str, sizeof(str), "Fisher");
			snprintf(str_q, sizeof(str_q), "q%d", OA->Descr->F->q);

		}

		else if (ST.stringcmp(Descr->family_name, "Mondello") == 0) {
			if (f_v) {
				cout << "BLT_set_create::init creating object of family Mondello" << endl;
			}
			set = NEW_lint(OA->Descr->F->q + 1);
			ABC = NEW_int(3 * (OA->Descr->F->q + 1));

			geometry::geometry_global GG;
			field_theory::finite_field *FQ;
			int q, Q;

			FQ = NEW_OBJECT(field_theory::finite_field);
			q = OA->Descr->F->q;
			Q = q * q;
			FQ->finite_field_init(Q, FALSE /* f_without_tables */, 0 /* verbose_level */);
			GG.create_Mondello_BLT_set(set, ABC,
							FQ, OA->Descr->F, verbose_level);

			FREE_OBJECT(FQ);
			snprintf(str, sizeof(str), "Mondello");
			snprintf(str_q, sizeof(str_q), "q%d", OA->Descr->F->q);

		}


		else if (ST.stringcmp(Descr->family_name, "FTWKB") == 0) {
			if (f_v) {
				cout << "BLT_set_create::init creating object of family FTWKB" << endl;
			}
			set = NEW_lint(OA->Descr->F->q + 1);
			ABC = NEW_int(3 * (OA->Descr->F->q + 1));
			OG.create_FTWKB_BLT_set(OA->O, set, ABC, verbose_level);
			// for q congruent 2 mod 3
			// a(t)= t, b(t) = 3*t^2, c(t) = 3*t^3, all t \in GF(q)
			// together with the point (0, 0, 0, 1, 0)
			snprintf(str, sizeof(str), "FTWKB");
			snprintf(str_q, sizeof(str_q), "q%d", OA->Descr->F->q);

		}
		else if (ST.stringcmp(Descr->family_name, "Kantor1") == 0) {
			if (f_v) {
				cout << "BLT_set_create::init creating object of family Kantor1" << endl;
			}
			set = NEW_lint(OA->Descr->F->q + 1);
			ABC = NEW_int(3 * (OA->Descr->F->q + 1));
			OG.create_K1_BLT_set(OA->O, set, ABC, verbose_level);
			// for a non-square m, and q=p^e
			// a(t)= t, b(t) = 0, c(t) = -m*t^p, all t \in GF(q)
			// together with the point (0, 0, 0, 1, 0)
			snprintf(str, sizeof(str), "Kantor1");
			snprintf(str_q, sizeof(str_q), "q%d", OA->Descr->F->q);
		}
		else if (ST.stringcmp(Descr->family_name, "Kantor2") == 0) {
			if (f_v) {
				cout << "BLT_set_create::init creating object of family Kantor2" << endl;
			}
			set = NEW_lint(OA->Descr->F->q + 1);
			ABC = NEW_int(3 * (OA->Descr->F->q + 1));
			OG.create_K2_BLT_set(OA->O, set, ABC, verbose_level);
			// for q congruent 2 or 3 mod 5
			// a(t)= t, b(t) = 5*t^3, c(t) = 5*t^5, all t \in GF(q)
			// together with the point (0, 0, 0, 1, 0)
			snprintf(str, sizeof(str), "Kantor2");
			snprintf(str_q, sizeof(str_q), "q%d", OA->Descr->F->q);
		}
		else if (ST.stringcmp(Descr->family_name, "LP_37_72") == 0) {
			if (f_v) {
				cout << "BLT_set_create::init creating object LP_37_72" << endl;
			}
			set = NEW_lint(OA->Descr->F->q + 1);
			OG.create_LP_37_72_BLT_set(OA->O, set, verbose_level);
			snprintf(str, sizeof(str), "LP_ago72");
			snprintf(str_q, sizeof(str_q), "q%d", OA->Descr->F->q);
		}
		else if (ST.stringcmp(Descr->family_name, "LP_37_4a") == 0) {
			if (f_v) {
				cout << "BLT_set_create::init creating object LP_37_4a" << endl;
			}
			set = NEW_lint(OA->Descr->F->q + 1);
			OG.create_LP_37_4a_BLT_set(OA->O, set, verbose_level);
			snprintf(str, sizeof(str), "LP_ago4a");
			snprintf(str_q, sizeof(str_q), "q%d", OA->Descr->F->q);
		}
		else if (ST.stringcmp(Descr->family_name, "LP_37_4b") == 0) {
			if (f_v) {
				cout << "BLT_set_create::init creating object LP_37_4b" << endl;
			}
			set = NEW_lint(OA->Descr->F->q + 1);
			OG.create_LP_37_4b_BLT_set(OA->O, set, verbose_level);
			snprintf(str, sizeof(str), "LP_ago4b");
			snprintf(str_q, sizeof(str_q), "q%d", OA->Descr->F->q);
		}
		else if (ST.stringcmp(Descr->family_name, "LP_71") == 0) {
			if (f_v) {
				cout << "BLT_set_create::init creating object LP_71" << endl;
			}
			set = NEW_lint(OA->Descr->F->q + 1);
			OG.create_Law_71_BLT_set(OA->O, set, verbose_level);
			snprintf(str, sizeof(str), "LP");
			snprintf(str_q, sizeof(str_q), "q%d", OA->Descr->F->q);
		}
		else {
			cout << "BLT_set_create::init family name is not recognized" << endl;
			exit(1);
		}

		prefix.assign(str);

		label_txt.assign(prefix);
		label_txt.append("_");
		label_txt.append(str_q);

		label_tex.assign(prefix);
		label_tex.append("\\_");
		label_tex.append(str_q);

	}


	else if (Descr->f_catalogue) {

		if (f_v) {
			cout << "BLT_set_create::init BLT set from catalogue" << endl;
		}
		int nb_iso;
		knowledge_base K;

		nb_iso = K.BLT_nb_reps(OA->Descr->F->q);
		if (Descr->iso >= nb_iso) {
			cout << "BLT_set_create::init iso >= nb_iso, "
					"this BLT set does not exist" << endl;
			exit(1);
		}

		set = NEW_lint(OA->Descr->F->q + 1);
		Lint_vec_copy(K.BLT_representative(OA->Descr->F->q, Descr->iso), set, OA->Descr->F->q + 1);

		Sg = NEW_OBJECT(groups::strong_generators);

		if (f_v) {
			cout << "BLT_set_create::init before "
					"Sg->BLT_set_from_catalogue_stabilizer" << endl;
		}

		Sg->BLT_set_from_catalogue_stabilizer(OA->A,
				OA->Descr->F, Descr->iso,
				verbose_level);
		f_has_group = TRUE;

		char str_q[1000];
		char str_iso[1000];

		snprintf(str_q, sizeof(str_q), "%d", OA->Descr->F->q);
		snprintf(str_iso, sizeof(str_iso), "%d", Descr->iso);

		prefix.assign("catalogue_q");
		prefix.append(str_q);
		prefix.append("_iso");
		prefix.append(str_iso);

		label_txt.assign("catalogue_q");
		label_txt.append(str_q);
		label_txt.append("_iso");
		label_txt.append(str_iso);

		label_tex.assign("catalogue\\_q");
		label_tex.append(str_q);
		label_tex.append("\\_iso");
		label_tex.append(str_iso);

		if (f_v) {
			cout << "BLT_set_create::init after "
					"Sg->BLT_set_from_catalogue_stabilizer" << endl;
		}
	}
	else {
		cout << "BLT_set_create::init we do not recognize "
				"the type of BLT-set" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "BLT_set_create::init set = ";
		Lint_vec_print(cout, set, OA->Descr->F->q + 1);
		cout << endl;
	}

	if (f_has_group) {
		cout << "BLT_set_create::init the stabilizer is:" << endl;
		Sg->print_generators_tex(cout);
	}


	BA = NEW_OBJECT(blt_set_with_action);

	if (f_v) {
		cout << "BLT_set_create::init before BA->init_set" << endl;
	}
	BA->init_set(
			OA->A,
			Blt_set_domain,
			set,
			Sg,
			verbose_level);
	if (f_v) {
		cout << "BLT_set_create::init after BA->init_set" << endl;
	}



	if (f_v) {
		cout << "BLT_set_create::init done" << endl;
	}
}

void BLT_set_create::apply_transformations(
		std::vector<std::string> transform_coeffs,
		std::vector<int> f_inverse_transform, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "BLT_set_create::apply_transformations done" << endl;
	}
}

void BLT_set_create::report(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "BLT_set_create::report" << endl;
	}

	string fname;

	fname.assign("BLT_");
	fname.append(label_txt);
	fname.append(".tex");
	ofstream ost(fname);

	report2(ost, verbose_level);


	if (f_v) {
		cout << "BLT_set_create::report done" << endl;
	}
}

void BLT_set_create::report2(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "BLT_set_create::report2" << endl;
	}

	int f_book = FALSE;
	int f_title = TRUE;
	char str[1000];

	string title, author, extra_praeamble;

	int f_toc = FALSE;
	int f_landscape = FALSE;
	int f_12pt = FALSE;
	int f_enlarged_page = TRUE;
	int f_pagenumbers = TRUE;
	orbiter_kernel_system::latex_interface L;

	snprintf(str, sizeof(str), "BLT-set %s", label_tex.c_str());
	title.assign(str);
	author.assign("Orbiter");

	L.head(ost, f_book, f_title,
		title,
		author,
		f_toc,
		f_landscape,
		f_12pt,
		f_enlarged_page,
		f_pagenumbers,
		extra_praeamble /* extra_praeamble */);


	OA->O->report_quadratic_form(ost, verbose_level - 1);

	if (ABC) {
		print_set_of_points_with_ABC(ost, set, OA->O->F->q + 1);
	}
	else {
		print_set_of_points(ost, set, OA->O->F->q + 1);
	}

	BA->report(ost, verbose_level);

	L.foot(ost);

	if (f_v) {
		cout << "BLT_set_create::report2 done" << endl;
	}
}

void BLT_set_create::print_set_of_points(std::ostream &ost, long int *Pts, int nb_pts)
{
	int h, I;
	int *v;
	int n = 4;

	v = NEW_int(n + 1);

	ost << "The BLT-set is:\\\\" << endl;
	for (I = 0; I < (nb_pts + 39) / 40; I++) {
		ost << "$$" << endl;
		ost << "\\begin{array}{|r|r|r|}" << endl;
		ost << "\\hline" << endl;
		ost << "i & \\mbox{Rank} & \\mbox{Point} \\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;
		for (h = 0; h < 40; h++) {
			if (I * 40 + h < nb_pts) {

				OA->O->Hyperbolic_pair->unrank_point(v, 1, Pts[I * 40 + h], 0 /* verbose_level */);

				ost << I * 40 + h << " & " << Pts[I * 40 + h] << " & ";
				Int_vec_print(ost, v, n + 1);
				ost << "\\\\" << endl;
			}
		}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;
	}
	FREE_int(v);
}

void BLT_set_create::print_set_of_points_with_ABC(std::ostream &ost, long int *Pts, int nb_pts)
{
	int h, I;
	int *v;
	int n = 4;
	int a, b, c;

	v = NEW_int(n + 1);

	ost << "The BLT-set is:\\\\" << endl;
	for (I = 0; I < (nb_pts + 39) / 40; I++) {
		ost << "$$" << endl;
		ost << "\\begin{array}{|r|r|r|r|}" << endl;
		ost << "\\hline" << endl;
		ost << "i & \\mbox{Rank} & \\mbox{Point} & (a,b,c) \\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;
		for (h = 0; h < 40; h++) {
			if (I * 40 + h < nb_pts) {

				OA->O->Hyperbolic_pair->unrank_point(v, 1, Pts[I * 40 + h], 0 /* verbose_level */);

				a = ABC[3 * (I * 40 + h) + 0];
				b = ABC[3 * (I * 40 + h) + 1];
				c = ABC[3 * (I * 40 + h) + 2];

				ost << I * 40 + h << " & " << Pts[I * 40 + h] << " & ";
				Int_vec_print(ost, v, n + 1);
				ost << " & ";
				ost << "(";
				ost << a;
				ost << ", ";
				ost << b;
				ost << ", ";
				ost << c;
				ost << ")";
				ost << "\\\\" << endl;
			}
		}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		ost << "$$" << endl;
	}
	FREE_int(v);
}



}}}

