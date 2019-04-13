/*
 * classify_cubic_curves.cpp
 *
 *  Created on: Mar 7, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


// global data:

int t0; // the system time when the program started



int main(int argc, const char **argv)
{

	int i;
	int verbose_level = 0;
	int f_q = FALSE;
	int q = 0;

	t0 = os_ticks();

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
	}
	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
	}

	const char *starter_directory_name = "";
	char base_fname[1000];

	sprintf(base_fname, "cubic_curves_%d", q);

	int f_v = (verbose_level >= 1);

	int f_semilinear = FALSE;
	number_theory_domain NT;

	if (!NT.is_prime(q)) {
		f_semilinear = TRUE;
	}
	finite_field *F;

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	cubic_curve *CC;

	CC = NEW_OBJECT(cubic_curve);

	CC->init(F, verbose_level);


	cubic_curve_with_action *CCA;

	CCA = NEW_OBJECT(cubic_curve_with_action);

	CCA->init(CC, f_semilinear, verbose_level);

	classify_cubic_curves *CCC;

	CCC = NEW_OBJECT(classify_cubic_curves);


	CCC->init(CCA,
			starter_directory_name,
			base_fname,
			argc, argv,
			verbose_level);

	CCC->compute_starter(verbose_level);

	CCC->test_orbits(verbose_level);

	CCC->do_classify(verbose_level);

	int f_with_stabilizers = TRUE;


	if (f_v) {
		cout << "surface_classify writing cheat sheet "
				"on cubic curves" << endl;
		}
	{
	char fname[1000];
	char title[1000];
	char author[1000];
	int *Pts_on_curve;
	int *inflexion_Pts;
	int *singular_Pts;
	int *type;

	Pts_on_curve = NEW_int(CCA->CC->P->N_points);
	inflexion_Pts = NEW_int(CCA->CC->P->N_points);
	singular_Pts = NEW_int(CCA->CC->P->N_points);
	type = NEW_int(CCA->CC->P->N_lines);

	sprintf(title, "Cubic Curves in PG$(2,%d)$", q);
	sprintf(author, "");
	sprintf(fname, "Cubic_curves_q%d.tex", q);

		{
		ofstream fp(fname);
		latex_interface L;

		//latex_head_easy(fp);
		L.head(fp,
			FALSE /* f_book */,
			TRUE /* f_title */,
			title, author,
			FALSE /*f_toc */,
			FALSE /* f_landscape */,
			FALSE /* f_12pt */,
			TRUE /*f_enlarged_page */,
			TRUE /* f_pagenumbers*/,
			NULL /* extra_praeamble */);

#if 0
		CCC->Curves->print_latex(fp, title, f_with_stabilizers);
#else
		fp << "\\subsection*{" << title << "}" << endl;



		{

		fp << "The order of the group is ";
		CCC->Curves->go.print_not_scientific(fp);
		fp << "\\\\" << endl;

		fp << "\\bigskip" << endl;
		}

		fp << "The group has " << CCC->Curves->nb_orbits
				<< " orbits: \\\\" << endl;

		int i;
		longinteger_domain D;
		longinteger_object go1, ol, Ol;
		Ol.create(0);

		vector<string> References;
		int *Ago;
		int *Nb_points;
		int *Nb_singular_points;
		int *Nb_inflexions;
		Ago = NEW_int(CCC->Curves->nb_orbits);
		Nb_points = NEW_int(CCC->Curves->nb_orbits);
		Nb_singular_points = NEW_int(CCC->Curves->nb_orbits);
		Nb_inflexions = NEW_int(CCC->Curves->nb_orbits);



		for (i = 0; i < CCC->Curves->nb_orbits; i++) {

			if (f_v) {
				cout << "Curve " << i << " / "
						<< CCC->Curves->nb_orbits << ": "
						"verbose_level=" << verbose_level << endl;
			}

			CCC->Curves->Orbit[i].gens->group_order(go1);

			if (f_v) {
				cout << "stab order " << go1 << endl;
				}

			Ago[i] = go1.as_int();

			D.integral_division_exact(CCC->Curves->go, go1, ol);

			if (f_v) {
				cout << "orbit length " << ol << endl;
				}

			int *data;
			int *eqn;
			int nb_pts_on_curve;
			int nb_singular_pts;
			int nb_inflection_pts;

			data = CCC->Curves->Rep + i * CCC->Curves->representation_sz;
			eqn = data + 9;

			fp << "\\subsection*{Curve " << i << " / "
					<< CCC->Curves->nb_orbits << "}" << endl;
			//fp << "$" << i << " / " << CCC->Curves->nb_orbits << "$ $" << endl;

			fp << "$";
			int_set_print_tex_for_inline_text(fp,
					data,
					9 /*CCC->Curves->representation_sz*/);
			fp << "_{";
			go1.print_not_scientific(fp);
			fp << "}$ orbit length $";
			ol.print_not_scientific(fp);
			fp << "$\\\\" << endl;


#if 0
			int_vec_zero(eqn, 10);
			// y = x^3 or X^3 - YZ^2
			eqn[0] = 1;
			eqn[8] = F->minus_one();
			eqn[2] = 0;
//0 & X^3 & ( 3, 0, 0 )
//1 & Y^3 & ( 0, 3, 0 )
//2 & Z^3 & ( 0, 0, 3 )
//3 & X^2Y & ( 2, 1, 0 )
//4 & X^2Z & ( 2, 0, 1 )
//5 & XY^2 & ( 1, 2, 0 )
//6 & Y^2Z & ( 0, 2, 1 )
//7 & XZ^2 & ( 1, 0, 2 )
//8 & YZ^2 & ( 0, 1, 2 )
//9 & XYZ & ( 1, 1, 1 )
#endif
#if 0
			int_vec_zero(eqn, 10);
			// y = x^3 + x + 3
			eqn[0] = 1;
			eqn[2] = 3;
			eqn[6] = 10;
			eqn[7] = 1;
#endif


			fp << "\\begin{eqnarray*}" << endl;
			fp << "&&";


			CCA->CC->Poly->enumerate_points(eqn,
					Pts_on_curve, nb_pts_on_curve,
					verbose_level - 2);

			Nb_points[i] = nb_pts_on_curve;


			CC->Poly->print_equation_with_line_breaks_tex(fp,
					eqn,
					5 /* nb_terms_per_line */,
					"\\\\\n&&");
			fp << "\\end{eqnarray*}" << endl;

			fp << "The curve has " << nb_pts_on_curve
					<< " points.\\\\" << endl;


			CC->compute_singular_points(
					eqn,
					Pts_on_curve, nb_pts_on_curve,
					singular_Pts, nb_singular_pts,
					verbose_level - 2);

			fp << "The curve has " << nb_singular_pts
					<< " singular points.\\\\" << endl;
			Nb_singular_points[i] = nb_singular_pts;


			CC->compute_inflexion_points(
					eqn,
					Pts_on_curve, nb_pts_on_curve,
					inflexion_Pts, nb_inflection_pts,
					verbose_level - 2);


			Nb_inflexions[i] = nb_inflection_pts;

			fp << "The curve has " << nb_inflection_pts << " inflexion points: $";
			int_vec_print(fp, inflexion_Pts, nb_inflection_pts);
			fp << "$\\\\" << endl;


			CCA->CC->P->line_intersection_type(
					Pts_on_curve, nb_pts_on_curve /* set_size */,
					type, 0 /*verbose_level*/);
			// type[N_lines]

			fp << "The line type is $";
			classify C;
			C.init(type, CCA->CC->P->N_lines, FALSE, 0);
			C.print_naked_tex(fp, TRUE /* f_backwards*/);
			fp << ".$ \\\\" << endl;


			if (f_with_stabilizers) {
				//ost << "Strong generators are:" << endl;
				CCC->Curves->Orbit[i].gens->print_generators_tex(fp);
				D.add_in_place(Ol, ol);
				}

#if 1
			if (nb_inflection_pts == 3) {
				int Basis[9];
				int Basis_t[9];
				int Basis_inv[9];
				int transformed_eqn[10];

				CC->P->unrank_point(Basis, inflexion_Pts[0]);
				CC->P->unrank_point(Basis + 3, inflexion_Pts[1]);

				CC->P->F->extend_basis(2, 3, Basis,
					verbose_level);

				//CC->P->unrank_point(Basis + 6, inflexion_Pts[2]);
				CC->F->transpose_matrix(Basis, Basis_t, 3, 3);
				CC->F->invert_matrix(Basis, Basis_inv, 3);
				CC->Poly->substitute_linear(eqn, transformed_eqn,
						Basis /* int *Mtx_inv */, 0 /* verbose_level */);


				fp << "The transformed equation is:\\\\" << endl;
				fp << "\\begin{eqnarray*}" << endl;
				fp << "&&";


				CCA->CC->Poly->enumerate_points(transformed_eqn,
						Pts_on_curve, nb_pts_on_curve,
						verbose_level - 2);


				CC->Poly->print_equation_with_line_breaks_tex(fp,
						transformed_eqn,
						5 /* nb_terms_per_line */,
						"\\\\\n&&");
				fp << "\\end{eqnarray*}" << endl;

				fp << "The transformed curve has " << nb_pts_on_curve
						<< " points.\\\\" << endl;

				CC->compute_singular_points(
						transformed_eqn,
						Pts_on_curve, nb_pts_on_curve,
						singular_Pts, nb_singular_pts,
						verbose_level - 2);

				fp << "The curve has " << nb_singular_pts
						<< " singular points.\\\\" << endl;


				CC->compute_inflexion_points(
						transformed_eqn,
						Pts_on_curve, nb_pts_on_curve,
						inflexion_Pts, nb_inflection_pts,
						verbose_level - 2);

				fp << "The transformed curve has " << nb_inflection_pts
						<< " inflexion points: $";
				int_vec_print(fp, inflexion_Pts, nb_inflection_pts);
				fp << "$\\\\" << endl;



			}
#endif


		} // next i
		fp << "The overall number of objects is: " << Ol << "\\\\" << endl;




		fp << "summary of the stabilizer orders:\\\\" << endl;


		for (i = 0; i < CCC->Curves->nb_orbits; i++) {
			string ref("");
			References.push_back(ref);
		}


		int *Iso_type1;
		int *Iso_type2;
		int *Iso_type3;
		int *Iso_typeE;
		int *Iso_typeH;
		int *Iso_typeG;
		int e, c, d;

		Iso_type1 = NEW_int(F->q);
		Iso_type2 = NEW_int(F->q);
		Iso_type3 = NEW_int(F->q);
		Iso_typeE = NEW_int(F->q);
		Iso_typeH = NEW_int(F->q);
		Iso_typeG = NEW_int(F->q * F->q);
		CCC->family1_recognize(Iso_type1, verbose_level - 1);
		CCC->family2_recognize(Iso_type2, verbose_level - 1);
		CCC->family3_recognize(Iso_type3, verbose_level - 1);
		CCC->familyE_recognize(Iso_typeE, verbose_level - 1);
		CCC->familyH_recognize(Iso_typeH, verbose_level - 1);
		CCC->familyG_recognize(Iso_typeG, verbose_level - 1);

		fp << "Families 1, 2, 3, E, H: \\\\" << endl;
		for (e = 0; e < F->q; e++) {
			fp << "e=" << e
					<< " iso1=" << Iso_type1[e]
					<< " iso2=" << Iso_type2[e]
					<< " iso3=" << Iso_type2[e]
					<< " isoE=" << Iso_typeE[e]
					<< " isoH=" << Iso_typeH[e]
					<< " \\\\" << endl;
		}
		for (c = 1; c < F->q; c++) {
			for (d = 1; d < F->q; d++) {
				fp << "c=" << c << " d=" << d
						<< " isoG=" << Iso_typeG[c * F->q + d]
						<< " \\\\" << endl;
			}
		}
		for (e = 0; e < F->q; e++) {
			if (Iso_type1[e] != -1) {
				string ref;
				char str[1000];
				ref = References[Iso_type1[e]];
				if (strlen(ref.c_str())) {
					ref.append(",");
				}
				sprintf(str, "F1_{%d}", e);
				ref.append(str);
				References[Iso_type1[e]] = ref;
			}
		}
		for (e = 0; e < F->q; e++) {
			if (Iso_type2[e] != -1) {
				string ref;
				char str[1000];
				ref = References[Iso_type2[e]];
				if (strlen(ref.c_str())) {
					ref.append(",");
				}
				sprintf(str, "F2_{%d}", e);
				ref.append(str);
				References[Iso_type2[e]] = ref;
			}
		}
		for (e = 0; e < F->q; e++) {
			if (Iso_type3[e] != -1) {
				string ref;
				char str[1000];
				ref = References[Iso_type3[e]];
				if (strlen(ref.c_str())) {
					ref.append(",");
				}
				sprintf(str, "F3_{%d}", e);
				ref.append(str);
				References[Iso_type3[e]] = ref;
			}
		}
		for (e = 0; e < F->q; e++) {
			if (Iso_typeE[e] != -1) {
				string ref;
				char str[1000];
				ref = References[Iso_typeE[e]];
				if (strlen(ref.c_str())) {
					ref.append(",");
				}
				sprintf(str, "E_{%d}", e);
				ref.append(str);
				References[Iso_typeE[e]] = ref;
			}
		}
		for (e = 0; e < F->q; e++) {
			if (Iso_typeH[e] != -1) {
				string ref;
				char str[1000];
				ref = References[Iso_typeH[e]];
				if (strlen(ref.c_str())) {
					ref.append(",");
				}
				sprintf(str, "H_{%d}", e);
				ref.append(str);
				References[Iso_typeH[e]] = ref;
			}
		}
		for (c = 1; c < F->q; c++) {
			for (d = 1; d < F->q; d++) {
				int iso;

				iso = Iso_typeG[c * F->q + d];
				if (iso == -1) {
					continue;
				}
				string ref;
				char str[1000];
				ref = References[iso];
				if (strlen(ref.c_str())) {
					ref.append(",");
				}
				sprintf(str, "G_{%d,%d}", c,d);
				ref.append(str);
				References[iso] = ref;
			}
		}



		classify C;

		C.init(Ago, CCC->Curves->nb_orbits, FALSE, 0);
		fp << "Distribution: $(";
		C.print_naked_tex(fp, TRUE /* f_backwards */);
		fp << ")$\\\\" << endl;


		fp << "$$" << endl;
		fp << "\\begin{array}{|c||c|c|c|c|c|}";
		fp << "\\hline";
		fp << "\\mbox{Curve} & ";
		fp << "\\mbox{Ago} & ";
		fp << "\\mbox{Pts} & ";
		fp << "\\mbox{s. Pts} & ";
		fp << "\\mbox{Infl} & ";
		fp << "\\mbox{References} \\\\";
		fp << "\\hline";
		for (i = 0; i < CCC->Curves->nb_orbits; i++) {
			fp << i;
			fp << " & " << Ago[i];
			fp << " & " << Nb_points[i];
			fp << " & " << Nb_singular_points[i];
			fp << " & " << Nb_inflexions[i];
			fp << " & " << References[i];
			fp << "\\\\";
		}
		fp << "\\hline";
		fp << "\\end{array}" << endl;
		fp << "$$" << endl;

		fp << "with canonical forms " << endl;
		fp << "\\begin{eqnarray*}" << endl;
		fp << "F1_e &=& X^2Y + XY^2 + eZ^3 \\\\" << endl;
		fp << "F2_e &=& X^2Y + XY^2 + XYZ + eZ^3 \\\\" << endl;
		fp << "F3_e &=& XYZ + e(X + Y + Z)^3 \\\\" << endl;
		fp << "E_d &=& Z^2Y + X^3 - dY^3 \\\\" << endl;
		fp << "H_e &=& Z^2Y + X^3 + eXY^2 \\\\" << endl;
		fp << "G_{c,d} &=&  Z^2Y + X^3 + cXY^2 + dY^3 \\\\" << endl;
		fp << "\\end{eqnarray*}" << endl;
		fp << "for $c,d,e \\in {\\mathbb F}_{" << F->q << "}$ \\\\" << endl;

		FREE_int(Iso_type1);
		FREE_int(Iso_type2);
		FREE_int(Iso_type3);
		FREE_int(Iso_typeE);
		FREE_int(Iso_typeH);
		FREE_int(Iso_typeG);
		FREE_int(Ago);
		FREE_int(Nb_points);
		FREE_int(Nb_singular_points);
		FREE_int(Nb_inflexions);

#endif

		L.foot(fp);
		FREE_int(Pts_on_curve);
		FREE_int(inflexion_Pts);
		FREE_int(singular_Pts);
		FREE_int(type);
		}
	cout << "Written file " << fname << " of size "
			<< file_size(fname) << endl;
	}
	if (f_v) {
		cout << "surface_classify writing cheat sheet on "
				"double sixes done" << endl;
		}




	the_end(t0);
}
