/*
 * projective_space_global.cpp
 *
 *  Created on: Oct 9, 2021
 *      Author: betten
 */

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



void projective_space_global::map(
		projective_space_with_action *PA,
		std::string &label,
		std::string &evaluate_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::map" << endl;
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
			formula *Formula;
			Formula = (formula *) The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx1].ptr;

			PA->map(Formula,
					evaluate_text,
					verbose_level);
		}
	}
	else if (The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].object_type == t_formula) {
		cout << "symbol table entry is a formula" << endl;

		formula *Formula;
		Formula = (formula *) The_Orbiter_top_level_session->Orbiter_session->Orbiter_symbol_table->Table[idx].ptr;

		PA->map(Formula,
				evaluate_text,
				verbose_level);
	}
	else {
		cout << "symbol table entry must be either a formula or a collection" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_global::map done" << endl;
	}
}


void projective_space_global::analyze_del_Pezzo_surface(
		projective_space_with_action *PA,
		std::string &label,
		std::string &evaluate_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::analyze_del_Pezzo_surface" << endl;
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
		cout << "projective_space_global::analyze_del_Pezzo_surface done" << endl;
	}
}

void projective_space_global::analyze_del_Pezzo_surface_formula_given(
		projective_space_with_action *PA,
		formula *F,
		std::string &evaluate_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::analyze_del_Pezzo_surface_formula_given" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::analyze_del_Pezzo_surface_formula_given before PA->analyze_del_Pezzo_surface" << endl;
	}

	PA->analyze_del_Pezzo_surface(F, evaluate_text, verbose_level);

	if (f_v) {
		cout << "projective_space_global::analyze_del_Pezzo_surface_formula_given after PA->analyze_del_Pezzo_surface" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::analyze_del_Pezzo_surface_formula_given done" << endl;
	}
}




void projective_space_global::canonical_form_of_code(
		projective_space_with_action *PA,
		std::string &label, int m, int n,
		std::string &data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::canonical_form_of_code" << endl;
		cout << "m=" << m << endl;
		cout << "n=" << n << endl;
		cout << "data=" << data << endl;
	}

	if (f_v) {
		cout << "projective_space_global::canonical_form_of_code before PA->canonical_form_of_code" << endl;
	}
	PA->canonical_form_of_code(
				label, m, n,
				data,
				verbose_level);
	if (f_v) {
		cout << "projective_space_global::canonical_form_of_code after PA->canonical_form_of_code" << endl;
	}


	if (f_v) {
		cout << "projective_space_global::canonical_form_of_code done" << endl;
	}
}



void projective_space_global::do_create_surface(
		projective_space_with_action *PA,
		surface_create_description *Surface_Descr,
		surface_with_action *&Surf_A,
		surface_create *&SC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::do_create_surface" << endl;
		cout << "projective_space_global::do_create_surface verbose_level=" << verbose_level << endl;
	}

	int q;
	surface_domain *Surf;

	if (f_v) {
		cout << "projective_space_global::do_create_surface before Surface_Descr->get_q" << endl;
	}
	q = Surface_Descr->get_q();
	if (f_v) {
		cout << "projective_space_global::do_create_surface q = " << q << endl;
	}

	if (PA->q != q) {
		cout << "projective_space_global::do_create_surface PA->q != q" << endl;
		exit(1);
	}
	if (PA->n != 3) {
		cout << "projective_space_global::do_create_surface we need a three-dimensional projective space" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_global::do_create_surface before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(PA->F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_global::do_create_surface after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "projective_space_global::do_create_surface before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_global::do_create_surface after Surf_A->init" << endl;
	}


	if (f_v) {
		cout << "projective_space_global::do_create_surface before Surf_A->create_surface_and_do_report" << endl;
	}

	Surf_A->create_surface(
			Surface_Descr,
			SC,
			verbose_level);

	if (f_v) {
		cout << "projective_space_global::do_create_surface after Surf_A->create_surface_and_do_report" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::do_create_surface done" << endl;
	}
}


void projective_space_global::table_of_quartic_curves(
		projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::table_of_quartic_curves" << endl;
	}

	PA->table_of_quartic_curves(verbose_level);

	if (f_v) {
		cout << "projective_space_global::table_of_quartic_curves done" << endl;
	}
}

void projective_space_global::table_of_cubic_surfaces(
		projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::table_of_cubic_surfaces" << endl;
	}

	PA->table_of_cubic_surfaces(verbose_level);

	if (f_v) {
		cout << "projective_space_global::table_of_cubic_surfaces done" << endl;
	}
}

void projective_space_global::do_create_quartic_curve(
		projective_space_with_action *PA,
		quartic_curve_create_description *Quartic_curve_descr,
		quartic_curve_create *&QC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::do_create_quartic_curve" << endl;
		cout << "projective_space_global::do_create_quartic_curve verbose_level=" << verbose_level << endl;
	}


	if (f_v) {
		cout << "projective_space_global::do_create_quartic_curve before PA->create_quartic_curve" << endl;
	}

	PA->create_quartic_curve(
				Quartic_curve_descr,
				QC,
				verbose_level);

	if (f_v) {
		cout << "projective_space_global::do_create_quartic_curve after PA->create_quartic_curve" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::do_create_quartic_curve done" << endl;
	}
}



void projective_space_global::do_spread_classify(
		projective_space_with_action *PA,
		int k,
		poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::do_spread_classify" << endl;
	}

	PA->do_spread_classify(k,
			Control,
			verbose_level);

	if (f_v) {
		cout << "projective_space_global::do_spread_classify done" << endl;
	}
}


void projective_space_global::do_classify_semifields(
		projective_space_with_action *PA,
		semifield_classify_description *Semifield_classify_description,
		poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::do_classify_semifields" << endl;
	}


	semifield_classify_with_substructure *S;

	S = NEW_OBJECT(semifield_classify_with_substructure);

	if (f_v) {
		cout << "projective_space_global::do_classify_semifields before S->init" << endl;
	}
	S->init(
			Semifield_classify_description,
			PA,
			Control,
			verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_semifields after S->init" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::do_classify_semifields done" << endl;
	}
}


void projective_space_global::do_cheat_sheet_PG(
		projective_space_with_action *PA,
		layered_graph_draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_global::do_cheat_sheet_PG verbose_level="
				<< verbose_level << endl;
	}


	PA->cheat_sheet(O, verbose_level);


	if (f_v) {
		cout << "projective_space_global::do_cheat_sheet_PG done" << endl;
	}

}


void projective_space_global::classify_quartic_curves_nauty(
		projective_space_with_action *PA,
		std::string &fname_mask, int nb,
		std::string &fname_classification,
		canonical_form_classifier *&Classifier,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_nauty" << endl;
	}


	canonical_form_classifier_description *Descr;

	Descr = NEW_OBJECT(canonical_form_classifier_description);

	Descr->fname_mask.assign(fname_mask);
	Descr->f_fname_base_out = TRUE;
	Descr->fname_base_out.assign(fname_classification);
	Descr->PA = PA;
	Descr->f_degree = TRUE;
	Descr->degree = 4;
	Descr->nb_files = nb;
	Descr->f_algorithm_nauty = TRUE;
	Descr->f_algorithm_substructure = FALSE;


	Classifier = NEW_OBJECT(canonical_form_classifier);

	Classifier->classify(Descr, verbose_level);

	cout << "The number of types of quartic curves is " << Classifier->CB->nb_types << endl;

	Descr->Canon_substructure = Classifier;



	int idx;

	cout << "idx : ago" << endl;
	for (idx = 0; idx < Classifier->CB->nb_types; idx++) {

		canonical_form_nauty *C1;
		longinteger_object go;

		C1 = (canonical_form_nauty *) Classifier->CB->Type_extra_data[idx];

		C1->Stab_gens_quartic->group_order(go);

		cout << idx << " : " << go << endl;


	}



	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_nauty done" << endl;
	}
}


void projective_space_global::classify_quartic_curves_with_substructure(
		projective_space_with_action *PA,
		std::string &fname_mask, int nb, int substructure_size, int degree,
		std::string &fname_classification,
		canonical_form_classifier *&Classifier,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_with_substructure" << endl;
	}

	canonical_form_classifier_description *Descr;

	Descr = NEW_OBJECT(canonical_form_classifier_description);


	Descr->fname_mask.assign(fname_mask);
	Descr->f_fname_base_out = TRUE;
	Descr->fname_base_out.assign(fname_classification);
	Descr->PA = PA;
	Descr->f_degree = TRUE;
	Descr->degree = degree;
	Descr->nb_files = nb;
	Descr->f_algorithm_nauty = FALSE;
	Descr->f_algorithm_substructure = TRUE;
	Descr->substructure_size = substructure_size;


	Classifier = NEW_OBJECT(canonical_form_classifier);

	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_with_substructure before Classifier.classify" << endl;
	}
	Classifier->classify(Descr, verbose_level);
	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_with_substructure after Classifier.classify" << endl;
	}

	Descr->Canon_substructure = Classifier;


	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_with_substructure before Classifier.report" << endl;
	}
	Classifier->report(fname_classification, verbose_level);
	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_with_substructure after Classifier.report" << endl;
	}


#if 0
	cout << "The number of types of quartic curves is " << Classifier.CB->nb_types << endl;
	int idx;

	cout << "idx : ago" << endl;
	for (idx = 0; idx < Classifier.CB->nb_types; idx++) {

		canonical_form *C1;
		longinteger_object go;

		C1 = (canonical_form *) Classifier.CB->Type_extra_data[idx];

		C1->Stab_gens_quartic->group_order(go);

		cout << idx << " : " << go << endl;


	}
#endif


	if (f_v) {
		cout << "projective_space_global::classify_quartic_curves_with_substructure done" << endl;
	}
}

void projective_space_global::set_stabilizer(
		projective_space_with_action *PA,
		int intermediate_subset_size,
		std::string &fname_mask, int nb, std::string &column_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_global::set_stabilizer" << endl;
	}

#if 0
	top_level_geometry_global T;

	T.set_stabilizer_projective_space(
				PA,
				intermediate_subset_size,
				fname_mask, nb, column_label,
				verbose_level);
#endif
	substructure_classifier *SubC;

	SubC = NEW_OBJECT(substructure_classifier);

	SubC->set_stabilizer_in_any_space(
			PA->A, PA->A, PA->A->Strong_gens,
			intermediate_subset_size,
			fname_mask, nb, column_label,
			verbose_level);
	FREE_OBJECT(SubC);

	if (f_v) {
		cout << "projective_space_global::set_stabilizer done" << endl;
	}

}



void projective_space_global::conic_type(
		projective_space_with_action *PA,
		int threshold,
		std::string &set_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_global::conic_type" << endl;
	}

	long int *Pts;
	int nb_pts;

	Orbiter->Lint_vec.scan(set_text, Pts, nb_pts);


	if (f_v) {
		cout << "projective_space_global::conic_type before PA->conic_type" << endl;
	}

	PA->conic_type(Pts, nb_pts, threshold, verbose_level);

	if (f_v) {
		cout << "projective_space_global::conic_type after PA->conic_type" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::conic_type done" << endl;
	}
}

void projective_space_global::do_lift_skew_hexagon(
		projective_space_with_action *PA,
		std::string &text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon" << endl;
	}

	int *Pluecker_coords;
	int sz;

	Orbiter->Int_vec.scan(text, Pluecker_coords, sz);

	long int *Pts;
	int nb_pts;

	nb_pts = sz / 6;

	if (nb_pts * 6 != sz) {
		cout << "projective_space_global::do_lift_skew_hexagon the number of coordinates must be a multiple of 6" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "Pluecker coordinates of lines:" << endl;
		Orbiter->Int_vec.matrix_print(Pluecker_coords, nb_pts, 6);
	}

	surface_domain *Surf;
	surface_with_action *Surf_A;

	if (PA->n != 3) {
		cout << "projective_space_global::do_lift_skew_hexagon we need a three-dimensional projective space" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(PA->F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon after Surf_A->init" << endl;
	}




	int i;

	Pts = NEW_lint(nb_pts);

	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Surf_A->Surf->Klein->Pluecker_to_line_rk(Pluecker_coords + i * 6, 0 /*verbose_level*/);
	}

	if (nb_pts != 6) {
		cout << "projective_space_global::do_lift_skew_hexagon nb_pts != 6" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "lines:" << endl;
		Orbiter->Lint_vec.print(cout, Pts, 6);
		cout << endl;
	}


	std::vector<std::vector<long int> > Double_sixes;

	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon before Surf_A->complete_skew_hexagon" << endl;
	}

	Surf_A->complete_skew_hexagon(Pts, Double_sixes, verbose_level);

	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon after Surf_A->complete_skew_hexagon" << endl;
	}

	cout << "We found " << Double_sixes.size() << " double sixes. They are:" << endl;
	for (i = 0; i < Double_sixes.size(); i++) {
		cout << Double_sixes[i][0] << ",";
		cout << Double_sixes[i][1] << ",";
		cout << Double_sixes[i][2] << ",";
		cout << Double_sixes[i][3] << ",";
		cout << Double_sixes[i][4] << ",";
		cout << Double_sixes[i][5] << ",";
		cout << Double_sixes[i][6] << ",";
		cout << Double_sixes[i][7] << ",";
		cout << Double_sixes[i][8] << ",";
		cout << Double_sixes[i][9] << ",";
		cout << Double_sixes[i][10] << ",";
		cout << Double_sixes[i][11] << "," << endl;

	}

	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon done" << endl;
	}
}


void projective_space_global::do_lift_skew_hexagon_with_polarity(
		projective_space_with_action *PA,
		std::string &polarity_36,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity" << endl;
	}

	int *Polarity36;
	int sz1;

	Orbiter->Int_vec.scan(polarity_36, Polarity36, sz1);

	if (sz1 != 36) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity I need exactly 36 coefficients for the polarity" << endl;
		exit(1);
	}


	surface_domain *Surf;
	surface_with_action *Surf_A;

	if (PA->n != 3) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity we need a three-dimensional projective space" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(PA->F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, PA, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_global::do_lift_skew_hexagon_with_polarity after Surf_A->init" << endl;
	}




	std::vector<std::vector<long int> > Double_sixes;

	int Pluecker_coords[36];
	int alpha, beta;
	int i, j;

	Orbiter->Int_vec.zero(Pluecker_coords, 36);
	// a1 = 1,0,0,0,0,0
	Pluecker_coords[0] = 1;

	for (alpha = 1; alpha < PA->F->q; alpha++) {



		for (beta = 1; beta < PA->F->q; beta++) {

			// a2 = 0,beta,0,alpha,alpha,0

			Pluecker_coords[6 + 1] = beta;
			Pluecker_coords[6 + 3] = alpha;
			Pluecker_coords[6 + 4] = alpha;

			// a3 = 0,beta,0,alpha,alpha,0

			Pluecker_coords[12 + 1] = alpha;
			Pluecker_coords[12 + 2] = beta;


			for (j = 0; j < 3; j++) {
				Surf->F->mult_matrix_matrix(Pluecker_coords + j * 6, Polarity36,
						Pluecker_coords + 18 + j * 6, 1, 6, 6, 0 /* verbose_level */);
			}

			int nb_pts;

			nb_pts = 6;

			if (f_v) {
				cout << "Pluecker coordinates of lines:" << endl;
				Orbiter->Int_vec.matrix_print(Pluecker_coords, nb_pts, 6);
			}


			long int *Pts;


			Pts = NEW_lint(nb_pts);

			for (i = 0; i < nb_pts; i++) {
				Pts[i] = Surf_A->Surf->Klein->Pluecker_to_line_rk(Pluecker_coords + i * 6, 0 /*verbose_level*/);
			}

			if (nb_pts != 6) {
				cout << "projective_space_global::do_lift_skew_hexagon_with_polarity nb_pts != 6" << endl;
				exit(1);
			}

			if (f_v) {
				cout << "lines:" << endl;
				Orbiter->Lint_vec.print(cout, Pts, 6);
				cout << endl;
			}


			string label;
			char str[1000];

			sprintf(str, "alpha=%d beta=%d", alpha, beta);

			label.assign(str);

			if (f_v) {
				cout << "projective_space_global::do_lift_skew_hexagon_with_polarity before Surf_A->complete_skew_hexagon_with_polarity" << endl;
			}

			Surf_A->complete_skew_hexagon_with_polarity(label, Pts, Polarity36, Double_sixes, verbose_level);

			if (f_v) {
				cout << "projective_space_global::do_lift_skew_hexagon_with_polarity after Surf_A->complete_skew_hexagon_with_polarity" << endl;
			}

			FREE_lint(Pts);


		}

	}



	cout << "We found " << Double_sixes.size() << " double sixes. They are:" << endl;
	for (i = 0; i < Double_sixes.size(); i++) {
		cout << Double_sixes[i][0] << ",";
		cout << Double_sixes[i][1] << ",";
		cout << Double_sixes[i][2] << ",";
		cout << Double_sixes[i][3] << ",";
		cout << Double_sixes[i][4] << ",";
		cout << Double_sixes[i][5] << ",";
		cout << Double_sixes[i][6] << ",";
		cout << Double_sixes[i][7] << ",";
		cout << Double_sixes[i][8] << ",";
		cout << Double_sixes[i][9] << ",";
		cout << Double_sixes[i][10] << ",";
		cout << Double_sixes[i][11] << "," << endl;

	}

	if (f_v) {
		cout << "projective_space_global::do_lift_do_lift_skew_hexagon_with_polarityskew_hexagon done" << endl;
	}
}


void projective_space_global::do_classify_arcs(
		projective_space_with_action *PA,
		arc_generator_description *Arc_generator_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::do_classify_arcs" << endl;
	}

#if 0
	Arc_generator_description->F = LG->F;
	Arc_generator_description->LG = LG;
	Arc_generator_description->Control = Descr->Control;

	if (Arc_generator_description->n != LG->A2->matrix_group_dimension()) {
		cout << "projective_space_global::do_classify_arcs the dimensions don't match" << endl;
		exit(1);
	}
#endif

	{
		arc_generator *Gen;

		Gen = NEW_OBJECT(arc_generator);



		if (f_v) {
			cout << "projective_space_global::do_classify_arcs before Gen->init" << endl;
		}
		Gen->init(
				Arc_generator_description,
				PA,
				PA->A->Strong_gens,
				verbose_level);

		if (f_v) {
			cout << "projective_space_global::do_classify_arcs after Gen->init" << endl;
		}



		if (f_v) {
			cout << "projective_space_global::do_classify_arcs before Gen->main" << endl;
		}
		Gen->main(verbose_level);
		if (f_v) {
			cout << "projective_space_global::do_classify_arcs after Gen->main" << endl;
		}


		FREE_OBJECT(Gen);
	}


	if (f_v) {
		cout << "projective_space_global::do_classify_arcs done" << endl;
	}
}


void projective_space_global::do_classify_cubic_curves(
		projective_space_with_action *PA,
		arc_generator_description *Arc_generator_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves" << endl;
	}



	cubic_curve *CC;

	CC = NEW_OBJECT(cubic_curve);

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves before CC->init" << endl;
	}
	CC->init(PA->F, verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves after CC->init" << endl;
	}


	cubic_curve_with_action *CCA;

	CCA = NEW_OBJECT(cubic_curve_with_action);

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves before CCA->init" << endl;
	}
	CCA->init(CC, PA->A, verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves after CCA->init" << endl;
	}


	classify_cubic_curves *CCC;

	CCC = NEW_OBJECT(classify_cubic_curves);


	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves before CCC->init" << endl;
	}
	CCC->init(
			PA,
			CCA,
			Arc_generator_description,
			verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves after CCC->init" << endl;
	}

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves before CCC->compute_starter" << endl;
	}
	CCC->compute_starter(verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves after CCC->compute_starter" << endl;
	}

#if 0
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves before CCC->test_orbits" << endl;
	}
	CCC->test_orbits(verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves after CCC->test_orbits" << endl;
	}
#endif

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves before CCC->do_classify" << endl;
	}
	CCC->do_classify(verbose_level);
	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves after CCC->do_classify" << endl;
	}


	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves creating cheat sheet" << endl;
	}
	char fname[1000];
	char title[1000];
	char author[1000];
	snprintf(title, 1000, "Cubic Curves in PG$(2,%d)$", PA->F->q);
	strcpy(author, "");
	snprintf(fname, 1000, "Cubic_curves_q%d.tex", PA->F->q);

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

		fp << "\\subsection*{" << title << "}" << endl;

		if (f_v) {
			cout << "projective_space_global::do_classify_cubic_curves before CCC->report" << endl;
		}
		CCC->report(fp, verbose_level);
		if (f_v) {
			cout << "projective_space_global::do_classify_cubic_curves after CCC->report" << endl;
		}

		L.foot(fp);
	}

	file_io Fio;

	cout << "Written file " << fname << " of size "
		<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves writing cheat sheet on "
				"cubic curves done" << endl;
	}


	if (f_v) {
		cout << "projective_space_global::do_classify_cubic_curves done" << endl;
	}
}





}}




