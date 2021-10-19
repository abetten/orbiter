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

		surface_domain_high_level SH;

		SH.do_cubic_surface_properties(
				PA,
				Descr->table_of_cubic_surfaces_compute_fname_csv,
				Descr->table_of_cubic_surfaces_compute_defining_q,
				Descr->table_of_cubic_surfaces_compute_column_offset,
				verbose_level);
	}
	else if (Descr->f_cubic_surface_properties_analyze) {


		surface_domain_high_level SH;

		SH.do_cubic_surface_properties_analyze(
				PA,
				Descr->cubic_surface_properties_fname_csv,
				Descr->cubic_surface_properties_defining_q,
				verbose_level);
	}
	else if (Descr->f_canonical_form_of_code) {

		projective_space_global G;

		G.canonical_form_of_code(
				PA,
				Descr->canonical_form_of_code_label,
				Descr->canonical_form_of_code_m, Descr->canonical_form_of_code_n,
				Descr->canonical_form_of_code_text,
				verbose_level);

	}
	else if (Descr->f_map) {

		projective_space_global G;

		G.map(
				PA,
				Descr->map_label,
				Descr->map_parameters,
				verbose_level);

	}
	else if (Descr->f_analyze_del_Pezzo_surface) {

		projective_space_global G;

		G.analyze_del_Pezzo_surface(
				PA,
				Descr->analyze_del_Pezzo_surface_label,
				Descr->analyze_del_Pezzo_surface_parameters,
				verbose_level);

	}

	else if (Descr->f_cheat_sheet_for_decomposition_by_element_PG) {

		PA->do_cheat_sheet_for_decomposition_by_element_PG(
				Descr->decomposition_by_element_power,
				Descr->decomposition_by_element_data,
				Descr->decomposition_by_element_fname,
				verbose_level);

	}

	else if (Descr->f_decomposition_by_subgroup) {

		PA->do_cheat_sheet_for_decomposition_by_subgroup(
				Descr->decomposition_by_subgroup_label,
				Descr->decomposition_by_subgroup_Descr,
				verbose_level);

	}


	else if (Descr->f_define_object) {
		cout << "-define_object " << Descr->define_object_label << endl;
		Descr->Object_Descr->print();

		combinatorial_object_create *CombObj;

		CombObj = NEW_OBJECT(combinatorial_object_create);

		CombObj->init(Descr->Object_Descr, PA->P, verbose_level);

		orbiter_symbol_table_entry *Symb;

		Symb = NEW_OBJECT(orbiter_symbol_table_entry);

		Symb->init_combinatorial_object(Descr->define_object_label, CombObj, verbose_level);
		if (f_v) {
			cout << "before Orbiter->add_symbol_table_entry " << Descr->define_surface_label << endl;
		}
		Orbiter->add_symbol_table_entry(Descr->define_surface_label, Symb, verbose_level);

	}
	else if (Descr->f_define_surface) {

		cout << "f_define_surface label = " << Descr->define_surface_label << endl;

		surface_with_action *Surf_A;
		surface_create *SC;

		projective_space_global G;

		G.do_create_surface(
			PA,
			Descr->Surface_Descr,
			Surf_A,
			SC,
			verbose_level);

		orbiter_symbol_table_entry *Symb;

		Symb = NEW_OBJECT(orbiter_symbol_table_entry);

		Symb->init_cubic_surface(Descr->define_surface_label, SC, verbose_level);
		if (f_v) {
			cout << "before Orbiter->add_symbol_table_entry " << Descr->define_surface_label << endl;
		}
		Orbiter->add_symbol_table_entry(Descr->define_surface_label, Symb, verbose_level);


		//FREE_OBJECT(SC);
		//FREE_OBJECT(Surf_A);
	}

	else if (Descr->f_table_of_quartic_curves) {

		cout << "table_of_quartic_curves" << endl;

		projective_space_global G;

		G.table_of_quartic_curves(PA, verbose_level);
	}

	else if (Descr->f_table_of_cubic_surfaces) {

		cout << "table_of_cubic_surfaces" << endl;

		projective_space_global G;

		G.table_of_cubic_surfaces(PA, verbose_level);
	}

	else if (Descr->f_define_quartic_curve) {

		cout << "f_define_quartic_curve label = " << Descr->f_define_quartic_curve << endl;

		quartic_curve_create *QC;

		projective_space_global G;

		G.do_create_quartic_curve(
			PA,
			Descr->Quartic_curve_descr,
			QC,
			verbose_level);

		orbiter_symbol_table_entry *Symb;

		Symb = NEW_OBJECT(orbiter_symbol_table_entry);

		Symb->init_quartic_curve(Descr->define_quartic_curve_label, QC, verbose_level);
		if (f_v) {
			cout << "before Orbiter->add_symbol_table_entry " << Descr->define_surface_label << endl;
		}
		Orbiter->add_symbol_table_entry(Descr->define_quartic_curve_label, Symb, verbose_level);


		//FREE_OBJECT(SC);
	}

	// surfaces:


	else if (Descr->f_classify_surfaces_with_double_sixes) {

		surface_domain_high_level SH;
		surface_classify_wedge *SCW;


		SH.classify_surfaces_with_double_sixes(
				PA,
				Descr->classify_surfaces_with_double_sixes_control,
				SCW,
				verbose_level);

		orbiter_symbol_table_entry *Symb;

		Symb = NEW_OBJECT(orbiter_symbol_table_entry);

		Symb->init_classification_of_cubic_surfaces_with_double_sixes(Descr->classify_surfaces_with_double_sixes_label, SCW, verbose_level);
		if (f_v) {
			cout << "before Orbiter->add_symbol_table_entry " << Descr->classify_surfaces_with_double_sixes_label << endl;
		}
		Orbiter->add_symbol_table_entry(Descr->classify_surfaces_with_double_sixes_label, Symb, verbose_level);

	}

	else if (Descr->f_classify_surfaces_through_arcs_and_two_lines) {

		if (!Descr->f_control_six_arcs) {
			cout << "please use option -control_six_arcs <description> -end" << endl;
			exit(1);
		}
		surface_domain_high_level SH;

		SH.do_classify_surfaces_through_arcs_and_two_lines(
				PA,
				Descr->Control_six_arcs,
				Descr->f_test_nb_Eckardt_points, Descr->nb_E,
				verbose_level);
	}

	else if (Descr->f_classify_surfaces_through_arcs_and_trihedral_pairs) {
		if (!Descr->f_trihedra1_control) {
			cout << "please use option -trihedra1_control <description> -end" << endl;
			exit(1);
		}
		if (!Descr->f_trihedra2_control) {
			cout << "please use option -trihedra2_control <description> -end" << endl;
			exit(1);
		}
		if (!Descr->f_control_six_arcs) {
			cout << "please use option -control_six_arcs <description> -end" << endl;
			exit(1);
		}
		surface_domain_high_level SH;

		SH.do_classify_surfaces_through_arcs_and_trihedral_pairs(
				PA,
				Descr->Trihedra1_control, Descr->Trihedra2_control,
				Descr->Control_six_arcs,
				Descr->f_test_nb_Eckardt_points, Descr->nb_E,
				verbose_level);
	}
	else if (Descr->f_sweep_4) {

		surface_domain_high_level SH;


		SH.do_sweep_4(
				PA,
				Descr->sweep_4_surface_description,
				Descr->sweep_4_fname,
				verbose_level);
	}
	else if (Descr->f_sweep_4_27) {

		surface_domain_high_level SH;


		SH.do_sweep_4_27(
				PA,
				Descr->sweep_4_27_surface_description,
				Descr->sweep_4_27_fname,
				verbose_level);
	}

#if 0
	else if (Descr->f_create_surface) {
		if (!Descr->f_control_six_arcs) {
			cout << "please use option -control_six_arcs <description> -end" << endl;
			exit(1);
		}

		surface_domain_high_level SH;

		SH.do_create_surface(
				PA,
				Descr->surface_description, Descr->Control_six_arcs,
				verbose_level);
	}
#endif
	else if (Descr->f_six_arcs_not_on_conic) {
		if (!Descr->f_control_six_arcs) {
			cout << "please use option -control_six_arcs <description> -end" << endl;
			exit(1);
		}
		surface_domain_high_level SH;

		SH.do_six_arcs(
				PA,
				Descr->Control_six_arcs,
				Descr->f_filter_by_nb_Eckardt_points,
				Descr->nb_Eckardt_points,
				verbose_level);
	}
	else if (Descr->f_make_gilbert_varshamov_code) {

		coding_theory_domain Coding;

		Coding.make_gilbert_varshamov_code(
				Descr->make_gilbert_varshamov_code_n,
				Descr->make_gilbert_varshamov_code_n - (PA->P->n + 1),
				Descr->make_gilbert_varshamov_code_d,
				PA->P->F->q,
				PA->P, verbose_level);
	}

	else if (Descr->f_spread_classify) {

		projective_space_global G;

		G.do_spread_classify(PA,
				Descr->spread_classify_k,
				Descr->spread_classify_Control,
				verbose_level);
	}
	else if (Descr->f_classify_semifields) {

		projective_space_global G;

		G.do_classify_semifields(
				PA,
				Descr->Semifield_classify_description,
				Descr->Semifield_classify_Control,
				verbose_level);

	}
	else if (Descr->f_cheat_sheet) {

		layered_graph_draw_options *O;

		if (Orbiter->f_draw_options) {
			O = Orbiter->draw_options;
		}
		else {
			cout << "please use -draw_options .. -end" << endl;
			exit(1);
		}

		projective_space_global G;

		G.do_cheat_sheet_PG(
				PA,
				O,
				verbose_level);
	}
	else if (Descr->f_classify_quartic_curves_nauty) {

		canonical_form_classifier *Classifier;

		projective_space_global G;

		G.classify_quartic_curves_nauty(PA,
				Descr->classify_quartic_curves_nauty_fname_mask,
				Descr->classify_quartic_curves_nauty_nb,
				Descr->classify_quartic_curves_nauty_fname_classification,
				Classifier,
				verbose_level);
	}


	else if (Descr->f_classify_quartic_curves_with_substructure) {

		canonical_form_classifier *Classifier;

		projective_space_global G;

		G.classify_quartic_curves_with_substructure(PA,
				Descr->classify_quartic_curves_with_substructure_fname_mask,
				Descr->classify_quartic_curves_with_substructure_nb,
				Descr->classify_quartic_curves_with_substructure_size,
				Descr->classify_quartic_curves_with_substructure_degree,
				Descr->classify_quartic_curves_with_substructure_fname_classification,
				Classifier,
				verbose_level);

		cout << "transversal:" << endl;
		Orbiter->Int_vec.print(cout, Classifier->transversal, Classifier->nb_types);
		cout << endl;

		int i, j;

		cout << "orbit frequencies:" << endl;
		for (i = 0; i < Classifier->nb_types; i++) {
			cout << i << " : ";

			j = Classifier->transversal[i];

			cout << j << " : ";

			if (Classifier->CFS_table[j]) {
				Orbiter->Int_vec.print(cout,
						Classifier->CFS_table[j]->SubSt->orbit_frequencies,
						Classifier->CFS_table[j]->SubSt->nb_orbits);
			}
			else {
				cout << "DNE";
			}

			cout << endl;

		}

		int *orbit_frequencies;
		int nb_orbits = 0;

		for (i = 0; i < Classifier->nb_types; i++) {
			cout << i << " : ";

			j = Classifier->transversal[i];

			cout << j << " : ";

			if (Classifier->CFS_table[j]) {
				nb_orbits = Classifier->CFS_table[j]->SubSt->nb_orbits;
				break;
			}
		}
		if (i == Classifier->nb_types) {
			cout << "cannot determine nb_orbits" << endl;
			exit(1);
		}
		orbit_frequencies = NEW_int(Classifier->nb_types * nb_orbits);

		Orbiter->Int_vec.zero(orbit_frequencies, Classifier->nb_types * nb_orbits);

		for (i = 0; i < Classifier->nb_types; i++) {

			j = Classifier->transversal[i];

			if (Classifier->CFS_table[j]) {
				Orbiter->Int_vec.copy(
						Classifier->CFS_table[j]->SubSt->orbit_frequencies,
						orbit_frequencies + i * nb_orbits,
						nb_orbits);
			}

		}

		tally_vector_data *T;
		int *transversal;
		int *frequency;
		int nb_types;

		T = NEW_OBJECT(tally_vector_data);

		T->init(orbit_frequencies, Classifier->nb_types, nb_orbits, verbose_level);



		T->get_transversal(transversal, frequency, nb_types, verbose_level);


		cout << "Classification of types:" << endl;
		cout << "nb_types=" << nb_types << endl;


		cout << "transversal:" << endl;
		Orbiter->Int_vec.print(cout, transversal, nb_types);
		cout << endl;

		cout << "frequency:" << endl;
		Orbiter->Int_vec.print(cout, frequency, nb_types);
		cout << endl;

		T->print_classes_bigger_than_one(verbose_level);


		file_io Fio;
		std::string fname;
		string_tools String;
		char str[1000];

		fname.assign(Descr->classify_quartic_curves_with_substructure_fname_mask);
		String.chop_off_extension(fname);
		sprintf(str, "_subset%d_types.csv", Descr->classify_quartic_curves_with_substructure_size);
		fname.append(str);


		cout << "preparing table" << endl;
		int *table;
		int h;

		table = NEW_int(Classifier->nb_types * (nb_orbits + 2));
		for (i = 0; i < Classifier->nb_types; i++) {

			cout << "preparing table i=" << i << endl;

			h = Classifier->transversal[i];

			cout << "preparing table i=" << i << " h=" << h << endl;

			table[i * (nb_orbits + 2) + 0] = i;

			for (j = 0; j < nb_orbits; j++) {
				table[i * (nb_orbits + 2) + 1 + j] = orbit_frequencies[i * nb_orbits + j];
			}

			table[i * (nb_orbits + 2) + 1 + nb_orbits] = Classifier->CFS_table[h]->SubSt->selected_orbit;

		}

		Fio.int_matrix_write_csv(fname, table, Classifier->nb_types, nb_orbits + 2);
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;


		if (Classifier->nb_types == 1) {
			cout << "preparing detailed information:" << endl;

			i = 0;
			int h;

			substructure_stats_and_selection *SubSt;

			h = Classifier->transversal[i];

			SubSt = Classifier->CFS_table[h]->SubSt;

			cout << "nb_interesting_subsets = "
					<< SubSt->nb_interesting_subsets << endl;
			cout << "interesting subsets: ";
			Orbiter->Lint_vec.print(cout, SubSt->interesting_subsets, SubSt->nb_interesting_subsets);
			cout << endl;

			cout << "selected_orbit=" << SubSt->selected_orbit << endl;

			cout << "generators for the canonical subset:" << endl;
			SubSt->gens->print_generators_tex();


			compute_stabilizer *CS;

			CS = Classifier->CFS_table[h]->CS;

			stabilizer_orbits_and_types *Stab_orbits;

			Stab_orbits = CS->Stab_orbits;

			cout << "reduced_set_size=" << Stab_orbits->reduced_set_size << endl;

			cout << "nb_orbits=" << Stab_orbits->Schreier->nb_orbits << endl;

			cout << "Orbit length:" << endl;
			Orbiter->Int_vec.print_integer_matrix_width(cout,
					Stab_orbits->Schreier->orbit_len,
					1,
					Stab_orbits->Schreier->nb_orbits,
					Stab_orbits->Schreier->nb_orbits,
					2);

			cout << "Orbit_patterns:" << endl;
#if 0
			Orbiter->Int_vec.print_integer_matrix_width(cout,
						Stab_orbits->Orbit_patterns,
						CS->SubSt->nb_interesting_subsets,
						Stab_orbits->Schreier->nb_orbits,
						Stab_orbits->Schreier->nb_orbits,
						2);
#endif

			cout << "minimal orbit pattern:" << endl;
			Stab_orbits->print_minimal_orbit_pattern();


			tally_vector_data *T_O;
			int *T_O_transversal;
			int *T_O_frequency;
			int T_O_nb_types;

			T_O = NEW_OBJECT(tally_vector_data);

			T_O->init(Stab_orbits->Orbit_patterns, CS->SubSt->nb_interesting_subsets,
					Stab_orbits->Schreier->nb_orbits, verbose_level);



			T_O->get_transversal(T_O_transversal, T_O_frequency, T_O_nb_types, verbose_level);

			cout << "T_O_nb_types = " << T_O_nb_types << endl;

			cout << "T_O_transversal:" << endl;
			Orbiter->Int_vec.print(cout, T_O_transversal, T_O_nb_types);
			cout << endl;

			cout << "T_O_frequency:" << endl;
			Orbiter->Int_vec.print(cout, T_O_frequency, T_O_nb_types);
			cout << endl;

			T_O->print_classes_bigger_than_one(verbose_level);

			cout << "Types classified:" << endl;
			int u, v;

			for (u = 0; u < T_O_nb_types; u++) {
				v = T_O_transversal[u];

				if (v == Stab_orbits->minimal_orbit_pattern_idx) {
					cout << "*";
				}
				else {
					cout << " ";
				}
				cout << setw(3) << u << " : " << setw(3) << v << " : " << setw(3) << T_O_frequency[u] << " : ";

				Orbiter->Int_vec.print_integer_matrix_width(cout,
							Stab_orbits->Orbit_patterns + v * Stab_orbits->Schreier->nb_orbits,
							1,
							Stab_orbits->Schreier->nb_orbits,
							Stab_orbits->Schreier->nb_orbits,
							2);

			}


			cout << "Types classified in lex order:" << endl;

			int *data;

			data = NEW_int(T_O_nb_types * Stab_orbits->Schreier->nb_orbits);
			for (u = 0; u < T_O_nb_types; u++) {

				cout << setw(3) << u << " : " << setw(3) << T_O->Frequency_in_lex_order[u] << " : ";

				Orbiter->Int_vec.print_integer_matrix_width(cout,
						T_O->Reps_in_lex_order[u],
						1,
						Stab_orbits->Schreier->nb_orbits,
						Stab_orbits->Schreier->nb_orbits,
						2);
				Orbiter->Int_vec.copy(T_O->Reps_in_lex_order[u], data + u * Stab_orbits->Schreier->nb_orbits, Stab_orbits->Schreier->nb_orbits);
			}

			fname.assign(Descr->classify_quartic_curves_with_substructure_fname_mask);
			String.chop_off_extension(fname);
			sprintf(str, "_subset%d_types_classified.csv", Descr->classify_quartic_curves_with_substructure_size);
			fname.append(str);

			Fio.int_matrix_write_csv(fname, data, T_O_nb_types, Stab_orbits->Schreier->nb_orbits);
			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;



			cout << "All canonical_forms:" << endl;
			Orbiter->Lint_vec.matrix_print_width(cout,
					CS->Canonical_forms,
					Stab_orbits->nb_interesting_subsets_reduced,
					Stab_orbits->reduced_set_size,
					Stab_orbits->reduced_set_size,
					2);

			cout << "All canonical_forms, with transporter" << endl;
			CS->print_canonical_sets();


			fname.assign(Descr->classify_quartic_curves_with_substructure_fname_mask);
			String.chop_off_extension(fname);
			sprintf(str, "_subset%d_cf_input.csv", Descr->classify_quartic_curves_with_substructure_size);
			fname.append(str);

#if 0
			Fio.lint_matrix_write_csv(fname, CS->Canonical_form_input,
					Stab_orbits->nb_interesting_subsets_reduced,
					Stab_orbits->reduced_set_size);
			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
#endif

			Fio.write_characteristic_matrix(fname,
					CS->Canonical_form_input,
					Stab_orbits->nb_interesting_subsets_reduced,
					Stab_orbits->reduced_set_size,
					Stab_orbits->nb_interesting_points,
					verbose_level);



			fname.assign(Descr->classify_quartic_curves_with_substructure_fname_mask);
			String.chop_off_extension(fname);
			sprintf(str, "_subset%d_cf_output.csv", Descr->classify_quartic_curves_with_substructure_size);
			fname.append(str);

#if 0
			Fio.lint_matrix_write_csv(fname, CS->Canonical_forms, Stab_orbits->nb_interesting_subsets_reduced, Stab_orbits->reduced_set_size);
			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
#endif

			Fio.write_characteristic_matrix(fname,
					CS->Canonical_forms,
					Stab_orbits->nb_interesting_subsets_reduced,
					Stab_orbits->reduced_set_size,
					Stab_orbits->nb_interesting_points,
					verbose_level);

			fname.assign(Descr->classify_quartic_curves_with_substructure_fname_mask);
			String.chop_off_extension(fname);
			sprintf(str, "_subset%d_cf_transporter.tex", Descr->classify_quartic_curves_with_substructure_size);
			fname.append(str);


			std::string title;

			title.assign("Transporter");
			PA->A->write_set_of_elements_latex_file(fname, title,
					CS->Canonical_form_transporter,
					Stab_orbits->nb_interesting_subsets_reduced);


			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		}

#if 0


		substructure_stats_and_selection *SubSt;

#if 0
		long int *interesting_subsets; // [selected_frequency]
		int nb_interesting_subsets;
			// interesting_subsets are the lvl-subsets of the given set
			// which are of the chosen type.
			// There is nb_interesting_subsets of them.

		strong_generators *gens;
#endif



		compute_stabilizer *CS;
#if 0
		action *A_on_the_set;
			// only used to print the induced action on the set
			// of the set stabilizer

		sims *Stab; // the stabilizer of the original set


		longinteger_object stab_order, new_stab_order;
		int nb_times_orbit_count_does_not_match_up;
		int backtrack_nodes_first_time;
		int backtrack_nodes_total_in_loop;

		stabilizer_orbits_and_types *Stab_orbits;
#if 0
		strong_generators *selected_set_stab_gens;
		sims *selected_set_stab;


		int reduced_set_size; // = set_size - level




		long int *reduced_set1; // [set_size]
		long int *reduced_set2; // [set_size]
		long int *reduced_set1_new_labels; // [set_size]
		long int *reduced_set2_new_labels; // [set_size]
		long int *canonical_set1; // [set_size]
		long int *canonical_set2; // [set_size]

		int *elt1, *Elt1, *Elt1_inv, *new_automorphism, *Elt4;
		int *elt2, *Elt2;
		int *transporter0; // = elt1 * elt2

		longinteger_object go_G;

		schreier *Schreier;
		int nb_orbits;
		int *orbit_count1; // [nb_orbits]
		int *orbit_count2; // [nb_orbits]


		int nb_interesting_subsets_reduced;
		long int *interesting_subsets_reduced;

		int *Orbit_patterns; // [nb_interesting_subsets * nb_orbits]


		int *orbit_to_interesting_orbit; // [nb_orbits]

		int nb_interesting_orbits;
		int *interesting_orbits;

		int nb_interesting_points;
		long int *interesting_points;

		int *interesting_orbit_first;
		int *interesting_orbit_len;

		int local_idx1, local_idx2;
#endif






		action *A_induced;
		longinteger_object induced_go, K_go;

		int *transporter_witness;
		int *transporter1;
		int *transporter2;
		int *T1, *T1v;
		int *T2;

		sims *Kernel_original;
		sims *K; // kernel for building up Stab



		sims *Aut;
		sims *Aut_original;
		longinteger_object ago;
		longinteger_object ago1;
		longinteger_object target_go;


		//union_find_on_k_subsets *U;


		long int *Canonical_forms; // [nb_interesting_subsets_reduced * reduced_set_size]
		int nb_interesting_subsets_rr;
		long int *interesting_subsets_rr;
#endif

		strong_generators *Gens_stabilizer_original_set;
		strong_generators *Gens_stabilizer_canonical_form;


		orbit_of_equations *Orb;

		strong_generators *gens_stab_of_canonical_equation;

		int *trans1;
		int *trans2;
		int *intermediate_equation;



		int *Elt;
		int *eqn2;

		int *canonical_equation;
		int *transporter_to_canonical_form;
#endif

	}
	else if (Descr->f_set_stabilizer) {

		projective_space_global G;

		G.set_stabilizer(PA,
				Descr->set_stabilizer_intermediate_set_size,
				Descr->set_stabilizer_fname_mask,
				Descr->set_stabilizer_nb,
				Descr->set_stabilizer_column_label,
				verbose_level);
	}

	else if (Descr->f_conic_type) {

		projective_space_global G;

		G.conic_type(PA,
				Descr->conic_type_threshold,
				Descr->conic_type_set_text,
				verbose_level);
	}

	else if (Descr->f_lift_skew_hexagon) {

		projective_space_global G;

		G.do_lift_skew_hexagon(PA,
				Descr->lift_skew_hexagon_text,
				verbose_level);
	}

	else if (Descr->f_lift_skew_hexagon_with_polarity) {

		projective_space_global G;

		G.do_lift_skew_hexagon_with_polarity(PA,
				Descr->lift_skew_hexagon_with_polarity_polarity,
				verbose_level);
	}
	else if (Descr->f_arc_with_given_set_as_s_lines_after_dualizing) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity f_arc_with_given_set_as_i_lines_after_dualizing" << endl;
		}
		diophant *D = NULL;
		int f_save_system = TRUE;

		long int *the_set_in;
		int set_size_in;

		Orbiter->Lint_vec.scan(Descr->arc_input_set, the_set_in, set_size_in);

		PA->P->arc_with_given_set_of_s_lines_diophant(
				the_set_in /*one_lines*/, set_size_in /* nb_one_lines */,
				Descr->arc_size /*target_sz*/, Descr->arc_d /* target_d */,
				Descr->arc_d_low, Descr->arc_s /* target_s */,
				TRUE /* f_dualize */,
				D,
				verbose_level);

		if (FALSE) {
			D->print_tight();
		}

		if (f_save_system) {

			string fname_system;

			fname_system.assign(Descr->arc_label);
			fname_system.append(".diophant");
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << endl;
			D->save_in_general_format(fname_system, 0 /* verbose_level */);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << " done" << endl;
			//D->print();
			//D->print_tight();
			}

		long int nb_backtrack_nodes;
		long int *Sol;
		int nb_sol;

		D->solve_all_mckay(nb_backtrack_nodes, INT_MAX, verbose_level);

		if (f_v) {
			cout << "before D->get_solutions" << endl;
			}
		D->get_solutions(Sol, nb_sol, verbose_level);
		if (f_v) {
			cout << "after D->get_solutions, nb_sol=" << nb_sol << endl;
			}
		string fname_solutions;

		fname_solutions.assign(Descr->arc_label);
		fname_solutions.append(".solutions");

		{
			ofstream fp(fname_solutions);
			int i, j, a;

			for (i = 0; i < nb_sol; i++) {
				fp << D->sum;
				for (j = 0; j < D->sum; j++) {
					a = Sol[i * D->sum + j];
					fp << " " << a;
					}
				fp << endl;
				}
			fp << -1 << " " << nb_sol << endl;
		}
		file_io Fio;

		cout << "Written file " << fname_solutions << " of size "
				<< Fio.file_size(fname_solutions) << endl;
		FREE_lint(Sol);
		FREE_lint(the_set_in);


	}
	else if (Descr->f_arc_with_two_given_sets_of_lines_after_dualizing) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity f_arc_with_two_given_sets_of_lines_after_dualizing" << endl;
		}

		long int *t_lines;
		int nb_t_lines;

		Orbiter->Lint_vec.scan(Descr->t_lines_string, t_lines, nb_t_lines);

		cout << "The t-lines, t=" << Descr->arc_t << " are ";
		Orbiter->Lint_vec.print(cout, t_lines, nb_t_lines);
		cout << endl;


		long int *the_set_in;
		int set_size_in;

		Orbiter->Lint_vec.scan(Descr->arc_input_set, the_set_in, set_size_in);


		diophant *D = NULL;
		int f_save_system = TRUE;

		PA->P->arc_with_two_given_line_sets_diophant(
				the_set_in /* s_lines */, set_size_in /* nb_s_lines */, Descr->arc_s,
				t_lines, nb_t_lines, Descr->arc_t,
				Descr->arc_size /*target_sz*/, Descr->arc_d, Descr->arc_d_low,
				TRUE /* f_dualize */,
				D,
				verbose_level);

		if (FALSE) {
			D->print_tight();
		}

		if (f_save_system) {
			string fname_system;

			fname_system.assign(Descr->arc_label);
			fname_system.append(".diophant");

			//sprintf(fname_system, "system_%d.diophant", back_end_counter);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << endl;
			D->save_in_general_format(fname_system, 0 /* verbose_level */);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << " done" << endl;
			//D->print();
		}

		long int nb_backtrack_nodes;
		long int *Sol;
		int nb_sol;

		D->solve_all_mckay(nb_backtrack_nodes, INT_MAX, verbose_level);

		if (f_v) {
			cout << "before D->get_solutions" << endl;
		}
		D->get_solutions(Sol, nb_sol, verbose_level);
		if (f_v) {
			cout << "after D->get_solutions, nb_sol=" << nb_sol << endl;
		}
		string fname_solutions;

		fname_solutions.assign(Descr->arc_label);
		fname_solutions.append(".solutions");

		{
			ofstream fp(fname_solutions);
			int i, j, a;

			for (i = 0; i < nb_sol; i++) {
				fp << D->sum;
				for (j = 0; j < D->sum; j++) {
					a = Sol[i * D->sum + j];
					fp << " " << a;
				}
				fp << endl;
			}
			fp << -1 << " " << nb_sol << endl;
		}

		file_io Fio;

		cout << "Written file " << fname_solutions << " of size "
				<< Fio.file_size(fname_solutions) << endl;
		FREE_lint(Sol);
		FREE_lint(the_set_in);


	}
	else if (Descr->f_arc_with_three_given_sets_of_lines_after_dualizing) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity f_arc_with_three_given_sets_of_lines_after_dualizing" << endl;
		}
		//int arc_size;
		//int arc_d;
		diophant *D = NULL;
		int f_save_system = TRUE;

		long int *t_lines;
		int nb_t_lines;
		long int *u_lines;
		int nb_u_lines;


		Orbiter->Lint_vec.scan(Descr->t_lines_string, t_lines, nb_t_lines);
		Orbiter->Lint_vec.scan(Descr->u_lines_string, u_lines, nb_u_lines);
		//lint_vec_print(cout, t_lines, nb_t_lines);
		//cout << endl;

		cout << "The t-lines, t=" << Descr->arc_t << " are ";
		Orbiter->Lint_vec.print(cout, t_lines, nb_t_lines);
		cout << endl;
		cout << "The u-lines, u=" << Descr->arc_u << " are ";
		Orbiter->Lint_vec.print(cout, u_lines, nb_u_lines);
		cout << endl;


		long int *the_set_in;
		int set_size_in;

		Orbiter->Lint_vec.scan(Descr->arc_input_set, the_set_in, set_size_in);


		PA->P->arc_with_three_given_line_sets_diophant(
				the_set_in /* s_lines */, set_size_in /* nb_s_lines */, Descr->arc_s,
				t_lines, nb_t_lines, Descr->arc_t,
				u_lines, nb_u_lines, Descr->arc_u,
				Descr->arc_size /*target_sz*/, Descr->arc_d, Descr->arc_d_low,
				TRUE /* f_dualize */,
				D,
				verbose_level);

		if (FALSE) {
			D->print_tight();
		}
		if (f_save_system) {
			string fname_system;

			fname_system.assign(Descr->arc_label);
			fname_system.append(".diophant");

			cout << "projective_space_activity::perform_activity saving the system "
					"to file " << fname_system << endl;
			D->save_in_general_format(fname_system, 0 /* verbose_level */);
			cout << "projective_space_activity::perform_activity saving the system "
					"to file " << fname_system << " done" << endl;
			//D->print();
			//D->print_tight();
		}

		long int nb_backtrack_nodes;
		long int *Sol;
		int nb_sol;

		D->solve_all_mckay(nb_backtrack_nodes, INT_MAX, verbose_level);

		if (f_v) {
			cout << "before D->get_solutions" << endl;
		}
		D->get_solutions(Sol, nb_sol, verbose_level);
		if (f_v) {
			cout << "after D->get_solutions, nb_sol=" << nb_sol << endl;
		}
		string fname_solutions;

		fname_solutions.assign(Descr->arc_label);
		fname_solutions.append(".solutions");

		{
			ofstream fp(fname_solutions);
			int i, j, a;

			for (i = 0; i < nb_sol; i++) {
				fp << D->sum;
				for (j = 0; j < D->sum; j++) {
					a = Sol[i * D->sum + j];
					fp << " " << a;
				}
				fp << endl;
			}
			fp << -1 << " " << nb_sol << endl;
		}
		file_io Fio;

		cout << "Written file " << fname_solutions << " of size "
				<< Fio.file_size(fname_solutions) << endl;
		FREE_lint(Sol);
		FREE_lint(the_set_in);


	}
	else if (Descr->f_dualize_hyperplanes_to_points) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity f_dualize_hyperplanes_to_points" << endl;
		}
		long int *the_set_in;
		int set_size_in;
		long int *the_set_out;
		int set_size_out;

		Orbiter->Lint_vec.scan(Descr->dualize_input_set, the_set_in, set_size_in);

		int i;
		long int a;

		set_size_out = set_size_in;
		the_set_out = NEW_lint(set_size_in);
		for (i = 0; i < set_size_in; i++) {
			a = the_set_in[i];
			the_set_out[i] = PA->P->Standard_polarity->Hyperplane_to_point[a];
		}

		// only if n = 2:
		//int *Polarity_point_to_hyperplane; // [N_points]
		//int *Polarity_hyperplane_to_point; // [N_points]

		FREE_lint(the_set_in);
		FREE_lint(the_set_out);

	}
	else if (Descr->f_dualize_points_to_hyperplanes) {
		if (f_v) {
			cout << "projective_space_activity::perform_activity f_dualize_points_to_hyperplanes" << endl;
		}
		long int *the_set_in;
		int set_size_in;
		long int *the_set_out;
		int set_size_out;

		Orbiter->Lint_vec.scan(Descr->dualize_input_set, the_set_in, set_size_in);

		int i;
		long int a;

		set_size_out = set_size_in;
		the_set_out = NEW_lint(set_size_in);
		for (i = 0; i < set_size_in; i++) {
			a = the_set_in[i];
			the_set_out[i] = PA->P->Standard_polarity->Point_to_hyperplane[a];
		}

		FREE_lint(the_set_in);
		FREE_lint(the_set_out);

	}
	else if (Descr->f_classify_arcs) {
		projective_space_global G;

		G.do_classify_arcs(
				PA,
				Descr->Arc_generator_description,
				verbose_level);
	}
	else if (Descr->f_classify_cubic_curves) {
		projective_space_global G;

		G.do_classify_cubic_curves(
				PA,
				Descr->Arc_generator_description,
				verbose_level);
	}
	else if (Descr->f_latex_homogeneous_equation) {
		int d = Descr->latex_homogeneous_equation_degree;
		int *eqn;
		int sz;
		homogeneous_polynomial_domain *Poly;

		Orbiter->Int_vec.scan(Descr->latex_homogeneous_equation_text, eqn, sz);
		Poly = NEW_OBJECT(homogeneous_polynomial_domain);

		if (f_v) {
			cout << "projective_space_activity::perform_activity before Poly->init" << endl;
		}
		Poly->init(PA->F,
				PA->d /* nb_vars */, d /* degree */,
				FALSE /* f_init_incidence_structure */,
				t_PART,
				verbose_level);

		Poly->remake_symbols(0 /* symbol_offset */,
					Descr->latex_homogeneous_equation_symbol_txt.c_str(),
					Descr->latex_homogeneous_equation_symbol_tex.c_str(),
					verbose_level);


		if (Poly->get_nb_monomials() != sz) {
			cout << "Poly->get_nb_monomials() = " << Poly->get_nb_monomials() << endl;
			cout << "number of coefficients given = " << sz << endl;
			exit(1);
		}
		Poly->print_equation_tex(cout, eqn);
		cout << endl;
		if (f_v) {
			cout << "projective_space_activity::perform_activity after Poly1->init" << endl;
		}
	}
	else if (Descr->f_lines_on_point_but_within_a_plane) {

		if (f_v) {
			cout << "projective_space_activity::perform_activity f_lines_on_point_but_within_a_plane" << endl;
		}

		long int point_rk = Descr->lines_on_point_but_within_a_plane_point_rk;
		long int plane_rk = Descr->lines_on_point_but_within_a_plane_plane_rk;
		long int *line_pencil;
		int q;

		q = PA->F->q;
		line_pencil = NEW_lint(q + 1);

		PA->P->create_lines_on_point_but_inside_a_plane(
				point_rk, plane_rk,
				line_pencil, verbose_level);
			// assumes that line_pencil[q + 1] has been allocated

		cout << "line_pencil: ";
		Orbiter->Lint_vec.print(cout, line_pencil, q + 1);
		cout << endl;

		if (f_v) {
			cout << "projective_space_activity::perform_activity f_lines_on_point_but_within_a_plane done" << endl;
		}
	}



	if (f_v) {
		cout << "projective_space_activity::perform_activity done" << endl;
	}

}


}}
