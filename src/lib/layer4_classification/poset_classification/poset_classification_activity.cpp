/*
 * poset_classification_activity.cpp
 *
 *  Created on: Feb 19, 2023
 *      Author: betten
 */


#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {

poset_classification_activity::poset_classification_activity()
{
	Descr = NULL;
	PC = NULL;
	actual_size = 0;
}

poset_classification_activity::~poset_classification_activity()
{
}

void poset_classification_activity::init(
		poset_classification_activity_description *Descr,
		poset_classification *PC,
		int actual_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification_activity::init" << endl;
	}
	poset_classification_activity::Descr = Descr;
	poset_classification_activity::PC = PC;
	poset_classification_activity::actual_size = actual_size;
	if (f_v) {
		cout << "poset_classification_activity::init done" << endl;
	}

}


void poset_classification_activity::perform_work(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification_activity::perform_work" << endl;
	}

	if (Descr->f_report) {

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_report" << endl;
		}

		PC->report(Descr->report_options, verbose_level);

	}

	if (Descr->f_export_level_to_cpp) {

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_export_level_to_cpp" << endl;
		}

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before generate_source_code" << endl;
		}
		generate_source_code(
				Descr->export_level_to_cpp_level, verbose_level);
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after generate_source_code" << endl;
		}

	}

	if (Descr->f_export_history_to_cpp) {

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_export_history_to_cpp" << endl;
		}

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before generate_history" << endl;
		}
		generate_history(
				Descr->export_history_to_cpp_level, verbose_level);
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after generate_history" << endl;
		}

	}


	if (f_v) {
		cout << "poset_classification_activity::perform_work "
				"problem_label_with_path="
				<< PC->get_problem_label_with_path()
				<< " verbose_level=" << verbose_level << endl;
	}

	if (Descr->f_write_tree) {
		PC->get_Poo()->print_tree();
		PC->write_treefile(
				PC->get_problem_label_with_path(), PC->get_depth(),
				orbiter_kernel_system::Orbiter->draw_options,
				verbose_level - 1);

		//return 0;
	}
	if (Descr->f_table_of_nodes) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_table_of_nodes" << endl;
		}
		PC->get_Poo()->make_tabe_of_nodes(verbose_level);
	}

	if (Descr->f_list_all) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_list_all" << endl;
		}

		int d;

		for (d = 0; d <= PC->get_depth(); d++) {
			cout << "There are " << PC->nb_orbits_at_level(d)
					<< " orbits on subsets of size " << d << ":" << endl;

#if 0
			if (d < Descr->orbits_on_subsets_size) {
				//continue;
			}
#endif

			PC->list_all_orbits_at_level(
					d,
					false /* f_has_print_function */,
					NULL /* void (*print_function)(std::ostream &ost, int len, int *S, void *data)*/,
					NULL /* void *print_function_data*/,
					Descr->f_show_orbit_decomposition /* f_show_orbit_decomposition */,
					Descr->f_show_stab /* f_show_stab */,
					Descr->f_save_stab /* f_save_stab */,
					Descr->f_show_whole_orbits /* f_show_whole_orbit*/);
		}
	}

	if (Descr->f_list) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work f_list" << endl;
		}
#if 1
		//int f_show_orbit_decomposition = true;
		//int f_show_stab = true;
		//int f_save_stab = true;
		//int f_show_whole_orbit = false;

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before list_all_orbits_at_level" << endl;
		}
		PC->list_all_orbits_at_level(
				actual_size,
			false,
			NULL,
			this,
			Descr->f_show_orbit_decomposition,
			Descr->f_show_stab,
			Descr->f_save_stab,
			Descr->f_show_whole_orbits);

		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after list_all_orbits_at_level" << endl;
		}

#if 0
		int d;
		for (d = 0; d < 3; d++) {
			gen->print_schreier_vectors_at_depth(d, verbose_level);
		}
#endif
#endif
	}

	if (Descr->f_level_summary_csv) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"preparing level spreadsheet" << endl;
		}
		{
			data_structures::spreadsheet *Sp;
			PC->make_spreadsheet_of_level_info(
					Sp, actual_size, verbose_level);
			string fname_csv;

			fname_csv = PC->get_problem_label_with_path() + "_levels_" + std::to_string(actual_size) + ".csv";

			Sp->save(fname_csv, verbose_level);
			FREE_OBJECT(Sp);
		}
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"preparing level spreadsheet done" << endl;
		}
	}


	if (Descr->f_orbit_reps_csv) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"preparing orbit spreadsheet" << endl;
		}
		{
			data_structures::spreadsheet *Sp;
			PC->make_spreadsheet_of_orbit_reps(
					Sp, actual_size);
			string fname_csv;

			fname_csv = PC->get_problem_label_with_path() + "_orbits_at_level_" + std::to_string(actual_size) + ".csv";
			Sp->save(fname_csv, verbose_level);
			FREE_OBJECT(Sp);
		}
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"preparing orbit spreadsheet done" << endl;
		}
	}


	if (Descr->f_draw_poset) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before draw_poset" << endl;
		}
#if 0
		if (!Descr->f_draw_options) {
			cout << "poset_classification_activity::perform_work "
					"Descr->f_draw_poset && !Control->f_draw_options" << endl;
			exit(1);
		}
#endif
		PC->draw_poset(
				PC->get_problem_label_with_path(), actual_size,
			0 /* data1 */,
			orbiter_kernel_system::Orbiter->draw_options,
			verbose_level);
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after draw_poset" << endl;
		}
	}

	if (Descr->f_draw_full_poset) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"before draw_full_poset" << endl;
		}
		PC->draw_poset_full(
				PC->get_problem_label_with_path(), actual_size,
				0 /* data1 */,
				orbiter_kernel_system::Orbiter->draw_options,
				1 /* x_stretch */, verbose_level);
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"after draw_full_poset" << endl;
		}
	}
	if (Descr->f_make_relations_with_flag_orbits) {
			string fname_prefix;


			fname_prefix = PC->get_problem_label_with_path() + "_flag_orbits";

			if (f_v) {
				cout << "poset_classification_activity::perform_work "
						"before make_flag_orbits_on_relations" << endl;
			}
			PC->make_flag_orbits_on_relations(
					PC->get_depth(), fname_prefix, verbose_level);
			if (f_v) {
				cout << "poset_classification_activity::perform_work "
						"after make_flag_orbits_on_relations" << endl;
			}
	}
	if (Descr->f_print_data_structure) {
		if (f_v) {
			cout << "poset_classification_activity::perform_work "
					"f_print_data_structure" << endl;
		}
		PC->print_data_structure_tex(
				actual_size, verbose_level);
	}


	if (Descr->f_test_multi_edge_in_decomposition_matrix) {
		test_for_multi_edge_in_classification_graph(
				PC->get_depth(), verbose_level);
	}


	if (Descr->recognize.size()) {
		int h;

		for (h = 0; h < Descr->recognize.size(); h++) {

			PC->recognize(Descr->recognize[h],
					h, Descr->recognize.size(),
					verbose_level);
		}
	}

	if (f_v) {
		cout << "poset_classification_activity::perform_work done" << endl;
	}
}


void poset_classification_activity::compute_Kramer_Mesner_matrix(
		int t, int k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "poset_classification_activity::compute_Kramer_Mesner_matrix" << endl;
	}


	// compute Stack of neighboring Kramer Mesner matrices
	long int **pM;
	int *Nb_rows, *Nb_cols;
	int h;

	pM = NEW_plint(k);
	Nb_rows = NEW_int(k);
	Nb_cols = NEW_int(k);
	for (h = 0; h < k; h++) {

		if (f_v) {
			cout << "poset_classification_activity::compute_Kramer_Mesner_matrix "
					"level " << h << " / " << k
					<< " before Kramer_Mesner_matrix_neighboring" << endl;
		}
		Kramer_Mesner_matrix_neighboring(
				h, pM[h], Nb_rows[h], Nb_cols[h],
				verbose_level - 2);

		if (f_v) {
			cout << "poset_classification_activity::compute_Kramer_Mesner_matrix "
					"matrix level " << h << " computed" << endl;
	#if 0
			int j;
			for (i = 0; i < Nb_rows[h]; i++) {
				for (j = 0; j < Nb_cols[h]; j++) {
					cout << pM[h][i * Nb_cols[h] + j];
					if (j < Nb_cols[h]) {
						cout << ",";
					}
				}
				cout << endl;
			}
	#endif
		}
	}

	long int *Mtk;
	int nb_r, nb_c;


	if (f_v) {
		cout << "poset_classification_activity::compute_Kramer_Mesner_matrix "
				"before Mtk_from_MM" << endl;
	}
	Mtk_from_MM(pM, Nb_rows, Nb_cols,
			t, k,
			Mtk,
			nb_r, nb_c,
			verbose_level);
	if (f_v) {
		cout << "poset_classification_activity::compute_Kramer_Mesner_matrix "
				"after Mtk_from_MM" << endl;
		cout << "poset_classification_activity::compute_Kramer_Mesner_matrix "
				"M_{" << t << "," << k << "} "
				"has size " << nb_r << " x " << nb_c << "." << endl;

#if 0
		int j;
		for (i = 0; i < nb_r; i++) {
			for (j = 0; j < nb_c; j++) {
				cout << Mtk[i * nb_c + j];
				if (j < nb_c - 1) {
					cout << ",";
				}
			}
			cout << endl;
		}
#endif

	}

	long int *Mtk_inf;


	if (f_v) {
		cout << "poset_classification_activity::compute_Kramer_Mesner_matrix "
				"before Asup_to_Ainf" << endl;
	}
	Asup_to_Ainf(
			t, k,
			Mtk, Mtk_inf, verbose_level);
	if (f_v) {
		cout << "poset_classification_activity::compute_Kramer_Mesner_matrix "
				"after Asup_to_Ainf" << endl;
	}


	orbiter_kernel_system::file_io Fio;
	int i;

	string fname;

	fname = PC->get_problem_label() + "_KM_" + std::to_string(t) + "_" + std::to_string(k) + ".csv";

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname, Mtk, nb_r, nb_c);

	//Mtk.print(cout);
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


	fname = PC->get_problem_label() + "_KM_inf_" + std::to_string(t) + "_" + std::to_string(k) + ".csv";

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname, Mtk_inf, nb_r, nb_c);

	//Mtk.print(cout);
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;



	if (f_v) {
		cout << "poset_classification_activity::compute_Kramer_Mesner_matrix "
				"computing Kramer Mesner matrices done" << endl;
	}
	for (i = 0; i < k; i++) {
		FREE_lint(pM[i]);
	}
	FREE_plint(pM);
	FREE_lint(Mtk);
	FREE_lint(Mtk_inf);

	if (f_v) {
		cout << "poset_classification_activity::compute_Kramer_Mesner_matrix done" << endl;
	}
}

void poset_classification_activity::Plesken_matrix_up(
		int depth,
		int *&P, int &N, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Nb;
	int *Fst;
	int *Pij;
	int i, j;
	int N1, N2;
	int a, b, cnt;

	if (f_v) {
		cout << "poset_classification_activity::Plesken_matrix_up" << endl;
	}
	N = 0;
	Nb = NEW_int(depth + 1);
	Fst = NEW_int(depth + 2);
	Fst[0] = 0;
	for (i = 0; i <= depth; i++) {
		Nb[i] = PC->nb_orbits_at_level(i);
		Fst[i + 1] = Fst[i] + Nb[i];
		N += Nb[i];
	}
	P = NEW_int(N * N);
	for (i = 0; i <= depth; i++) {
		for (j = 0; j <= depth; j++) {

			Plesken_submatrix_up(
					i, j, Pij, N1, N2, verbose_level - 1);

			for (a = 0; a < N1; a++) {
				for (b = 0; b < N2; b++) {
					cnt = Pij[a * N2 + b];
					P[(Fst[i] + a) * N + Fst[j] + b] = cnt;
				}
			}
			FREE_int(Pij);
		}
	}
	if (f_v) {
		cout << "poset_classification_activity::Plesken_matrix_up done" << endl;
	}
}

void poset_classification_activity::Plesken_matrix_down(
		int depth,
		int *&P, int &N, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Nb;
	int *Fst;
	int *Pij;
	int i, j;
	int N1, N2;
	int a, b, cnt;

	if (f_v) {
		cout << "poset_classification_activity::Plesken_matrix_down" << endl;
	}
	N = 0;
	Nb = NEW_int(depth + 1);
	Fst = NEW_int(depth + 2);
	Fst[0] = 0;
	for (i = 0; i <= depth; i++) {
		Nb[i] = PC->nb_orbits_at_level(i);
		Fst[i + 1] = Fst[i] + Nb[i];
		N += Nb[i];
	}
	P = NEW_int(N * N);
	for (i = 0; i <= depth; i++) {
		for (j = 0; j <= depth; j++) {

			Plesken_submatrix_down(i, j,
					Pij, N1, N2, verbose_level - 1);

			for (a = 0; a < N1; a++) {
				for (b = 0; b < N2; b++) {
					cnt = Pij[a * N2 + b];
					P[(Fst[i] + a) * N + Fst[j] + b] = cnt;
				}
			}
			FREE_int(Pij);
		}
	}
	if (f_v) {
		cout << "poset_classification_activity::Plesken_matrix_down done" << endl;
	}
}

void poset_classification_activity::Plesken_submatrix_up(
		int i, int j,
		int *&Pij, int &N1, int &N2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b;

	if (f_v) {
		cout << "poset_classification_activity::Plesken_submatrix_up "
				"i=" << i << " j=" << j << endl;
	}
	N1 = PC->nb_orbits_at_level(i);
	N2 = PC->nb_orbits_at_level(j);
	Pij = NEW_int(N1 * N2);
	for (a = 0; a < N1; a++) {
		for (b = 0; b < N2; b++) {
			Pij[a * N2 + b] = count_incidences_up(
					i, a, j, b, verbose_level - 1);
		}
	}
	if (f_v) {
		cout << "poset_classification_activity::Plesken_submatrix_up done" << endl;
	}
}

void poset_classification_activity::Plesken_submatrix_down(
		int i, int j,
		int *&Pij, int &N1, int &N2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b;

	if (f_v) {
		cout << "poset_classification_activity::Plesken_submatrix_down "
				"i=" << i << " j=" << j << endl;
	}
	N1 = PC->nb_orbits_at_level(i);
	N2 = PC->nb_orbits_at_level(j);
	Pij = NEW_int(N1 * N2);
	for (a = 0; a < N1; a++) {
		for (b = 0; b < N2; b++) {
			Pij[a * N2 + b] = count_incidences_down(
					i, a, j, b, verbose_level - 1);
		}
	}
	if (f_v) {
		cout << "poset_classification_activity::Plesken_submatrix_down done" << endl;
	}
}

int poset_classification_activity::count_incidences_up(
		int lvl1, int po1,
		int lvl2, int po2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *set;
	long int *set1;
	long int *set2;
	int ol, i, cnt = 0;
	int f_contained;

	if (f_v) {
		cout << "poset_classification_activity::count_incidences_up "
				"lvl1=" << lvl1 << " po1=" << po1
				<< " lvl2=" << lvl2 << " po2=" << po2 << endl;
	}
	if (lvl1 > lvl2) {
		return 0;
	}
	set = NEW_lint(lvl2 + 1);
	set1 = NEW_lint(lvl2 + 1);
	set2 = NEW_lint(lvl2 + 1);

	PC->orbit_element_unrank(
			lvl1, po1, 0 /*el1 */,
			set1, 0 /* verbose_level */);

	ol = PC->orbit_length_as_int(po2, lvl2);

	if (f_vv) {
		cout << "set1=";
		Lint_vec_print(cout, set1, lvl1);
		cout << endl;
	}

	for (i = 0; i < ol; i++) {

		Lint_vec_copy(set1, set, lvl1);


		PC->orbit_element_unrank(
				lvl2, po2, i, set2, 0 /* verbose_level */);

		if (f_vv) {
			cout << "set2 " << i << " / " << ol << "=";
			Lint_vec_print(cout, set2, lvl2);
			cout << endl;
		}

		f_contained = PC->poset_structure_is_contained(
				set, lvl1, set2, lvl2, verbose_level - 2);

		if (f_vv) {
			cout << "f_contained=" << f_contained << endl;
		}


		if (f_contained) {
			cnt++;
		}
	}


	FREE_lint(set);
	FREE_lint(set1);
	FREE_lint(set2);
	if (f_v) {
		cout << "poset_classification_activity::count_incidences_up "
				"lvl1=" << lvl1 << " po1=" << po1
				<< " lvl2=" << lvl2 << " po2=" << po2
				<< " cnt=" << cnt << endl;
	}
	return cnt;
}

int poset_classification_activity::count_incidences_down(
		int lvl1, int po1, int lvl2, int po2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *set;
	long int *set1;
	long int *set2;
	int ol, i, cnt = 0;
	int f_contained;

	if (f_v) {
		cout << "poset_classification_activity::count_incidences_down "
				"lvl1=" << lvl1 << " po1=" << po1
				<< " lvl2=" << lvl2 << " po2=" << po2 << endl;
	}
	if (lvl1 > lvl2) {
		return 0;
	}
	set = NEW_lint(lvl2 + 1);
	set1 = NEW_lint(lvl2 + 1);
	set2 = NEW_lint(lvl2 + 1);

	PC->orbit_element_unrank(
			lvl2, po2, 0 /*el1 */, set2,
			0 /* verbose_level */);

	ol = PC->orbit_length_as_int(po1, lvl1);

	if (f_vv) {
		cout << "set2=";
		Lint_vec_print(cout, set2, lvl2);
		cout << endl;
	}

	for (i = 0; i < ol; i++) {

		Lint_vec_copy(set2, set, lvl2);


		PC->orbit_element_unrank(
				lvl1, po1, i, set1,
				0 /* verbose_level */);

		if (f_vv) {
			cout << "set1 " << i << " / " << ol << "=";
			Lint_vec_print(cout, set1, lvl1);
			cout << endl;
		}


		f_contained = PC->poset_structure_is_contained(
				set1, lvl1, set, lvl2, verbose_level - 2);

		if (f_vv) {
			cout << "f_contained=" << f_contained << endl;
		}

		if (f_contained) {
			cnt++;
		}
	}


	FREE_lint(set);
	FREE_lint(set1);
	FREE_lint(set2);
	if (f_v) {
		cout << "poset_classification_activity::count_incidences_down "
				"lvl1=" << lvl1 << " po1=" << po1
				<< " lvl2=" << lvl2 << " po2=" << po2
				<< " cnt=" << cnt << endl;
	}
	return cnt;
}

void poset_classification_activity::Asup_to_Ainf(
		int t, int k,
		long int *M_sup, long int *&M_inf, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object quo, rem, aa, bb, cc;
	ring_theory::longinteger_object go;
	ring_theory::longinteger_object *go_t;
	ring_theory::longinteger_object *go_k;
	ring_theory::longinteger_object *ol_t;
	ring_theory::longinteger_object *ol_k;
	int Nt, Nk;
	int i, j;
	long int a, c;

	if (f_v) {
		cout << "poset_classification_activity::Asup_to_Ainf" << endl;
	}
	Nt = PC->nb_orbits_at_level(t);
	Nk = PC->nb_orbits_at_level(k);
	PC->get_stabilizer_order(0, 0, go);

	M_inf = NEW_lint(Nt * Nk);

	if (f_v) {
		cout << "poset_classification_activity::Asup_to_Ainf go=" << go << endl;
	}
	go_t = NEW_OBJECTS(ring_theory::longinteger_object, Nt);
	go_k = NEW_OBJECTS(ring_theory::longinteger_object, Nk);
	ol_t = NEW_OBJECTS(ring_theory::longinteger_object, Nt);
	ol_k = NEW_OBJECTS(ring_theory::longinteger_object, Nk);
	if (f_v) {
		cout << "poset_classification_activity::Asup_to_Ainf "
				"computing orbit lengths t-orbits" << endl;
	}
	for (i = 0; i < Nt; i++) {
		PC->get_stabilizer_order(t, i, go_t[i]);
		D.integral_division_exact(go, go_t[i], ol_t[i]);
	}
	if (f_v) {
		cout << "i : go_t[i] : ol_t[i]" << endl;
		for (i = 0; i < Nt; i++) {
			cout << i << " : " << go_t[i] << " : " << ol_t[i] << endl;
		}
	}
	if (f_v) {
		cout << "poset_classification_activity::Asup_to_Ainf "
				"computing orbit lengths k-orbits" << endl;
	}
	for (i = 0; i < Nk; i++) {
		PC->get_stabilizer_order(k, i, go_k[i]);
		D.integral_division_exact(go, go_k[i], ol_k[i]);
	}
	if (f_v) {
		cout << "i : go_k[i] : ol_k[i]" << endl;
		for (i = 0; i < Nk; i++) {
			cout << i << " : " << go_k[i] << " : " << ol_k[i] << endl;
		}
	}
	if (f_v) {
		cout << "poset_classification_activity::Asup_to_Ainf "
				"computing Ainf" << endl;
	}
	for (i = 0; i < Nt; i++) {
		for (j = 0; j < Nk; j++) {
			a = M_sup[i * Nk + j];
			aa.create(a);
			D.mult(ol_t[i], aa, bb);
			D.integral_division(bb, ol_k[j], cc, rem, 0);
			if (!rem.is_zero()) {
				cout << "poset_classification_activity::Asup_to_Ainf "
						"stabilizer order does not "
						"divide group order" << endl;
				cout << "i=" << i << " j=" << j
						<< " M_sup[i,j] = " << a
						<< " ol_t[i]=" << ol_t[i]
						<< " ol_k[j]=" << ol_k[j] << endl;
				exit(1);
			}
			c = cc.as_lint();
			M_inf[i * Nk + j] = c;
		}
	}
	if (f_v) {
		cout << "poset_classification_activity::Asup_to_Ainf "
				"computing Ainf done" << endl;
	}
	FREE_OBJECTS(go_t);
	FREE_OBJECTS(go_k);
	FREE_OBJECTS(ol_t);
	FREE_OBJECTS(ol_k);
	if (f_v) {
		cout << "poset_classification_activity::Asup_to_Ainf done" << endl;
	}
}

void poset_classification_activity::test_for_multi_edge_in_classification_graph(
		int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, f, l, j, h1;

	if (f_v) {
		cout << "poset_classification_activity::test_for_multi_edge_in_classification_graph "
				"depth=" << depth << endl;
	}
	for (i = 0; i <= depth; i++) {
		f = PC->get_Poo()->first_node_at_level(i);
		l = PC->nb_orbits_at_level(i);
		if (f_v) {
			cout << "poset_classification_activity::test_for_multi_edge_in_classification_graph "
					"level=" << i << " with " << l << " nodes" << endl;
		}
		for (j = 0; j < l; j++) {
			poset_orbit_node *O;

			//O = &root[f + j];
			O = PC->get_node_ij(i, j);
			for (h1 = 0; h1 < O->get_nb_of_extensions(); h1++) {
				extension *E1 = O->get_E(h1); // O->E + h1;

				if (E1->get_type() != EXTENSION_TYPE_FUSION) {
					continue;
				}

				//cout << "fusion (" << f + j << "/" << h1 << ") ->
				// (" << E1->data1 << "/" << E1->data2 << ")" << endl;
				if (E1->get_data1() == f + j) {
					cout << "multi_edge detected ! level "
							<< i << " with " << l << " nodes, "
							"fusion (" << j << "/" << h1 << ") -> "
							"(" << E1->get_data1() - f << "/"
							<< E1->get_data2() << ")" << endl;
				}

#if 0
				for (h2 = 0; h2 < O->get_nb_of_extensions(); h2++) {
					extension *E2 = O->E + h2;

					if (E2->get_type() != EXTENSION_TYPE_FUSION) {
						continue;

					if (E2->data1 == E1->data1 && E2->data2 == E1->data2) {
						cout << "multiedge detected!" << endl;
						cout << "fusion (" << f + j << "/" << h1
								<< ") -> (" << E1->get_data1() << "/"
								<< E1->get_data2() << ")" << endl;
						cout << "fusion (" << f + j << "/" << h2
								<< ") -> (" << E2->get_data1() << "/"
								<< E2->get_data2() << ")" << endl;
					}
				}
#endif

			}
		}
		if (f_v) {
			cout << "poset_classification_activity::test_for_multi_edge_in_classification_graph "
					"level=" << i << " with " << l << " nodes done" << endl;
		}
	}
	if (f_v) {
		cout << "poset_classification_activity::test_for_multi_edge_in_classification_graph "
				"done" << endl;
	}
}

void poset_classification_activity::Kramer_Mesner_matrix_neighboring(
		int level, long int *&M,
		int &nb_rows, int &nb_cols, int verbose_level)
// we assume that we don't use implicit fusion nodes
{
	int f_v = (verbose_level >= 1);
	int f_vv = false;//(verbose_level >= 2);
	int f1, f2, i, j, k, I, J, len;
	poset_orbit_node *O;

	if (f_v) {
		cout << "poset_classification_activity::Kramer_Mesner_matrix_neighboring "
				"level=" << level << endl;
	}

	f1 = PC->first_node_at_level(level);
	f2 = PC->first_node_at_level(level + 1);
	nb_rows = PC->nb_orbits_at_level(level);
	nb_cols = PC->nb_orbits_at_level(level + 1);

	M = NEW_lint(nb_rows * nb_cols);

	for (i = 0; i < nb_rows * nb_cols; i++) {
		M[i] = 0;
	}


	if (f_v) {
		cout << "poset_classification_activity::Kramer_Mesner_matrix_neighboring "
				"the size of the matrix is " << nb_rows << " x " << nb_cols << endl;
	}

	for (i = 0; i < nb_rows; i++) {
		if (f_vv) {
			cout << "poset_classification_activity::Kramer_Mesner_matrix_neighboring "
					"i=" << i << " / " << nb_rows << endl;
		}
		I = f1 + i;
		O = PC->get_node(I);
		for (k = 0; k < O->get_nb_of_extensions(); k++) {
			if (f_vv) {
				cout << "poset_classification_activity::Kramer_Mesner_matrix_neighboring "
						"i=" << i << " / " << nb_rows << " extension "
						<< k << " / " << O->get_nb_of_extensions() << endl;
			}
			if (O->get_E(k)->get_type() == EXTENSION_TYPE_EXTENSION) {
				if (f_vv) {
					cout << "poset_classification_activity::Kramer_Mesner_matrix_neighboring "
							"i=" << i << " / " << nb_rows << " extension "
							<< k << " / " << O->get_nb_of_extensions()
							<< " type extension node" << endl;
				}
				len = O->get_E(k)->get_orbit_len();
				J = O->get_E(k)->get_data();
				j = J - f2;
				M[i * nb_cols + j] += len;
			}
			if (O->get_E(k)->get_type() == EXTENSION_TYPE_FUSION) {
				if (f_vv) {
					cout << "poset_classification_activity::Kramer_Mesner_matrix_neighboring "
							"i=" << i << " / " << nb_rows << " extension "
							<< k << " / " << O->get_nb_of_extensions()
							<< " type fusion" << endl;
				}
				// fusion node
				len = O->get_E(k)->get_orbit_len();

				int I1, ext1;
				poset_orbit_node *O1;

				I1 = O->get_E(k)->get_data1();
				ext1 = O->get_E(k)->get_data2();
				O1 = PC->get_node(I1);
				if (O1->get_E(ext1)->get_type() != EXTENSION_TYPE_EXTENSION) {
					cout << "poset_classification_activity::Kramer_Mesner_matrix_neighboring "
							"O1->get_E(ext1)->type != EXTENSION_TYPE_EXTENSION "
							"something is wrong" << endl;
					exit(1);
				}
				J = O1->get_E(ext1)->get_data();

#if 0
				O->store_set(gen, level - 1);
					// stores a set of size level to gen->S
				gen->S[level] = O->E[k].pt;

				for (ii = 0; ii < level + 1; ii++) {
					gen->set[level + 1][ii] = gen->S[ii];
				}

				gen->A->element_one(gen->transporter->ith(level + 1), 0);

				J = O->apply_isomorphism(gen,
					level, I /* current_node */,
					//0 /* my_node */, 0 /* my_extension */, 0 /* my_coset */,
					k /* current_extension */, level + 1,
					false /* f_tolerant */,
					0/*verbose_level - 2*/);
				if (false) {
					cout << "after apply_isomorphism J=" << J << endl;
				}
#else

#endif




#if 0
				//cout << "fusion node:" << endl;
				//int_vec_print(cout, gen->S, level + 1);
				//cout << endl;
				gen->A->element_retrieve(O->E[k].data, gen->Elt1, 0);

				gen->A2->map_a_set(gen->S, gen->S0, level + 1, gen->Elt1, 0);
				//int_vec_print(cout, gen->S0, level + 1);
				//cout << endl;

				int_vec_heapsort(gen->S0, level + 1); //int_vec_sort(level + 1, gen->S0);

				//int_vec_print(cout, gen->S0, level + 1);
				//cout << endl;

				J = gen->find_poset_orbit_node_for_set(level + 1, gen->S0, 0);
#endif
				j = J - f2;
				M[i * nb_cols + j] += len;
			}
		}
	}
	if (f_v) {
		cout << "poset_classification_activity::Kramer_Mesner_matrix_neighboring "
				"level=" << level << " done" << endl;
	}
}

void poset_classification_activity::Mtk_via_Mtr_Mrk(
		int t, int r, int k,
		long int *Mtr, long int *Mrk, long int *&Mtk,
		int nb_r1, int nb_c1, int nb_r2, int nb_c2,
		int &nb_r3, int &nb_c3,
		int verbose_level)
// Computes $M_{tk}$ via a recursion formula:
// $M_{tk} = {{k - t} \choose {k - r}} \cdot M_{t,r} \cdot M_{r,k}$.
{
	int f_v = (verbose_level >= 1);
	int i, j, h, a, b, c, s = 0;
	combinatorics::combinatorics_domain C;

	if (f_v) {
		cout << "poset_classification_activity::Mtk_via_Mtr_Mrk "
				"t = " << t << ", r = "
				<< r << ", k = " << k << endl;
	}
	if (nb_c1 != nb_r2) {
		cout << "poset_classification_activity::Mtk_via_Mtr_Mrk "
				"nb_c1 != nb_r2" << endl;
		exit(1);
	}

	nb_r3 = nb_r1;
	nb_c3 = nb_c2;
	Mtk = NEW_lint(nb_r3 * nb_c3);
	for (i = 0; i < nb_r3; i++) {
		for (j = 0; j < nb_c3; j++) {
			c = 0;
			for (h = 0; h < nb_c1; h++) {
				a = Mtr[i * nb_c1 + h];
				b = Mrk[h * nb_c2 + j];
				c += a * b;
			}
			Mtk[i * nb_c3 + j] = c;
		}
	}


	//Mtk.mult(Mtr, Mrk);

	// Mtk := {(k - t) \atop (k - r)} * M_t,k


	ring_theory::longinteger_object S;

	if (PC->get_poset()->f_subset_lattice) {
		C.binomial(
				S, k - t, k - r,
				0/* verbose_level*/);
		s = S.as_lint();
	}
	else if (PC->get_poset()->f_subspace_lattice) {
		C.q_binomial(
				S, k - t, r - t, PC->get_poset()->VS->F->q,
				0/* verbose_level*/);
		s = S.as_lint();
	}
	else {
		cout << "poset_classification_activity::Mtk_via_Mtr_Mrk "
				"unknown type of lattice" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "poset_classification_activity::Mtk_via_Mtr_Mrk "
				"dividing by " << s << endl;
	}


	for (i = 0; i < nb_r3; i++) {
		for (j = 0; j < nb_c3; j++) {
			Mtk[i * nb_c3 + j] /= s;
		}
	}
	if (f_v) {
		cout << "poset_classification_activity::Mtk_via_Mtr_Mrk matrix "
				"M_{" << t << "," << k << "} "
						"of format " << nb_r3 << " x " << nb_c3
						<< " has been computed" << endl;
		}
}

void poset_classification_activity::Mtk_from_MM(
		long int **pM,
	int *Nb_rows, int *Nb_cols,
	int t, int k,
	long int *&Mtk, int &nb_r, int &nb_c,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "poset_classification_activity::Mtk_from_MM "
				"t = " << t << ", k = " << k << endl;
	}
	if (k == t) {
		cout << "poset_classification_activity::Mtk_from_MM "
				"k == t" << endl;
		exit(1);
	}

	long int *T;
	long int *T2;
	int Tr, Tc;
	int T2r, T2c;

	Tr = Nb_rows[t];
	Tc = Nb_cols[t];

	T = NEW_lint(Tr * Tc);
	for (i = 0; i < Tr; i++) {
		for (j = 0; j < Tc; j++) {
			T[i * Tc + j] = pM[t][i * Tc + j];
		}
	}
	if (f_v) {
		cout << "poset_classification_activity::Mtk_from_MM "
				"Tr=" << Tr << " Tc=" << Tc << endl;
	}

	if (t + 1 < k) {
		for (i = t + 2; i <= k; i++) {

			if (f_v) {
				cout << "poset_classification_activity::Mtk_from_MM "
						"i = " << i << " calling Mtk_via_Mtr_Mrk" << endl;
			}

			Mtk_via_Mtr_Mrk(
					t, i - 1, i,
				T, pM[i - 1], T2,
				Tr, Tc, Nb_rows[i - 1], Nb_cols[i - 1], T2r, T2c,
				verbose_level - 1);

			FREE_lint(T);
			T = T2;
			Tr = T2r;
			Tc = T2c;
			T2 = NULL;
		}
		Mtk = T;
		nb_r = Tr;
		nb_c = Tc;
	}
	else {
		Mtk = T;
		nb_r = Tr;
		nb_c = Tc;
	}


	if (f_v) {
		cout << "poset_classification_activity::Mtk_from_MM "
				"nb_r=" << nb_r << " nb_c=" << nb_c << endl;
	}

	if (f_v) {
		cout << "poset_classification_activity::Mtk_from_MM "
				"t = " << t << ", k = " << k << " done" << endl;
	}
}



}}}


