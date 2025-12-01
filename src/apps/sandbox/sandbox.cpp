/*
 * sandbox.cpp
 *
 *  Created on: Apr 2, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;
using namespace orbiter;


int main()
{

	int verbose_level = 2;
	int f_v = (verbose_level >= 1);

	orbiter::layer5_applications::user_interface::orbiter_top_level_session Top_level_session;
	orbiter::layer5_applications::user_interface::The_Orbiter_top_level_session = &Top_level_session;

	algebra::field_theory::finite_field *F0, *F1;

	F0 = NEW_OBJECT(algebra::field_theory::finite_field);
	F1 = NEW_OBJECT(algebra::field_theory::finite_field);

	//int q = 16; int e = 4;
	int q = 27; int e = 3;
	int f_without_tables = false;
	int f_compute_related_fields = true;

	F0->finite_field_init_small_order(
			q,
			false /*f_without_tables*/, f_compute_related_fields,
			verbose_level);

	F1->finite_field_init_small_order(
			q,
			true /*f_without_tables*/, f_compute_related_fields,
			verbose_level);


	int i, j, k1, k2;

	cout << "checking addition..." << endl;
	for (i = 0; i < q; i++) {
		for (j = 0; j < q; j++) {
			k1 = F0->add(i, j);
			k2 = F1->add(i, j);
			if (k1 != k2) {
				cout << "error in i+j=k, i=" << i << " j=" << j << " k1=" << k1 << " k2=" << k2 << endl;
				exit(1);
			}
		}
	}

	cout << "checking multiplication..." << endl;
	for (i = 0; i < q; i++) {
		for (j = 0; j < q; j++) {
			k1 = F0->mult(i, j);
			k2 = F1->mult(i, j);
			if (k1 != k2) {
				cout << "error in i*j=k, i=" << i << " j=" << j << " k1=" << k1 << " k2=" << k2 << endl;
				exit(1);
			}
		}
	}

	cout << "checking inverse..." << endl;
	for (i = 1; i < q; i++) {
		k1 = F0->inverse(i);
		k2 = F1->inverse_v(i, 0);
		if (k1 != k2) {
			cout << "error in i^-1=k, i=" << i << " k1=" << k1 << " k2=" << k2 << endl;
			exit(1);
		}
	}

	cout << "checking alpha_power..." << endl;
	for (i = 1; i < q; i++) {
		k1 = F0->alpha_power(i);
		k2 = F1->alpha_power(i);
		if (k1 != k2) {
			cout << "error in alpha^i=k, i=" << i << " k1=" << k1 << " k2=" << k2 << endl;
			exit(1);
		}
	}

	cout << "checking frob_power..." << endl;
	for (i = 0; i < q; i++) {
		for (j = 0; j < e; j++) {
			k1 = F0->frobenius_power(i, j, 0 /* verbose_level */);
			k2 = F1->frobenius_power(i, j, 0 /* verbose_level */);
			if (k1 != k2) {
				cout << "error in frobenius_power(i,j)=k, i=" << i << " j=" << j << " k1=" << k1 << " k2=" << k2 << endl;
				exit(1);
			}
		}
	}

#if 0

	// compute quartics in characteristic 2 by brute force:

	int verbose_level = 2;
	int f_v = (verbose_level >= 1);

	orbiter::layer5_applications::user_interface::orbiter_top_level_session Top_level_session;
	orbiter::layer5_applications::user_interface::The_Orbiter_top_level_session = &Top_level_session;

	algebra::field_theory::finite_field *F;

	F = NEW_OBJECT(algebra::field_theory::finite_field);

	int q = 4;
	int f_without_tables = false;
	int f_compute_related_fields = true;

	F->finite_field_init_small_order(
			q,
			f_without_tables, f_compute_related_fields,
			verbose_level);


	layer5_applications::projective_geometry::projective_space_with_action *PA;

	PA = NEW_OBJECT(layer5_applications::projective_geometry::projective_space_with_action);

	if (f_v) {
		cout << "symbol_definition::definition_of_projective_space "
				"before PA->init_from_description" << endl;
	}

	int n = 2;
	int f_semilinear = true;

	PA->init(
			F,
			n, f_semilinear,
			true /* f_init_incidence_structure */,
			verbose_level);


	algebra::ring_theory::homogeneous_polynomial_domain *HPD;

	HPD = NEW_OBJECT(algebra::ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "symbol_definition::definition_of_polynomial_ring "
				"before HPD->init" << endl;
	}

	int number_of_variables = n + 1;
	int homogeneous_of_degree = 4;
	monomial_ordering_type Monomial_ordering_type = t_PART;

	HPD->init_with_or_without_variables(
			F,
			number_of_variables,
			homogeneous_of_degree,
			Monomial_ordering_type,
			false, NULL, NULL,
			verbose_level);


	other::orbiter_kernel_system::file_io Fio;
	std::string fname_base;
	std::string fname;
	std::string col_label;

	other::data_structures::set_of_sets *SoS;
	other::data_structures::set_of_sets *SoS_stab_order;
	other::data_structures::set_of_sets *SoS_orbit_length;

	fname_base = "poly_orbits_d4_n2_q4";
	fname = fname_base + ".csv";
	col_label = "Rep";


	Fio.Csv_file_support->read_column_as_set_of_sets(
			fname, col_label,
			SoS,
			verbose_level);


	cout << "number of orbits: " << SoS->nb_sets << endl;

	col_label = "StabOrder";

	Fio.Csv_file_support->read_column_as_set_of_sets(
			fname, col_label,
			SoS_stab_order,
			verbose_level);

	col_label = "OrbitLength";

	Fio.Csv_file_support->read_column_as_set_of_sets(
			fname, col_label,
			SoS_orbit_length,
			verbose_level);



	int nb_rows;
	int nb_cols;

	string *Table;
	string *Table_ns;

	nb_rows = SoS->nb_sets;
	nb_cols = 6;

	Table = new string [nb_rows * nb_cols];
	Table_ns = new string [nb_rows * nb_cols];


	int cnt_ns = 0;
	int cnt = 0;
	int po_go = 0;
	int po_index = 0;
	int po = 0;
	int so = 0;

	geometry::algebraic_geometry::variety_description *Variety_description;


	Variety_description = NEW_OBJECT(geometry::algebraic_geometry::variety_description);

	cnt_ns = 0;
	for (cnt = 0; cnt < SoS->nb_sets; cnt++) {

		layer5_applications::canonical_form::variety_object_with_action *Variety;

		Variety = NEW_OBJECT(layer5_applications::canonical_form::variety_object_with_action);


		Variety_description->f_projective_space_pointer = true;
		Variety_description->Projective_space_pointer = PA->P;

		Variety_description->f_ring_pointer = true;
		Variety_description->Ring_pointer = HPD;

		Variety_description->f_equation_by_rank = true;
		Variety_description->equation_by_rank_text = std::to_string(SoS->Sets[cnt][0]);

		if (f_v) {
			cout << "symbol_definition::definition_of_variety "
					"before Variety->create_variety" << endl;
		}
		Variety->create_variety(
				PA, cnt, po_go, po_index, po, so, Variety_description,
				verbose_level);

		int nb_s;

		Variety->Variety_object->compute_singular_points(
					verbose_level - 2);

		if (Variety->Variety_object->f_has_singular_points) {


			nb_s = Variety->Variety_object->Singular_points.size();

		}
		else {
			nb_s = -1;
		}

		Table[cnt * nb_cols + 0] = std::to_string(cnt);
		Table[cnt * nb_cols + 1] = std::to_string(SoS->Sets[cnt][0]);
		Table[cnt * nb_cols + 2] = std::to_string(SoS_stab_order->Sets[cnt][0]);
		Table[cnt * nb_cols + 3] = std::to_string(SoS_orbit_length->Sets[cnt][0]);
		Table[cnt * nb_cols + 4] = std::to_string(Variety->Variety_object->get_nb_points());
		Table[cnt * nb_cols + 5] = std::to_string(nb_s);


		if (nb_s == 0) {

			Table_ns[cnt_ns * nb_cols + 0] = std::to_string(cnt);
			Table_ns[cnt_ns * nb_cols + 1] = std::to_string(SoS->Sets[cnt][0]);
			Table_ns[cnt_ns * nb_cols + 2] = std::to_string(SoS_stab_order->Sets[cnt][0]);
			Table_ns[cnt_ns * nb_cols + 3] = std::to_string(SoS_orbit_length->Sets[cnt][0]);
			Table_ns[cnt_ns * nb_cols + 4] = std::to_string(Variety->Variety_object->get_nb_points());
			Table_ns[cnt_ns * nb_cols + 5] = std::to_string(nb_s);
			cnt_ns++;

		}

		cout << "cnt = " << cnt << " nb_s = " << nb_s << endl;

	}


	std::string fname_out;

	fname_out = fname_base + "_properties.csv";

	std::string *Col_headings;

	Col_headings = new string [nb_cols];

	Col_headings[0] = "idx";
	Col_headings[1] = "rep";
	Col_headings[2] = "stab";
	Col_headings[3] = "orbitlength";
	Col_headings[4] = "nbpts";
	Col_headings[5] = "nbsingular";


	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname_out,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	if (f_v) {
		cout << "sandbox "
				"written file " << fname_out << " of size "
				<< Fio.file_size(fname_out) << endl;
	}

	fname_out = fname_base + "_ns.csv";

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname_out,
			cnt_ns, nb_cols, Table_ns,
			Col_headings,
			verbose_level);

	if (f_v) {
		cout << "sandbox "
				"written file " << fname_out << " of size "
				<< Fio.file_size(fname_out) << endl;
	}

	delete [] Col_headings;
	delete [] Table;
#endif




#if 0
	orbiter::layer5_applications::user_interface::orbiter_top_level_session Top_level_session;
	orbiter::layer5_applications::user_interface::The_Orbiter_top_level_session = &Top_level_session;

	other::orbiter_kernel_system::file_io Fio;
	std::string fname;
	std::string col_label;

	other::data_structures::set_of_sets *SoS;

	fname = "qc_select_EK_tally.csv";
	col_label = "TYPE";


	Fio.Csv_file_support->read_column_as_set_of_sets(
			fname, col_label,
			SoS,
			verbose_level);
	int nb_E[] = {18,10,9,6,4,3,2,1,0};
	int nb_K[] = {63,21,15,9,7,5,3,1,0};
	int nb_E_sz = sizeof(nb_E) / sizeof(int);
	int nb_K_sz = sizeof(nb_K) / sizeof(int);
	int *T;
	int h;
	int a, b;
	int idx1, idx2;
	other::data_structures::sorting Sorting;

	T = NEW_int(nb_E_sz * nb_K_sz);
	Int_vec_zero(T, nb_E_sz * nb_K_sz);
	for (h = 0; h < SoS->nb_sets; h++) {
		if (SoS->Set_size[h] != 2) {
			cout << "SoS->Set_size[h] != 2" << endl;
			exit(1);
		}
		a = SoS->Sets[h][0];
		b = SoS->Sets[h][1];


		if (!Sorting.int_vec_search_linear(
				nb_E, nb_E_sz, a, idx1)) {
			cout << "cannot find number of Eckardt points, a=" << a << endl;
			exit(1);
		}
		if (!Sorting.int_vec_search_linear(
				nb_K, nb_K_sz, b, idx2)) {
			cout << "cannot find number of Kovalevski points, b=" << b << endl;
			exit(1);
		}
		T[idx1 * nb_K_sz + idx2] = 1;
	}
	string *Table;
	int nb_rows = nb_E_sz + 1;
	int nb_cols = nb_K_sz + 1;
	int i, j;

	Table = new string[nb_rows * nb_cols];

	for (i = 0; i < nb_E_sz; i++) {
		Table[(i + 1) * nb_cols + 0] = std::to_string(nb_E[i]);
	}
	for (j = 0; j < nb_K_sz; j++) {
		Table[0 * nb_cols + 1 + j] = std::to_string(nb_K[j]);
	}
	for (i = 0; i < nb_E_sz; i++) {
		for (j = 0; j < nb_K_sz; j++) {
			Table[(i + 1) * nb_cols + 1 + j] = std::to_string(T[i * nb_K_sz + j]);
		}
	}

	std::string *Col_headings;

	Col_headings = new string[nb_cols];

	Col_headings[0] = "E";
	for (j = 0; j < nb_K_sz; j++) {
		Col_headings[1 + j] = "K" + std::to_string(nb_K[j]);

	}

	string fname_out;


	fname_out = "qc_select_EK_tally_table.csv";


	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname_out,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	delete [] Table;
#endif


#if 0
	int List[] = {
			4,
			7,
			8,
			9,
			11,
			13,
			16,
			17,
			19,
			23,
			25,
			27,
			29,
			31,
			32,
			37,
			41,
			43,
			47,
			49,
			53,
			59,
			61,
			64,
			67,
			71,
			73,
			79,
			81,
			83,
			89,
			97,
			101,
			103,
			107,
			109,
			113,
			121,
			127,
			128};

	int i, q;
	int nb;

	orbiter::layer5_applications::user_interface::orbiter_top_level_session Top_level_session;
	orbiter::layer5_applications::user_interface::The_Orbiter_top_level_session = &Top_level_session;
	combinatorics::knowledge_base::knowledge_base K;

	for (i = 0; i < sizeof(List) / sizeof(int); i++) {
		q = List[i];
		nb = K.cubic_surface_nb_reps(q);
		cout << "NB_CUBIC_SURFACES_Q" << q << "=" << nb << endl;
	}
#endif
#if 0
	int i, j;
	int n = 7;
	int *v;

	orbiter::layer5_applications::user_interface::orbiter_top_level_session Top_level_session;
	orbiter::layer5_applications::user_interface::The_Orbiter_top_level_session = &Top_level_session;
	geometry::other_geometry::geometry_global Gg;

	v = NEW_int(n);
	for (i = 0; i < 128; i++) {
		Gg.AG_element_unrank(2, v, 1, n, i);
		cout << i << " : ";
		other::orbiter_kernel_system::Orbiter->Int_vec->print_dense_bare(
				cout, v, n);
		cout << "\\\\" << endl;
	}
#endif

#if 0
	int i, j;
	for (i = 0; i < 32; i++) {
		cout << "split_" << i << ":" << endl;
		for (j = 0; j < 4096; j++) {
			if ((j % 32) == i) {
				cout << "\tmake -f split_4096.mk PG_3_3_packings_canonicize_" << j << "_mod_4096" << endl;
			}
		}
		cout << endl;
	}
#endif


#if 0
	string mask;

	mask =
	"PG_3_3_packings_canonicize_%d_mod_4096:\n\
	$(ORBITER) -v 2 \\\n\
		-define C -combinatorial_object \\\n\
			-label case_%d_mod_4096 \\\n\
			\"{\\\\rm PG\\_3\\_3\\_packings}\" \\\n\
			-file_of_packings_through_spread_table \\\n\
				spread_9_disjoint_starter_sz_2_combined_sol_split_%d_mod_4096.csv \\\n\
				SPREAD_TABLES_3/spread_9_spreads.csv \\\n\
				3 \\\n\
		-end \\\n\
		-define F -finite_field -q 3 -end \\\n\
		-define P3 -projective_space -n 3 -field F -v 0 -end \\\n\
		-with C -do \\\n\
		-combinatorial_object_activity \\\n\
			-canonical_form_PG P3 \\\n\
			-nauty_control -end \\\n\
			-end \\\n\
		-end\n";

	char str[10000];


	int i;
	for (i = 0; i < 4096; i++) {
		snprintf(str, sizeof(str), mask.c_str(), i,i,i,i,i,i);

		cout << endl;
		cout << str << endl;
		cout << endl;

	}
#endif


#if 0
	algebra::field_theory::finite_field_description Descr;
	algebra::field_theory::finite_field Fq;


	Descr.f_q = true;
	Descr.q_text.assign("11");
	Fq.init(&Descr, 1 /* verbose_level */);
#endif


#if 0
	string mask;

	mask =
	"Dickson1_q2_eqn_%d:\n\
		$(ORBITER) -v 6 \\\n\
			-define F -finite_field -q 2 -end \\\n\
			-define P -projective_space -n 3 -field F -v 0 -end \\\n\
			-define R -polynomial_ring \\\n\
				-field F \\\n\
				-number_of_variables 4 \\\n\
				-homogeneous_of_degree 3 \\\n\
				-monomial_ordering_partition \\\n\
				-variables \"X0,X1,X2,X3\" \"X_0,X_1,X_2,X_3\" \\\n\
			-end \\\n\
			-define V -variety \\\n\
				-projective_space P \\\n\
				-ring R \\\n\
				-equation_by_coefficients \\\n\
					$(DICKSON_1_Q2_EQN_%d) \\\n\
				-label_txt Dickson_1_q2_eqn_%d \\\n\
				-label_tex \"{\\rm Dickson\\_1\\_q2\\_eqn\\_%d}\" \\\n\
				-compute_lines \\\n\
			-end \\\n\
			-with V -do -variety_activity \\\n\
				-compute_group \\\n\
				-nauty_control -save_orbit_of_equations eqn_ -end \\\n\
			-end \\\n\
			-with V -do -variety_activity \\\n\
				-compute_set_stabilizer \\\n\
				-nauty_control -save_orbit_of_equations eqn_ -end \\\n\
			-end \\\n\
			-with V -do -variety_activity -singular_points -end \\\n\
			-with V -do -variety_activity -report -end \\\n\
			-with V -do -variety_activity -export -end\n\
		pdflatex variety_Dickson_1_q2_eqn_%d_report.tex\n\
		open variety_Dickson_1_q2_eqn_%d_report.pdf\n";

	char str[10000];


	int i;
	for (i = 1; i < 28; i++) {
		snprintf(str, sizeof(str), mask.c_str(), i,i,i,i,i,i);

		cout << endl;
		cout << str << endl;
		cout << endl;

	}
#endif

#if 0
	int i;
	for (i = 1; i < 28; i++) {
		cout << "\tmake Dickson1_q2_eqn_" << i << endl;
	}
#endif



#if 0
	string mask;

	mask =
	"Dickson%d_group:\n\
		$(ORBITER) -v 6 \\\n\
			-define F -finite_field -q 16 -end \\\n\
			-define P -projective_space -n 3 -field F -v 0 -end \\\n\
			-define R -polynomial_ring \\\n\
				-field F \\\n\
				-number_of_variables 4 \\\n\
				-homogeneous_of_degree 3 \\\n\
				-monomial_ordering_partition \\\n\
				-variables \"X0,X1,X2,X3\" \"X_0,X_1,X_2,X_3\" \\\n\
			-end \\\n\
			-define V -variety \\\n\
				-projective_space P \\\n\
				-ring R \\\n\
				-equation_in_algebraic_form \\\n\
					$(Dickson%d_eqn_af) \\\n\
				-label_txt Dickson_%d \\\n\
				-label_tex \"{\\rm Dickson\\_%d}\" \\\n\
				-compute_lines \\\n\
			-end \\\n\
			-with V -do -variety_activity \\\n\
				-compute_group \\\n\
				-nauty_control -save_orbit_of_equations eqn_ -end \\\n\
			-end \\\n\
			-with V -do -variety_activity \\\n\
				-compute_set_stabilizer \\\n\
				-nauty_control -save_orbit_of_equations eqn_ -end \\\n\
			-end \\\n\
			-with V -do -variety_activity -singular_points -end \\\n\
			-with V -do -variety_activity -report -end \\\n\
			-with V -do -variety_activity -export -end\n\
		pdflatex variety_Dickson_%d_report.tex\n\
		open variety_Dickson_%d_report.pdf\n";

	char str[10000];


	int i;
	for (i = 1; i < 48; i++) {
		snprintf(str, sizeof(str), mask.c_str(), i,i,i,i,i,i);

		cout << endl;
		cout << str << endl;
		cout << endl;

	}
#endif

#if 0
	int i;
	for (i = 1; i < 48; i++) {
		cout << "\t\t\t-define Dickson" << i << " -text -here $(Dickson" << i << "_eqn_af) -end \\" << endl;
	}
#endif


#if 0
	int i;
	for (i = 1; i < 48; i++) {
		cout << "\tmake Dickson" << i << "_group" << endl;
	}
#endif

#if 0
	int i;
	for (i = 1; i < 48; i++) {
		cout << "\t\tvariety_Dickson_" << i << "_report.pdf \\" << endl;
	}
#endif

#if 0
	char str[10000];
	int i;

	for (i = 0; i < 10000; i++) {
		snprintf(str, sizeof(str), "/scratch2/betten/COMPILE/orbiter/src/apps/orbiter/orbiter.out -v 3 -seed %d "
				"-introduce_errors "
				"-input log_c1_0_crc32.bin "
				"-output log_c1_0_crc32_e.bin "
				"-block_based_error_generator "
				"-block_length 771 "
				"-threshold 100000 "
				"-file_based_error_generator 100000 "
				"-nb_repeats 30 "
				"-end",
				1000 + i);
		system(str);
		snprintf(str, sizeof(str), "/scratch2/betten/COMPILE/orbiter/src/apps/orbiter/orbiter.out -v 3 "
				"-check_errors "
				"-input log_c1_0_crc32_e.bin "
				"-output log_c1_0_recovered.txt "
				"-error_log log_c1_0_crc32_e_pattern.csv "
				"-block_length 771 "
				"-end");
		system(str);
	}
#endif
#if 0
	int verbose_level = 10;
	data_structures::algorithms Algo;

	std::string fname_set_of_sets, fname_input, fname_output;

	fname_set_of_sets.assign("doily.csv");
	fname_input.assign("doily_cliques.csv");
	fname_output.assign("doily_cliques_union.csv");

	Algo.union_of_sets(fname_set_of_sets,
			fname_input, fname_output, verbose_level);

#endif
#if 0
	finite_field F;
	F.finite_field_init(16, false /* f_without_tables */, 0);

	cout << "8 x 15 = " << F.mult(8, 15) << endl;
#endif
#if 0
	int verbose_level = 2;
	int nb_gens = 3;
	int base_len = 2;
	long int given_base[] = {0, 8};
	int generators[] = {
			0, 4, 3, 2, 1, 6, 5, 11, 9, 8, 18, 7, 15, 16, 17, 12, 13, 14, 10, 20, 19, 22, 21, 23, 25, 24, 26, 34, 33, 31, 32, 29, 30, 28, 27, 37, 38, 35, 36, 40, 39, 46, 42, 45, 47, 43, 41, 44,
			0, 2, 1, 6, 5, 4, 3, 11, 12, 13, 14, 7, 8, 9, 10, 16, 15, 20, 19, 18, 17, 22, 21, 23, 24, 26, 25, 29, 30, 27, 28, 37, 38, 36, 35, 34, 33, 31, 32, 43, 41, 40, 44, 39, 42, 46, 45, 47,
			1, 0, 2, 8, 7, 9, 10, 4, 3, 5, 6, 13, 14, 11, 12, 18, 21, 17, 15, 22, 23, 16, 19, 20, 24, 27, 28, 25, 26, 30, 29, 33, 39, 31, 35, 34, 40, 41, 42, 32, 36, 37, 38, 44, 43, 46, 45, 47,
	};
	int degree = 48;
	int target_go_lint = 144;
	int nb_rows = 24;
	ring_theory::longinteger_object target_go;

	target_go.create(target_go_lint, __FILE__, __LINE__);
	actions::action *A;
	int f_no_base = false;

	A = NEW_OBJECT(actions::action);

	A->init_permutation_group_from_generators(degree,
			true /*  f_target_go */, target_go,
			nb_gens, generators,
			base_len, given_base,
			f_no_base,
			verbose_level);

	A->Strong_gens->print_generators_in_latex_individually(cout);
	A->Strong_gens->print_generators_in_source_code();
	A->print_base();
	A->print_info();

	actions::action *A2;

	groups::schreier *Sch;


	A2 = A->induced_action_on_interior_direct_product(nb_rows, verbose_level);

	A2->print_info();
	A2->compute_orbits_on_points(Sch, A->Strong_gens->gens, verbose_level);

	cout << "Orbit:" << endl;
	Sch->print_and_list_all_orbits_and_stabilizers_with_list_of_elements_tex(
			cout, A /*default_action*/, A->Strong_gens,
			verbose_level);

#endif
}

