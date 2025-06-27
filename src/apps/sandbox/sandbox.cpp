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

