/*
 * geometric_object_description.cpp
 *
 *  Created on: Nov 9, 2019
 *      Author: anton
 */






#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {




geometric_object_description::geometric_object_description()
{
	P = NULL;

	f_subiaco_oval = FALSE;
	f_short = FALSE;
	f_subiaco_hyperoval = FALSE;
	f_adelaide_hyperoval = FALSE;

	f_hyperoval = FALSE;
	f_translation = FALSE;
	translation_exponent = 0;
	f_Segre = FALSE;
	f_Payne = FALSE;
	f_Cherowitzo = FALSE;
	f_OKeefe_Penttila = FALSE;

	f_BLT_database = FALSE;
	BLT_k = 0;
	f_BLT_in_PG = FALSE;

#if 0
	f_BLT_Linear = FALSE;
	f_BLT_Fisher = FALSE;
	f_BLT_Mondello = FALSE;
	f_BLT_FTWKB = FALSE;
#endif

	f_ovoid = FALSE;
	f_ovoid_ST = FALSE;

	f_Baer = FALSE;

	f_orthogonal = FALSE;
	orthogonal_epsilon = 0;

	f_hermitian = FALSE;

	f_cuspidal_cubic = FALSE; // cuspidal cubic in PG(2,q)
	f_twisted_cubic = FALSE; // twisted cubic in PG(3,q)

	f_elliptic_curve = FALSE;
	elliptic_curve_b = 0;
	elliptic_curve_c = 0;

	//f_Hill_cap_56 = FALSE;

	f_ttp_code = FALSE;
	f_ttp_construction_A = FALSE;
	f_ttp_hyperoval = FALSE;
	f_ttp_construction_B = FALSE;

	f_unital_XXq_YZq_ZYq = FALSE;

	f_desarguesian_line_spread_in_PG_3_q = FALSE;
	f_embedded_in_PG_4_q = FALSE;

	f_Buekenhout_Metz = FALSE;
	f_classical = FALSE;
	f_Uab = FALSE;
	parameter_a = 0;
	parameter_b = 0;

	f_whole_space = FALSE;
	f_hyperplane = FALSE;
	pt = 0;

	f_segre_variety = FALSE;
	segre_variety_a = 0;
	segre_variety_b = 0;

	f_Maruta_Hamada_arc = FALSE;

	f_projective_variety = FALSE;
	//variety_label;
	//variety_label_txt
	variety_degree = 0;
	variety_n = 0;
	//variety_coeffs;
	Monomial_ordering_type = t_PART;

	f_intersection_of_zariski_open_sets = FALSE;
	//Variety_coeffs

	f_number_of_conditions_satisfied = FALSE;
	//std::string number_of_conditions_satisfied_fname;


	f_projective_curve = FALSE;
	//curve_label_txt;
	//curve_label_tex;
	curve_nb_vars = 0;
	curve_degree = 0;
	//curve_coeffs = NULL;

	f_set = FALSE;
	//set_text;

}

geometric_object_description::~geometric_object_description()
{

}


int geometric_object_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int i;
	data_structures::string_tools ST;

	cout << "geometric_object_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-hyperoval") == 0) {
			f_hyperoval = TRUE;
			cout << "-hyperoval " << endl;
		}
		else if (ST.stringcmp(argv[i], "-subiaco_oval") == 0) {
			f_subiaco_oval = TRUE;
			f_short = ST.strtoi(argv[++i]);
			cout << "-subiaco_oval " << f_short << endl;
		}
		else if (ST.stringcmp(argv[i], "-subiaco_hyperoval") == 0) {
			f_subiaco_hyperoval = TRUE;
			cout << "-subiaco_hyperoval " << endl;
		}
		else if (ST.stringcmp(argv[i], "-adelaide_hyperoval") == 0) {
			f_adelaide_hyperoval = TRUE;
			cout << "-adelaide_hyperoval " << endl;
		}
		else if (ST.stringcmp(argv[i], "-translation") == 0) {
			f_translation = TRUE;
			translation_exponent = ST.strtoi(argv[++i]);
			cout << "-translation " << translation_exponent << endl;
		}
		else if (ST.stringcmp(argv[i], "-Segre") == 0) {
			f_Segre = TRUE;
			cout << "-segre" << endl;
		}
		else if (ST.stringcmp(argv[i], "-Payne") == 0) {
			f_Payne = TRUE;
			cout << "-Payne" << endl;
		}
		else if (ST.stringcmp(argv[i], "-Cherowitzo") == 0) {
			f_Cherowitzo = TRUE;
			cout << "-Cherowitzo" << endl;
		}
		else if (ST.stringcmp(argv[i], "-OKeefe_Penttila") == 0) {
			f_OKeefe_Penttila = TRUE;
			cout << "-OKeefe_Penttila" << endl;
		}


		else if (ST.stringcmp(argv[i], "-BLT_database") == 0) {
			f_BLT_database = TRUE;
			BLT_k = ST.strtoi(argv[++i]);
			cout << "-BLT_database " << BLT_k << endl;
		}
		else if (ST.stringcmp(argv[i], "-BLT_in_PG") == 0) {
			f_BLT_in_PG = TRUE;
			cout << "-BLT_in_PG " << endl;
		}

#if 0
		else if (stringcmp(argv[i], "-BLT_Linear") == 0) {
			f_BLT_Linear = TRUE;
			cout << "-BLT_Linear " << endl;
		}
		else if (stringcmp(argv[i], "-BLT_Fisher") == 0) {
			f_BLT_Fisher = TRUE;
			cout << "-BLT_Fisher " << endl;
		}
		else if (stringcmp(argv[i], "-BLT_Mondello") == 0) {
			f_BLT_Mondello = TRUE;
			cout << "-BLT_Mondello " << endl;
		}
		else if (stringcmp(argv[i], "-BLT_FTWKB") == 0) {
			f_BLT_FTWKB = TRUE;
			cout << "-BLT_FTWKB " << endl;
		}
#endif

		else if (ST.stringcmp(argv[i], "-ovoid") == 0) {
			f_ovoid = TRUE;
			cout << "-ovoid " << endl;
		}
		else if (ST.stringcmp(argv[i], "-ovoid_ST") == 0) {
			f_ovoid_ST = TRUE;
			cout << "-ovoid_ST " << endl;
		}
		else if (ST.stringcmp(argv[i], "-Baer") == 0) {
			f_Baer = TRUE;
			cout << "-Baer " << endl;
		}
		else if (ST.stringcmp(argv[i], "-orthogonal") == 0) {
			f_orthogonal = TRUE;
			orthogonal_epsilon = ST.strtoi(argv[++i]);
			cout << "-orthogonal " << orthogonal_epsilon << endl;
		}
		else if (ST.stringcmp(argv[i], "-hermitian") == 0) {
			f_hermitian = TRUE;
			cout << "-hermitian" << endl;
		}
		else if (ST.stringcmp(argv[i], "-cuspidal_cubic") == 0) {
			f_cuspidal_cubic = TRUE;
			cout << "-cuspidal_cubic " << endl;
		}
		else if (ST.stringcmp(argv[i], "-twisted_cubic") == 0) {
			f_twisted_cubic = TRUE;
			cout << "-twisted_cubic " << endl;
		}
		else if (ST.stringcmp(argv[i], "-elliptic_curve") == 0) {
			f_elliptic_curve = TRUE;
			elliptic_curve_b = ST.strtoi(argv[++i]);
			elliptic_curve_c = ST.strtoi(argv[++i]);
			cout << "-elliptic_curve " << elliptic_curve_b
					<< " " << elliptic_curve_c << endl;
		}
#if 0
		else if (stringcmp(argv[i], "-Hill_cap_56") == 0) {
			f_Hill_cap_56 = TRUE;
			cout << "-Hill_cap_56 " << endl;
		}
#endif
		else if (ST.stringcmp(argv[i], "-ttp_construction_A") == 0) {
			f_ttp_code = TRUE;
			f_ttp_construction_A = TRUE;
			cout << "-ttp_construction_A" << endl;
		}
		else if (ST.stringcmp(argv[i], "-ttp_construction_A_hyperoval") == 0) {
			f_ttp_code = TRUE;
			f_ttp_construction_A = TRUE;
			f_ttp_hyperoval = TRUE;
			cout << "-ttp_construction_A_hyperoval" << endl;
		}
		else if (ST.stringcmp(argv[i], "-ttp_construction_B") == 0) {
			f_ttp_code = TRUE;
			f_ttp_construction_B = TRUE;
			cout << "-ttp_construction_B" << endl;
		}
		else if (ST.stringcmp(argv[i], "-unital_XXq_YZq_ZYq") == 0) {
			f_unital_XXq_YZq_ZYq = TRUE;
			cout << "-unital_XXq_YZq_ZYq" << endl;
		}
		else if (ST.stringcmp(argv[i], "-desarguesian_line_spread_in_PG_3_q") == 0) {
			f_desarguesian_line_spread_in_PG_3_q = TRUE;
			cout << "-desarguesian_line_spread_in_PG_3_q" << endl;
		}
		else if (ST.stringcmp(argv[i], "-embedded_in_PG_4_q") == 0) {
			f_embedded_in_PG_4_q = TRUE;
			cout << "-embedded_in_PG_4_q" << endl;
		}
		else if (ST.stringcmp(argv[i], "-Buekenhout_Metz") == 0) {
			f_Buekenhout_Metz = TRUE;
			cout << "-Buekenhout_Metz " << endl;
		}
		else if (ST.stringcmp(argv[i], "-classical") == 0) {
			f_classical = TRUE;
			cout << "-classical " << endl;
		}
		else if (ST.stringcmp(argv[i], "-Uab") == 0) {
			f_Uab = TRUE;
			parameter_a = ST.strtoi(argv[++i]);
			parameter_b = ST.strtoi(argv[++i]);
			cout << "-Uab " << parameter_a << " " << parameter_b << endl;
		}
		else if (ST.stringcmp(argv[i], "-whole_space") == 0) {
			f_whole_space = TRUE;
			cout << "-whole_space " << endl;
		}
		else if (ST.stringcmp(argv[i], "-hyperplane") == 0) {
			f_hyperplane = TRUE;
			pt = ST.strtoi(argv[++i]);
			cout << "-hyperplane " << pt << endl;
		}
		else if (ST.stringcmp(argv[i], "-segre_variety") == 0) {
			f_segre_variety = TRUE;
			segre_variety_a = ST.strtoi(argv[++i]);
			segre_variety_b = ST.strtoi(argv[++i]);
			cout << "-segre_variety " << segre_variety_a
					<< " " << segre_variety_b << endl;
		}
		else if (ST.stringcmp(argv[i], "-Maruta_Hamada_arc") == 0) {
			f_Maruta_Hamada_arc = TRUE;
			cout << "-Maruta_Hamada_arc " << endl;
		}
		else if (ST.stringcmp(argv[i], "-projective_variety") == 0) {
			f_projective_variety = TRUE;
			variety_label_txt.assign(argv[++i]);
			variety_label_tex.assign(argv[++i]);
			variety_degree = ST.strtoi(argv[++i]);


			os_interface Os;

			i++;
			Os.get_string_from_command_line(variety_coeffs, argc, argv, i, verbose_level);
			i--;

			cout << "-projective_variety "
					<< variety_label_txt << " "
					<< variety_label_tex << " "
					<< variety_degree << " "
					<< variety_coeffs << endl;
		}
		else if (ST.stringcmp(argv[i], "-intersection_of_zariski_open_sets") == 0) {
			f_intersection_of_zariski_open_sets = TRUE;
			variety_label_txt.assign(argv[++i]);
			variety_label_tex.assign(argv[++i]);
			variety_degree = ST.strtoi(argv[++i]);
			variety_n = ST.strtoi(argv[++i]);

			int j;


			os_interface Os;

			i++;

			for (j = 0; j < variety_n; j++) {
				string s;

				cout << "reading argument " << j << " / " << variety_n << " : " << argv[i] << endl;

				Os.get_string_from_command_line(s, argc, argv, i, verbose_level);
				Variety_coeffs.push_back(s);
			}

			i--;


			cout << "-intersection_of_zariski_open_sets "
					<< variety_label_txt << " "
					<< variety_label_tex << " "
					<< variety_degree << " "
					<< variety_n << endl;
			for (j = 0; j < variety_n; j++) {
				cout << j << " : " << Variety_coeffs[j] << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-number_of_conditions_satisfied") == 0) {
			f_number_of_conditions_satisfied = TRUE;
			variety_label_txt.assign(argv[++i]);
			variety_label_tex.assign(argv[++i]);
			variety_degree = ST.strtoi(argv[++i]);
			variety_n = ST.strtoi(argv[++i]);

			number_of_conditions_satisfied_fname.assign(argv[++i]);


			int j;

			os_interface Os;

			i++;

			for (j = 0; j < variety_n; j++) {
				string s;

				cout << "reading argument " << j << " / " << variety_n << " : " << argv[i] << endl;

				Os.get_string_from_command_line(s, argc, argv, i, verbose_level);
				Variety_coeffs.push_back(s);
			}

			i--;


			cout << "-number_of_conditions_satisfied "
					<< variety_label_txt << " "
					<< variety_label_tex << " "
					<< variety_degree << " "
					<< variety_n << endl;
			for (j = 0; j < variety_n; j++) {
				cout << j << " : " << Variety_coeffs[j] << endl;
			}
		}



		else if (ST.stringcmp(argv[i], "-projective_curve") == 0) {
			f_projective_curve = TRUE;
			curve_label_txt.assign(argv[++i]);
			curve_label_tex.assign(argv[++i]);
			curve_nb_vars = ST.strtoi(argv[++i]);
			curve_degree = ST.strtoi(argv[++i]);
			curve_coeffs.assign(argv[++i]);
			cout << "-projective_curve "
					<< curve_label_txt << " "
					<< curve_label_tex << " "
					<< curve_nb_vars << " "
					<< curve_degree << " "
					<< curve_coeffs << endl;
		}
		else if (ST.stringcmp(argv[i], "-monomial_type_LEX") == 0) {
			Monomial_ordering_type = t_LEX;
			cout << "-monomial_type_LEX " << endl;
		}
		else if (ST.stringcmp(argv[i], "-monomial_type_PART") == 0) {
			Monomial_ordering_type = t_PART;
			cout << "-monomial_type_PART " << endl;
		}
		else if (ST.stringcmp(argv[i], "-set") == 0) {
			f_set = TRUE;
			set_text.assign(argv[++i]);
			cout << "-set " << set_text << endl;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "geometric_object_description::read_arguments unknown command " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "geometric_object_description::read_arguments done" << endl;
	return i + 1;
}

void geometric_object_description::print()
{
	if (f_hyperoval) {
		cout << "-hyperoval " << endl;
	}
	if (f_subiaco_oval) {
		cout << "-subiaco_oval " << f_short << endl;
	}
	if (f_subiaco_hyperoval) {
		cout << "-subiaco_hyperoval " << endl;
	}
	if (f_adelaide_hyperoval) {
		cout << "-adelaide_hyperoval " << endl;
	}
	if (f_translation) {
		cout << "-translation " << translation_exponent << endl;
	}
	if (f_Segre) {
		cout << "-segre" << endl;
	}
	if (f_Payne) {
		cout << "-Payne" << endl;
	}
	if (f_Cherowitzo) {
		cout << "-Cherowitzo" << endl;
	}
	if (f_OKeefe_Penttila) {
		cout << "-OKeefe_Penttila" << endl;
	}


	if (f_BLT_database) {
		cout << "-BLT_database " << BLT_k << endl;
	}
	if (f_BLT_in_PG) {
		cout << "-BLT_in_PG " << endl;
	}

	if (f_ovoid) {
		cout << "-ovoid " << endl;
	}
	if (f_ovoid_ST) {
		cout << "-ovoid_ST " << endl;
	}
	if (f_Baer) {
		cout << "-Baer " << endl;
	}
	if (f_orthogonal) {
		cout << "-orthogonal " << orthogonal_epsilon << endl;
	}
	if (f_hermitian) {
		cout << "-hermitian" << endl;
	}
	if (f_cuspidal_cubic) {
		cout << "-cuspidal_cubic " << endl;
	}
	if (f_twisted_cubic) {
		f_twisted_cubic = TRUE;
		cout << "-twisted_cubic " << endl;
	}
	if (f_elliptic_curve) {
		cout << "-elliptic_curve " << elliptic_curve_b
				<< " " << elliptic_curve_c << endl;
	}
	if (f_ttp_construction_A) {
		cout << "-ttp_construction_A" << endl;
	}
	if (f_ttp_construction_A && f_hyperoval) {
		cout << "-ttp_construction_A_hyperoval" << endl;
	}
	if (f_ttp_construction_B) {
		cout << "-ttp_construction_B" << endl;
	}
	if (f_unital_XXq_YZq_ZYq) {
		cout << "-unital_XXq_YZq_ZYq" << endl;
	}
	if (f_desarguesian_line_spread_in_PG_3_q) {
		cout << "-desarguesian_line_spread_in_PG_3_q" << endl;
	}
	if (f_embedded_in_PG_4_q) {
		cout << "-embedded_in_PG_4_q" << endl;
	}
	if (f_Buekenhout_Metz) {
		cout << "-Buekenhout_Metz " << endl;
	}
	if (f_classical) {
		cout << "-classical " << endl;
	}
	if (f_Uab) {
		cout << "-Uab " << parameter_a << " " << parameter_b << endl;
	}
	if (f_whole_space) {
		cout << "-whole_space " << endl;
	}
	if (f_hyperplane) {
		cout << "-hyperplane " << pt << endl;
	}
	if (f_segre_variety) {
		cout << "-segre_variety " << segre_variety_a
				<< " " << segre_variety_b << endl;
	}
	if (f_Maruta_Hamada_arc) {
		cout << "-Maruta_Hamada_arc " << endl;
	}
	if (f_projective_variety) {
		cout << "-projective_variety "
				<< variety_label_txt << " "
				<< variety_label_tex << " "
				<< variety_degree << " "
				<< variety_coeffs << endl;
	}
	if (f_intersection_of_zariski_open_sets) {
		cout << "-intersection_of_zariski_open_sets "
				<< variety_label_txt << " "
				<< variety_label_tex << " "
				<< variety_degree << " "
				<< variety_n << endl;
		int j;
		for (j = 0; j < variety_n; j++) {
			cout << j << " : " << Variety_coeffs[j] << endl;
		}
	}


	if (f_number_of_conditions_satisfied) {
		cout << "-number_of_conditions_satisfied "
				<< variety_label_txt << " "
				<< variety_label_tex << " "
				<< variety_degree << " "
				<< variety_n << endl;
		int j;
		for (j = 0; j < variety_n; j++) {
			cout << j << " : " << Variety_coeffs[j] << endl;
		}
	}



	if (f_projective_curve) {
		cout << "-projective_curve "
				<< curve_label_txt << " "
				<< curve_label_tex << " "
				<< curve_nb_vars << " "
				<< curve_degree << " "
				<< curve_coeffs << endl;
	}
	if (f_set) {
		cout << "-set " << set_text << endl;
	}
}




}}}


