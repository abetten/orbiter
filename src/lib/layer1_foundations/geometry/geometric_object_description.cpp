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

	f_subiaco_oval = false;
	f_short = false;
	f_subiaco_hyperoval = false;
	f_adelaide_hyperoval = false;

	f_hyperoval = false;
	f_translation = false;
	translation_exponent = 0;
	f_Segre = false;
	f_Payne = false;
	f_Cherowitzo = false;
	f_OKeefe_Penttila = false;

	f_BLT_database = false;
	BLT_database_k = 0;
	f_BLT_database_embedded = false;
	BLT_database_embedded_k = 0;

#if 0
	f_BLT_Linear = false;
	f_BLT_Fisher = false;
	f_BLT_Mondello = false;
	f_BLT_FTWKB = false;
#endif

	f_elliptic_quadric_ovoid = false;
	f_ovoid_ST = false;

	f_Baer_substructure = false;

	f_orthogonal = false;
	orthogonal_epsilon = 0;

	f_hermitian = false;

	f_cuspidal_cubic = false; // cuspidal cubic in PG(2,q)
	f_twisted_cubic = false; // twisted cubic in PG(3,q)

	f_elliptic_curve = false;
	elliptic_curve_b = 0;
	elliptic_curve_c = 0;

	//f_Hill_cap_56 = false;

	f_ttp_code = false;
	f_ttp_construction_A = false;
	f_ttp_hyperoval = false;
	f_ttp_construction_B = false;

	f_unital_XXq_YZq_ZYq = false;

	f_desarguesian_line_spread_in_PG_3_q = false;
	f_embedded_in_PG_4_q = false;

	f_Buekenhout_Metz = false;
	f_classical = false;
	f_Uab = false;
	parameter_a = 0;
	parameter_b = 0;

	f_whole_space = false;
	f_hyperplane = false;
	pt = 0;

	f_segre_variety = false;
	segre_variety_a = 0;
	segre_variety_b = 0;

	f_Maruta_Hamada_arc = false;

	f_projective_variety = false;
	//projective_variety_ring_label
	//variety_label;
	//variety_label_txt
	//variety_coeffs;
	//Monomial_ordering_type = t_PART;


	f_intersection_of_zariski_open_sets = false;
	//Variety_coeffs

	f_number_of_conditions_satisfied = false;
	//std::string number_of_conditions_satisfied_fname;


	f_projective_curve = false;
	//projective_curve_ring_label
	//curve_label_txt;
	//curve_label_tex;
	//curve_coeffs = NULL;

	f_set = false;
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
			f_hyperoval = true;
			cout << "-hyperoval " << endl;
		}
		else if (ST.stringcmp(argv[i], "-subiaco_oval") == 0) {
			f_subiaco_oval = true;
			f_short = ST.strtoi(argv[++i]);
			cout << "-subiaco_oval " << f_short << endl;
		}
		else if (ST.stringcmp(argv[i], "-subiaco_hyperoval") == 0) {
			f_subiaco_hyperoval = true;
			cout << "-subiaco_hyperoval " << endl;
		}
		else if (ST.stringcmp(argv[i], "-adelaide_hyperoval") == 0) {
			f_adelaide_hyperoval = true;
			cout << "-adelaide_hyperoval " << endl;
		}
		else if (ST.stringcmp(argv[i], "-translation") == 0) {
			f_translation = true;
			translation_exponent = ST.strtoi(argv[++i]);
			cout << "-translation " << translation_exponent << endl;
		}
		else if (ST.stringcmp(argv[i], "-Segre") == 0) {
			f_Segre = true;
			cout << "-segre" << endl;
		}
		else if (ST.stringcmp(argv[i], "-Payne") == 0) {
			f_Payne = true;
			cout << "-Payne" << endl;
		}
		else if (ST.stringcmp(argv[i], "-Cherowitzo") == 0) {
			f_Cherowitzo = true;
			cout << "-Cherowitzo" << endl;
		}
		else if (ST.stringcmp(argv[i], "-OKeefe_Penttila") == 0) {
			f_OKeefe_Penttila = true;
			cout << "-OKeefe_Penttila" << endl;
		}


		else if (ST.stringcmp(argv[i], "-BLT_database") == 0) {
			f_BLT_database = true;
			BLT_database_k = ST.strtoi(argv[++i]);
			cout << "-BLT_database " << BLT_database_k << endl;
		}
		else if (ST.stringcmp(argv[i], "-BLT_database_embedded") == 0) {
			f_BLT_database_embedded = true;
			BLT_database_embedded_k = ST.strtoi(argv[++i]);
			cout << "-BLT_database_embedded " << BLT_database_embedded_k << endl;
		}

#if 0
		else if (stringcmp(argv[i], "-BLT_Linear") == 0) {
			f_BLT_Linear = true;
			cout << "-BLT_Linear " << endl;
		}
		else if (stringcmp(argv[i], "-BLT_Fisher") == 0) {
			f_BLT_Fisher = true;
			cout << "-BLT_Fisher " << endl;
		}
		else if (stringcmp(argv[i], "-BLT_Mondello") == 0) {
			f_BLT_Mondello = true;
			cout << "-BLT_Mondello " << endl;
		}
		else if (stringcmp(argv[i], "-BLT_FTWKB") == 0) {
			f_BLT_FTWKB = true;
			cout << "-BLT_FTWKB " << endl;
		}
#endif

		else if (ST.stringcmp(argv[i], "-elliptic_quadric_ovoid") == 0) {
			f_elliptic_quadric_ovoid = true;
			cout << "-elliptic_quadric_ovoid " << endl;
		}
		else if (ST.stringcmp(argv[i], "-ovoid_ST") == 0) {
			f_ovoid_ST = true;
			cout << "-ovoid_ST " << endl;
		}
		else if (ST.stringcmp(argv[i], "-Baer_substructure") == 0) {
			f_Baer_substructure = true;
			cout << "-Baer_substructure " << endl;
		}
		else if (ST.stringcmp(argv[i], "-orthogonal") == 0) {
			f_orthogonal = true;
			orthogonal_epsilon = ST.strtoi(argv[++i]);
			cout << "-orthogonal " << orthogonal_epsilon << endl;
		}
		else if (ST.stringcmp(argv[i], "-hermitian") == 0) {
			f_hermitian = true;
			cout << "-hermitian" << endl;
		}
		else if (ST.stringcmp(argv[i], "-cuspidal_cubic") == 0) {
			f_cuspidal_cubic = true;
			cout << "-cuspidal_cubic " << endl;
		}
		else if (ST.stringcmp(argv[i], "-twisted_cubic") == 0) {
			f_twisted_cubic = true;
			cout << "-twisted_cubic " << endl;
		}
		else if (ST.stringcmp(argv[i], "-elliptic_curve") == 0) {
			f_elliptic_curve = true;
			elliptic_curve_b = ST.strtoi(argv[++i]);
			elliptic_curve_c = ST.strtoi(argv[++i]);
			cout << "-elliptic_curve " << elliptic_curve_b
					<< " " << elliptic_curve_c << endl;
		}
#if 0
		else if (stringcmp(argv[i], "-Hill_cap_56") == 0) {
			f_Hill_cap_56 = true;
			cout << "-Hill_cap_56 " << endl;
		}
#endif
		else if (ST.stringcmp(argv[i], "-ttp_construction_A") == 0) {
			f_ttp_code = true;
			f_ttp_construction_A = true;
			cout << "-ttp_construction_A" << endl;
		}
		else if (ST.stringcmp(argv[i], "-ttp_construction_A_hyperoval") == 0) {
			f_ttp_code = true;
			f_ttp_construction_A = true;
			f_ttp_hyperoval = true;
			cout << "-ttp_construction_A_hyperoval" << endl;
		}
		else if (ST.stringcmp(argv[i], "-ttp_construction_B") == 0) {
			f_ttp_code = true;
			f_ttp_construction_B = true;
			cout << "-ttp_construction_B" << endl;
		}
		else if (ST.stringcmp(argv[i], "-unital_XXq_YZq_ZYq") == 0) {
			f_unital_XXq_YZq_ZYq = true;
			cout << "-unital_XXq_YZq_ZYq" << endl;
		}
		else if (ST.stringcmp(argv[i], "-desarguesian_line_spread_in_PG_3_q") == 0) {
			f_desarguesian_line_spread_in_PG_3_q = true;
			cout << "-desarguesian_line_spread_in_PG_3_q" << endl;
		}
		else if (ST.stringcmp(argv[i], "-embedded_in_PG_4_q") == 0) {
			f_embedded_in_PG_4_q = true;
			cout << "-embedded_in_PG_4_q" << endl;
		}
		else if (ST.stringcmp(argv[i], "-Buekenhout_Metz") == 0) {
			f_Buekenhout_Metz = true;
			cout << "-Buekenhout_Metz " << endl;
		}
		else if (ST.stringcmp(argv[i], "-classical") == 0) {
			f_classical = true;
			cout << "-classical " << endl;
		}
		else if (ST.stringcmp(argv[i], "-Uab") == 0) {
			f_Uab = true;
			parameter_a = ST.strtoi(argv[++i]);
			parameter_b = ST.strtoi(argv[++i]);
			cout << "-Uab " << parameter_a << " " << parameter_b << endl;
		}
		else if (ST.stringcmp(argv[i], "-whole_space") == 0) {
			f_whole_space = true;
			cout << "-whole_space " << endl;
		}
		else if (ST.stringcmp(argv[i], "-hyperplane") == 0) {
			f_hyperplane = true;
			pt = ST.strtoi(argv[++i]);
			cout << "-hyperplane " << pt << endl;
		}
		else if (ST.stringcmp(argv[i], "-segre_variety") == 0) {
			f_segre_variety = true;
			segre_variety_a = ST.strtoi(argv[++i]);
			segre_variety_b = ST.strtoi(argv[++i]);
			cout << "-segre_variety " << segre_variety_a
					<< " " << segre_variety_b << endl;
		}
		else if (ST.stringcmp(argv[i], "-Maruta_Hamada_arc") == 0) {
			f_Maruta_Hamada_arc = true;
			cout << "-Maruta_Hamada_arc " << endl;
		}
		else if (ST.stringcmp(argv[i], "-projective_variety") == 0) {
			f_projective_variety = true;
			projective_variety_ring_label.assign(argv[++i]);
			variety_label_txt.assign(argv[++i]);
			variety_label_tex.assign(argv[++i]);


			orbiter_kernel_system::os_interface Os;

			i++;
			Os.get_string_from_command_line(variety_coeffs, argc, argv, i, verbose_level);
			i--;

			cout << "-projective_variety "
					<< variety_label_txt << " "
					<< variety_label_tex << " "
					<< variety_coeffs << endl;
		}



		else if (ST.stringcmp(argv[i], "-intersection_of_zariski_open_sets") == 0) {
			f_intersection_of_zariski_open_sets = true;
			intersection_of_zariski_open_sets_ring_label.assign(argv[++i]);
			variety_label_txt.assign(argv[++i]);
			variety_label_tex.assign(argv[++i]);

			int nb;

			nb = ST.strtoi(argv[++i]);

			int j;


			orbiter_kernel_system::os_interface Os;

			i++;

			for (j = 0; j < nb; j++) {
				string s;

				cout << "reading argument " << j << " / " << nb << " : " << argv[i] << endl;

				Os.get_string_from_command_line(s, argc, argv, i, verbose_level);
				Variety_coeffs.push_back(s);
			}

			i--;


			cout << "-intersection_of_zariski_open_sets "
					<< intersection_of_zariski_open_sets_ring_label << " "
					<< variety_label_txt << " "
					<< variety_label_tex << " "
					<< endl;
			for (j = 0; j < Variety_coeffs.size(); j++) {
				cout << j << " : " << Variety_coeffs[j] << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-number_of_conditions_satisfied") == 0) {
			f_number_of_conditions_satisfied = true;
			number_of_conditions_satisfied_ring_label.assign(argv[++i]);
			variety_label_txt.assign(argv[++i]);
			variety_label_tex.assign(argv[++i]);

			number_of_conditions_satisfied_fname.assign(argv[++i]);

			int nb = ST.strtoi(argv[++i]);


			int j;

			orbiter_kernel_system::os_interface Os;

			i++;

			for (j = 0; j < nb; j++) {
				string s;

				cout << "reading argument " << j << " / " << nb << " : " << argv[i] << endl;

				Os.get_string_from_command_line(s, argc, argv, i, verbose_level);
				Variety_coeffs.push_back(s);
			}

			i--;


			cout << "-number_of_conditions_satisfied "
					<< number_of_conditions_satisfied_ring_label << " "
					<< variety_label_txt << " "
					<< variety_label_tex << endl;
			for (j = 0; j < Variety_coeffs.size(); j++) {
				cout << j << " : " << Variety_coeffs[j] << endl;
			}
		}



		else if (ST.stringcmp(argv[i], "-projective_curve") == 0) {
			f_projective_curve = true;
			projective_curve_ring_label.assign(argv[++i]);
			curve_label_txt.assign(argv[++i]);
			curve_label_tex.assign(argv[++i]);
			//curve_nb_vars = ST.strtoi(argv[++i]);
			//curve_degree = ST.strtoi(argv[++i]);
			curve_coeffs.assign(argv[++i]);
			cout << "-projective_curve "
					<< projective_curve_ring_label << " "
					<< curve_label_txt << " "
					<< curve_label_tex << " "
					<< curve_coeffs << endl;
		}
#if 0
		else if (ST.stringcmp(argv[i], "-monomial_type_LEX") == 0) {
			Monomial_ordering_type = t_LEX;
			cout << "-monomial_type_LEX " << endl;
		}
		else if (ST.stringcmp(argv[i], "-monomial_type_PART") == 0) {
			Monomial_ordering_type = t_PART;
			cout << "-monomial_type_PART " << endl;
		}
#endif
		else if (ST.stringcmp(argv[i], "-set") == 0) {
			f_set = true;
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
		cout << "-BLT_database " << BLT_database_k << endl;
	}
	if (f_BLT_database_embedded) {
		cout << "-BLT_database_embedded " << BLT_database_embedded_k << endl;
	}

	if (f_elliptic_quadric_ovoid) {
		cout << "-elliptic_quadric_ovoid " << endl;
	}
	if (f_ovoid_ST) {
		cout << "-ovoid_ST " << endl;
	}
	if (f_Baer_substructure) {
		cout << "-Baer_substructure " << endl;
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
		f_twisted_cubic = true;
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
				<< projective_variety_ring_label << " "
				<< variety_label_txt << " "
				<< variety_label_tex << " "
				<< variety_coeffs << endl;
	}

	if (f_intersection_of_zariski_open_sets) {
		cout << "-intersection_of_zariski_open_sets "
				<< intersection_of_zariski_open_sets_ring_label << " "
				<< variety_label_txt << " "
				<< variety_label_tex << " "
				<< endl;
		for (int j = 0; j < Variety_coeffs.size(); j++) {
			cout << j << " : " << Variety_coeffs[j] << endl;
		}
	}


	if (f_number_of_conditions_satisfied) {
		cout << "-number_of_conditions_satisfied "
				<< number_of_conditions_satisfied_ring_label << " "
				<< variety_label_txt << " "
				<< variety_label_tex << endl;
		int j;
		for (j = 0; j < Variety_coeffs.size(); j++) {
			cout << j << " : " << Variety_coeffs[j] << endl;
		}
	}



	if (f_projective_curve) {
		cout << "-projective_curve "
				<< projective_curve_ring_label << " "
				<< curve_label_txt << " "
				<< curve_label_tex << " "
				<< curve_coeffs << endl;
	}
	if (f_set) {
		cout << "-set " << set_text << endl;
	}
}




}}}


