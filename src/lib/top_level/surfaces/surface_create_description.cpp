// surface_create_description.cpp
// 
// Anton Betten
//
// December 8, 2017
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


surface_create_description::surface_create_description()
{
	f_q = FALSE;
	q = 0;
	f_catalogue = FALSE;
	iso = 0;
	f_by_coefficients = FALSE;
	//coefficients_text = NULL;
	f_family_HCV = FALSE;
	family_HCV_a = 0;
	family_HCV_b = 0;
	f_family_G13 = FALSE;
	family_G13_a = 0;
	f_family_F13 = FALSE;
	family_F13_a = 0;
	f_family_bes = FALSE;
	family_bes_a = 0;
	family_bes_c = 0;
	f_family_general_abcd = FALSE;
	family_general_abcd_a = 0;
	family_general_abcd_b = 0;
	family_general_abcd_c = 0;
	family_general_abcd_d = 0;
	f_arc_lifting = FALSE;
	//arc_lifting_text = NULL;
	//arc_lifting_two_lines_text = NULL;
	f_arc_lifting_with_two_lines = FALSE;
	//nb_select_double_six = 0;
	//select_double_six_string[];

	//std::vector<std::string> transform_coeffs;
	//std::vector<int> f_inverse_transform;

	//null();
}

surface_create_description::~surface_create_description()
{
	freeself();
}

void surface_create_description::null()
{
}

void surface_create_description::freeself()
{
	null();
}

int surface_create_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "surface_create_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = strtoi(argv[++i]);
			cout << "-q " << q << endl;
		}
		else if (stringcmp(argv[i], "-catalogue") == 0) {
			f_catalogue = TRUE;
			iso = strtoi(argv[++i]);
			cout << "-catalogue " << iso << endl;
		}
		else if (stringcmp(argv[i], "-by_coefficients") == 0) {
			f_by_coefficients = TRUE;
			coefficients_text.assign(argv[++i]);
			cout << "-by_coefficients " << coefficients_text << endl;
		}
		else if (stringcmp(argv[i], "-family_HCV") == 0) {
			f_family_HCV = TRUE;
			family_HCV_a = strtoi(argv[++i]);
			family_HCV_b = strtoi(argv[++i]);
			cout << "-family_HCV " << family_HCV_a << " " << family_HCV_b << endl;
		}
		else if (stringcmp(argv[i], "-family_G13") == 0) {
			f_family_G13 = TRUE;
			family_G13_a = strtoi(argv[++i]);
			cout << "-family_G13 " << family_G13_a << endl;
		}
		else if (stringcmp(argv[i], "-family_F13") == 0) {
			f_family_F13 = TRUE;
			family_F13_a = strtoi(argv[++i]);
			cout << "-family_F13 " << family_F13_a << endl;
		}
		else if (stringcmp(argv[i], "-family_bes") == 0) {
			f_family_bes = TRUE;
			family_bes_a = strtoi(argv[++i]);
			family_bes_c = strtoi(argv[++i]);
			cout << "-family_bes " << family_bes_a << " " << family_bes_c << endl;
		}
		else if (stringcmp(argv[i], "-family_general_abcd") == 0) {
			f_family_general_abcd = TRUE;
			family_general_abcd_a = strtoi(argv[++i]);
			family_general_abcd_b = strtoi(argv[++i]);
			family_general_abcd_c = strtoi(argv[++i]);
			family_general_abcd_d = strtoi(argv[++i]);
			cout << "-family_general_abcd "
					<< family_general_abcd_a << " " << family_general_abcd_b
					<< family_general_abcd_c << " " << family_general_abcd_d
					<< endl;
		}
		else if (stringcmp(argv[i], "-arc_lifting") == 0) {
			f_arc_lifting = TRUE;
			arc_lifting_text.assign(argv[++i]);
			cout << "-arc_lifting " << arc_lifting_text << endl;
		}
		else if (stringcmp(argv[i], "-arc_lifting_with_two_lines") == 0) {
			f_arc_lifting_with_two_lines = TRUE;
			arc_lifting_text.assign(argv[++i]);
			arc_lifting_two_lines_text.assign(argv[++i]);
			cout << "-arc_lifting_with_two_lines " << arc_lifting_text
					<< " " << arc_lifting_two_lines_text << endl;
		}
		else if (stringcmp(argv[i], "-select_double_six") == 0) {
			//f_select_double_six = TRUE;
			string s;

			s.assign(argv[++i]);
			select_double_six_string.push_back(s);
			cout << "-select_double_six "
					<< select_double_six_string[select_double_six_string.size() - 1] << endl;
		}
		else if (stringcmp(argv[i], "-transform") == 0) {

			string s;

			s.assign(argv[++i]);
			transform_coeffs.push_back(s);
			f_inverse_transform.push_back(FALSE);
			cout << "-transform " << transform_coeffs[transform_coeffs.size() - 1]
					<< " " << f_inverse_transform[transform_coeffs.size() - 1] << endl;
		}
		else if (stringcmp(argv[i], "-transform_inverse") == 0) {

			string s;

			s.assign(argv[++i]);
			transform_coeffs.push_back(s);
			f_inverse_transform.push_back(TRUE);
			cout << "-transform_inverse " << transform_coeffs[transform_coeffs.size() - 1]
					<< " " << f_inverse_transform[transform_coeffs.size() - 1] << endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
	} // next i
	cout << "surface_create_description::read_arguments done" << endl;
	return i + 1;
}


int surface_create_description::get_q()
{
	if (!f_q) {
		cout << "surface_create_description::get_q "
				"q has not been set yet" << endl;
		exit(1);
	}
	return q;
}

}}



