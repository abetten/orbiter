// spread_create_description.cpp
// 
// Anton Betten
//
// March 22, 2018
//
//
// 
//
//

#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace layer5_applications {
namespace spreads {


spread_create_description::spread_create_description()
{
	Record_birth();
	f_kernel_field = false;
	//std::string kernel_field_label;

	f_group = false;
	//std::string group_label;

	f_group_on_subspaces = false;
	//std::string group_on_subspaces_label;

	f_k = false;
	k = 0;

	f_catalogue = false;
	iso = 0;

	f_family = false;
	//family_name;

	f_spread_set = false;
	//std::string spread_set_label;
	//std::string spread_set_label_tex;
	//std::string spread_set_data;

	f_transform = false;
	//std::vector<std::string> transform_text;
	//std::vector<int> transform_f_inv;
}

spread_create_description::~spread_create_description()
{
	Record_death();
}


int spread_create_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int i;
	other::data_structures::string_tools ST;

	cout << "spread_create_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-kernel_field") == 0) {
			f_kernel_field = true;
			kernel_field_label.assign(argv[++i]);
			cout << "-kernel_field " << kernel_field_label << endl;
		}
		else if (ST.stringcmp(argv[i], "-group") == 0) {
			f_group = true;
			group_label.assign(argv[++i]);
			cout << "-group " << group_label << endl;
		}
		else if (ST.stringcmp(argv[i], "-group_on_subspaces") == 0) {
			f_group_on_subspaces = true;
			group_on_subspaces_label.assign(argv[++i]);
			cout << "-group_on_subspaces " << group_on_subspaces_label << endl;
		}
		else if (ST.stringcmp(argv[i], "-k") == 0) {
			f_k = true;
			k = ST.strtoi(argv[++i]);
			cout << "-k " << k << endl;
		}
		else if (ST.stringcmp(argv[i], "-catalogue") == 0) {
			f_catalogue = true;
			iso = ST.strtoi(argv[++i]);
			cout << "-catalogue " << iso << endl;
		}
		else if (ST.stringcmp(argv[i], "-family") == 0) {
			f_family = true;
			family_name.assign(argv[++i]);
			cout << "-family " << family_name << endl;
		}
		else if (ST.stringcmp(argv[i], "-spread_set") == 0) {
			f_spread_set = true;
			spread_set_label.assign(argv[++i]);
			spread_set_label_tex.assign(argv[++i]);
			spread_set_data.assign(argv[++i]);
			cout << "-spread_set " << spread_set_label
					<< " " << spread_set_label_tex
					<< " " << spread_set_data << endl;
		}
		else if (ST.stringcmp(argv[i], "-transform") == 0) {
			f_transform = true;

			string text;

			text.assign(argv[++i]);

			transform_text.push_back(text);
			transform_f_inv.push_back(false);

			cout << "-transform " << transform_text[transform_text.size() - 1] << endl;
		}
		else if (ST.stringcmp(argv[i], "-transform_inv") == 0) {
			f_transform = true;

			string text;

			text.assign(argv[++i]);

			transform_text.push_back(text);
			transform_f_inv.push_back(true);

			cout << "-transform_inv " << transform_text[transform_text.size() - 1] << endl;
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "spread_create_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "spread_create_description::read_arguments done" << endl;
	return i + 1;
}

void spread_create_description::print()
{
	if (f_kernel_field) {
		cout << "-kernel_field " << kernel_field_label << endl;
	}
	if (f_group) {
		cout << "-group " << group_label << endl;
	}
	if (f_group_on_subspaces) {
		cout << "-group_on_subspaces " << group_on_subspaces_label << endl;
	}
	if (f_k) {
		cout << "-k " << k << endl;
	}
	if (f_catalogue) {
		cout << "-catalogue " << iso << endl;
	}
	if (f_family) {
		cout << "-family " << family_name << endl;
	}
	if (f_spread_set) {
		cout << "-spread_set " << spread_set_label
				<< " " << spread_set_label_tex
				<< " " << spread_set_data << endl;
	}
	if (f_transform) {
		int i;

		for (i = 0; i < transform_text.size(); i++) {
			if (transform_f_inv[i]) {
				cout << "-transform_inv " << transform_text[i] << endl;
			}
			else {
				cout << "-transform " << transform_text[i] << endl;
			}
		}
	}
}


}}}
