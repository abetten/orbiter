/*
 * text_builder_description.cpp
 *
 *  Created on: Apr 20, 2025
 *      Author: betten
 */


#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace data_structures {


text_builder_description::text_builder_description()
{
	Record_birth();

	f_here = false;
	//std::string here_text;

}

text_builder_description::~text_builder_description()
{
	Record_death();
}


int text_builder_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i = 0;
	string_tools ST;

	if (f_v) {
		cout << "text_builder_description::read_arguments" << endl;
		cout << "text_builder_description::read_arguments i = " << i << endl;
		cout << "text_builder_description::read_arguments argc = " << argc << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-here") == 0) {
			f_here = true;
			here_text.assign(argv[++i]);
			if (f_v) {
				cout << "-here " << here_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "vector_builder_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "vector_builder_description::read_arguments done" << endl;
	}
	return i + 1;
}

void text_builder_description::print()
{
	if (f_here) {
		cout << "-here " << here_text << endl;
	}

}


}}}}





