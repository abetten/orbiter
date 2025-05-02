/*
 * text_builder.cpp
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


text_builder::text_builder()
{
	Record_birth();
	Descr = NULL;

	f_has_text = false;
	//std::string text;

}

text_builder::~text_builder()
{
	Record_death();
}

void text_builder::init(
		text_builder_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "text_builder::init" << endl;
	}

	text_builder::Descr = Descr;

	if (Descr->f_here) {
		if (f_v) {
			cout << "text_builder::init -here" << endl;
		}
		text = Descr->here_text;
		f_has_text = true;
		if (f_v) {
			cout << "text_builder::init" << endl;
		}

	}

	if (f_v) {
		if (f_has_text) {
			cout << "text_builder::init "
					"created text: " << text << endl;
			cout << endl;
		}
	}


	if (f_v) {
		cout << "text_builder::init done" << endl;
	}
}


void text_builder::print(
		std::ostream &ost)
{

	if (f_has_text) {
		ost << text << endl;
	}
	else {
		ost << "no text" << endl;
	}
}


}}}}




