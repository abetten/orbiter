/*
 * activity_output.cpp
 *
 *  Created on: May 22, 2024
 *      Author: betten
 */





#include "foundations.h"

using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace orbiter_kernel_system {


activity_output::activity_output()
{

	//std::string fname_base;
	//std::vector<std::vector<std::string> > Feedback;
	//std::string description_txt;
	//std::string headings;
	nb_cols = 0;


}

activity_output::~activity_output()
{

}

void activity_output::save(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "activity_output::save" << endl;
	}
	std::string *Table;
	int m, n, i, j;

	m = Feedback.size();
	n = nb_cols;

	if (f_v) {
		cout << "activity_output::save "
				"m = " << m
				<< endl;
	}

	Table = new string[m * n];
	for (i = 0; i < m; i++) {
		if (false) {
			cout << "activity_output::save "
					"i = " << i
					<< endl;
		}
		for (j = 0; j < n; j++) {
			Table[i * n + j] = Feedback[i][j];
		}
	}

	orbiter_kernel_system::file_io Fio;

	string fname_out;

	fname_out = fname_base + "_" + description_txt + "_out.csv";

	if (f_v) {
		cout << "activity_output::save "
				"fname_out = " << fname_out
				<< endl;
	}
	if (f_v) {
		cout << "activity_output::save "
				"before Fio.Csv_file_support->write_table_of_strings "
				<< endl;
	}
	Fio.Csv_file_support->write_table_of_strings(
			fname_out,
			m, n, Table,
			headings,
			verbose_level);
	if (f_v) {
		cout << "activity_output::save "
				"after Fio.Csv_file_support->write_table_of_strings "
				<< endl;
	}


}


}}}

