/*
 * trace_record.cpp
 *
 *  Created on: May 15, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



trace_record::trace_record()
{
	coset = 0;
	trace_po = 0;
	f_skip = FALSE;
	solution_idx = -1;
	nb_sol = -1;
	go = -1;
	pos = -1;
	so = -1;
	orbit_len = 0;
	f2 = -1;
}

trace_record::~trace_record()
{

}


void save_trace_record(
		trace_record *T,
		int f_trace_record_prefix, std::string &trace_record_prefix,
		int iso, int f, int po, int so, int N)
{
	long int *M;
	int w = 10;
	string fname;
	const char *column_label[] = {
		"coset",
		"trace_po",
		"f_skip",
		"solution_idx",
		"nb_sol",
		"go",
		"pos",
		"so",
		"orbit_len",
		"f2"
		};
	int i;
	file_io Fio;

	M = NEW_lint(N * w);
	for (i = 0; i < N; i++) {
		M[i * w + 0] = T[i].coset;
		M[i * w + 1] = T[i].trace_po;
		M[i * w + 2] = T[i].f_skip;
		M[i * w + 3] = T[i].solution_idx;
		M[i * w + 4] = T[i].nb_sol;
		M[i * w + 5] = T[i].go;
		M[i * w + 6] = T[i].pos;
		M[i * w + 7] = T[i].so;
		M[i * w + 8] = T[i].orbit_len;
		M[i * w + 9] = T[i].f2;
		}

	char str[1000];

	sprintf(str, "trace_record_%03d_f%05d_po%d_so%d.csv",
			iso, f, po, so);


	if (f_trace_record_prefix) {
		fname.assign(trace_record_prefix);
		fname.append(str);
	}
	else {
		fname.assign(str);
	}
	Fio.lint_matrix_write_csv_with_labels(fname, M, N, w, column_label);
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
}


}}


