/*
 * regular_packing.cpp
 *
 *  Created on: Sep 19, 2020
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




regular_packing::regular_packing()
{
		PW = NULL;

		spread_to_external_line_idx = NULL;
		external_line_to_spread = NULL;
}

regular_packing::~regular_packing()
{
	if (spread_to_external_line_idx) {
		FREE_lint(spread_to_external_line_idx);
	}
	if (external_line_to_spread) {
		FREE_lint(external_line_to_spread);
	}
}

void regular_packing::init(packing_was *PW, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "regular_packing::init" << endl;
	}
	regular_packing::PW = PW;

	if (f_v) {
		cout << "regular_packing::init before PW->P->T->Klein->compute_external_lines" << endl;
	}
	PW->P->T->Klein->compute_external_lines(External_lines, verbose_level);
	if (f_v) {
		cout << "regular_packing::init after PW->P->T->Klein->compute_external_lines" << endl;
	}

	if (f_v) {
		cout << "regular_packing::init before PW->P->T->Klein->identify_external_lines_and_spreads" << endl;
	}
	PW->P->T->Klein->identify_external_lines_and_spreads(
			PW->Spread_tables_reduced,
			External_lines,
			spread_to_external_line_idx,
			external_line_to_spread,
			verbose_level);
	if (f_v) {
		cout << "regular_packing::init after PW->P->T->Klein->identify_external_lines_and_spreads" << endl;
	}

}



}}
