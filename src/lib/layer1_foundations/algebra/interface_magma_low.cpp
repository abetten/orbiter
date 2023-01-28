/*
 * interface_magma_low.cpp
 *
 *  Created on: Jan 27, 2023
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {


interface_magma_low::interface_magma_low()
{
}

interface_magma_low::~interface_magma_low()
{
}

void interface_magma_low::magma_set_stabilizer_in_collineation_group(
		field_theory::finite_field *F,
		int d, long int *Pts, int nb_pts,
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_magma_low::magma_set_stabilizer_in_collineation_group" << endl;
	}

	string fname2;
	int *v;
	int h, i, a, b;
	data_structures::string_tools ST;

	v = NEW_int(d);
	fname2.assign(fname);
	ST.replace_extension_with(fname2, ".magma");

	{
		ofstream fp(fname2);

		fp << "G,I:=PGammaL(" << d << "," << F->q
				<< ");F:=GF(" << F->q << ");" << endl;
		fp << "S:={};" << endl;
		fp << "a := F.1;" << endl;
		for (h = 0; h < nb_pts; h++) {
			F->PG_element_unrank_modified_lint(v, 1, d, Pts[h]);

			F->PG_element_normalize_from_front(v, 1, d);

			fp << "Include(~S,Index(I,[";
			for (i = 0; i < d; i++) {
				a = v[i];
				if (a == 0) {
					fp << "0";
				}
				else if (a == 1) {
					fp << "1";
				}
				else {
					b = F->log_alpha(a);
					fp << "a^" << b;
				}
				if (i < d - 1) {
					fp << ",";
				}
			}
			fp << "]));" << endl;
		}
		fp << "Stab := Stabilizer(G,S);" << endl;
		fp << "Size(Stab);" << endl;
		fp << endl;
	}
	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname2 << " of size "
			<< Fio.file_size(fname2) << endl;

	FREE_int(v);
	if (f_v) {
		cout << "interface_magma_low::magma_set_stabilizer_in_collineation_group done" << endl;
	}
}

}}}



