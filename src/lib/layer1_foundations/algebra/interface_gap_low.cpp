/*
 * interface_gap_low.cpp
 *
 *  Created on: Jan 27, 2023
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {


interface_gap_low::interface_gap_low()
{
}

interface_gap_low::~interface_gap_low()
{
}

void interface_gap_low::fining_set_stabilizer_in_collineation_group(
		field_theory::finite_field *F,
		int d, long int *Pts, int nb_pts,
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_magma_low::fining_set_stabilizer_in_collineation_group" << endl;
	}

	string fname2;
	int *v;
	int h, i, a, b;
	data_structures::string_tools ST;

	v = NEW_int(d);
	fname2.assign(fname);
	ST.replace_extension_with(fname2, ".gap");

	{
		ofstream fp(fname2);

		fp << "LoadPackage(\"fining\");" << endl;
		fp << "pg := ProjectiveSpace(" << d - 1 << "," << F->q << ");" << endl;
		fp << "S:=[" << endl;
		for (h = 0; h < nb_pts; h++) {
			F->PG_element_unrank_modified_lint(v, 1, d, Pts[h]);

			F->PG_element_normalize_from_front(v, 1, d);

			fp << "[";
			for (i = 0; i < d; i++) {
				a = v[i];
				if (a == 0) {
					fp << "0*Z(" << F->q << ")";
				}
				else if (a == 1) {
					fp << "Z(" << F->q << ")^0";
				}
				else {
					b = F->log_alpha(a);
					fp << "Z(" << F->q << ")^" << b;
				}
				if (i < d - 1) {
					fp << ",";
				}
			}
			fp << "]";
			if (h < nb_pts - 1) {
				fp << ",";
			}
			fp << endl;
		}
		fp << "];" << endl;
		fp << "S := List(S,x -> VectorSpaceToElement(pg,x));" << endl;
		fp << "g := CollineationGroup(pg);" << endl;
		fp << "stab := Stabilizer(g,Set(S),OnSets);" << endl;
		fp << "Size(stab);" << endl;
	}
	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname2 << " of size "
			<< Fio.file_size(fname2) << endl;

#if 0
LoadPackage("fining");
pg := ProjectiveSpace(2,4);
#points := Points(pg);
#pointslist := AsList(points);
#Display(pointslist[1]);
frame := [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]*Z(2)^0;
frame := List(frame,x -> VectorSpaceToElement(pg,x));
pairs := Combinations(frame,2);
secants := List(pairs,p -> Span(p[1],p[2]));
leftover := Filtered(pointslist,t->not ForAny(secants,s->t in s));
hyperoval := Union(frame,leftover);
g := CollineationGroup(pg);
stab := Stabilizer(g,Set(hyperoval),OnSets);
StructureDescription(stab);
#endif


	FREE_int(v);
	if (f_v) {
		cout << "interface_magma_low::fining_set_stabilizer_in_collineation_group done" << endl;
	}
}

}}}


