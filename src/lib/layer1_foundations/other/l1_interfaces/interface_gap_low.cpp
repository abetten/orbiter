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
namespace other {
namespace l1_interfaces {


interface_gap_low::interface_gap_low()
{
	Record_birth();
}

interface_gap_low::~interface_gap_low()
{
	Record_death();
}

void interface_gap_low::fining_set_stabilizer_in_collineation_group(
		algebra::field_theory::finite_field *F,
		int d, long int *Pts, int nb_pts,
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_gap_low::fining_set_stabilizer_in_collineation_group" << endl;
	}

	string fname2;
	data_structures::string_tools ST;

	fname2.assign(fname);
	ST.replace_extension_with(fname2, ".gap");

	{
		ofstream ost(fname2);

		collineation_set_stabilizer(
					ost,
					F,
					d, Pts, nb_pts,
					verbose_level - 2);
	}
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Written file " << fname2 << " of size "
				<< Fio.file_size(fname2) << endl;
	}

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


	if (f_v) {
		cout << "interface_gap_low::fining_set_stabilizer_in_collineation_group done" << endl;
	}
}

void interface_gap_low::collineation_set_stabilizer(
		std::ostream &ost,
		algebra::field_theory::finite_field *F,
		int d, long int *Pts, int nb_pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_gap_low::collineation_set_stabilizer" << endl;
	}
	int h, i, a;
	int *v;

	v = NEW_int(d);
	ost << "LoadPackage(\"fining\");" << endl;
	ost << "pg := ProjectiveSpace(" << d - 1 << "," << F->q << ");" << endl;
	ost << "S:=[" << endl;
	for (h = 0; h < nb_pts; h++) {
		F->Projective_space_basic->PG_element_unrank_modified_lint(v, 1, d, Pts[h]);

		F->Projective_space_basic->PG_element_normalize_from_front(v, 1, d);

		ost << "[";
		for (i = 0; i < d; i++) {
			a = v[i];

			write_element_of_finite_field(ost, F, a);

			if (i < d - 1) {
				ost << ",";
			}
		}
		ost << "]";
		if (h < nb_pts - 1) {
			ost << ",";
		}
		ost << endl;
	}

	ost << "];" << endl;
	ost << "S := List(S,x -> VectorSpaceToElement(pg,x));" << endl;
	ost << "g := CollineationGroup(pg);" << endl;
	ost << "stab := Stabilizer(g,Set(S),OnSets);" << endl;
	ost << "Size(stab);" << endl;

	FREE_int(v);
	if (f_v) {
		cout << "interface_gap_low::collineation_set_stabilizer done" << endl;
	}
}

void interface_gap_low::write_matrix(
		std::ostream &ost,
		algebra::field_theory::finite_field *F,
		int *Mtx, int d,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_gap_low::write_matrix" << endl;
	}
	int i, j, a;

	ost << "[";
	for (i = 0; i < d; i++) {
		ost << "[";
		for (j = 0; j < d; j++) {
			a = Mtx[i * d + j];

			write_element_of_finite_field(ost, F, a);

			if (j < d - 1) {
				ost << ",";
			}
		}
		ost << "]";
		if (i < d - 1) {
			ost << "," << endl;
		}
	}
	ost << "]";

	if (f_v) {
		cout << "interface_gap_low::write_matrix done" << endl;
	}
}



void interface_gap_low::write_element_of_finite_field(
		std::ostream &ost,
		algebra::field_theory::finite_field *F, int a)
{
	int b;

	if (a == 0) {
		ost << "0*Z(" << F->q << ")";
	}
	else if (a == 1) {
		ost << "Z(" << F->q << ")^0";
	}
	else {
		b = F->log_alpha(a);
		ost << "Z(" << F->q << ")^" << b;
	}

}

}}}}



