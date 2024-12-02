/*
 * ug_3_2_F_2.cpp
 *
 *  Created on: Jan 15, 2023
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;
using namespace orbiter;


void first();
void second();
void third();

int main()
{
	orbiter::layer5_applications::user_interface::orbiter_top_level_session Orbiter;

	first();
	second();
	third();

}

void first()
{
	algebra::field_theory::finite_field_description Descr;
	algebra::field_theory::finite_field Fq;

	int verbose_level = 2;

	Descr.f_q = true;
	Descr.q_text.assign("2");
	Fq.init(&Descr, verbose_level);

	cout << "in F_2, 1 + 1 = " << Fq.add(1, 1) << endl;

	algebra::basic_algebra::algebra_global Algebra;

	Algebra.do_cheat_sheet_GF(&Fq, verbose_level);
}

void second()
{
	int q = 2;
	int verbose_level = 2;
	int f_without_tables = false;
	algebra::field_theory::finite_field Fq;

	Fq.finite_field_init_small_order(q,
			f_without_tables,
			true /* f_compute_related_fields */,
			verbose_level);

	cout << "in F_2, 1 + 1 = " << Fq.add(1, 1) << endl;

	algebra::basic_algebra::algebra_global Algebra;

	Algebra.do_cheat_sheet_GF(&Fq, verbose_level);
}

void third()
{
	int q = 2;
	int verbose_level = 2;
	int f_without_tables = false;
	algebra::field_theory::finite_field *Fq;

	Fq = NEW_OBJECT(algebra::field_theory::finite_field);

	Fq->finite_field_init_small_order(q,
			f_without_tables,
			true /* f_compute_related_fields */,
			verbose_level);

	cout << "in F_2, 1 + 1 = " << Fq->add(1, 1) << endl;


	algebra::basic_algebra::algebra_global Algebra;

	Algebra.do_cheat_sheet_GF(Fq, verbose_level);

	FREE_OBJECT(Fq);
}



