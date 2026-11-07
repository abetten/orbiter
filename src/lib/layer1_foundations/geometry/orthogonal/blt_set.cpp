/*
 * blt_set.cpp
 *
 *  Created on: Nov 6, 2026
 *      Author: betten
 */






#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace orthogonal_geometry {



blt_set::blt_set()
{
	Record_birth();

	//string label_txt;
	//string label_tex;

	Blt_set_domain = NULL;

	set = NULL;

	ABC = NULL;

	T = NULL;
	Pi_ij = NULL;

	Invariants = NULL;

}

blt_set::~blt_set()
{
	Record_death();

	if (set) {
		FREE_lint(set);
	}
	if (ABC) {
		FREE_int(ABC);
	}
	if (T) {
		FREE_lint(T);
	}
	if (Pi_ij) {
		FREE_lint(Pi_ij);
	}
	if (Invariants) {
		FREE_OBJECT(Invariants);
	}
}



void blt_set::init(
		blt_set_domain *Blt_set_domain,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "blt_set::init" << endl;
	}

	blt_set::Blt_set_domain = Blt_set_domain;

	set = NEW_lint(Blt_set_domain->target_size);

	ABC = NEW_int(3 * (Blt_set_domain->q + 1));


	if (f_v) {
		cout << "blt_set::init finished" << endl;
	}

}


void blt_set::compute_properties(
		int f_invariants,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set::compute_properties" << endl;
	}
	if (f_v) {
		cout << "blt_set::compute_properties f_invariants = " << f_invariants << endl;
	}


	if (f_v) {
		cout << "blt_set::compute_properties "
				"before compute_T" << endl;
	}
	T = Blt_set_domain->compute_T(
			set,
			verbose_level - 1);
	if (f_v) {
		cout << "blt_set::compute_properties "
				"after compute_T" << endl;
	}

	if (f_v) {
		cout << "blt_set::compute_properties "
				"before compute_Pi_ij" << endl;
	}
	Pi_ij = Blt_set_domain->compute_Pi_ij(
			set, T,
			verbose_level - 1);
	if (f_v) {
		cout << "blt_set::compute_properties "
				"after compute_Pi_ij" << endl;
	}


	Invariants = NEW_OBJECT(geometry::orthogonal_geometry::blt_set_invariants);
	if (f_v) {
		cout << "blt_set::compute_properties "
				"before Invariants->init" << endl;
	}
	Invariants->init(
			Blt_set_domain,
			set,
			verbose_level - 1);
	if (f_v) {
		cout << "blt_set::compute_properties "
				"after Invariants->init" << endl;
	}

	if (f_invariants) {
		if (f_v) {
			cout << "blt_set::compute_properties "
					"before Invariants->compute" << endl;
		}
		Invariants->compute(verbose_level - 1);
		if (f_v) {
			cout << "blt_set::compute_properties "
					"after Invariants->compute" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "blt_set::compute_properties "
					"We don't compute invariants." << endl;
		}

	}


	if (f_v) {
		cout << "blt_set::compute_properties done" << endl;
	}

}


void blt_set::report(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set::report" << endl;
	}

	ost << "\\bigskip" << endl;
	ost << "BLT-set and tangent hyperplanes $T$:\\\\" << endl;
	ost << "\\bigskip" << endl;


	Blt_set_domain->report_given_point_set(
			ost,
			set, Blt_set_domain->target_size,
			verbose_level);

	ost << "\\bigskip" << endl;


	ost << "Tangent hyperplanes:\\\\" << endl;

	Blt_set_domain->G54->print_set_tex(
			ost, T, Blt_set_domain->target_size,
			verbose_level);

	ost << "\\bigskip" << endl;


	other::l1_interfaces::latex_interface Li;

	ost << "$\\Pi_{ij}$ matrix:" << endl;
	ost << "$$" << endl;
	ost << "\\left[" << endl;

	Li.print_lint_matrix_tex(
			ost,
			Pi_ij,
			Blt_set_domain->target_size,
			Blt_set_domain->target_size);

	ost << "\\right]" << endl;
	ost << "$$" << endl;

	ost << "\\bigskip" << endl;

	ost << "First row of $\\Pi_{ij}$ (with the diagonal entry removed):\\\\" << endl;

	Blt_set_domain->G53->print_set_tex(
			ost,
			Pi_ij + 1,
			Blt_set_domain->q,
			verbose_level);

	ost << "\\bigskip" << endl;

	if (Invariants) {
		if (f_v) {
			cout << "blt_set::report "
					"before Invariants->latex" << endl;
		}
		Invariants->latex(ost, verbose_level);
		if (f_v) {
			cout << "blt_set::report "
					"after Invariants->latex" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "blt_set::report "
					"Invariants not available" << endl;
		}

	}


	if (f_v) {
		cout << "blt_set::report done" << endl;
	}

}


void blt_set::print_Pi_ij_one_row(
		int row_idx)
{

	int target_size;

	target_size = Blt_set_domain->target_size;

	cout << "blt_set::print_Pi_ij_one_row Pi_ij with row " << row_idx << endl;
	Lint_vec_print(cout, Pi_ij + row_idx * target_size, target_size);
	cout << endl;

	int j;
	long int plane;
	int B[3 * 5];

	for (j = 0; j < Blt_set_domain->target_size; j++) {
		plane = Pi_ij[row_idx * target_size + j];

		Blt_set_domain->G53->unrank_lint_here(
				B, plane, 0 /*verbose_level*/);
		cout << "blt_set::print_Pi_ij_one_row row_idx = " << row_idx << " j=" << j << " / "
				<< target_size << " plane=" << plane << " : " << endl;
		Int_matrix_print(B, 3, 5);
	}

}

void blt_set::report_points_and_tangent_hyperplanes(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set::report_points_and_tangent_hyperplanes" << endl;
	}


	int width;

	width = Blt_set_domain->F->log10_of_q + 1;

	int target_size;

	target_size = Blt_set_domain->target_size;


	long int pt_rk, hyperplane_rk, plane_rk;
	int i;

	ost << "A set of points of size " << target_size << "\\\\" << endl;
	ost << "The Points:\\\\" << endl;
	for (i = 0; i < target_size; i++) {
		pt_rk = set[i];


		Blt_set_domain->O->Hyperbolic_pair->unrank_point(
				Blt_set_domain->O->Hyperbolic_pair->v1, 1, pt_rk,
				0 /*verbose_level*/);

		hyperplane_rk = T[i];

		ost << i << " : $P_{" << pt_rk << "} = ";
		Int_vec_print(ost, Blt_set_domain->O->Hyperbolic_pair->v1, Blt_set_domain->O->Quadratic_form->n);
		ost << ", T_{" << hyperplane_rk << "}$\\\\" << endl;


		int B[5 * 5];

		Blt_set_domain->O->Hyperbolic_pair->unrank_point(B, 1, pt_rk,
				0 /*verbose_level*/);
		if (f_v) {
			cout << "blt_set_domain::compute_tangent_hyperplane "
					"the point is:" << endl;
			Int_vec_print_integer_matrix_width_easy(cout, B, 1, 5, width);
		}

		if (f_v) {
			cout << "blt_set_domain::compute_tangent_hyperplane "
					"before F->Linear_algebra->perp" << endl;
		}
		Blt_set_domain->F->Linear_algebra->perp(
				5, 1, B,
				Blt_set_domain->O->Quadratic_form->Gram_matrix,
				0 /* verbose_level */);
		if (f_v) {
			cout << "blt_set_domain::compute_tangent_hyperplane "
					"after F->Linear_algebra->perp" << endl;
		}
		if (f_v) {
			cout << "blt_set_domain::compute_tangent_hyperplane "
					"the matrix B is:" << endl;
			Int_vec_print_integer_matrix_width_easy(cout, B, 5, 5, width);
		}

		plane_rk = Blt_set_domain->G54->rank_lint_here(B + 5, 0 /*verbose_level*/);

		ost << i << " plane\\_rk=" << plane_rk << "\\\\" << endl;

		ost << i << " hyperplane\\_rk=" << hyperplane_rk << "\\\\" << endl;

	}

	if (f_v) {
		cout << "blt_set::report_points_and_tangent_hyperplanes done" << endl;
	}
}



void blt_set::export_set_in_PG(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "blt_set::export_set_in_PG" << endl;
	}
	int sz;
	int i;
	long int j;
	int v[5];
	long int *Pts;
	algebra::field_theory::finite_field *F;

	sz = Blt_set_domain->target_size;

	F = Blt_set_domain->O->F;

	Pts = NEW_lint(sz);
	for (i = 0; i < sz; i++) {

		Blt_set_domain->O->Hyperbolic_pair->unrank_point(
				v,
				1, set[i],
				0 /*verbose_level - 1*/);

		F->Projective_space_basic->PG_element_rank_modified(
				v, 1, 5, j);

		Pts[i] = j;

		if (f_v) {
			cout << setw(4) << i << " : ";
			Int_vec_print(cout, v, 5);
			cout << " : " << setw(5) << j << endl;
		}
	}

	string fname_csv;
	other::orbiter_kernel_system::file_io Fio;


	fname_csv = label_txt + "_in_PG.csv";

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname_csv, Pts, sz, 1);
	cout << "written file " << fname_csv << " of size "
			<< Fio.file_size(fname_csv) << endl;


	FREE_lint(Pts);

	if (f_v) {
		cout << "blt_set::export_set_in_PG done" << endl;
	}
}




}}}}


