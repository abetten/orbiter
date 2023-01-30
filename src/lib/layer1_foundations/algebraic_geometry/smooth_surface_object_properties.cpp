/*
 * smooth_surface_object_properties.cpp
 *
 *  Created on: Jan 9, 2023
 *      Author: betten
 */







#include "foundations.h"


using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {




smooth_surface_object_properties::smooth_surface_object_properties()
{

	SO = NULL;

	Tritangent_plane_rk = NULL;
	nb_tritangent_planes = 0;

	Lines_in_tritangent_planes = NULL;

	Trihedral_pairs_as_tritangent_planes = NULL;

	All_Planes = NULL;
	Dual_point_ranks = NULL;

}

smooth_surface_object_properties::~smooth_surface_object_properties()
{
	if (Tritangent_plane_rk) {
		FREE_lint(Tritangent_plane_rk);
	}


	if (Lines_in_tritangent_planes) {
		FREE_lint(Lines_in_tritangent_planes);
	}

	if (Trihedral_pairs_as_tritangent_planes) {
		FREE_lint(Trihedral_pairs_as_tritangent_planes);
	}

	if (All_Planes) {
		FREE_lint(All_Planes);
	}
	if (Dual_point_ranks) {
		FREE_int(Dual_point_ranks);
	}
}

void smooth_surface_object_properties::init(surface_object *SO, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "smooth_surface_object_properties::init" << endl;
	}

	smooth_surface_object_properties::SO = SO;

	if (f_v) {
		cout << "smooth_surface_object_properties::init "
				"before compute_tritangent_planes_by_rank" << endl;
	}
	compute_tritangent_planes_by_rank(0 /*verbose_level*/);
	if (f_v) {
		cout << "smooth_surface_object_properties::init "
				"after compute_tritangent_planes_by_rank" << endl;
	}

	if (f_v) {
		cout << "smooth_surface_object_properties::init "
				"before compute_Lines_in_tritangent_planes" << endl;
	}
	compute_Lines_in_tritangent_planes(0 /*verbose_level*/);
	if (f_v) {
		cout << "smooth_surface_object_properties::init "
				"after compute_Lines_in_tritangent_planes" << endl;
	}

	if (f_v) {
		cout << "smooth_surface_object_properties::init "
				"before compute_Trihedral_pairs_as_tritangent_planes" << endl;
	}
	compute_Trihedral_pairs_as_tritangent_planes(0 /*verbose_level*/);
	if (f_v) {
		cout << "smooth_surface_object_properties::init "
				"after compute_Trihedral_pairs_as_tritangent_planes" << endl;
	}

	if (f_v) {
		cout << "smooth_surface_object_properties::init done" << endl;
	}

}

void smooth_surface_object_properties::compute_tritangent_planes_by_rank(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "smooth_surface_object_properties::compute_tritangent_planes_by_rank" << endl;
	}


	if (SO->nb_lines != 27) {
		cout << "smooth_surface_object_properties::compute_tritangent_planes_by_rank "
				"SO->nb_lines != 27 we should not be here" << endl;
		//nb_tritangent_planes = 0;
		exit(1);
	}


	nb_tritangent_planes = 45;
	Tritangent_plane_rk = NEW_lint(45);


	int tritangent_plane_idx;
	int three_lines_idx[3];
	long int three_lines[3];
	int i, r;
	int Basis[6 * 4];
	int base_cols[4];

	for (tritangent_plane_idx = 0;
			tritangent_plane_idx < 45;
			tritangent_plane_idx++) {
		SO->Surf->Schlaefli->Eckardt_points[tritangent_plane_idx].three_lines(
				SO->Surf, three_lines_idx);

		for (i = 0; i < 3; i++) {

			three_lines[i] = SO->Lines[three_lines_idx[i]];

			SO->Surf->Gr->unrank_lint_here(Basis + i * 8,
					three_lines[i], 0 /* verbose_level */);

		}
		r = SO->F->Linear_algebra->Gauss_simple(Basis, 6, 4,
			base_cols, 0 /* verbose_level */);
		if (r != 3) {
			cout << "smooth_surface_object_properties::compute_tritangent_planes_by_rank r != 3" << endl;
			exit(1);
		}
		Tritangent_plane_rk[tritangent_plane_idx] =
				SO->Surf->Gr3->rank_lint_here(Basis, 0 /* verbose_level */);
	}
	if (f_vv) {
		cout << "smooth_surface_object_properties::compute_tritangent_planes_by_rank" << endl;
		for (tritangent_plane_idx = 0;
				tritangent_plane_idx < 45;
				tritangent_plane_idx++) {
			cout << tritangent_plane_idx << " : "
					<< Tritangent_plane_rk[tritangent_plane_idx] << endl;
		}
	}
	if (f_v) {
		cout << "smooth_surface_object_properties::compute_tritangent_planes_by_rank done" << endl;
	}
}


void smooth_surface_object_properties::compute_Lines_in_tritangent_planes(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int tritangent_plane_idx, j;

	if (f_v) {
		cout << "smooth_surface_object_properties::compute_Lines_in_tritangent_planes" << endl;
	}
	Lines_in_tritangent_planes = NEW_lint(45 * 3);
	for (tritangent_plane_idx = 0;
			tritangent_plane_idx < 45;
			tritangent_plane_idx++) {
		for (j = 0; j < 3; j++) {
			Lines_in_tritangent_planes[tritangent_plane_idx * 3 + j] =
				SO->Lines[SO->Surf->Schlaefli->Lines_in_tritangent_planes[tritangent_plane_idx * 3 + j]];
		}
	}

	if (f_v) {
		cout << "smooth_surface_object_properties::compute_Lines_in_tritangent_planes done" << endl;
	}
}

void smooth_surface_object_properties::compute_Trihedral_pairs_as_tritangent_planes(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "smooth_surface_object_properties::compute_Trihedral_pairs_as_tritangent_planes" << endl;
	}
	Trihedral_pairs_as_tritangent_planes = NEW_lint(120 * 6);
	for (i = 0; i < 120; i++) {
		for (j = 0; j < 6; j++) {
			Trihedral_pairs_as_tritangent_planes[i * 6 + j] =
					Tritangent_plane_rk[SO->Surf->Schlaefli->Trihedral_to_Eckardt[i * 6 + j]];
		}
	}

	if (f_v) {
		cout << "smooth_surface_object_properties::compute_Trihedral_pairs_as_tritangent_planes done" << endl;
	}
}

void smooth_surface_object_properties::compute_planes_and_dual_point_ranks(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "smooth_surface_object_properties::compute_planes_and_dual_point_ranks" << endl;
	}

	All_Planes = NEW_lint(SO->Surf->Schlaefli->nb_trihedral_pairs * 6);
	Dual_point_ranks = NEW_int(SO->Surf->Schlaefli->nb_trihedral_pairs * 6);
	//Iso_trihedral_pair = NEW_int(Surf->nb_trihedral_pairs);


	SO->Surf->Trihedral_pairs_to_planes(SO->Lines, All_Planes, 0 /*verbose_level*/);


	for (i = 0; i < SO->Surf->Schlaefli->nb_trihedral_pairs; i++) {
		//cout << "trihedral pair " << i << " / "
		// << Surf->nb_trihedral_pairs << endl;
		for (j = 0; j < 6; j++) {
			Dual_point_ranks[i * 6 + j] =
				SO->Surf->P->Solid->dual_rank_of_plane_in_three_space(
						All_Planes[i * 6 + j], 0 /* verbose_level */);
		}

	}
	if (f_v) {
		cout << "smooth_surface_object_properties::compute_planes_and_dual_point_ranks done" << endl;
	}
}

void smooth_surface_object_properties::print_planes_in_trihedral_pairs(std::ostream &ost)
{
	orbiter_kernel_system::latex_interface L;

	ost << "\\clearpage\n\\subsection*{All planes "
			"in trihedral pairs}" << endl;

	ost << "All planes by plane rank:" << endl;

	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels(ost,
		All_Planes, 30, 6, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		All_Planes + 30 * 6, 30, 6, 30, 0, TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		All_Planes + 60 * 6, 30, 6, 60, 0, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	L.print_lint_matrix_with_standard_labels_and_offset(ost,
		All_Planes + 90 * 6, 30, 6, 90, 0, TRUE /* f_tex */);
	ost << "$$" << endl;



	ost << "All planes by dual point rank:" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels(ost,
		Dual_point_ranks, 30, 6, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
		Dual_point_ranks + 30 * 6, 30, 6, 30, 0, TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
		Dual_point_ranks + 60 * 6, 30, 6, 60, 0, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	L.print_integer_matrix_with_standard_labels_and_offset(ost,
		Dual_point_ranks + 90 * 6, 30, 6, 90, 0, TRUE /* f_tex */);
	ost << "$$" << endl;
}

void smooth_surface_object_properties::print_tritangent_planes(std::ostream &ost)
{
	int i;
	//int plane_rk, b, v4[4];
	//int Mtx[16];

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Tritangent planes}" << endl;
	ost << "The " << nb_tritangent_planes << " tritangent "
			"planes are:\\\\" << endl;
	for (i = 0; i < nb_tritangent_planes; i++) {
		print_single_tritangent_plane(ost, i);
	}

#if 0
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Tritangent_planes, 9, 5, TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "Their dual point ranks are:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Tritangent_plane_dual, 9, 5, TRUE /* f_tex */);
	ost << "$$" << endl;

	for (i = 0; i < nb_tritangent_planes; i++) {
		a = Tritangent_planes[i];
		b = Tritangent_plane_dual[i];
		//b = Surf->P->dual_rank_of_plane_in_three_space(a, 0);
		ost << "plane " << i << " / " << nb_tritangent_planes
				<< " : rank " << a << " is $";
		ost << "\\left[" << endl;
		Surf->Gr3->print_single_generator_matrix_tex(ost, a);
		ost << "\\right]" << endl;
		ost << "$, dual pt rank = $" << b << "$ ";
		PG_element_unrank_modified(*F, v4, 1, 4, b);
		ost << "$=";
		int_vec_print(ost, v4, 4);
		ost << "$\\\\" << endl;
		}

	ost << "The iso types of the tritangent planes are:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			iso_type_of_tritangent_plane, nb_tritangent_planes / 5, 5,
			TRUE /* f_tex */);
	ost << "$$" << endl;

	ost << "Type iso of tritangent planes: ";
	ost << "$$" << endl;
	Type_iso_tritangent_planes->print_naked_tex(ost, TRUE);
	ost << endl;
	ost << "$$" << endl;
	ost << endl;

	ost << "Tritangent\\_plane\\_to\\_Eckardt:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Tritangent_plane_to_Eckardt, nb_tritangent_planes / 5, 5,
			TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "Eckardt\\_to\\_Tritangent\\_plane:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Eckardt_to_Tritangent_plane, nb_tritangent_planes / 5, 5,
			TRUE /* f_tex */);
	ost << "$$" << endl;

	ost << "Trihedral\\_pairs\\_as\\_tritangent\\_planes:" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost,
			Trihedral_pairs_as_tritangent_planes, 30, 6, TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
			Trihedral_pairs_as_tritangent_planes + 30 * 6, 30, 6, 30, 0,
			TRUE /* f_tex */);
	ost << "$$" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
			Trihedral_pairs_as_tritangent_planes + 60 * 6, 30, 6, 60, 0,
			TRUE /* f_tex */);
	ost << "\\;\\;" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost,
			Trihedral_pairs_as_tritangent_planes + 90 * 6, 30, 6, 90, 0,
			TRUE /* f_tex */);
	ost << "$$" << endl;
#endif

}

void smooth_surface_object_properties::print_single_tritangent_plane(
		std::ostream &ost, int plane_idx)
{
	long int plane_rk, b;
	int v4[4];
	int Mtx[16];

#if 0
	j = Eckardt_to_Tritangent_plane[plane_idx];
	plane_rk = Tritangent_planes[j];
	b = Tritangent_plane_dual[j];
#else
	plane_rk = Tritangent_plane_rk[plane_idx];
	b = SO->Surf->P->Solid->dual_rank_of_plane_in_three_space(
			plane_rk, 0 /* verbose_level */);
#endif
	ost << "$$" << endl;
	ost << "\\pi_{" << SO->Surf->Schlaefli->Eckard_point_label_tex[plane_idx] << "} = ";
	ost << "\\pi_{" << plane_idx << "} = " << plane_rk << " = ";
	//ost << "\\left[" << endl;
	SO->Surf->Gr3->print_single_generator_matrix_tex(ost, plane_rk);
	//ost << "\\right]" << endl;
	ost << " = ";
	SO->Surf->Gr3->print_single_generator_matrix_tex_numerical(ost, plane_rk);

	SO->Surf->Gr3->unrank_lint_here_and_compute_perp(Mtx, plane_rk,
		0 /*verbose_level */);
	SO->F->Projective_space_basic->PG_element_normalize(Mtx + 12, 1, 4);
	ost << "$$" << endl;
	ost << "$$" << endl;
	ost << "=V\\big(" << endl;
	SO->Surf->PolynomialDomains->Poly1_4->print_equation(ost, Mtx + 12);
	ost << "\\big)" << endl;
	ost << "=V\\big(" << endl;
	SO->Surf->PolynomialDomains->Poly1_4->print_equation_numerical(ost, Mtx + 12);
	ost << "\\big)" << endl;
	ost << "$$" << endl;
	ost << "dual pt rank = $" << b << "$ ";
	SO->F->Projective_space_basic->PG_element_unrank_modified(v4, 1, 4, b);
	ost << "$=";
	Int_vec_print(ost, v4, 4);
	ost << "$.\\\\" << endl;

}


void smooth_surface_object_properties::latex_table_of_trihedral_pairs(
		std::ostream &ost,
		int *T, int nb_T)
{
	int h, i, j, t_idx;

	cout << "smooth_surface_object_properties::latex_table_of_trihedral_pairs" << endl;
	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Trihedral Pairs}" << endl;
	//ost << "\\begin{multicols}{2}" << endl;
	ost << "\\noindent" << endl;
	for (h = 0; h < nb_T; h++) {
		ost << "$" << h << " / " << nb_T << "$ ";
		t_idx = T[h];
		ost << "$T_{" << t_idx << "} = T_{"
				<< SO->Surf->Schlaefli->Trihedral_pair_labels[t_idx]
				<< "} = \\\\" << endl;
		latex_trihedral_pair(ost, t_idx);
		ost << "$\\\\" << endl;
		ost << "$";
		make_and_print_equation_in_trihedral_form(ost, t_idx);
		ost << "$\\\\" << endl;
	}
	ost << "Dual point ranks: \\\\" << endl;
	for (i = 0; i < SO->Surf->Schlaefli->nb_trihedral_pairs; i++) {
		ost << "$T_{" << i << "} = T_{"
			<< SO->Surf->Schlaefli->Trihedral_pair_labels[i]
			<< "}: \\quad " << endl;
		for (j = 0; j < 6; j++) {
			ost << Dual_point_ranks[i * 6 + j];
			if (j < 6 - 1) {
				ost << ", ";
			}
		}
		ost << "$\\\\" << endl;
	}


#if 0
	ost << "Planes by generator matrix: \\\\" << endl;;
	for (i = 0; i < Surf->nb_trihedral_pairs; i++) {
		ost << "$T_{" << i << "} = T_{"
			<< Surf->Trihedral_pair_labels[i] << "}$" << endl;
		for (j = 0; j < 6; j++) {
			int d;

			d = All_Planes[i * 6 + j];
			ost << "Plane " << j << " has rank " << d << "\\\\" << endl;
			Surf->Gr3->unrank_int(d, 0 /* verbose_level */);
			ost << "$";
			ost << "\\left[";
			print_integer_matrix_tex(ost, Surf->Gr3->M, 3, 4);
			ost << "\\right]";
			ost << "$\\\\" << endl;
		}
	}
#endif
	//ost << "\\end{multicols}" << endl;
	cout << "smooth_surface_object_properties::latex_table_of_trihedral_pairs done" << endl;
}

void smooth_surface_object_properties::latex_trihedral_pair(
		std::ostream &ost, int t_idx)
{
	int i, j, e, a;

	//ost << "\\left[" << endl;
	ost << "\\begin{array}{c||ccc|cc}" << endl;
	ost << " & G_0 & G_1 & G_2 & \\mbox{plane} & "
			"\\mbox{dual rank} \\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < 3; i++) {
		ost << "F_" << i;
		for (j = 0; j < 3; j++) {
			a = SO->Surf->Schlaefli->Trihedral_pairs[t_idx * 9 + i * 3 + j];
			ost << " & {" << SO->Surf->Schlaefli->Labels->Line_label_tex[a] << "=\\atop";
			ost << "\\left[" << endl;
			SO->Surf->Gr->print_single_generator_matrix_tex(ost, SO->Lines[a]);
			ost << "\\right]}" << endl;
		}
		e = SO->Surf->Schlaefli->Trihedral_to_Eckardt[t_idx * 6 + i];
		ost << " & {\\pi_{" << e << "} =\\atop";
#if 0
		t = Eckardt_to_Tritangent_plane[e];
		a = Tritangent_planes[t];
#else
		a = Tritangent_plane_rk[e];
#endif
		ost << "\\left[" << endl;
		SO->Surf->Gr3->print_single_generator_matrix_tex(ost, a);
		ost << "\\right]}" << endl;
		ost << " & ";
		a = Dual_point_ranks[t_idx * 6 + i];
		ost << a << "\\\\" << endl;
	}
	ost << "\\hline" << endl;
	for (j = 0; j < 3; j++) {
		e = SO->Surf->Schlaefli->Trihedral_to_Eckardt[t_idx * 6 + 3 + j];
		ost << " & {\\pi_{" << e << "} =\\atop";
#if 0
		t = Eckardt_to_Tritangent_plane[e];
		a = Tritangent_planes[t];
#else
		a = Tritangent_plane_rk[e];
#endif
		ost << "\\left[" << endl;
		SO->Surf->Gr3->print_single_generator_matrix_tex(ost, a);
		ost << "\\right]}" << endl;
	}
	ost << " & & \\\\" << endl;
	for (j = 0; j < 3; j++) {
		a = Dual_point_ranks[t_idx * 6 + 3 + j];
		ost << " & " << a;
	}
	ost << " & & \\\\" << endl;
	//Surf->latex_trihedral_pair(ost, Surf->Trihedral_pairs + h * 9);
	ost << "\\end{array}" << endl;
	//ost << "\\right]" << endl;
}

void smooth_surface_object_properties::make_equation_in_trihedral_form(int t_idx,
	int *F_planes, int *G_planes, int &lambda, int *equation,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, c, h;
	int row_col_Eckardt_points[6];
	int plane_rk[6];
	//int plane_idx[6];

	if (f_v) {
		cout << "smooth_surface_object_properties::make_equation_in_trihedral_form "
				"t_idx=" << t_idx << endl;
	}

	if (f_v) {
		cout << "Trihedral pair T_{"
			<< SO->Surf->Schlaefli->Trihedral_pair_labels[t_idx] << "}"
			<< endl;
	}

	for (h = 0; h < 6; h++) {
		row_col_Eckardt_points[h] =
				SO->Surf->Schlaefli->Trihedral_to_Eckardt[t_idx * 6 + h];
	}
	//int_vec_copy(Surf->Trihedral_to_Eckardt + t_idx * 6, row_col_Eckardt_points, 6);
	for (i = 0; i < 6; i++) {
		//plane_idx[i] = Eckardt_to_Tritangent_plane[row_col_Eckardt_points[i]];
		//plane_rk[i] = Tritangent_planes[plane_idx[i]];
		plane_rk[i] = Tritangent_plane_rk[row_col_Eckardt_points[i]];
	}
	for (i = 0; i < 3; i++) {
		c = SO->Surf->P->Solid->dual_rank_of_plane_in_three_space(
				plane_rk[i], 0 /* verbose_level */);
		//c = Tritangent_plane_dual[plane_idx[i]];
		SO->F->Projective_space_basic->PG_element_unrank_modified(F_planes + i * 4, 1, 4, c);
	}
	for (i = 0; i < 3; i++) {
		c = SO->Surf->P->Solid->dual_rank_of_plane_in_three_space(
				plane_rk[3 + i], 0 /* verbose_level */);
		//c = Tritangent_plane_dual[plane_idx[3 + i]];
		SO->F->Projective_space_basic->PG_element_unrank_modified(G_planes + i * 4, 1, 4, c);
	}
	int evals[6];
	int pt_on_surface[4];
	int a, b, ma, bv, pt;
	int eqn_F[20];
	int eqn_G[20];
	int eqn_G2[20];

	for (h = 0; h < SO->nb_pts; h++) {
		pt = SO->Pts[h];
		SO->F->Projective_space_basic->PG_element_unrank_modified(pt_on_surface, 1, 4, pt);
		for (i = 0; i < 3; i++) {
			evals[i] = SO->Surf->PolynomialDomains->Poly1_4->evaluate_at_a_point(
					F_planes + i * 4, pt_on_surface);
		}
		for (i = 0; i < 3; i++) {
			evals[3 + i] = SO->Surf->PolynomialDomains->Poly1_4->evaluate_at_a_point(
					G_planes + i * 4, pt_on_surface);
		}
		a = SO->F->mult3(evals[0], evals[1], evals[2]);
		b = SO->F->mult3(evals[3], evals[4], evals[5]);
		if (b) {
			ma = SO->F->negate(a);
			bv = SO->F->inverse(b);
			lambda = SO->F->mult(ma, bv);
			break;
		}
	}
	if (h == SO->nb_pts) {
		cout << "smooth_surface_object_properties::make_equation_in_trihedral_form could "
				"not determine lambda" << endl;
		exit(1);
	}

	SO->Surf->PolynomialDomains->multiply_linear_times_linear_times_linear_in_space(F_planes,
		F_planes + 4, F_planes + 8,
		eqn_F, FALSE /* verbose_level */);
	SO->Surf->PolynomialDomains->multiply_linear_times_linear_times_linear_in_space(G_planes,
		G_planes + 4, G_planes + 8,
		eqn_G, FALSE /* verbose_level */);

	Int_vec_copy(eqn_G, eqn_G2, 20);
	SO->F->Linear_algebra->scalar_multiply_vector_in_place(lambda, eqn_G2, 20);
	SO->F->Linear_algebra->add_vector(eqn_F, eqn_G2, equation, 20);
	SO->F->Projective_space_basic->PG_element_normalize(equation, 1, 20);



	if (f_v) {
		cout << "smooth_surface_object_properties::make_equation_in_trihedral_form done" << endl;
	}
}

void smooth_surface_object_properties::print_equation_in_trihedral_form(
		std::ostream &ost,
	int *F_planes, int *G_planes, int lambda)
{

	ost << "\\begin{align*}" << endl;
	ost << "0 & = F_0F_1F_2 + \\lambda G_0G_1G_2\\\\" << endl;
	ost << "& = " << endl;

	print_equation_in_trihedral_form_equation_only(ost, F_planes, G_planes, lambda);
}

void smooth_surface_object_properties::print_equation_in_trihedral_form_equation_only(
	std::ostream &ost,
	int *F_planes, int *G_planes, int lambda)
{

	ost << "\\Big(";
	SO->Surf->PolynomialDomains->Poly1_4->print_equation(ost, F_planes);
	ost << "\\Big)";
	ost << "\\Big(";
	SO->Surf->PolynomialDomains->Poly1_4->print_equation(ost, F_planes + 4);
	ost << "\\Big)";
	ost << "\\Big(";
	SO->Surf->PolynomialDomains->Poly1_4->print_equation(ost, F_planes + 8);
	ost << "\\Big)";
	ost << "+ " << lambda;
	ost << "\\Big(";
	SO->Surf->PolynomialDomains->Poly1_4->print_equation(ost, G_planes);
	ost << "\\Big)";
	ost << "\\Big(";
	SO->Surf->PolynomialDomains->Poly1_4->print_equation(ost, G_planes + 4);
	ost << "\\Big)";
	ost << "\\Big(";
	SO->Surf->PolynomialDomains->Poly1_4->print_equation(ost, G_planes + 8);
	ost << "\\Big)";
}

void smooth_surface_object_properties::make_and_print_equation_in_trihedral_form(
	std::ostream &ost, int t_idx)
{
	int F_planes[12];
	int G_planes[12];
	int lambda;
	int equation[20];
	//int *system;

	make_equation_in_trihedral_form(t_idx, F_planes, G_planes,
		lambda, equation, 0 /* verbose_level */);
	print_equation_in_trihedral_form_equation_only(ost,
		F_planes, G_planes, lambda);
	//FREE_int(system);
}

void smooth_surface_object_properties::latex_table_of_trihedral_pairs_and_clebsch_system(
	std::ostream &ost, int *T, int nb_T)
{
	int t_idx, t;

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Trihedral Pairs and the Clebsch System}" << endl;

	for (t = 0; t < nb_T; t++) {

		t_idx = T[t];


		int F_planes[12];
		int G_planes[12];
		int lambda;
		int equation[20];
		int *system;

		make_equation_in_trihedral_form(t_idx,
			F_planes, G_planes, lambda, equation,
			0 /* verbose_level */);

#if 0
		if (t_idx == 71) {
			int_vec_swap(F_planes, F_planes + 8, 4);
			}
#endif

		SO->Surf->prepare_system_from_FG(F_planes, G_planes,
				lambda, system, 0 /*verbose_level*/);


		ost << "$" << t << " / " << nb_T << "$ ";
		ost << "$T_{" << t_idx << "} = T_{"
			<< SO->Surf->Schlaefli->Trihedral_pair_labels[t_idx]
			<< "} = \\\\" << endl;
		latex_trihedral_pair(ost, t_idx);
		ost << "$\\\\" << endl;
		ost << "$";
		print_equation_in_trihedral_form_equation_only(ost,
				F_planes, G_planes, lambda);
		ost << "$\\\\" << endl;
		//ost << "$";
		SO->Surf->PolynomialDomains->print_system(ost, system);
		//ost << "$\\\\" << endl;
		FREE_int(system);


	}
}

void smooth_surface_object_properties::latex_trihedral_pair(
		std::ostream &ost, int *T, long int *TE)
{
	int i, j, plane_rk;
	int Mtx[16];

	ost << "\\begin{array}{*{" << 3 << "}{c}|c}" << endl;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			SO->Surf->Schlaefli->print_line(ost, T[i * 3 + j]);
			ost << " & ";
		}
		ost << "\\pi_{";
		SO->Surf->Schlaefli->Eckardt_points[TE[i]].latex_index_only(ost);
		ost << "}=" << endl;
#if 0
		t = Eckardt_to_Tritangent_plane[TE[i]];
		plane_rk = Tritangent_planes[t];
#else
		plane_rk = Tritangent_plane_rk[TE[i]];
#endif
		SO->Surf->Gr3->unrank_lint_here_and_compute_perp(Mtx, plane_rk,
			0 /*verbose_level */);
		SO->F->Projective_space_basic->PG_element_normalize(Mtx + 12, 1, 4);
		ost << "V\\big(";
		SO->Surf->PolynomialDomains->Poly1_4->print_equation(ost, Mtx + 12);
		ost << "\\big)=" << plane_rk;
		ost << "\\\\" << endl;
	}
	ost << "\\hline" << endl;
	for (j = 0; j < 3; j++) {
		ost << "\\pi_{";
		SO->Surf->Schlaefli->Eckardt_points[TE[3 + j]].latex_index_only(ost);
		ost << "} & ";
	}
	ost << "\\\\" << endl;
	for (j = 0; j < 3; j++) {
		ost << "\\multicolumn{4}{l}{" << endl;
#if 0
		t = Eckardt_to_Tritangent_plane[TE[3 + j]];
		plane_rk = Tritangent_planes[t];
#else
		plane_rk = Tritangent_plane_rk[TE[i]];
#endif
		ost << "\\pi_{";
		SO->Surf->Schlaefli->Eckardt_points[TE[3 + j]].latex_index_only(ost);
		ost << "}=" << endl;
		ost << "V\\big(" << endl;
		SO->Surf->Gr3->unrank_lint_here_and_compute_perp(Mtx, plane_rk,
			0 /*verbose_level */);
		SO->F->Projective_space_basic->PG_element_normalize(Mtx + 12, 1, 4);
		SO->Surf->PolynomialDomains->Poly1_4->print_equation(ost, Mtx + 12);
		ost << "\\big)=" << plane_rk << "}\\\\" << endl;
	}
	ost << "\\\\" << endl;
	ost << "\\end{array}" << endl;
}

void smooth_surface_object_properties::latex_table_of_trihedral_pairs(
		std::ostream &ost)
{
	int i;

	cout << "surface_object_properties::latex_table_of_trihedral_pairs" << endl;
	//ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < SO->Surf->Schlaefli->nb_trihedral_pairs; i++) {
		ost << "$T_{" << i << "} = T_{"
				<< SO->Surf->Schlaefli->Trihedral_pair_labels[i]
				<< "} = $\\\\" << endl;
		ost << "$" << endl;
		//ost << "\\left[" << endl;
		//ost << "\\begin{array}" << endl;
		latex_trihedral_pair(ost,
				SO->Surf->Schlaefli->Trihedral_pairs + i * 9,
				SO->Surf->Schlaefli->Trihedral_to_Eckardt + i * 6);
		//ost << "\\end{array}" << endl;
		//ost << "\\right]" << endl;
		ost << "$\\\\" << endl;
#if 0
		ost << "planes: $";
		int_vec_print(ost, Trihedral_to_Eckardt + i * 6, 6);
		ost << "$\\\\" << endl;
#endif
	}
	//ost << "\\end{multicols}" << endl;

	//print_trihedral_pairs(ost);

	cout << "surface_object_properties::latex_table_of_trihedral_pairs done" << endl;
}

void smooth_surface_object_properties::print_Steiner_and_Eckardt(
		std::ostream &ost)
{
#if 0
	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Eckardt Points}" << endl;
	latex_table_of_Eckardt_points(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Tritangent Planes}" << endl;
	latex_table_of_tritangent_planes(ost);
#endif

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Steiner Trihedral Pairs}" << endl;
	latex_table_of_trihedral_pairs(ost);

}


}}}


