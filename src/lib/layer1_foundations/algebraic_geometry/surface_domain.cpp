// surface_domain.cpp
// 
// Anton Betten
// Jul 25, 2016
//
// 
//
//

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {


surface_domain::surface_domain()
{
	q = 0;
	n = 0;
	n2 = 0;

	F = NULL;
	P = NULL;
	P2 = NULL;
	Gr = NULL;
	Gr3 = NULL;
	nb_lines_PG_3 = 0;
	nb_pts_on_surface_with_27_lines = 0;

	O = NULL;
	Klein = NULL;

	//int Basis0[16];
	//int Basis1[16];
	//int Basis2[16];

	v = NULL;
	v2 = NULL;
	w2 = NULL;

	nb_monomials = 0;

	Schlaefli = NULL;

	Poly1 = NULL;
	Poly2 = NULL;
	Poly3 = NULL;
	Poly1_x123 = NULL;
	Poly2_x123 = NULL;
	Poly3_x123 = NULL;
	Poly4_x123 = NULL;
	Poly1_4 = NULL;
	Poly2_4 = NULL;
	Poly3_4 = NULL;

	Partials = NULL;

	f_has_large_polynomial_domains = FALSE;
	Poly2_27 = NULL;
	Poly4_27 = NULL;
	Poly6_27 = NULL;
	Poly3_24 = NULL;

	nb_monomials2 = nb_monomials4 = nb_monomials6 = 0;
	nb_monomials3 = 0;

	Clebsch_Pij = NULL;
	Clebsch_P = NULL;
	Clebsch_P3 = NULL;
	Clebsch_coeffs = NULL;
	CC = NULL;

}





surface_domain::~surface_domain()
{
	int f_v = FALSE;

	if (f_v) {
		cout << "surface_domain::~surface_domain" << endl;
	}
	if (v) {
		FREE_int(v);
	}
	if (v2) {
		FREE_int(v2);
	}
	if (w2) {
		FREE_int(w2);
	}
	if (P) {
		FREE_OBJECT(P);
	}
	if (P2) {
		FREE_OBJECT(P2);
	}
	if (Gr) {
		FREE_OBJECT(Gr);
	}
	if (Gr3) {
		FREE_OBJECT(Gr3);
	}
	if (O) {
		FREE_OBJECT(O);
	}
	if (Klein) {
		FREE_OBJECT(Klein);
	}

	if (Schlaefli) {
		FREE_OBJECT(Schlaefli);
	}

	if (f_v) {
		cout << "before Poly1" << endl;
	}

	if (Poly1) {
		FREE_OBJECT(Poly1);
	}
	if (Poly2) {
		FREE_OBJECT(Poly2);
	}
	if (Poly3) {
		FREE_OBJECT(Poly3);
	}
	if (Poly1_x123) {
		FREE_OBJECT(Poly1_x123);
	}
	if (Poly2_x123) {
		FREE_OBJECT(Poly2_x123);
	}
	if (Poly3_x123) {
		FREE_OBJECT(Poly3_x123);
	}
	if (Poly4_x123) {
		FREE_OBJECT(Poly4_x123);
	}
	if (Poly1_4) {
		FREE_OBJECT(Poly1_4);
	}
	if (Poly2_4) {
		FREE_OBJECT(Poly2_4);
	}
	if (Poly3_4) {
		FREE_OBJECT(Poly3_4);
	}

	if (Partials) {
		FREE_OBJECTS(Partials);
	}

	if (f_has_large_polynomial_domains) {
		if (Poly2_27) {
			FREE_OBJECT(Poly2_27);
		}
		if (Poly4_27) {
			FREE_OBJECT(Poly4_27);
		}
		if (Poly6_27) {
			FREE_OBJECT(Poly6_27);
		}
		if (Poly3_24) {
			FREE_OBJECT(Poly3_24);
		}
	}
	if (Clebsch_Pij) {
		FREE_int(Clebsch_Pij);
	}
	if (Clebsch_P) {
		FREE_pint(Clebsch_P);
	}
	if (Clebsch_P3) {
		FREE_pint(Clebsch_P3);
	}
	if (Clebsch_coeffs) {
		FREE_int(Clebsch_coeffs);
	}
	if (CC) {
		FREE_pint(CC);
	}
	if (f_v) {
		cout << "surface_domain::~surface_domain done" << endl;
	}
}

void surface_domain::init(field_theory::finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::init" << endl;
	}
	
	n = 4;
	n2 = 2 * n;
	surface_domain::F = F;
	q = F->q;
	nb_pts_on_surface_with_27_lines = q * q + 7 * q + 1;
	if (f_v) {
		cout << "surface::init nb_pts_on_surface_with_27_lines = "
				<< nb_pts_on_surface_with_27_lines << endl;
	}

	v = NEW_int(n);
	v2 = NEW_int(6);
	w2 = NEW_int(6);
	
	P = NEW_OBJECT(geometry::projective_space);
	if (f_v) {
		cout << "surface::init before P->projective_space_init" << endl;
	}
	P->projective_space_init(3, F,
		TRUE /*f_init_incidence_structure */, 
		verbose_level - 2);
	if (f_v) {
		cout << "surface::init after P->projective_space_init" << endl;
	}

	P2 = NEW_OBJECT(geometry::projective_space);
	if (f_v) {
		cout << "surface::init before P2->projective_space_init" << endl;
	}
	P2->projective_space_init(2, F,
		TRUE /*f_init_incidence_structure */, 
		verbose_level - 2);
	if (f_v) {
		cout << "surface::init after P2->projective_space_init" << endl;
	}

	Gr = NEW_OBJECT(geometry::grassmann);
	Gr->init(n, 2, F, 0 /* verbose_level */);
	nb_lines_PG_3 = Gr->nCkq->as_lint();
	if (f_v) {
		cout << "surface::init nb_lines_PG_3 = "
				<< nb_lines_PG_3 << endl;
	}

	Gr3 = NEW_OBJECT(geometry::grassmann);
	Gr3->init(4, 3, F, 0 /* verbose_level*/);


	if (f_v) {
		cout << "surface::init "
				"initializing orthogonal" << endl;
	}
	O = NEW_OBJECT(orthogonal_geometry::orthogonal);
	O->init(1 /* epsilon */, 6 /* n */, F, verbose_level - 2);
	if (f_v) {
		cout << "surface::init "
				"initializing orthogonal done" << endl;
	}

	Klein = NEW_OBJECT(geometry::klein_correspondence);

	if (f_v) {
		cout << "surface::init before Klein->init" << endl;
	}
	Klein->init(F, O, verbose_level - 2);
	if (f_v) {
		cout << "surface::init after Klein->init" << endl;
	}



	if (f_v) {
		cout << "surface::init before init_polynomial_domains" << endl;
	}
	init_polynomial_domains(verbose_level - 2);
	if (f_v) {
		cout << "surface::init after init_polynomial_domains" << endl;
	}

	//init_large_polynomial_domains(verbose_level);



	if (f_v) {
		cout << "surface::init before init_Schlaefli" << endl;
	}
	init_Schlaefli(verbose_level - 2);
	if (f_v) {
		cout << "surface::init after init_Schlaefli" << endl;
	}



	//clebsch_cubics(verbose_level);

	if (f_v) {
		cout << "surface::init done" << endl;
	}
}


void surface_domain::init_polynomial_domains(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains" << endl;
	}
	Poly1 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly2 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly3 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly1->init" << endl;
	}
	Poly1->init(F,
			3 /* nb_vars */, 1 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly1->init" << endl;
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly2->init" << endl;
	}
	Poly2->init(F,
			3 /* nb_vars */, 2 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly2->init" << endl;
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly3->init" << endl;
	}
	Poly3->init(F,
			3 /* nb_vars */, 3 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly3->init" << endl;
	}

	Poly1_x123 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly2_x123 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly3_x123 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly4_x123 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly1_x123->init" << endl;
	}
	Poly1_x123->init(F,
			3 /* nb_vars */, 1 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly1_x123->init" << endl;
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly2_x123->init" << endl;
	}
	Poly2_x123->init(F,
			3 /* nb_vars */, 2 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly2_x123->init" << endl;
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly3_x123->init" << endl;
	}
	Poly3_x123->init(F,
			3 /* nb_vars */, 3 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly3_x123->init" << endl;
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly4_x123->init" << endl;
	}
	Poly4_x123->init(F,
			3 /* nb_vars */, 4 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly4_x123->init" << endl;
	}


	label_variables_3(Poly1, 0 /* verbose_level */);
	label_variables_3(Poly2, 0 /* verbose_level */);
	label_variables_3(Poly3, 0 /* verbose_level */);

	label_variables_x123(Poly1_x123, 0 /* verbose_level */);
	label_variables_x123(Poly2_x123, 0 /* verbose_level */);
	label_variables_x123(Poly3_x123, 0 /* verbose_level */);
	label_variables_x123(Poly4_x123, 0 /* verbose_level */);

	Poly1_4 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly2_4 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly3_4 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly1_4->init" << endl;
	}
	Poly1_4->init(F,
			4 /* nb_vars */, 1 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly1_4->init" << endl;
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly2_4->init" << endl;
	}
	Poly2_4->init(F,
			4 /* nb_vars */, 2 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly2_4->init" << endl;
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly3_4->init" << endl;
	}
	Poly3_4->init(F,
			4 /* nb_vars */, 3 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly3_4->init" << endl;
	}

	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before label_variables_4" << endl;
	}
	label_variables_4(Poly1_4, 0 /* verbose_level */);
	label_variables_4(Poly2_4, 0 /* verbose_level */);
	label_variables_4(Poly3_4, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after label_variables_4" << endl;
	}

	nb_monomials = Poly3_4->get_nb_monomials();

	if (f_v) {
		cout << "Poly3_4->nb_monomials = " << nb_monomials << endl;
		cout << "Poly2_4->nb_monomials = " << Poly2_4->get_nb_monomials() << endl;
	}



	Partials = NEW_OBJECTS(ring_theory::partial_derivative, 4);

	int i;

	if (f_v) {
		cout << "surface_domain::init_polynomial_domains initializing partials" << endl;
	}
	for (i = 0; i < 4; i++) {
		Partials[i].init(Poly3_4, Poly2_4, i, verbose_level);
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains initializing partials done" << endl;
	}




	if (f_v) {
		cout << "surface_domain::init_polynomial_domains done" << endl;
	}
}

void surface_domain::init_large_polynomial_domains(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_domain::init_large_polynomial_domains" << endl;
	}
	f_has_large_polynomial_domains = TRUE;
	Poly2_27 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly4_27 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly6_27 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);
	Poly3_24 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	Poly2_27->init(F,
			27 /* nb_vars */, 2 /* degree */,
			t_PART,
			verbose_level);
	Poly4_27->init(F,
			27 /* nb_vars */, 4 /* degree */,
			t_PART,
			verbose_level);
	Poly6_27->init(F,
			27 /* nb_vars */, 6 /* degree */,
			t_PART,
			verbose_level);
	Poly3_24->init(F,
			24 /* nb_vars */, 3 /* degree */,
			t_PART,
			verbose_level);

	nb_monomials2 = Poly2_27->get_nb_monomials();
	nb_monomials4 = Poly4_27->get_nb_monomials();
	nb_monomials6 = Poly6_27->get_nb_monomials();
	nb_monomials3 = Poly3_24->get_nb_monomials();

	label_variables_27(Poly2_27, 0 /* verbose_level */);
	label_variables_27(Poly4_27, 0 /* verbose_level */);
	label_variables_27(Poly6_27, 0 /* verbose_level */);
	label_variables_24(Poly3_24, 0 /* verbose_level */);

	if (f_v) {
		cout << "nb_monomials2 = " << nb_monomials2 << endl;
		cout << "nb_monomials4 = " << nb_monomials4 << endl;
		cout << "nb_monomials6 = " << nb_monomials6 << endl;
		cout << "nb_monomials3 = " << nb_monomials3 << endl;
	}

	if (f_v) {
		cout << "surface_domain::init_large_polynomial_domains "
				"before clebsch_cubics" << endl;
	}
	clebsch_cubics(verbose_level - 1);
	if (f_v) {
		cout << "surface_domain::init_large_polynomial_domains "
				"after clebsch_cubics" << endl;
	}

	if (f_v) {
		cout << "surface::init_large_polynomial_domains done" << endl;
	}
}

void surface_domain::label_variables_3(
		ring_theory::homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::label_variables_3" << endl;
	}
	if (HPD->nb_variables != 3) {
		cout << "surface_domain::label_variables_3 HPD->nb_variables != 3" << endl;
		exit(1);
	}

	HPD->remake_symbols(0 /* symbol_offset */,
			"y_%d", "y_{%d}", verbose_level);

	if (f_v) {
		cout << "surface_domain::label_variables_3 done" << endl;
	}
}

void surface_domain::label_variables_x123(
		ring_theory::homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_domain::label_variables_x123" << endl;
	}
	if (HPD->nb_variables != 3) {
		cout << "surface_domain::label_variables_x123 "
				"HPD->nb_variables != 3" << endl;
		exit(1);
	}


	HPD->remake_symbols(1 /* symbol_offset */,
			"x_%d", "x_{%d}", verbose_level);

	if (f_v) {
		cout << "surface_domain::label_variables_x123 done" << endl;
	}
}

void surface_domain::label_variables_4(
		ring_theory::homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i, l;
	//char label[1000];
	
	if (f_v) {
		cout << "surface_domain::label_variables_4" << endl;
		}
	if (HPD->nb_variables != 4) {
		cout << "surface_domain::label_variables_4 HPD->nb_variables != 4" << endl;
		exit(1);
		}


	HPD->remake_symbols(0 /* symbol_offset */,
			"X_%d", "X_{%d}", verbose_level);


	if (f_v) {
		cout << "surface::label_variables_4 done" << endl;
		}
	
}

void surface_domain::label_variables_27(
		ring_theory::homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_domain::label_variables_27" << endl;
	}
	if (HPD->nb_variables != 27) {
		cout << "surface_domain::label_variables_27 HPD->n != 27" << endl;
		exit(1);
	}

	HPD->remake_symbols_interval(0 /* symbol_offset */,
			0, 3,
			"y_%d", "y_{%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			3, 4,
			"f_0%d", "f_{0%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			7, 4,
			"f_1%d", "f_{1%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			11, 4,
			"f_2%d", "f_{2%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			15, 4,
			"g_0%d", "g_{0%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			19, 4,
			"g_1%d", "g_{1%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			23, 4,
			"g_2%d", "g_{2%d}",
			verbose_level);

	if (f_v) {
		cout << "surface_domain::label_variables_27 done" << endl;
	}
}

void surface_domain::label_variables_24(
		ring_theory::homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_domain::label_variables_24" << endl;
	}
	if (HPD->nb_variables != 24) {
		cout << "surface_domain::label_variables_24 HPD->n != 24" << endl;
		exit(1);
	}

	HPD->remake_symbols_interval(0 /* symbol_offset */,
			0, 4,
			"f_0%d", "f_{0%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			4, 4,
			"f_1%d", "f_{1%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			8, 4,
			"f_2%d", "f_{2%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			12, 4,
			"g_0%d", "g_{0%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			16, 4,
			"g_1%d", "g_{1%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			20, 4,
			"g_2%d", "g_{2%d}",
			verbose_level);

	if (f_v) {
		cout << "surface_domain::label_variables_24 done" << endl;
	}
}


int surface_domain::index_of_monomial(int *v)
{
	return Poly3_4->index_of_monomial(v);
}

void surface_domain::unrank_point(int *v, long int rk)
{
	P->unrank_point(v, rk);
}

long int surface_domain::rank_point(int *v)
{
	long int rk;

	rk = P->rank_point(v);
	return rk;
}

void surface_domain::unrank_plane(int *v, long int rk)
{
	Gr3->unrank_lint_here(v, rk, 0 /* verbose_level */);
}

long int surface_domain::rank_plane(int *v)
{
	long int rk;

	rk = Gr3->rank_lint_here(v, 0 /* verbose_level */);
	return rk;
}

void surface_domain::enumerate_points(int *coeff,
		std::vector<long int> &Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::enumerate_points" << endl;
	}

	Poly3_4->enumerate_points(coeff, Pts, verbose_level);
	if (f_v) {
		cout << "surface_domain::enumerate_points done" << endl;
	}
}

void surface_domain::substitute_semilinear(
	int *coeff_in, int *coeff_out,
	int f_semilinear, int frob, int *Mtx_inv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::substitute_semilinear" << endl;
	}
	Poly3_4->substitute_semilinear(coeff_in, coeff_out, 
		f_semilinear, frob, Mtx_inv, verbose_level);
	if (f_v) {
		cout << "surface_domain::substitute_semilinear done" << endl;
	}
}

void surface_domain::list_starter_configurations(
	long int *Lines, int nb_lines,
	data_structures::set_of_sets *line_intersections, int *&Table, int &N,
	int verbose_level)
// goes over all lines and considers all 5-subsets
// of the set of lines which intersect
// Then filters those 5-subsets which together
// with the original line give 19 linearly independent conditions
{
	int f_v = (verbose_level >= 1);
	combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "surface_domain::list_starter_configurations" << endl;
	}

	vector<vector<int>> V;

	{
		int subset[5];
		long int subset2[5];
		long int S6[6];
		int nCk, h;
		int i, j, r;
		N = 0;
		for (i = 0; i < nb_lines; i++) {
			if (line_intersections->Set_size[i] < 5) {
				continue;
			}
			nCk = Combi.int_n_choose_k(line_intersections->Set_size[i], 5);
			for (j = 0; j < nCk; j++) {
				Combi.unrank_k_subset(j, subset,
					line_intersections->Set_size[i], 5);
				for (h = 0; h < 5; h++) {
					subset2[h] =
					line_intersections->Sets[i][subset[h]];
					S6[h] = Lines[subset2[h]];
				}
				S6[5] = Lines[i];
				r = rank_of_system(6, S6, 0 /*verbose_level*/);
				if (r == 19) {
					vector<int> v;

					v.push_back(i);
					v.push_back(j);
					V.push_back(v);
					N++;
				}
			}
		}
	}

	if (f_v) {
		cout << "surface_domain::list_starter_configurations We found "
			<< N << " starter configurations on this surface" 
			<< endl;
	}

	if (N != V.size()) {
		cout << "surface_domain::list_starter_configurations N != V.size()" << endl;
		exit(1);
	}

	Table = NEW_int(N * 2);
	int i;

	for (i = 0; i < V.size(); i++) {
		Table[i * 2 + 0] = V[i][0];
		Table[i * 2 + 1] = V[i][1];
	}
#if 0
	N1 = 0;
	for (i = 0; i < nb_lines; i++) {
		if (line_intersections->Set_size[i] < 5) {
			continue;
		}
		nCk = Combi.int_n_choose_k(line_intersections->Set_size[i], 5);
		for (j = 0; j < nCk; j++) {
			Combi.unrank_k_subset(j, subset,
				line_intersections->Set_size[i], 5);
			for (h = 0; h < 5; h++) {
				subset2[h] = 
				line_intersections->Sets[i][subset[h]];
				S6[h] = Lines[subset2[h]];
			}
			S6[5] = Lines[i];
			r = rank_of_system(6, S6, 0 /*verbose_level*/);
			if (r == 19) {
				Table[N1 * 2 + 0] = i;
				Table[N1 * 2 + 1] = j;
				N1++;
			}
		}
	}
	if (N1 != N) {
		cout << "N1 != N" << endl;
		exit(1);
	}
#endif
	if (f_v) {
		cout << "surface_domain::list_starter_configurations done" << endl;
	}
}

void surface_domain::create_starter_configuration(
	int line_idx, int subset_idx, 
	data_structures::set_of_sets *line_neighbors, long int *Lines, long int *S,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int subset[5];
	int subset2[5];
	int h; //, nCk;
	combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "surface_domain::create_starter_configuration" << endl;
	}
	//nCk = int_n_choose_k(line_neighbors->Set_size[line_idx], 5);
	Combi.unrank_k_subset(subset_idx, subset,
		line_neighbors->Set_size[line_idx], 5);
	for (h = 0; h < 5; h++) {
		subset2[h] = line_neighbors->Sets[line_idx][subset[h]];
		S[h] = Lines[subset2[h]];
	}
	S[5] = Lines[line_idx];
	if (f_v) {
		cout << "surface_domain::create_starter_configuration done" << endl;
	}
}

void surface_domain::wedge_to_klein(int *W, int *K)
{
	F->wedge_to_klein(W, K);
#if 0
	K[0] = W[0];
	K[1] = W[5];
	K[2] = W[1];
	K[3] = F->negate(W[4]);
	K[4] = W[2];
	K[5] = W[3];
#endif
}

void surface_domain::klein_to_wedge(int *K, int *W)
{
	F->klein_to_wedge(K, W);
#if 0
	W[0] = K[0];
	W[1] = K[2];
	W[2] = K[4];
	W[3] = K[5];
	W[4] = F->negate(K[3]);
	W[5] = K[1];
#endif
}

long int surface_domain::line_to_wedge(long int line_rk)
{
	long int a, b;
	
	a = Klein->line_to_point_on_quadric(line_rk, 0 /* verbose_level*/);
	O->Hyperbolic_pair->unrank_point(w2, 1, a, 0 /* verbose_level*/);
	klein_to_wedge(w2, v2);
	F->PG_element_rank_modified_lint(v2, 1, 6 /*wedge_dimension*/, b);
	//b = AW->rank_point(v);
	return b;
}

void surface_domain::line_to_wedge_vec(
		long int *Line_rk, long int *Wedge_rk, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		Wedge_rk[i] = line_to_wedge(Line_rk[i]);
	}
}

void surface_domain::line_to_klein_vec(
		long int *Line_rk, long int *Klein_rk, int len)
{
	//int_vec_apply(Line_rk, Klein->Line_to_point_on_quadric,
	//		Klein_rk, len);
	//from through to
	//for (i = 0; i < len; i++) {
	//	to[i] = through[from[i]];
	//	}
	int i;

	for (i = 0; i < len; i++) {
		Klein_rk[i] = Klein->line_to_point_on_quadric(Line_rk[i], 0 /* verbose_level*/);
	}
}

long int surface_domain::klein_to_wedge(long int klein_rk)
{
	long int b;
	
	O->Hyperbolic_pair->unrank_point(w2, 1, klein_rk, 0 /* verbose_level*/);
	klein_to_wedge(w2, v2);
	F->PG_element_rank_modified_lint(v2, 1, 6 /*wedge_dimension*/, b);
	//b = AW->rank_point(v);
	return b;
}

void surface_domain::klein_to_wedge_vec(
		long int *Klein_rk, long int *Wedge_rk, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		Wedge_rk[i] = klein_to_wedge(Klein_rk[i]);
	}
}

void surface_domain::save_lines_in_three_kinds(std::string &fname_csv,
	long int *Lines_wedge, long int *Lines, long int *Lines_klein, int nb_lines)
{
	data_structures::spreadsheet *Sp;
	
	make_spreadsheet_of_lines_in_three_kinds(Sp, 
		Lines_wedge, Lines, Lines_klein, nb_lines,
		0 /* verbose_level */);

	Sp->save(fname_csv, 0 /*verbose_level*/);
	FREE_OBJECT(Sp);
}


int surface_domain::build_surface_from_double_six_and_count_Eckardt_points(long int *double_six, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points" << endl;
	}
	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points before Surf->build_cubic_surface_from_lines" << endl;
	}

	int coeffs20[20];
	long int Lines27[27];
	int nb_E;

	build_cubic_surface_from_lines(
		12, double_six,
		coeffs20, 0/* verbose_level*/);

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points after Surf->build_cubic_surface_from_lines" << endl;
	}

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points "
				"coeffs20:" << endl;
		Int_vec_print(cout, coeffs20, 20);
		cout << endl;

		Poly3_4->print_equation(cout, coeffs20);
		cout << endl;
	}


	Lint_vec_copy(double_six, Lines27, 12);


	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points before Surf->create_the_fifteen_other_lines" << endl;
	}
	create_the_fifteen_other_lines(Lines27,
			Lines27 + 12, verbose_level);
	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points after Surf->create_the_fifteen_other_lines" << endl;
	}



	surface_object *SO;

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points before SO->init_with_27_lines" << endl;
	}

	SO->init_with_27_lines(this,
		Lines27, coeffs20,
		FALSE /* f_find_double_six_and_rearrange_lines */,
		verbose_level);
	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points after SO->init_with_27_lines" << endl;
	}

	nb_E = SO->SOP->nb_Eckardt_points;
	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points the surface has " << nb_E << " Eckardt points" << endl;
	}

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points deleting SO" << endl;
	}
	FREE_OBJECT(SO);

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points done" << endl;
	}
	return nb_E;

}



}}}

