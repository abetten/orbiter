// eckardt_point_info.cpp
//
// Anton Betten
// October 20, 2018

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace algebraic_geometry {


static void intersection_matrix_entry_print(
		int *p,
	int m, int n, int i, int j, int val,
	std::string &output, void *data);

eckardt_point_info::eckardt_point_info()
{
	Record_birth();
	P2 = NULL;
	//long int arc6[6];
	bisecants = NULL;
	Intersections = NULL;
	B_pts = NULL;
	B_pts_label = NULL;
	nb_B_pts = 0;
	E2 = NULL;
	nb_E2 = 0;
	conic_coefficients = NULL;
	E = NULL;
	nb_E = 0;
}


eckardt_point_info::~eckardt_point_info()
{
	Record_death();
	if (bisecants) {
		FREE_int(bisecants);
		}
	if (Intersections) {
		FREE_int(Intersections);
		}
	if (B_pts) {
		FREE_int(B_pts);
		}
	if (B_pts_label) {
		FREE_int(B_pts_label);
		}
	if (E2) {
		FREE_int(E2);
		}
	if (conic_coefficients) {
		FREE_int(conic_coefficients);
		}

	if (E) {
		FREE_OBJECTS(E);
		}
}

void eckardt_point_info::init(
		geometry::projective_geometry::projective_space *P2,
		long int *arc6, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, h, pi, pj, bi, bj, p;
	int multiplicity = 6;
	int t, f, l, s, u;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	long int arc5[5];
	int *H1; // [6]
	int *H; // [12]


	if (f_v) {
		cout << "eckardt_point_info::init" << endl;
		}
	//eckardt_point_info::Surf = Surf;
	eckardt_point_info::P2 = P2;
	Lint_vec_copy(arc6, eckardt_point_info::arc6, 6);

	if (P2->Subspaces->n != 2) {
		cout << "eckardt_point_info::init "
				"P->Subspaces->n != 2" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "arc: ";
		Lint_vec_print(cout, arc6, 6);
		cout << endl;
	}

	if (f_v) {
		cout << "eckardt_point_info::init "
				"computing E_{ij,kl,mn}:" << endl;
	}



	// bisecants
	bisecants = NEW_int(15);
	h = 0;
	for (i = 0; i < 6; i++) {
		pi = arc6[i];
		for (j = i + 1; j < 6; j++, h++) {
			pj = arc6[j];
			bisecants[h] = P2->Subspaces->line_through_two_points(pi, pj);
		}
	}
	if (f_v) {
		cout << "eckardt_point_info::init bisecants: ";
		Int_vec_print(cout, bisecants, 15);
		cout << endl;
	}
	Intersections = NEW_int(15 * 15);
	for (i = 0; i < 15; i++) {
		bi = bisecants[i];
		for (j = 0; j < 15; j++) {
			bj = bisecants[j];
			if (i == j) {
				p = -1;
			}
			else {
				p = P2->Subspaces->intersection_of_two_lines(bi, bj);
			}
			Intersections[i * 15 + j] = p;
		}
	}
	//int_matrix_print(Intersections, 15, 15);


	other::data_structures::tally C;
	C.init(Intersections, 15 * 15, false, 0);
	C.get_data_by_multiplicity(B_pts, nb_B_pts,
		multiplicity, 0 /* verbose_level */);

	if (f_vv) {
		cout << "eckardt_point_info::init "
				"We found " << nb_B_pts << " B-pts: ";
		Int_vec_print(cout, B_pts, nb_B_pts);
		cout << endl;
	}

	B_pts_label = NEW_int(nb_B_pts * 3);
	H1 = NEW_int(6);
	H = NEW_int(12);

	s = 0;
	for (t = 0; t < C.nb_types; t++) {
		f = C.type_first[t];
		l = C.type_len[t];
		if (l == multiplicity) {
			if (B_pts[s] != C.data_sorted[f]) {
				cout << "Pts[s] != C.data_sorted[f]" << endl;
				exit(1);
			}
			//len = 0;
			for (u = 0; u < l; u++) {
				h = C.sorting_perm_inv[f + u];
				H1[u] = h;
			}

			if (f_vv) {
				cout << "eckardt_point_info::init H1=";
				Int_vec_print(cout, H1, 6);
				cout << endl;
			}
			for (u = 0; u < 6; u++) {
				h = H1[u];
				H[2 * u + 0] = h % 15;
				H[2 * u + 1] = h / 15;
				}
			if (f_vv) {
				cout << "eckardt_point_info::init H=";
				Int_vec_print(cout, H, 12);
				cout << endl;
			}

			other::data_structures::tally C2;
			int *Labels;
			int nb_labels;

			C2.init(H, 12, false, 0);
			C2.get_data_by_multiplicity(
				Labels, nb_labels,
				4 /*multiplicity*/,
				0 /* verbose_level */);

			if (f_vv) {
				cout << "eckardt_point_info::init "
					"We found " << nb_labels << " labels: ";
				Int_vec_print(cout, Labels, nb_labels);
				cout << endl;
			}

			if (nb_labels != 3) {
				cout << "nb_labels != 3" << endl;
				exit(1);
				}
			Int_vec_copy(Labels, B_pts_label + 3 * s, 3);

			FREE_int(Labels);
			s++;
		} // if
	} // next t

	//int_matrix_print(B_pts_label, nb_B_pts, 3);

	if (f_v) {
		cout << "eckardt_point_info::init "
			"We found " << nb_B_pts << " Eckardt points of the Brianchon type:" << endl;
		for (s = 0; s < nb_B_pts; s++) {
			cout << "E_{";
			for (l = 0; l < 3; l++) {
				h = B_pts_label[s * 3 + l];
				Combi.k2ij(h, i, j, 6);
				cout << i + 1 << j + 1;
				if (l < 2) {
					cout << ",";
				}
			}
			cout << "} B-pt=" << B_pts[s] << endl;
		}
	}

	if (f_v) {
		cout << "eckardt_point_info::init computing Eckardt "
				"points of the second type E_ij:" << endl;
	}


	E2 = NEW_int(6 * 5 * 2);
	conic_coefficients = NEW_int(6 * 6);

	for (j = 0; j < 6; j++) {

		if (f_v) {
			cout << "eckardt_point_info::init j=" << j << " / 6" << endl;
		}

		int deleted_point, rk, i1;
		int *six_coeffs;
		long int tangents[5];
		int Basis[9];

		six_coeffs = conic_coefficients + j * 6;

		deleted_point = arc6[j];
		Lint_vec_copy(arc6, arc5, j);
		Lint_vec_copy(arc6 + j + 1, arc5 + j, 5 - j);

#if 0
		cout << "deleting point " << j << " / 6:";
		int_vec_print(cout, arc5, 5);
		cout << endl;
#endif
		if (f_v) {
			cout << "eckardt_point_info::init j=" << j << " / 6 arc5 : ";
			Lint_vec_print(cout, arc5, 5);
			cout << endl;
		}

		if (f_v) {
			cout << "eckardt_point_info::init j=" << j << " / 6 "
					"before P2->determine_conic_in_plane" << endl;
		}
		P2->Plane->determine_conic_in_plane(arc5, 5,
			six_coeffs, verbose_level);
		if (f_v) {
			cout << "eckardt_point_info::init j=" << j << " / 6 "
					"after P2->determine_conic_in_plane" << endl;
		}

		P2->Subspaces->F->Projective_space_basic->PG_element_normalize_from_front(six_coeffs, 1, 6);

		if (f_v) {
			cout << "eckardt_point_info::init j=" << j << " / 6 "
					"coefficients of the conic : ";
			Int_vec_print(cout, six_coeffs, 6);
			cout << endl;
		}

		if (f_v) {
			cout << "eckardt_point_info::init j=" << j << " / 6 "
					"before P2->find_tangent_lines_to_conic" << endl;
		}
		P2->Plane->find_tangent_lines_to_conic(six_coeffs,
			arc5, 5,
			tangents, verbose_level);
		if (f_v) {
			cout << "eckardt_point_info::init j=" << j << " / 6 "
					"after P2->find_tangent_lines_to_conic" << endl;
		}

		for (i = 0; i < 5; i++) {
			if (f_v) {
				cout << "eckardt_point_info::init j=" << j << " / 6 "
						"i=" << i << " / 5 tangents[i] = " << tangents[i] << endl;
			}

			P2->unrank_line(Basis, tangents[i]);

#if 0
			cout << "The tangent line at " << arc5[i] << " is:" << endl;
			int_matrix_print(Basis, 2, 3);
#endif

			P2->unrank_point(Basis + 6, deleted_point);
			rk = P2->Subspaces->F->Linear_algebra->Gauss_easy(Basis, 3, 3);
			if (rk == 2) {
				if (i >= j) {
					i1 = i + 1;
				}
				else {
					i1 = i;
				}
				if (f_v) {
					cout << "eckardt_point_info::init Found Eckardt point "
							"E_{" << i1 + 1 << j + 1 << "}" << endl;
				}
				E2[nb_E2 * 2 + 0] = i1;
				E2[nb_E2 * 2 + 1] = j;
				nb_E2++;
			}
		} // next i
	} // next j
	if (f_v) {
		cout << "eckardt_point_info::init We found " << nb_E2
				<< " Eckardt points of the second type" << endl;
		}

	nb_E = nb_B_pts + nb_E2;
	E = NEW_OBJECTS(eckardt_point, nb_E);
	for (i = 0; i < nb_B_pts; i++) {
		E[i].len = 3;
		E[i].pt = B_pts[i];
		Int_vec_copy(B_pts_label + i * 3, E[i].index, 3);
		}
	for (i = 0; i < nb_E2; i++) {
		E[nb_B_pts + i].len = 2;
		E[nb_B_pts + i].pt = -1;
		E[nb_B_pts + i].index[0] = E2[i * 2 + 0];
		E[nb_B_pts + i].index[1] = E2[i * 2 + 1];
		E[nb_B_pts + i].index[2] = -1;
		}


	FREE_int(H1);
	FREE_int(H);
	if (f_v) {
		cout << "eckardt_point_info::init done" << endl;
		}
}


void eckardt_point_info::print_bisecants(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h, a;
	int Mtx[9];
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "eckardt_point_info::print_bisecants" << endl;
	}


	algebra::ring_theory::homogeneous_polynomial_domain *Poly1;


	Poly1 = NEW_OBJECT(algebra::ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "eckardt_point_info::print_bisecants "
				"before Poly1->init" << endl;
	}
	Poly1->init(P2->Subspaces->F,
			3 /* nb_vars */, 1 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "eckardt_point_info::print_bisecants "
				"after Poly1->init" << endl;
	}


	ost << "The 15 bisecants are:\\\\" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|r|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "h & P_iP_j & \\mbox{rank} & \\mbox{line} "
			"& \\mbox{equation}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (h = 0; h < 15; h++) {
		a = bisecants[h];
		Combi.k2ij(h, i, j, 6);
		ost << h << " & P_{" << i + 1 << "}P_{" << j + 1
				<< "} & " << a << " & " << endl;
		//ost << "\\left[ " << endl;
		P2->Subspaces->Grass_lines->print_single_generator_matrix_tex(ost, a);
		//ost << "\\right] ";

		P2->Subspaces->Grass_lines->unrank_lint_here_and_compute_perp(Mtx, a,
			0 /*verbose_level */);
		P2->Subspaces->F->Projective_space_basic->PG_element_normalize(Mtx + 6, 1, 3);

		ost << " & ";
		Poly1->print_equation(ost, Mtx + 6); // ToDo
		ost << "\\\\" << endl;
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;

	FREE_OBJECT(Poly1);

	if (f_v) {
		cout << "eckardt_point_info::print_bisecants done" << endl;
	}
}

void eckardt_point_info::print_intersections(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::l1_interfaces::latex_interface L;
	int labels[15];
	int fst[1];
	int len[1];
	fst[0] = 0;
	len[0] = 15;
	int i;

	if (f_v) {
		cout << "eckardt_point_info::print_intersections" << endl;
	}
	ost << "The intersections of bisecants are:\\\\" << endl;
	for (i = 0; i < 15; i++) {
		labels[i] = i;
	}
	ost << "{\\small \\arraycolsep=1pt" << endl;
	ost << "$$" << endl;
	L.int_matrix_print_with_labels_and_partition(ost,
		Intersections, 15, 15,
		labels, labels,
		fst, len, 1,
		fst, len, 1,
		intersection_matrix_entry_print, (void *) this,
		true /* f_tex */);
	ost << "$$}" << endl;
	if (f_v) {
		cout << "eckardt_point_info::print_intersections done" << endl;
	}
}

void eckardt_point_info::print_conics(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h;
	algebra::ring_theory::homogeneous_polynomial_domain *Poly2;

	if (f_v) {
		cout << "eckardt_point_info::print_conics" << endl;
	}


	Poly2 = NEW_OBJECT(algebra::ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "eckardt_point_info::print_conics "
				"before Poly2->init" << endl;
	}
	Poly2->init(P2->Subspaces->F,
			3 /* nb_vars */, 2 /* degree */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "eckardt_point_info::print_conics "
				"after Poly2->init" << endl;
	}



	ost << "The 6 conics are:\\\\" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "i & C_i & \\mbox{equation}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (h = 0; h < 6; h++) {
		ost << h + 1 << " & C_" << h + 1 << " & " << endl;

		Poly2->print_equation(ost,
				conic_coefficients + h * 6);

		ost << "\\\\" << endl;
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;


	FREE_OBJECT(Poly2);

	if (f_v) {
		cout << "eckardt_point_info::print_conics done" << endl;
	}
}

void eckardt_point_info::print_Eckardt_points(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int s;

	if (f_v) {
		cout << "eckardt_point_info::print_Eckardt_points" << endl;
	}
	ost << "We found " << nb_E << " Eckardt points:\\\\" << endl;
	for (s = 0; s < nb_E; s++) {
		ost << s << " / " << nb_E << " : $";
		E[s].latex(ost);
		ost << "= E_{" << E[s].rank() << "}$\\\\" << endl;
	}
	//ost << "by rank: ";
	//int_vec_print(ost, E_idx, nb_E);
	//ost << "\\\\" << endl;
	if (f_v) {
		cout << "eckardt_point_info::print_Eckardt_points done" << endl;
	}
}


static void intersection_matrix_entry_print(
		int *p,
	int m, int n, int i, int j, int val,
	std::string &output, void *data)
{
	//eckardt_point_info *E;
	//E = (eckardt_point_info *) data;
	int a, b;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (i == -1) {
		Combi.k2ij(j, a, b, 6);
		output = "P_" + std::to_string(a + 1) + "P_" + std::to_string(b + 1);
	}
	else if (j == -1) {
		Combi.k2ij(i, a, b, 6);
		output = "P_" + std::to_string(a + 1) + "P_" + std::to_string(b + 1);
	}
	else {
		if (val == -1) {
			output = ".";
		}
		else {
			output = std::to_string(val);
		}
	}
}



}}}}



