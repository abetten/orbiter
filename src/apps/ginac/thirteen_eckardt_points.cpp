/*
 * thirteen_eckardt_points.cpp
 *
 *  Created on: Jan 3, 2020
 *      Author: betten
 */





#include "orbiter.h"
// Include Orbiter definitions


using namespace std;
using namespace orbiter;
using namespace orbiter::top_level;
// use orbiter's namespaces


// We rely on a package called ginac.
// Ginac is a C++ package for computer algebra
// despite the misleading acronym "Ginac is not a computer algebra system"

#include "ginac/ginac.h"

using namespace GiNaC;
// use ginac's namespace

#include <iostream>
// standard C++ stuff
using namespace std;
// use namespace std which countains things like cout


#include "ginac_linear_algebra.cpp"


void thirteen_Eckardt_points(int argc, const char **argv);


void thirteen_Eckardt_points(int argc, const char **argv)
{
	int verbose_level = 0;

	int i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else {
			cout << "unrecognized option " << argv[i] << endl;
		}
	}



	char fname[1000];
	char title[1000];
	const char *author = "Orbiter";
	const char *extras_for_preamble = "";

	vsnprintf(fname, 1000, "13_report.tex", 0);
	vsnprintf(title, 1000, "Thirteen Eckardt Points", 0);

	{
		ofstream fp(fname);
		latex_interface L;
		//latex_head_easy(fp);
		L.head(fp,
			FALSE /* f_book */, TRUE /* f_title */,
			title, author,
			FALSE /*f_toc*/, FALSE /* f_landscape*/, FALSE /* f_12pt*/,
			TRUE /*f_enlarged_page*/, TRUE /* f_pagenumbers*/,
			extras_for_preamble);

		//LG->report(fp, f_sylow, f_group_table, verbose_level);



		// create symbolic variables for u, v
		// Later, u and v can be specified.
		// For instance, u=1 and v=2 leads to the standard form of the
		// Hilbert Cohn-Vossen surface

		symbol a("a"); //', b("b"), c("c"), d("d");

		// a matrix for 20 points:
		matrix V(6,3);

		// the 5 points for the conic:
		matrix P1(1,3);
		matrix P2(1,3);
		matrix P3(1,3);
		matrix P4(1,3);
		matrix P5(1,3);
		matrix Pi(1,3);
		matrix Pj(1,3);
		matrix *P[5];

		// we create a matrix of point coordinates:
		V = matrix{
			{1,0,0},
			{0,1,0},
			{0,0,1},
			{1,1,1},
			{a,a+1,1},
			{a+1,a,1},
		};
		fp << "$$V=" << endl << latex << V << endl << "$$" << endl;

		int conic_idx[] = {
				1,2,3,4,5,
				0,2,3,4,5,
				0,1,3,4,5,
				0,1,2,4,5,
				0,1,2,3,5,
				0,1,2,3,4,
		};
		int Pj_idx[] = {
				0,1,2,3,4,5
		};

		int j, r, col;
		int *idx;


		for (j = 0; j < 6; j++) {

			idx = conic_idx + j * 5;
			P1 = matrix{{V(idx[0], 0), V(idx[0], 1), V(idx[0], 2)}};
			fp << "$P_" << idx[0] << "=" << P1 << "$\\\\" << endl;
			P2 = matrix{{V(idx[1], 0), V(idx[1], 1), V(idx[1], 2)}};
			fp << "$P_" << idx[1] << "=" << P2 << "$\\\\" << endl;
			P3 = matrix{{V(idx[2], 0), V(idx[2], 1), V(idx[2], 2)}};
			fp << "$P_" << idx[2] << "=" << P3 << "$\\\\" << endl;
			P4 = matrix{{V(idx[3], 0), V(idx[3], 1), V(idx[3], 2)}};
			fp << "$P_" << idx[3] << "=" << P4 << "$\\\\" << endl;
			P5 = matrix{{V(idx[4], 0), V(idx[4], 1), V(idx[4], 2)}};
			fp << "$P_" << idx[4] << "=" << P5 << "$\\\\" << endl;


			Pj = matrix{{V(j, 0), V(j, 1), V(j, 2)}};
			fp << "$P_" << j << "=" << Pj << "$\\\\" << endl;

			// make an array of pointers so that we can access the 19 points
			// though this indexed array in loops later on:

			P[0] = &P1;
			P[1] = &P2;
			P[2] = &P3;
			P[3] = &P4;
			P[4] = &P5;

			// create the homogeneous component of degree 2
			// in the polynomial ring F_2[X0,X1,X2]:
			// We don't need the fact that it is over F_2,
			// but we have to pick a finite field in order to create the ring
			// in Orbiter:

			homogeneous_polynomial_domain *HPD;
			finite_field *F;

			// create the finite field F_2:

			F = NEW_OBJECT(finite_field);
			F->init(2);



				// create the homogeneous polynomial ring
			// of degree 3 in X0,X1,X2,X3:
			// Orbiter will create the 6 monomials

			HPD = NEW_OBJECT(homogeneous_polynomial_domain);
			HPD->init(F, 3 /* nb_vars */, 2 /* degree */,
					FALSE /* f_init_incidence_structure */, verbose_level);

			//print the monomials in Orbiter ordering:
			for (col = 0; col < HPD->nb_monomials; col++) {
				cout << col << " : " << HPD->monomial_symbols[col] << endl;
			}

			// Create the 6 monomials as objects in ginac:
			// We use Orbiter created strings which represent each monomial
			ex M[6];
			parser reader;

			for (col = 0; col < HPD->nb_monomials; col++) {
				M[col] = reader(HPD->monomial_symbols[col]);
				ostringstream s;
				s << latex << M[col];
				cout << "M[" << col << "]=" << s.str() << endl;
			}

			// During the previous loop, ginac has created 3 symbolic variables.
			// We just need to identify them in the symbol table:

			symtab table = reader.get_syms();
			symbol X0 = ex_to<symbol>(table["X0"]);
			symbol X1 = ex_to<symbol>(table["X1"]);
			symbol X2 = ex_to<symbol>(table["X2"]);


			// Create the coefficient matrix which forces
			// the surface to pass through the 19 points:
			matrix S(5, 6);
			for (col = 0; col < HPD->nb_monomials; col++) {
				for (r = 0; r < 5; r++) {

					// the (r,col)-entry of S
					// is the col-th monomial evaluated at the r-th point P[r].

					ex e = M[col].subs(lst{X0, X1, X2},
							lst{(*P[r])(0, 0), (*P[r])(0, 1), (*P[r])(0, 2)});

					// produce latex output:
					ostringstream s;
					s << latex << e;
					cout << "M[" << r << "," << col << "]=" << s.str() << endl;

					S(r, col) = e;
				}
			}


			// typeset the coefficient matrix in latex:
			//fp << "\\begin{sidewaysfigure}" << endl;
			//fp << "{\\small \\arraycolsep=1pt" << endl;
			fp << "$$" << endl << S << endl << "$$" << endl;
			//fp << "}" << endl;
			//fp << "\\end{sidewaysfigure}" << endl;





			// create 6 variables in ginac representing the monomials in the ring:
			// We are using labels such as Xij,
			// which stands for the monomial Xi*Xj
			// We have Orbiter create these labels for us
			// and store them in HPD->monomial_symbols_easy

			symbol m0(HPD->monomial_symbols_easy[0]);
			symbol m1(HPD->monomial_symbols_easy[1]);
			symbol m2(HPD->monomial_symbols_easy[2]);
			symbol m3(HPD->monomial_symbols_easy[3]);
			symbol m4(HPD->monomial_symbols_easy[4]);
			symbol m5(HPD->monomial_symbols_easy[5]);

			lst eqns, vars;
			ex sol;

			// Create the linear system in ginac form:
			// Create the linear system [a*x+b*y==3,x-y==b]...
			for (r = 0; r < 5; r++) {

				// Create one equation for each point.
				// Here, the equation for the r-th point.
				// The right hand side is 0:

				eqns.append(S(r,0)*m0+S(r,1)*m1+S(r,2)*m2+S(r,3)*m3+S(r,4)*m4
						+S(r,5)*m5
						==0);
			}

			// specify the list of variables for which we want to solve:

			vars.append(m0).append(m1).append(m2).append(m3).append(m4);
			vars.append(m5);

			// Solve it:
			cout << "solving " << eqns << " for " << vars << endl;
			sol = my_lsolve(fp, eqns, vars);

			fp << "\\clearpage" << endl;

			// Print the solution (there should be only one):
			fp << "solution " << endl << "$$" << endl << sol << endl << "$$" << endl;

			int f_first;
			double Eqn[6];

			f_first = TRUE;
			fp << "The equation of the conic $C_" << j << "$ is" << endl;
			fp << "$$" << endl;
			for (col = 0; col < 6; col++) {
				ex solc = sol.op(col);  // solution for col-th variable

				Eqn[col] = 0;

				if (!solc.is_zero()) {
					numeric value = ex_to<numeric>(evalf(solc));

					Eqn[col] = value.to_double();

					if (f_first) {
						f_first = FALSE;
					}
					else {
						fp << " + ";
					}
					fp << solc << HPD->monomial_symbols_latex[col];
				}
			}
			fp << " = 0" << endl;
			fp << "$$" << endl;
			cout << endl;

			for (i = 0; i < 6; i++) {
				if (i == j) {
					continue;
				}

				Pi = matrix{{V(i, 0), V(i, 1), V(i, 2)}};
				fp << "$P_" << i << "=" << Pi << "$\\\\" << endl;

			} // next i

		} // next j


		L.foot(fp);
	}

}

int main(int argc, const char **argv)
{

	thirteen_Eckardt_points(argc, argv);

}


