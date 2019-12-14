/*
 * hilbert_cohn_vossen.cpp
 *
 *  Created on: Dec 8, 2019
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


// local functions copied from ginac:

ex my_lsolve(ostream &ost, const ex &eqns, const ex &symbols, unsigned options);
matrix solve(matrix *M, const matrix & vars,
                     const matrix & rhs,
                     unsigned algo);
matrix right_nullspace(matrix *M);
int pivot(matrix *M, unsigned ro, unsigned co, bool symbolic);

// the function which computes the equation of the Hilbert, Cohn-Vossen surface:
void surface();


// ginac stuff:
class my_symbolset {
	exset s;
	void insert_symbols(const ex &e)
	{
		if (is_a<symbol>(e)) {
			s.insert(e);
		} else {
			for (const ex &sube : e) {
				insert_symbols(sube);
			}
		}
	}
public:
	explicit my_symbolset(const ex &e)
	{
		insert_symbols(e);
	}
	bool has(const ex &e) const
	{
		return s.find(e) != s.end();
	}
};



ex my_lsolve(ostream &ost, const ex &eqns, const ex &symbols, unsigned options = solve_algo::gauss)
{
	// solve a system of linear equations
	if (eqns.info(info_flags::relation_equal)) {
		if (!symbols.info(info_flags::symbol))
			throw(std::invalid_argument("lsolve(): 2nd argument must be a symbol"));
		const ex sol = lsolve(lst{eqns}, lst{symbols});

		GINAC_ASSERT(sol.nops()==1);
		GINAC_ASSERT(is_exactly_a<relational>(sol.op(0)));

		return sol.op(0).op(1); // return rhs of first solution
	}

	// syntax checks
	if (!(eqns.info(info_flags::list) || eqns.info(info_flags::exprseq))) {
		throw(std::invalid_argument("lsolve(): 1st argument must be a list, a sequence, or an equation"));
	}
	for (size_t i=0; i<eqns.nops(); i++) {
		if (!eqns.op(i).info(info_flags::relation_equal)) {
			throw(std::invalid_argument("lsolve(): 1st argument must be a list of equations"));
		}
	}
	if (!(symbols.info(info_flags::list) || symbols.info(info_flags::exprseq))) {
		throw(std::invalid_argument("lsolve(): 2nd argument must be a list, a sequence, or a symbol"));
	}
	for (size_t i=0; i<symbols.nops(); i++) {
		if (!symbols.op(i).info(info_flags::symbol)) {
			throw(std::invalid_argument("lsolve(): 2nd argument must be a list or a sequence of symbols"));
		}
	}

	// build matrix from equation system
	matrix sys(eqns.nops(),symbols.nops());
	matrix rhs(eqns.nops(),1);
	matrix vars(symbols.nops(),1);

	for (size_t r=0; r<eqns.nops(); r++) {
		const ex eq = eqns.op(r).op(0)-eqns.op(r).op(1); // lhs-rhs==0
		const my_symbolset syms(eq);
		ex linpart = eq;
		for (size_t c=0; c<symbols.nops(); c++) {
			if (!syms.has(symbols.op(c)))
				continue;
			const ex co = eq.coeff(ex_to<symbol>(symbols.op(c)),1);
			linpart -= co*symbols.op(c);
			sys(r,c) = co;
		}
		linpart = linpart.expand();
		rhs(r,0) = -linpart;
	}

#if 0
	ost << "\\begin{sidewaysfigure}" << endl;
	ost << "$$" << endl << sys << endl << "$$" << endl;
	ost << "\\end{sidewaysfigure}" << endl;
	ost << "\\begin{figure}" << endl;
	ost << "$$" << endl << rhs << endl << "$$" << endl;
	ost << "\\end{figure}" << endl;
#endif


	// test if system is linear and fill vars matrix
	const my_symbolset sys_syms(sys);
	const my_symbolset rhs_syms(rhs);
	for (size_t i=0; i<symbols.nops(); i++) {
		vars(i,0) = symbols.op(i);
		if (sys_syms.has(symbols.op(i)))
			throw(std::logic_error("lsolve: system is not linear"));
		if (rhs_syms.has(symbols.op(i)))
			throw(std::logic_error("lsolve: system is not linear"));
	}

	matrix solution;
	try {
		solution = solve(&sys, vars,rhs,options);
	} catch (const std::runtime_error & e) {
		// Probably singular matrix or otherwise overdetermined system:
		// It is consistent to return an empty list
		return lst{};
	}
	GINAC_ASSERT(solution.cols()==1);
	GINAC_ASSERT(solution.rows()==symbols.nops());

#if 0
	// return list of equations of the form lst{var1==sol1,var2==sol2,...}
	lst sollist;
	for (size_t i=0; i<symbols.nops(); i++)
		sollist.append(symbols.op(i)==solution(i,0));

	return sollist;
#endif
	return solution;
}

matrix solve(matrix *M, const matrix & vars,
                     const matrix & rhs,
                     unsigned algo)
{
	const unsigned m = M->rows();
	const unsigned n = M->cols();
	const unsigned p = rhs.cols();

	cout << "solve M=" << endl;
	//cout << *M << endl;

	cout << "solve rhs=" << endl;
	//cout << rhs << endl;

	// syntax checks
	if ((rhs.rows() != m) || (vars.rows() != n) || (vars.cols() != p))
		throw (std::logic_error("matrix::solve(): incompatible matrices"));
	for (unsigned ro=0; ro<n; ++ro)
		for (unsigned co=0; co<p; ++co)
			if (!vars(ro,co).info(info_flags::symbol))
				throw (std::invalid_argument("matrix::solve(): 1st argument must be matrix of symbols"));

#if 0
	// build the augmented matrix of *this with rhs attached to the right
	matrix aug(m,n+p);
	for (unsigned r=0; r<m; ++r) {
		for (unsigned c=0; c<n; ++c)
			aug.m[r*(n+p)+c] = M->m[r*n+c];
		for (unsigned c=0; c<p; ++c)
			aug.m[r*(n+p)+c+n] = rhs.m[r*p+c];
	}
#endif

	//cout << "solve before gauss_elimination, M=" << endl;
	//cout << *M << endl;


	// Eliminate the augmented matrix:
	//auto colid = aug.echelon_form(algo, n);

	matrix K;

	K = right_nullspace(M);

	//cout << "solve after gauss_elimination, M=" << endl;
	//cout << *M << endl;
	cout << "solve after gauss_elimination, K=" << endl;
	cout << K << endl;

	return K;
#if 0
	// assemble the solution matrix:
	matrix sol(n,p);
	for (unsigned co=0; co<p; ++co) {
		unsigned last_assigned_sol = n+1;
		for (int r=m-1; r>=0; --r) {
			unsigned fnz = 1;    // first non-zero in row
			while ((fnz<=n) && (aug.m[r*(n+p)+(fnz-1)].normal().is_zero()))
				++fnz;
			if (fnz>n) {
				// row consists only of zeros, corresponding rhs must be 0, too
				if (!aug.m[r*(n+p)+n+co].normal().is_zero()) {
					throw (std::runtime_error("matrix::solve(): inconsistent linear system"));
				}
			} else {
				// assign solutions for vars between fnz+1 and
				// last_assigned_sol-1: free parameters
				for (unsigned c=fnz; c<last_assigned_sol-1; ++c)
					sol(colid[c],co) = vars.m[colid[c]*p+co];
				ex e = aug.m[r*(n+p)+n+co];
				for (unsigned c=fnz; c<n; ++c)
					e -= aug.m[r*(n+p)+c]*sol.m[colid[c]*p+co];
				sol(colid[fnz-1],co) = (e/(aug.m[r*(n+p)+fnz-1])).normal();
				last_assigned_sol = fnz;
			}
		}
		// assign solutions for vars between 1 and
		// last_assigned_sol-1: free parameters
		for (unsigned ro=0; ro<last_assigned_sol-1; ++ro)
			sol(colid[ro],co) = vars(colid[ro],co);
	}

	return sol;
#endif
}


matrix right_nullspace(matrix *M)
{
	//ensure_if_modifiable();
	const unsigned m = M->rows();
	const unsigned n = M->cols();
	GINAC_ASSERT(!det || n==m);
	int sign = 1;
	int *base_cols;
	int *kernel_cols;

	base_cols = NEW_int(n);
	kernel_cols = NEW_int(n);

	unsigned r0 = 0;
	for (unsigned c0=0; c0<n; ++c0) {


		cout << "right_nullspace, c0=" << c0 << " / " << n << " r0=" << r0 << endl;
		//cout << "right_nullspace, M=" << endl;
		//cout << *M << endl;



		int indx = pivot(M, r0, c0, true);

		cout << "right_nullspace, after pivot, c0=" << c0 << " r0=" << r0 << " indx=" << indx << endl;
		//cout << "right_nullspace, M=" << endl;
		//cout << *M << endl;


		if (indx == -1) {
			cout << "right_nullspace, did not find pivot element" << endl;
			sign = 0;
		}
		if (indx>=0) {

			base_cols[r0] = c0;

			if (indx > 0)
				sign = -sign;
			{
				ex piv = 1 / M->m[r0*n+c0];
				cout << "piv=" << piv << endl;
				for (unsigned c=c0/*+1*/; c<n; ++c) {
					M->m[r0*n+c] *= piv;

				}
			}
			for (unsigned r2=r0+1; r2<m; ++r2) {
				if (!M->m[r2*n+c0].is_zero()) {
					cout << "right_nullspace, cleaning c0=" << c0 << " r0=" << r0 << " r2=" << r2 << endl;
					// yes, there is something to do in this row
					ex piv = M->m[r2*n+c0];
					cout << "piv=" << piv << endl;
					for (unsigned c=c0/*+1*/; c<n; ++c) {
						M->m[r2*n+c] -= piv * M->m[r0*n+c];
					}
				}

			}
			cout << "right_nullspace after cleaning column " << c0 << ", M=" << endl;
			//cout << *M << endl;
			++r0;
		}
	} // next c0

	cout << "right_nullspace, rank = " << r0 << endl;
	cout << "right_nullspace, base_cols=";
	int_vec_print(cout, base_cols, r0);
	cout << endl;
	//cout << "right_nullspace, M=" << endl;
	//cout << *M << endl;
	cout << "right_nullspace, second part" << endl;

	int i, j, c;
	for (i = r0 - 1; i >= 0; i--) {
		c = base_cols[i];
		cout << "i=" << i << " c=" << c << " piv=" << M->m[i*n+c] << endl;
		for (int r2 = i - 1; r2 >= 0; r2--) {
			ex entry = M->m[r2*n+c];
			cout << "cleaning row " << r2 << " entry=" << entry << endl;
			if (!M->m[r2*n+c].is_zero()) {
				for (j = c; j < n; j++) {
					cout << "j=" << j << endl;
					M->m[r2*n+j] -= entry * M->m[i*n+j];
					cout << "done" << endl;
				}
			}
		}
		//cout << "right_nullspace after cleaning column " << c << ", M=" << endl;
		//cout << *M << endl;

	}
	cout << "right_nullspace finished, M=" << endl;
	//cout << *M << endl;

	int_vec_complement(base_cols, kernel_cols, n, r0 /* nb_base_cols */);

	int r, k, ii, b, iii, a;
	int kernel_n, kernel_m;
	r = r0;
	k = n - r;

	cout << "right_nullspace, kernel_cols=";
	int_vec_print(cout, kernel_cols, k);
	cout << endl;

	kernel_m = n;
	kernel_n = k;

	matrix K(kernel_m, kernel_n);


	ii = 0;
	j = 0;
	if (j < r) {
		b = base_cols[j];
		}
	else {
		b = -1;
		}
	for (i = 0; i < n; i++) {
		cout << "i=" << i << " / " << n << endl;
		if (i == b) {
			for (iii = 0; iii < k; iii++) {
				a = kernel_cols[iii];
				K(i, iii) = M->m[j * n + a];
				}
			j++;
			if (j < r) {
				b = base_cols[j];
				}
			else {
				b = -1;
				}
			}
		else {
			for (iii = 0; iii < k; iii++) {
				if (iii == ii) {
					K(i, iii) = -1;
					}
				else {
					K(i, iii) = 0;
					}
				}
			ii++;
			}
		}

	cout << "right_nullspace finished, K=" << endl;
	cout << K << endl;

	FREE_int(base_cols);
	FREE_int(kernel_cols);
#if 0
	// clear remaining rows
	for (unsigned r=r0+1; r<m; ++r) {
		for (unsigned c=0; c<n; ++c)
			M->m[r*n+c] = _ex0;
	}
#endif

	//return sign;
	return K;
}

int pivot(matrix *M, unsigned ro, unsigned co, bool symbolic)
{
	unsigned k = ro;
	if (symbolic) {
		// search first non-zero element in column co beginning at row ro
		while ((k<M->row) && (M->m[k*M->col+co].expand().is_zero()))
			++k;
	} else {
		// search largest element in column co beginning at row ro
		GINAC_ASSERT(is_exactly_a<numeric>(this->m[k*col+co]));
		unsigned kmax = k+1;
		numeric mmax = abs(ex_to<numeric>(M->m[kmax*M->col+co]));
		while (kmax<M->row) {
			GINAC_ASSERT(is_exactly_a<numeric>(M->m[kmax*col+co]));
			numeric tmp = ex_to<numeric>(M->m[kmax*M->col+co]);
			if (abs(tmp) > mmax) {
				mmax = tmp;
				k = kmax;
			}
			++kmax;
		}
		if (!mmax.is_zero())
			k = kmax;
	}
	if (k==M->row)
		// all elements in column co below row ro vanish
		return -1;
	if (k==ro)
		// matrix needs no pivoting
		return 0;
	// matrix needs pivoting, so swap rows k and ro
	//ensure_if_modifiable();
	for (unsigned c=0; c<M->col; ++c)
		M->m[k*M->col+c].swap(M->m[ro*M->col+c]);

	return k;
}

void surface()
// Computes the equation of the Hilbert, Cohn-Vossen surface
{



	char fname[1000];
	char title[1000];
	const char *author = "Orbiter";
	const char *extras_for_preamble = "";

	sprintf(fname, "HCV_report.tex");
	sprintf(title, "The Hilbert, Cohn-Vossen Cubic Surface");

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

		symbol u("u"), v("v");

		// a matrix for 20 points:
		matrix V(20,3);

		// the 19 points on the double six which force the surface:
		matrix P1(1,3);
		matrix P2(1,3);
		matrix P3(1,3);
		matrix P4(1,3);
		matrix P5(1,3);
		matrix P6(1,3);
		matrix P7(1,3);
		matrix P8(1,3);
		matrix P9(1,3);
		matrix P10(1,3);
		matrix P11(1,3);
		matrix P12(1,3);
		matrix P13(1,3);
		matrix P14(1,3);
		matrix P15(1,3);
		matrix P16(1,3);
		matrix P17(1,3);
		matrix P18(1,3);
		matrix P19(1,3);
		matrix *P[19];

		// we create a matrix of point coordinates.
		// the first 8 points are the vertices of the cube,
		// with coordinates u * (\pm 1, \pm 1, \pm 1)
		// The next twelve vertices are the extruded points
		// with three points from every vertex of the tetrahedron
		// chosen from the 8 original vertices:
		// The coordinates  of the extruded vertices have \pm v in them.
		V = matrix{
			{u, u, u},
			{u,u,-u},
			{u,-u,u},
			{u,-u,-u},
			{-u,u,u},
			{-u,u,-u},
			{-u,-u,u},
			{-u,-u,-u},
			{v,u,u},
			{u,v,u},
			{u,u,v},
			{-v,-u,u},
			{-u,-v,u},
			{-u,-u,v},
			{v,-u,-u},
			{u,-v,-u},
			{u,-u,-v},
			{-v,u,-u},
			{-u,v,-u},
			{-u,u,-v},
		};
		//v1.append(u).append(u).append(u);
		//v1.push_back(ex("u",lst{u});
		fp << "$$V=" << endl << latex << V << endl << "$$" << endl;

		// the double six:
		// the 6 red lines, in pairs of two:
		// the numbers are row-indices into the V matrix
		// (plus one because this is pulled from earlier Maple code)
		int red_idx[] = {9, 18, 10, 13, 11, 17, 12, 15, 16, 19, 14, 20};
		// the 6 blue lines, in pairs of two:
		int blue_idx[] = {14, 17, 15, 18, 13, 19, 11, 20, 9, 12, 10, 16};

		int i, j;

		// subtract one so that the indices work for C-code:
		for (i = 0; i < 12; i++) {
			red_idx[i]--;
		}
		for (i = 0; i < 12; i++) {
			blue_idx[i]--;
		}
		int idx, idx1, idx2;

		int a, b;

		// pick four points P1,P2,P3,P4 on a1:
		idx = red_idx[0 * 2 + 0];
		P1 = matrix{{V(idx, 0), V(idx, 1), V(idx, 2)}};
		fp << "$P_1=" << P1 << "$\\\\" << endl;
		idx = red_idx[0 * 2 + 1];
		P2 = matrix{{V(idx, 0), V(idx, 1), V(idx, 2)}};
		fp << "$P_2=" << P2 << "$\\\\" << endl;
		a = -1; b = 2;
		P3 = matrix {{
			a * P1(0,0) + b * P2(0,0),
			a * P1(0,1) + b * P2(0,1),
			a * P1(0,2) + b * P2(0,2)
		}};
		fp << "$P_3=" << P3 << "$\\\\" << endl;
		a = -2; b = 3;
		P4 = matrix {{
			a * P1(0,0) + b * P2(0,0),
			a * P1(0,1) + b * P2(0,1),
			a * P1(0,2) + b * P2(0,2)
		}};
		fp << "$P_4=" << P4 << "$\\\\" << endl;


		// pick three points P5,P6,P7 on b2:
		idx1 = blue_idx[1 * 2 + 1];
		idx2 = blue_idx[1 * 2 + 0];
		a = 0; b = 1;
		P5 = matrix {{
			a * V(idx1,0) + b * V(idx2,0),
			a * V(idx1,1) + b * V(idx2,1),
			a * V(idx1,2) + b * V(idx2,2)
		}};
		fp << "$P_5=" << P5 << "$\\\\" << endl;
		a = -1; b = 2;
		P6 = matrix {{
			a * V(idx1,0) + b * V(idx2,0),
			a * V(idx1,1) + b * V(idx2,1),
			a * V(idx1,2) + b * V(idx2,2)
		}};
		fp << "$P_6=" << P6 << "$\\\\" << endl;
		a = -2; b = 3;
		P7 = matrix {{
			a * V(idx1,0) + b * V(idx2,0),
			a * V(idx1,1) + b * V(idx2,1),
			a * V(idx1,2) + b * V(idx2,2)
		}};
		fp << "$P_7=" << P7 << "$\\\\" << endl;

		// pick three points P8,P9,P10 on b3:
		idx1 = blue_idx[2 * 2 + 0];
		idx2 = blue_idx[2 * 2 + 1];
		a = 1; b = 0;
		P8 = matrix {{
			a * V(idx1,0) + b * V(idx2,0),
			a * V(idx1,1) + b * V(idx2,1),
			a * V(idx1,2) + b * V(idx2,2)
		}};
		fp << "$P_8=" << P8 << "$\\\\" << endl;
		a = 0; b = 1;
		P9 = matrix {{
			a * V(idx1,0) + b * V(idx2,0),
			a * V(idx1,1) + b * V(idx2,1),
			a * V(idx1,2) + b * V(idx2,2)
		}};
		fp << "$P_9=" << P9 << "$\\\\" << endl;
		a = -2; b = 3;
		P10 = matrix {{
			a * V(idx1,0) + b * V(idx2,0),
			a * V(idx1,1) + b * V(idx2,1),
			a * V(idx1,2) + b * V(idx2,2)
		}};
		fp << "$P_{10}=" << P10 << "$\\\\" << endl;

		// pick three points P11,P12,P13 on b4:
		idx1 = blue_idx[3 * 2 + 0];
		idx2 = blue_idx[3 * 2 + 1];
		a = 1; b = 0;
		P11 = matrix {{
			a * V(idx1,0) + b * V(idx2,0),
			a * V(idx1,1) + b * V(idx2,1),
			a * V(idx1,2) + b * V(idx2,2)
		}};
		fp << "$P_{11}=" << P11 << "$\\\\" << endl;
		a = 0; b = 1;
		P12 = matrix {{
			a * V(idx1,0) + b * V(idx2,0),
			a * V(idx1,1) + b * V(idx2,1),
			a * V(idx1,2) + b * V(idx2,2)
		}};
		fp << "$P_{12}=" << P12 << "$\\\\" << endl;
		a = -1; b = 2;
		P13 = matrix {{
			a * V(idx1,0) + b * V(idx2,0),
			a * V(idx1,1) + b * V(idx2,1),
			a * V(idx1,2) + b * V(idx2,2)
		}};
		fp << "$P_{13}=" << P13 << "$\\\\" << endl;

		// pick three points P14,P15,P16 on b5:
		idx1 = blue_idx[4 * 2 + 0];
		idx2 = blue_idx[4 * 2 + 1];
		a = 0; b = 1;
		P14 = matrix {{
			a * V(idx1,0) + b * V(idx2,0),
			a * V(idx1,1) + b * V(idx2,1),
			a * V(idx1,2) + b * V(idx2,2)
		}};
		fp << "$P_{14}=" << P14 << "$\\\\" << endl;
		a = -1; b = 2;
		P15 = matrix {{
			a * V(idx1,0) + b * V(idx2,0),
			a * V(idx1,1) + b * V(idx2,1),
			a * V(idx1,2) + b * V(idx2,2)
		}};
		fp << "$P_{15}=" << P15 << "$\\\\" << endl;
		a = -2; b = 3;
		P16 = matrix {{
			a * V(idx1,0) + b * V(idx2,0),
			a * V(idx1,1) + b * V(idx2,1),
			a * V(idx1,2) + b * V(idx2,2)
		}};
		fp << "$P_{16}=" << P16 << "$\\\\" << endl;

		// pick three points P17,P18,P19 on b6:
		idx1 = blue_idx[5 * 2 + 0];
		idx2 = blue_idx[5 * 2 + 1];
		a = 0; b = 1;
		P17 = matrix {{
			a * V(idx1,0) + b * V(idx2,0),
			a * V(idx1,1) + b * V(idx2,1),
			a * V(idx1,2) + b * V(idx2,2)
		}};
		fp << "$P_{17}=" << P17 << "$\\\\" << endl;
		a = -1; b = 2;
		P18 = matrix {{
			a * V(idx1,0) + b * V(idx2,0),
			a * V(idx1,1) + b * V(idx2,1),
			a * V(idx1,2) + b * V(idx2,2)
		}};
		fp << "$P_{18}=" << P18 << "$\\\\" << endl;
		a = -2; b = 3;
		P19 = matrix {{
			a * V(idx1,0) + b * V(idx2,0),
			a * V(idx1,1) + b * V(idx2,1),
			a * V(idx1,2) + b * V(idx2,2)
		}};
		fp << "$P_{19}=" << P19 << "$\\\\" << endl;

		// make an array of pointers so that we can access the 19 points
		// though this indexed array in loops later one:

		P[0] = &P1;
		P[1] = &P2;
		P[2] = &P3;
		P[3] = &P4;
		P[4] = &P5;
		P[5] = &P6;
		P[6] = &P7;
		P[7] = &P8;
		P[8] = &P9;
		P[9] = &P10;
		P[10] = &P11;
		P[11] = &P12;
		P[12] = &P13;
		P[13] = &P14;
		P[14] = &P15;
		P[15] = &P16;
		P[16] = &P17;
		P[17] = &P18;
		P[18] = &P19;


		// create the homogeneous component of degree 3
		// in the polynomial ring F_2[X0,X1,X2,X3]:
		// We don't need the fact that it is over F_2,
		// but we have to pick a finite field in order to create the ring
		// in Orbiter:

		homogeneous_polynomial_domain *HPD;
		finite_field *F;
		int verbose_level = 1;

		// create the finite field F_2:

		F = NEW_OBJECT(finite_field);
		F->init(2);

		// create the homogeneous polynomial ring
		// of degree 3 in X0,X1,X2,X3:
		// Orbiter will create the 20 monomials

		HPD = NEW_OBJECT(homogeneous_polynomial_domain);
		HPD->init(F, 4 /* nb_vars */, 3 /* degree */,
				FALSE /* f_init_incidence_structure */, verbose_level);

		//print the monomials in Orbiter ordering:
		for (i = 0; i < HPD->nb_monomials; i++) {
			cout << i << " : " << HPD->monomial_symbols[i] << endl;
		}

		// Create the 20 monomials as objects in ginac:
		// We use Orbiter created strings which represent each monomial
		ex M[20];
		parser reader;

		for (i = 0; i < HPD->nb_monomials; i++) {
			M[i] = reader(HPD->monomial_symbols[i]);
			ostringstream s;
			s << latex << M[i];
			cout << "M[" << i << "]=" << s.str() << endl;
		}

		// During the previous loop, ginac has created 4 symbolic variables.
		// We just need to identify them in the symbol table:

		symtab table = reader.get_syms();
		symbol X0 = ex_to<symbol>(table["X0"]);
		symbol X1 = ex_to<symbol>(table["X1"]);
		symbol X2 = ex_to<symbol>(table["X2"]);
		symbol X3 = ex_to<symbol>(table["X3"]);


		// Create the coefficient matrix which forces
		// the surface to pass through the 19 points:
		matrix S(19, 20);
		for (j = 0; j < HPD->nb_monomials; j++) {
			for (i = 0; i < 19; i++) {

				// the (i,j)-entry of S
				// is the j-th monomial evaluated at the i-th point P[i].
				// Note that we are in the affine chart X3 \neq 0, so
				// we put X3=1:

				ex e = M[j].subs(lst{X0, X1, X2, X3},
						lst{(*P[i])(0, 0), (*P[i])(0, 1), (*P[i])(0, 2), 1});

				// And now we specify u=1, v = 2:

				ex e2 = e.subs(lst{u, v}, lst{1, 2});

				// produce latex output:
				ostringstream s;
				s << latex << e;
				cout << "M[" << i << "," << j << "]=" << s.str() << endl;

				S(i, j) = e2;
				//S(i, j) = e;
			}
		}

		// typeset the coefficient matrix in latex:
		fp << "\\begin{sidewaysfigure}" << endl;
		fp << "$$" << endl << S << endl << "$$" << endl;
		fp << "\\end{sidewaysfigure}" << endl;

		// create 20 variables in ginac representing the monomials in the ring:
		// We are using labels such as Xijk,
		// which stands for the cubic monomial Xi*Xj*Xk
		// We have Orbiter create these labels for us
		// and store them in HPD->monomial_symbols_easy

		symbol m0(HPD->monomial_symbols_easy[0]);
		symbol m1(HPD->monomial_symbols_easy[1]);
		symbol m2(HPD->monomial_symbols_easy[2]);
		symbol m3(HPD->monomial_symbols_easy[3]);
		symbol m4(HPD->monomial_symbols_easy[4]);
		symbol m5(HPD->monomial_symbols_easy[5]);
		symbol m6(HPD->monomial_symbols_easy[6]);
		symbol m7(HPD->monomial_symbols_easy[7]);
		symbol m8(HPD->monomial_symbols_easy[8]);
		symbol m9(HPD->monomial_symbols_easy[9]);
		symbol m10(HPD->monomial_symbols_easy[10]);
		symbol m11(HPD->monomial_symbols_easy[11]);
		symbol m12(HPD->monomial_symbols_easy[12]);
		symbol m13(HPD->monomial_symbols_easy[13]);
		symbol m14(HPD->monomial_symbols_easy[14]);
		symbol m15(HPD->monomial_symbols_easy[15]);
		symbol m16(HPD->monomial_symbols_easy[16]);
		symbol m17(HPD->monomial_symbols_easy[17]);
		symbol m18(HPD->monomial_symbols_easy[18]);
		symbol m19(HPD->monomial_symbols_easy[19]);

		lst eqns, vars;
		ex sol;

		// Create the linear system in ginac form:
		// Create the linear system [a*x+b*y==3,x-y==b]...
		for (i = 0; i < 19; i++) {

			// Create one equation for each point.
			// Here, the equation for the i-th point.
			// The right hand side is 0:

			eqns.append(S(i,0)*m0+S(i,1)*m1+S(i,2)*m2+S(i,3)*m3+S(i,4)*m4
					+S(i,5)*m5+S(i,6)*m6+S(i,7)*m7+S(i,8)*m8+S(i,9)*m9
					+S(i,10)*m10+S(i,11)*m11+S(i,12)*m12+S(i,13)*m13+S(i,14)*m14
					+S(i,15)*m15+S(i,16)*m16+S(i,17)*m17+S(i,18)*m18+S(i,19)*m19
					==0);
		}

		// specify the list of variables for which we want to solve:

		vars.append(m0).append(m1).append(m2).append(m3).append(m4);
		vars.append(m5).append(m6).append(m7).append(m8).append(m9);
		vars.append(m10).append(m11).append(m12).append(m13).append(m14);
		vars.append(m15).append(m16).append(m17).append(m18).append(m19);

		// Solve it:
		cout << "solving " << eqns << " for " << vars << endl;
		sol = my_lsolve(fp, eqns, vars);

		fp << "\\clearpage" << endl;

		// Print the solution (there should be only one):
		fp << "solution " << endl << "$$" << endl << sol << endl << "$$" << endl;

		int f_first;

		f_first = TRUE;
		fp << "The equation of the surface is" << endl;
		fp << "$$" << endl;
		for (j = 0; j < 20; j++) {
			ex solj = sol.op(j);  // solution for j-th variable

			if (!solj.is_zero()) {
				if (f_first) {
					f_first = FALSE;
				}
				else {
					fp << " + ";
				}
				fp << solj << HPD->monomial_symbols_latex[j];
			}
			//if (j < 20) {
			//	cout << ", " << endl;
			//}
		}
		fp << " = 0" << endl;
		fp << "$$" << endl;
		cout << endl;

		L.foot(fp);
	}

}




int main() {

	surface();

}
