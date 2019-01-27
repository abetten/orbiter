// matrix.C
//
// Anton Betten
// 10.11.1999
// moved from D2 to ORBI Nov 15, 2007
//
// change in smith normal form: 4/21/2010
// problems with gcd:
// when the two elements x,y are constants, 
// u should not be 0.

#include "orbiter.h"

//#include "partition_backtrack.h"


namespace orbiter {
namespace discreta {

#undef MATRIX_COPY_VERBOSE
#undef DEBUG_S_IJ

#undef DEBUG_CONTENT

static int gfq_dep(int n, matrix& A, matrix& P,
		Vector& v, int m, permutation& rho, int verbose_level);





matrix::matrix()
{
	k = MATRIX;
	self.matrix_pointer = NULL;
}

matrix::matrix(const discreta_base &x)
	// copy constructor:    this := x
{
	cout << "matrix::copy constructor for object: "
			<< const_cast<discreta_base &>(x) << "\n";
	clearself();
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

matrix& matrix::operator = (const discreta_base &x)
	// copy assignment
{
	cout << "matrix::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void matrix::settype_matrix()
{
	OBJECTSELF s;
	
	s = self;
	new(this) matrix;
	k = MATRIX;
	self = s;
}

matrix::~matrix()
{
	freeself_matrix();
}

void matrix::freeself_matrix()
{
	if (self.matrix_pointer == NULL)
		return;
	free_m_times_n_objects(self.matrix_pointer);
	self.matrix_pointer = NULL;
}

kind matrix::s_virtual_kind()
{
	return MATRIX;
}

void matrix::copyobject_to(discreta_base &x)
{
	int i, j, m, n;
	
#ifdef MATRIX_COPY_VERBOSE
	cout << "in matrix::copyobject_to()\n";
#endif
	x.freeself();
	if (x.s_kind() != MATRIX) {
#ifdef MATRIX_CHANGE_KIND_VERBOSE
		cout << "waring: matrix::copyobject_to x not a vector\n";
#endif
		x.c_kind(MATRIX);
		x.clearself();
		// x.printobjectkindln();
		}
	m = s_m();
	n = s_n();
	matrix & xx = x.as_matrix();
	xx.m_mn(m, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			xx.s_ij(i, j) = s_ij(i, j);
			}
		}
}


ostream& matrix::print(ostream& ost)
{
	int i, j, k, m, n, l1, l2, l3;
	Vector col_width;
	
	m = s_m();
	n = s_n();
#ifdef PRint_WITH_TYPE
	ost << "(MATRIX of size " << m << " x " << n << ", \n";
#endif
	col_width.m_l_n(n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ostringstream s;
	
			s << s_ij(i, j) << ends;
			l1 = s.str().length();
			l2 = col_width.s_ii(j);
			l3 = MAXIMUM(l1, l2);
			col_width.m_ii(j, l3);
			}
		}
	if (current_printing_mode() == printing_mode_ascii) {
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				int l;
				ostringstream s;
	
				s << s_ij(i, j);
				l = s.str().length();
				for (k = l; k < col_width.s_ii(j); k++) 
					ost << ' ';
				ost << s.str();
				if (j < n - 1)
					ost << " ";
				}
			ost << endl;
			}
		}
	else if (current_printing_mode() == printing_mode_latex) {
		ost << "\\begin{array}{*{" << n << "}{c}}\n";
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				int l;
				ostringstream s;
	
				s << s_ij(i, j);
				l = s.str().length();
				for (k = l; k < col_width.s_ii(j); k++) 
					ost << ' ';
				ost << s.str();
				if (j < n - 1)
					ost << " & ";
				}
			ost << " \\\\" << endl;
			}
		ost << "\\end{array}\n";
		}
	else {
		ost << "current_printing_mode() = "
				<< current_printing_mode() << " not yet implemented" << endl;
		}
#ifdef PRint_WITH_TYPE
	ost << ")";
#endif
	//ost << "\n";
	return ost;
}

int matrix::compare_with(discreta_base &a)
{
	int i, j, m1, n1, m2, n2, r;
	
	if (a.s_kind() != MATRIX) {
		cout << "matrix::compare_with() a is not a matrix object" << endl;
		exit(1);
		}
	matrix &b = a.as_matrix();
	m1 = s_m();
	n1 = s_n();
	m2 = b.s_m();
	n2 = b.s_n();
	if (m1 < m2)
		return -1;
	if (m1 > m2)
		return 1;
	if (n1 < n2)
		return -1;
	if (n1 > n2)
		return 1;
	for (i = 0; i < m1; i++) {
		for (j = 0; j < n1; j++) {
			r = s_ij(i, j).compare_with(b.s_ij(i, j));
			if (r != 0)
				return r;
			}
		}
	return 0;
}

matrix& matrix::m_mn(int m, int n)
{
	freeself();
	self.matrix_pointer = calloc_m_times_n_objects(m, n, BASE);
	return *this;
}

matrix& matrix::m_mn_n(int m, int n)
{
	int i, j;
	
	m_mn(m, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			s_ij(i, j).m_i_i(0);
			}
		}
	return *this;
}

matrix& matrix::realloc(int m, int n)
{
	matrix M;
	int i, j, m1, n1;
	
	m1 = s_m();
	n1 = s_n();
	M.m_mn(m, n);
	for (i = 0; i < MINIMUM(m, m1); i++) {
		for (j = 0; j < MINIMUM(n, n1); j++) {
			M.s_ij(i, j).swap(s_ij(i, j));
			}
		}
	swap(M);
	return *this;
}

int matrix::s_m()
{
	if (self.matrix_pointer == NULL)
		return 0;
	return self.matrix_pointer[-2].s_i_i();
}

int matrix::s_n()
{
	if (self.matrix_pointer == NULL)
		return 0;
	return self.matrix_pointer[-1].s_i_i();
}

discreta_base & matrix::s_ij(int i, int j)
{
	int m, n;
	
#ifdef DEBUG_S_IJ
	cout << "matrix::s_ij(" << i << ", " << j << ")" << endl;
#endif
	if (self.matrix_pointer == NULL) {
		cout << "matrix::s_ij() matrix_pointer == NULL\n";
		exit(1);
		}
	m = self.matrix_pointer[-2].s_i_i();
	n = self.matrix_pointer[-1].s_i_i();
	if ( i < 0 || i >= m ) {
		cout << "matrix::s_ij() addressing error, i = " << i << ", m = " << m << "\n";
		exit(1);		
		}
	if ( j < 0 || j >= n ) {
		cout << "matrix::s_ij() addressing error, j = " << j << ", n = " << n << "\n";
		exit(1);		
		}
	return self.matrix_pointer[i * n + j];
}

discreta_base & matrix_access::operator [](int j)
{
	return p->s_ij(i, j);
}



void matrix::mult_to(discreta_base &x, discreta_base &y)
{
	if (x.s_kind() == MATRIX) {
		matrix& px = x.as_matrix();
		matrix_mult_to(px, y);
		}
	else if (x.s_kind() == VECTOR) {
		Vector& px = x.as_vector();
		vector_mult_to(px, y);
		}
	else {
		cout << "matrix::mult_to() object x is of bad type\n";
		exit(1);
		}
}

void matrix::matrix_mult_to(matrix &x, discreta_base &y)
{
	matrix py;
	int i, j, k, m, n, l;
	
	if (s_kind() != MATRIX) {
		cout << "matrix::matrix_mult_to() this is not a matrix\n";
		exit(1);
		}
	if (x.s_kind() != MATRIX) {
		cout << "matrix::matrix_mult_to() x is not a matrix\n";
		exit(1);
		}
	m = s_m();
	l = s_n();
	if (l != x.s_m()) {
		cout << "matrix::matrix_mult_to() l != x.s_m(), cannot multiply\n";
		exit(1);
		}
	n = x.s_n();
	
	py.m_mn(m, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			discreta_base a, b;
			for (k = 0; k < l; k++) {
				if (k == 0) {
					a.mult(s_ij(i, k), x[k][j]);
					}
				else {
					b.mult(s_ij(i, k), x[k][j]);
					a += b;
					}
				}
			py[i][j].swap(a);
			}
		}
	y.swap(py);
}

void matrix::vector_mult_to(Vector &x, discreta_base &y)
{
	Vector py;
	int i, j, m, l;
	
	if (s_kind() != MATRIX) {
		cout << "matrix::vector_mult_to() this is not a matrix\n";
		exit(1);
		}
	if (x.s_kind() != VECTOR) {
		cout << "matrix::vector_mult_to() x is not a vector\n";
		exit(1);
		}
	m = s_m();
	l = s_n();
	if (l != x.s_l()) {
		cout << "matrix::vector_mult_to() l != x.s_l(), cannot multiply\n";
		exit(1);
		}
	
	py.m_l(m);
	for (i = 0; i < m; i++) {
		discreta_base a, b;
		for (j = 0; j < l; j++) {
			if (j == 0) {
				a.mult(s_ij(i, j), x[j]);
				}
			else {
				b.mult(s_ij(i, j), x[j]);
				a += b;
				}
			}
		py[i].swap(a);
		}
	y.swap(py);
}

void matrix::multiply_vector_from_left(Vector &x, Vector &y)
{
	int l, i, j, m, n;
	
	l = x.s_l();
	m = s_m();
	n = s_n();
	if (l != m) {
		cout << "matrix::multiply_vector_from_left() l != m, cannot multiply\n";
		exit(1);
		}
	if (y.s_l() != n)
		y.m_l(n);
	for (j = 0; j < n; j++) {
		discreta_base a, b;
		for (i = 0; i < m; i++) {
			if (i == 0) {
				a.mult(x[i], s_ij(i, j));
				}
			else {
				b.mult(x[i], s_ij(i, j));
				a += b;
				}
			}
		y[j].swap(a);
		}	
}

int matrix::invert_to(discreta_base &x)
{
	int m, n, rank;
	matrix P;
	Vector base_cols;
	
	if (s_kind() != MATRIX) {
		cout << "matrix::invert_to() this not a matrix\n";
		exit(1);
		}
	m = s_m();
	n = s_n();
	if (m != n) {
		cout << "matrix::invert_to() m != n\n";
		exit(1);
		}
	P = *this;
	P.one();
	
	rank = Gauss(FALSE /* f_special */, TRUE /* f_complete */, base_cols, 
		TRUE /* f_P */, P, FALSE /* f_v */);
	P.swap(x);
	if (rank < m) {
		cout << "warning: matrix is not invertible\n";
		return FALSE;
		}
	return TRUE;
}

void matrix::add_to(discreta_base &x, discreta_base &y)
{
	int i, j, m, n;
	
	y.freeself();
	if (s_kind() != MATRIX) {
		cout << "matrix::add_to() this is not a matrix\n";
		exit(1);
		}
	if (x.s_kind() != MATRIX) {
		cout << "matrix::add_to() x is not a matrix\n";
		exit(1);
		}
	matrix& px = x.as_matrix();
	matrix py;
	
	m = s_m();
	n = s_n();
	if (m != px.s_m()) {
		cout << "matrix::add_to() m != px.s_m()\n";
		exit(1);
		}
	if (n != px.s_n()) {
		cout << "matrix::add_to() n != px.s_n()\n";
		exit(1);
		}
	py.m_mn(m, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			py[i][j].add(s_ij(i, j), px[i][j]);
			}
		}
	py.swap(y);
}

void matrix::negate_to(discreta_base &x)
{
	int i, j, m, n;
	
	if (s_kind() != MATRIX) {
		cout << "matrix::negate_to() this not a matrix\n";
		exit(1);
		}
	matrix py;
	
	m = s_m();
	n = s_n();
	py.m_mn(m, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			py[i][j] = s_ij(i, j);
			py[i][j].negate();
			}
		}
	py.swap(x);
}


void matrix::one()
{
	int i, j, m, n;
	
	m = s_m();
	n = s_n();
	if (m != n) {
		cout << "matrix::one() m != n\n";
		exit(1);
		}
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (i == j)
				s_ij(i, j).one();
			else
				s_ij(i, j).zero();
			}
		}
}

void matrix::zero()
{
	int i, j, m, n;
	
	m = s_m();
	n = s_n();
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			s_ij(i, j).zero();
			}
		}
}

int matrix::is_zero()
{
	matrix B;
	int m, n;
	
	m = s_m();
	n = s_n();
	B.m_mn_n(m, n);
	if (compare_with(B) == 0)
		return TRUE;
	else
		return FALSE;
	
}

int matrix::is_one()
{
	matrix B;
	int m, n, i, j;
	
	m = s_m();
	n = s_n();
	B.m_mn_n(m, n);
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			if (i == j)
				((integer)B.s_ij(i, j)).m_i(1);
		}
	}
	if (compare_with(B) == 0)
		return TRUE;
	else
		return FALSE;
	
}


int matrix::Gauss(int f_special, int f_complete, Vector& base_cols, 
	int f_P, matrix& P, int f_v)
// returns the rank
{
	int rank, i, j, k, jj, m, n, Pn = 0;
	discreta_base pivot, pivot_inv, a, b, c, z, f;
	
	base_cols.m_l(0);
	m = s_m();
	n = s_n();
	if (f_P) {
		if (m != P.s_m()) {
			cout << "matrix::Gauss() m != P.s_m()\n";
			exit(1);
			}
		Pn = P.s_n();
		}
	
	i = 0;
	for (j = 0; j < n; j++) {
	
		/* search for pivot element: */
		for (k = i; k < m; k++) {
			if (!s_ij(k, j).is_zero()) {
				// pivot element found: 
				if (k != i) {
					for (jj = j; jj < n; jj++) {
						s_ij(i, jj).swap(s_ij(k, jj));
						}
					if (f_P) {
						for (jj = 0; jj < Pn; jj++) {
							P.s_ij(i, jj).swap(P.s_ij(k, jj));
							}
						}
					}
				break;
				} // if != 0 
			} // next k
		
		if (k == m) // no pivot found 
			continue; // increase j, leave i constant
		
		if (f_v) {
			cout << "row " << i << " pivot in row " << k << " colum " << j << endl;
			}
		
		base_cols.append_integer(j);

		pivot = s_ij(i, j);
		pivot.invert_to(pivot_inv);
		if (!f_special) {
			// make pivot to 1: 
			for (jj = j; jj < n; jj++) {
				s_ij(i, jj) *= pivot_inv;
				}
			if (f_P) {
				for (jj = 0; jj < Pn; jj++) {
					P.s_ij(i, jj) *= pivot_inv;
					}
				}
			if (f_v) {
				cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv 
					<< " made to one: " << s_ij(i, j) << endl;
				}
			}
		
		/* do the gaussian elimination: */
		for (k = i + 1; k < m; k++) {
			z = s_ij(k, j);
			if (z.is_zero())
				continue;
			if (f_special) {
				f.mult(z, pivot_inv);
				}
			else {
				f = z;
				}
			f.negate();
			s_ij(k, j).zero();
			if (f_v) {
				cout << "eliminating row " << k << endl;
				}
			for (jj = j + 1; jj < n; jj++) {
				a = s_ij(i, jj);
				b = s_ij(k, jj);
				// c := b + f * a
				//    = b - z * a              if !f_special 
				//      b - z * pivot_inv * a  if f_special 
				c.mult(f, a);
				c += b;
				s_ij(k, jj) = c;
				if (f_v) {
					cout << s_ij(k, jj) << " ";
					}
				}
			if (f_P) {
				for (jj = 0; jj < Pn; jj++) {
					a = P.s_ij(i, jj);
					b = P.s_ij(k, jj);
					// c := b - z * a
					c.mult(f, a);
					c += b;
					P.s_ij(k, jj) = c;
					}
				}
			if (f_v) {
				cout << endl;
				}
			}
		i++;
		} // next j 
	rank = i;
	if (f_complete) {
		for (i = rank - 1; i >= 0; i--) {
			j = base_cols.s_ii(i);
			if (!f_special) {
				a = s_ij(i, j);
				}
			else {
				pivot = s_ij(i, j);
				pivot.invert_to(pivot_inv);
				}
			// do the gaussian elimination in the upper part: 
			for (k = i - 1; k >= 0; k--) {
				z = s_ij(k, j);
				if (z.is_zero())
					continue;
				s_ij(k, j).zero();
				for (jj = j + 1; jj < n; jj++) {
					a = s_ij(i, jj);
					b = s_ij(k, jj);
					if (f_special) {
						a *= pivot_inv;
						}
					c.mult(z, a);
					c.negate();
					c += b;
					s_ij(k, jj) = c;
					}
				if (f_P) {
					for (jj = 0; jj < Pn; jj++) {
						a = P.s_ij(i, jj);
						b = P.s_ij(k, jj);
						if (f_special) {
							a *= pivot_inv;
							}
						c.mult(z, a);
						c.negate();
						c += b;
						P.s_ij(k, jj) = c;
						}
					}
				} // next k
			} // next i
		}
	return rank;
}

int matrix::rank()
{
	Vector base_cols;
	matrix P;
	
	return Gauss(FALSE, FALSE, base_cols, FALSE, P, FALSE);
}

int matrix::get_kernel(Vector& base_cols, matrix& kernel)
{
	int r, n, k, i, j, ii, iii, a, b;
	Vector kcol;
	
	//m = s_m();
	n = s_n();
	r = base_cols.s_l();
	k = n - r;
	kernel.m_mn_n(n, k);
	kcol.m_l(k);
	ii = 0;
	j = 0;
	if (j < r)
		b = base_cols.s_ii(j);
	else
		b = -1;
	for (i = 0; i < n; i++) {
		if (i == b) {
			j++;
			if (j < r)
				b = base_cols.s_ii(j);
			else
				b = -1;
			}
		else {
			kcol.m_ii(ii, i);
			ii++;
			}
		}
	if (ii != k) {
		cout << "matrix::get_kernel() ii != k" << endl;
		exit(1);
		}
	//cout << "kcol = " << kcol << endl;
	ii = 0;
	j = 0;
	if (j < r)
		b = base_cols.s_ii(j);
	else
		b = -1;
	for (i = 0; i < n; i++) {
		if (i == b) {
			for (iii = 0; iii < k; iii++) {
				a = kcol.s_ii(iii);
				kernel[i][iii] = s_ij(j, a);
				}
			j++;
			if (j < r)
				b = base_cols.s_ii(j);
			else
				b = -1;
			}
		else {
			for (iii = 0; iii < k; iii++) {
				if (iii == ii) {
					kernel.s_ij(i, iii).m_one();
					}
				else {
					kernel.s_ij(i, iii).zero();
					}
				}
			ii++;
			}
		}
	return TRUE;
}

matrix& matrix::transpose()
{
	matrix A;
	int i, j, m, n;
	
	m = s_m();
	n = s_n();
	A.m_mn(n, m);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			s_ij(i, j).swap(A.s_ij(j, i));
			}
		}
	swap(A);
	return *this;
}

int matrix::Asup2Ainf()
//Computes the Plesken matrix $A^\wedge$ (Ainf)  from $A^\vee$ (Asup).
//Compare Plesken~\cite{Plesken82}.
{
	Vector orbit_size;
	int m, n, i, j;
	
	m = s_m();
	n = s_n();
	if (m != n) {
		cout << "matrix::Asup2Ainf() not rectangular\n";
		exit(1);
		}
	orbit_size.m_l(m);
	for (i = 0; i < m; i++) {
		orbit_size[i] = s_ij(i, m - 1);
		if (orbit_size[i].is_zero()) {
			cout << "matrix::Asup2Ainf() orbit_size[i].is_zero()\n";
			exit(1);
			}
		}
	for (j = 0; j < m; j++) {
		for (i = 0; i < m; i++) {
			s_ij(i, j) *= orbit_size[j];
			}
		}
	for (i = 0; i < m; i++) {
		for (j = 0; j < m; j++) {
			s_ij(i, j).divide_by_exact(orbit_size[i]);
			}
		}
	return TRUE;
}


int matrix::Ainf2Asup()
{
	transpose();
	Asup2Ainf();
	transpose();
	return TRUE;
}

int matrix::Asup2Acover()
//Computes the cover relations of the poset defined by this (=Asup).
//Assumes that Asup is upper triangular.
{
	int m, n, i, j, k;
	
	m = s_m();
	n = s_n();
	/* replace the numbers by a 0/1 flag */
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (s_iji(i, j))
				m_iji(i, j, 1);
			}
		}
	for (i = m - 1; i >= 0; i--) {
		for (j = i + 1; j < n; j++) {
			if (s_iji(i, j)) {
				/* B_i < B_j in the orbit - 
				 * poset of the lattice */
				for (k = 0; k < i; k++) {
					if (s_iji(k, i)) {
						/* B_k < B_i < B_j
						 * therefore the entry 
						 * B_k < B_j can be deleted. */
						m_iji(k, j, 0);
						}
					}
				}
			}
		}
	return TRUE;
}

int matrix::Acover2nl(Vector& nl)
//Computes the \lq neighbour-list\rq of the poset whose cover relations are given 
//in this (=Acover). This list nl is used as input for the lattice-placement 
//program \lq vbp \rq.
{
	int m, n, i, j, k, len, cur;
	
	m = s_m();
	n = s_n();
	/* count the non diagonal entries: */
	k = 0;
	for (i = 0; i < m; i++) {
		for (j = i + 1; j < n; j++) {
			if (s_iji(i, j))
				k++;
			}
		}
	len = m + 1 + k; // length of nl vector
	nl.m_l(len);
	nl.m_ii(m, len);
	cur = m + 1;
	for (i = 0; i < m; i++) {
		nl.m_ii(i, cur);
			/* start of neighbour list of i in nl */
		for (j = i + 1; j < n; j++) {
			if (s_iji(i, j)) {
				nl.m_ii(cur, j);
					/* found a neighbour of i */
				cur++;
				}
			}
		}
	return TRUE;
}

void matrix::Frobenius(unipoly& m, int p, int verbose_level)
//computes a $d \times d$ matrix whose j-th column 
//contains the coefficients of $x^{p^j} \mod m$. 
//Here, $d = \deg m.$  
{
	int f_v = (verbose_level >= 1);
	unipoly a, b, c;
	int i, j, d, d1;
	
	d = m.degree();
	if (f_v) {
		cout << "matrix::Frobenius() d=" << d << " p=" << p << endl;
		}
	m_mn_n(d, d);
	s_ij(0, 0).one();
	a.x();
	if (f_v) {
		cout << "matrix::Frobenius() x=" << a << endl;
		}
	a.power_int_mod(p, m); // a := x^p mod m
	if (f_v) {
		cout << "matrix::Frobenius() a = x^p=" << a << endl;
		}
	b.one();
	for (j = 1; j < d; j++) {
		c.mult_mod(b, a, m);
		b = c;
		// now b = x^{p^j}
		if (f_v) {
			cout << "matrix::Frobenius() x^{p^" << j << "}=" << b << endl;
			}
		d1 = b.degree();
		for (i = 0; i <= d1; i++) {
			s_ij(i, j) = b[i];
			}
		}
}

void matrix::Berlekamp(unipoly& m, int p, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l;
	integer m1;
	
	Frobenius(m, p, FALSE);
	if (f_v) {
		cout << "Frobenius matrix=\n" << *this;
		}
	l = s_m();
	m1.m_one();
	for (i = 0; i < l; i++) {
		s_ij(i, i) += m1;
		}
	if (f_v) {
		cout << "Berlekamp matrix=\n" << *this;
		}
}

void matrix::companion_matrix(unipoly& m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, d;
	
	if (f_v) {
		cout << "companion_matrix() of the polynomial " << m << endl;
		}
	d = m.degree();
	m_mn_n(d, d);
	zero();
	for (i = 0; i < d - 1; i++) {
		s_ij(i + 1, i).one();
		}
	for (i = 0; i < d; i++) {
		s_ij(i, d - 1) = m.s_i(i);
		s_ij(i, d - 1).negate();
		}
	if (f_v) {
		cout << "companion_matrix=\n" << *this;
		}
}

void matrix::elements_to_unipoly()
{
	int i, j, m, n;
	
	// cout << "matrix::elements_to_unipoly()" << endl;
	m = s_m();
	n = s_n();
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			// cout << "i=" << i << " j=" << j << endl;
			unipoly a;
			a.m_l(1);
			a[0] = s_ij(i, j);
			// cout << "a=" << a << endl;
			s_ij(i, j) = a;
			// cout << "s_ij=" << s_ij(i,j) << endl;
			}
		}
}

void matrix::minus_X_times_id()
{
	unipoly a;
	int i, m, n, l;
	
	a.x();
	a.negate();
	// cout << a << endl;
	m = s_m();
	n = s_n();
	l = MINIMUM(m, n);
	for (i = 0; i < l; i++) {
		s_ij(i, i) += a;
		}
}

void matrix::X_times_id_minus_self()
{
	unipoly a;
	int i, j, m, n, l;
	
	a.x();
	//a.negate();
	// cout << a << endl;
	m = s_m();
	n = s_n();
	l = MINIMUM(m, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			s_ij(i, j).negate();
			}
		}
	for (i = 0; i < l; i++) {
		s_ij(i, i) += a;
		}
}

void matrix::smith_normal_form(matrix& P, matrix& Pv,
		matrix& Q, matrix& Qv, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m, n, i, j, l, ii, jj, stable;
	discreta_base a0, a1, am1;
	
	if (f_v) {
		cout << "matrix::smith_normal_form" << endl;
		cout << *this;
		}
	m = s_m();
	n = s_n();
	a0 = s_ij(0, 0);
	a1 = s_ij(0, 0);
	a0.zero();
	a1.one();
	P.m_mn(m, m);
	Q.m_mn(n, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < m; j++) {
			if (i == j) {
				P[i][j] = a1;
				}
			else {
				P[i][j] = a0;
				}
			}
		}
	Pv = P;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) {
				Q[i][j] = a1;
				}
			else {
				Q[i][j] = a0;
				}
			}
		}
	Qv = Q;
	if (f_v) {
		cout << "this=" << endl << *this << endl;
		cout << "P=" << endl << P << endl;
		cout << "Q=" << endl << Q << endl;
		}
	l = MINIMUM(m, n);
	for (i = 0; i < l; i++) {
		if (f_v) {
			cout << "pivot column is " << i << endl;
			}
		stable = FALSE;
		while (!stable) {
			stable = TRUE;
			if (f_v) {
				cout << "before smith_eliminate_column " << i << endl;
				cout << "this=" << endl << *this << endl;
				cout << "P=" << endl << P << endl;
				cout << "Q=" << endl << Q << endl;
				}
			if (smith_eliminate_column(P, Pv, i, verbose_level)) {
				stable = FALSE;
				}
			if (f_v) {
				cout << "before smith_eliminate_row " << i << endl;
				cout << "this=" << endl << *this << endl;
				cout << "P=" << endl << P << endl;
				cout << "Q=" << endl << Q << endl;
				}
			if (smith_eliminate_row(Q, Qv, i, verbose_level - 1)) {
				stable = FALSE;
				}
			for (jj = i + 1; jj < n; jj++) {
				for (ii = i + 1; ii < m; ii++) {
					if (!s_ij(i, i).is_divisor(s_ij(ii, jj))) {
						break;
						}
					}
				if (ii < m) {
					if (f_v) {
						cout << "adding column " << jj
								<< " to column " << i << endl;
						}
					multiply_2by2_from_right(i, jj, a1, a0, a1, a1, verbose_level - 2);
					Q.multiply_2by2_from_right(i, jj, a1, a0, a1, a1, 0);
					am1 = a1;
					am1.negate();
					Qv.multiply_2by2_from_left(i, jj, a1, a0, am1, a1, 0);
					if (f_v) {
						cout << *this;
						}
					stable = FALSE;
					break;
					}
				}
			}
		}
	if (f_v) {
		cout << "smith normal form reached: " << endl;
		cout << "this=" << endl << *this << endl;
		cout << "P=" << endl << P << endl;
		cout << "Pv=" << endl << Pv << endl;
		cout << "Q=" << endl << Q << endl;
		cout << "Qv=" << endl << Qv << endl;
		}
}

int matrix::smith_eliminate_column(matrix& P,
		matrix& Pv, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int m, j, action = FALSE;
	discreta_base x, y, u, v, g, x1, y1;
	
	if (f_v) {
		cout << "matrix::smith_eliminate_column column " << i << endl;
		cout << "this=" << endl << *this << endl;
		}
	m = s_m();
	for (j = i + 1; j < m; j++) {
		x = s_ij(i, i);
		y = s_ij(j, i);
		if (f_v) {
			cout << "smith_eliminate_column() j=" << j
					<< " x=" << x << " y=" << y << endl;
			cout << "this=" << endl << *this << endl;
			}
		if (y.is_zero()) {
			continue;
			}
		if (f_v) {
			cout << "before extended_gcd" << endl;
			cout << "x=" << x << endl;
			cout << "y=" << y << endl;
			}
		x.extended_gcd(y, u, v, g, verbose_level);
		if (f_v) {
			cout << *this;
			cout << "i=" << i << " j=" << j << ": ";
			cout << g << " = (" << u << ") * (" << x << ") + "
					"(" << v << ") * (" << y << ")" << endl;
			}
		if (u.is_zero() && x.compare_with_euklidean(y) == 0) {
			u.swap(v);
			g = x;
			if (f_v) {
				cout << "after switch:" << endl;
				cout << "this=" << endl << *this << endl;
				cout << "i=" << i << " j=" << j << ": ";
				cout << g << " = (" << u << ") * (" << x << ") + "
						"(" << v << ") * (" << y << ")" << endl;
				}
			}
		x.integral_division_exact(g, x1);
		y.integral_division_exact(g, y1);
		y1.negate();
		multiply_2by2_from_left(i, j, u, v, y1, x1, verbose_level - 2);
		if (f_v) {
			cout << "After multiply_2by2_from_left, "
					"this=" << endl << *this << endl;
			}
		P.multiply_2by2_from_left(i, j, u, v, y1, x1, 0);
		if (f_v) {
			cout << "After P.multiply_2by2_from_left, "
					"P=" << endl << P << endl;
			}
		v.negate();
		y1.negate();
		Pv.multiply_2by2_from_right(i, j, x1, v, y1, u, 0);
		if (f_v) {
			cout << "After Pv.multiply_2by2_from_right, "
					"Pv=" << endl << Pv << endl;
			}
		action = TRUE;
		}
	return action;
}

int matrix::smith_eliminate_row(matrix& Q,
		matrix& Qv, int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int n, j, action = FALSE;
	discreta_base x, y, u, v, g, x1, y1;
	
	if (f_v) {
		cout << "matrix::smith_eliminate_row row " << i << endl;
		}
	n = s_n();
	for (j = i + 1; j < n; j++) {
		x = s_ij(i, i);
		y = s_ij(i, j);
		if (f_v) {
			cout << "smith_eliminate_row() "
					"j=" << j << " x=" << x << " y=" << y << endl;
			}
		if (y.is_zero())
			continue;
		x.extended_gcd(y, u, v, g, verbose_level - 2);
		if (f_vv) {
			cout << *this;
			cout << "i=" << i << " j=" << j << ": ";
			cout << g << " = (" << u << ") * (" << x << ") + "
					"(" << v << ") * (" << y << ")" << endl;
			}
		if (u.is_zero() && x.compare_with_euklidean(y) == 0) {
			u.swap(v);
			g = x;
			if (f_vv) {
				cout << "after switch:" << endl;
				cout << *this;
				cout << "i=" << i << " j=" << j << ": ";
				cout << g << " = (" << u << ") * (" << x << ") + "
						"(" << v << ") * (" << y << ")" << endl;
				}
			}
		x.integral_division_exact(g, x1);
		y.integral_division_exact(g, y1);
		y1.negate();
		multiply_2by2_from_right(i, j, u, y1, v, x1, verbose_level - 2);
		if (f_vv) {
			cout << *this;
			}
		Q.multiply_2by2_from_right(i, j, u, y1, v, x1, 0);
		v.negate();
		y1.negate();
		Qv.multiply_2by2_from_left(i, j, x1, y1, v, u, 0);
		action = TRUE;
		}
	return action;
}

void matrix::multiply_2by2_from_left(int i, int j, 
	discreta_base& aii, discreta_base& aij,
	discreta_base& aji, discreta_base& ajj, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int k, n;
	discreta_base x, y, xx, yy;
	
	if (f_v) {
		cout << "from left: i=" << i << " j=" << j << endl;
		cout << "(" << aii << ", " << aij << ")" << endl;
		cout << "(" << aji << ", " << ajj << ")" << endl;
		}
	n = s_n();
	for (k = 0; k < n; k++) {
		// cout << "k=" << k << endl;
		x.mult(aii, s_ij(i, k));
		y.mult(aij, s_ij(j, k));
		x += y;
		xx.mult(aji, s_ij(i, k));
		yy.mult(ajj, s_ij(j, k));
		yy += xx;
		x.swap(s_ij(i, k));
		yy.swap(s_ij(j, k));
		}
}

void matrix::multiply_2by2_from_right(int i, int j, 
	discreta_base& aii, discreta_base& aij,
	discreta_base& aji, discreta_base& ajj, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int k, m;
	discreta_base x, y, xx, yy;
	
	if (f_v) {
		cout << "from right: i=" << i << " j=" << j << endl;
		cout << "(" << aii << ", " << aij << ")" << endl;
		cout << "(" << aji << ", " << ajj << ")" << endl;
		}
	m = s_m();
	for (k = 0; k < m; k++) {
		// cout << "k=" << k << endl;
		x.mult(aii, s_ij(k, i));
		y.mult(aji, s_ij(k, j));
		x += y;
		xx.mult(aij, s_ij(k, i));
		yy.mult(ajj, s_ij(k, j));
		yy += xx;
		x.swap(s_ij(k, i));
		yy.swap(s_ij(k, j));
		}
}

void matrix::to_vector_of_rows(Vector& v)
{
	Vector vv;
	int i, j, m, n;
	
	m = s_m();
	n = s_n();
	v.m_l(m);
	for (i = 0; i < m; i++) {
		vv.m_l(n);
		for (j = 0; j < n; j++) {
			vv[j] = s_ij(i, j);
			}
		v[i] = vv;
		}
}

void matrix::from_vector_of_rows(Vector& v)
{
	int i, j, m, n;
	
	m = v.s_l();
	if (m <= 0) {
		cout << "matrix::from_vector_of_rows() m <= 0\n";
		exit(1);
		}
	n = v[0].as_vector().s_l();
	
	m_mn(m, n);
	for (i = 0; i < m; i++) {
		Vector &vv = v.s_i(i).as_vector();
		for (j = 0; j < n; j++) {
			s_ij(i, j) = vv[j];
			}
		}
}

void matrix::to_vector_of_columns(Vector& v)
{
	Vector vv;
	int i, j, m, n;
	
	m = s_m();
	n = s_n();
	v.m_l(n);
	for (j = 0; j < n; j++) {
		vv.m_l(m);
		for (i = 0; i < m; i++) {
			vv[i] = s_ij(i, j);
			}
		v[j] = vv;
		}
}

void matrix::from_vector_of_columns(Vector& v)
{
	int i, j, m, n;
	
	n = v.s_l();
	if (n <= 0) {
		cout << "matrix::from_vector_of_columns n <= 0" << endl;
		exit(1);
		}
	m = v[0].as_vector().s_l();
	
	m_mn(m, n);
	for (j = 0; j < n; j++) {
		Vector &vv = v.s_i(j).as_vector();
		for (i = 0; i < m; i++) {
			s_ij(i, j) = vv[i];
			}
		}
}

void matrix::evaluate_at(discreta_base& x)
{
	int i, j, m, n;
	discreta_base y;
	
	m = s_m();
	n = s_n();
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			s_ij(i, j).as_unipoly().evaluate_at(x, y);
			s_ij(i, j) = y;
			}
		}
}


static int gfq_dep(int n, matrix& A,
		matrix& P, Vector& v, int m, permutation& rho,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, k, f_null, pp;
	discreta_base a0, a1, c;

	if (f_v) {
		cout << "gfq_dep" << endl;
		}
	a0 = v[0];
	a1 = v[0];
	a0.zero();
	a1.one();
		
	for (j = 0; j < n; j++)
		A[m][j] = v[rho[j]];
	
	for (k = 0; k < m; k++) {
		if (f_vv) {
			cout << "k=" << k << endl;
			}
		c = A[k][k];
		c.invert();
		c *= A[m][k];
		c.negate();
		A.multiply_2by2_from_left(k, m, a1, a0, c, a1, FALSE);
		P.multiply_2by2_from_left(k, m, a1, a0, c, a1, FALSE);
		if (f_vv) {
			cout << "A=\n" << A << endl;
			cout << "P=\n" << P << endl;
			}
		} // next k
	
	f_null = (m == n);
	if (!f_null) {
		// search for a non-zero entry in row m starting in column m.
		// exchange that column with column m, and change the column
		// permutation rho.
		j = m;
		while (A[m][j].is_zero() && j < n - 1)
			j++;
		f_null = (A[m][j].is_zero());
		if (!f_null && j > m) {
			for (i = 0; i <= m; i++) {
				A[i][m].swap(A[i][j]);
				}
			pp = rho[m];
			rho[m] = rho[j];
			rho[j] = pp;
			}
		}
	return f_null;
}

void matrix::KX_module_order_ideal(int i, unipoly& mue, int verbose_level)
// Lueneburg~\cite{Lueneburg87a} p. 105
// determines the order ideal of $e_i$, the $i$-th unit vector,
// in the module over the polynomial ring $K[X]$ ($K$ a field), 
// which is given by substituting the matrix $F$ into the polynomial.
// the matrix $F$ (in this) has dimensions $n \times n$
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	matrix A, P;
	Vector v, v1;
	int n, m, f_null, j;
	permutation rho;
	discreta_base c;
	
	n = s_m();
	A.m_mn_n(n + 1, n);
	P.m_mn_n(n + 1, n + 1);
	P.one();
	v.m_l_n(n);
	rho.m_l(n);
	rho.one();
	v[i].m_i_i(1);
	
	m = 0;
	if (f_v) {
		cout << "matrix::KX_module_order_ideal() m=" << m << endl;
		cout << "v=\n" << v << endl;
		}
	f_null = gfq_dep(n, A, P, v, m, rho, verbose_level - 2);
	if (f_vv) {
		cout << "A=\n" << A << endl;
		}
	while (!f_null) {
		
		v1.mult(*this, v);
		v = v1;
		m++;
		if (f_vv) {
			cout << "matrix::KX_module_order_ideal m=" << m << endl;
			cout << "v=\n" << v << endl;
			}
		f_null = gfq_dep(n, A, P, v, m, rho, verbose_level - 2);
		if (f_vv) {
			cout << "A=\n" << A << endl;
			}
		if (m == n && !f_null) {
			cout << "gfq_order_ideal m == n && !f_null" << endl;
			exit(1);
			}
		}
	
	mue.m_l(m + 1);
	mue.s_i(m).one();
	for (j = m - 1; j >= 0; j--) {
		mue[j] = P[m][j];
		}

	if (f_v) {
		cout << "matrix::KX_module_order_ideal "
				"order ideal of e_" << i << ": " << mue << endl;
#if 0
		v.m_l_n(n);
		v[i].one();
		KX_module_apply(mue, v);
		cout << "mue(v)=" << v << endl;
#endif
		}
}

void matrix::KX_module_apply(unipoly& p, Vector& v)
{
	int i, d;
	Vector w, ww, vv;
	
	d = p.degree();
	w = v;
	w.mult_scalar(p[d]);
	for (i = d - 1; i >= 0; i--) {
		// cout << "i=" << i << " ; w=" << w << endl;
		ww.mult(*this, w);
		// cout << "i=" << i << " ; F * w=" << ww << endl;
		vv = v;
		vv.mult_scalar(p[i]);
		ww += vv;
		w.swap(ww);
		}
	v.swap(w);
}

void matrix::KX_module_join(Vector& v1, unipoly& mue1, 
	Vector& v2, unipoly& mue2, Vector& v3, unipoly& mue3,
	int verbose_level)
// compare Lueneburg~\cite{Lueneburg87a} p. 106.
{
	int f_v = (verbose_level >= 1);
	unipoly u, v, g, gg, r, rr, rrr, r4, m1, m2;
	Vector vv1, vv2;
	int dg, dm2;
	
	if (f_v) {
		cout << "matrix::KX_module_join()" << endl;
		cout << "v1=" << v1 << " mue1=" << mue1 << endl;
		cout << "v2=" << v2 << " mue2=" << mue2 << endl;
		}
	dm2 = mue2.degree();
	mue1.extended_gcd(mue2, u, v, g, verbose_level - 2);
	if (f_v) {
		cout << "g=" << g << endl;
		}
	dg = g.degree();
	if (dg < dm2) {
		vv1 = v1;
		mue2.integral_division_exact(g, m2);
		mue1.largest_divisor_prime_to(m2, r);
		mue1.integral_division_exact(r, m1);
		KX_module_apply(m1, vv1);
		// now: order(vv1) = (r)
		if (f_v) {
			cout << "vv1=" << vv1 << " r=" << r << endl;
			}
		
		vv2 = v2;
		mue1.integral_division_exact(g, m1);
		mue2.largest_divisor_prime_to(m1, rr);
		r.extended_gcd(rr, u, v, gg, verbose_level - 2);
		rr.integral_division_exact(gg, rrr);
		// now (gcd(r, rrr) = 1 and r*rrr = lcm(mue1,mue2)
		if (f_v) {
			cout << "r=" << r << endl;
			cout << "rrr=" << rrr << endl;
			r.extended_gcd(rrr, u, v, gg, verbose_level - 2);
			cout << "gcd(r,rrr)=" << gg << " (should be 1)" << endl;
			r4.mult(r, rrr);
			cout << "r*rrr=" << r4 << " (should be = lcd(mue1, mue2))" << endl;
			}
		
			
		mue2.integral_division_exact(rrr, m2);
		KX_module_apply(m2, vv2);
		// now: order(vv2) = (rrr) and gcd(r, rrr) = 1.
		if (f_v) {
			cout << "vv2=" << vv2 << " rrr=" << rrr << endl;
			}
		
		v3.add(vv1, vv2);
		mue3.mult(r, rrr);
		if (f_v) {
			cout << "matrix::KX_module_join()" << endl;
			cout << "v3=" << v3 << endl;
			cout << "mue3=" << mue3 << endl;
			vv1 = v3;
			KX_module_apply(mue3, vv1);
			cout << "mue3(v3)=" << vv1 << " (should be zero)" << endl;
			}
		}
	else {
		v3 = v1;
		mue3 = mue1;
		}
}

void matrix::KX_cyclic_module_generator(Vector& v,
		unipoly& mue, int verbose_level)
{
	int f_v = (verbose_level > 1);
	int f_vv = (verbose_level > 2);
	int i, f;
	Vector v1, v2, v3;
	unipoly mue1, mue2, mue3;
	
	if (f_v) {
		cout << "matrix::KX_cyclic_module_generator" << endl;
		}
	f = s_m();
	v1.m_l_n(f);
	v1[0].one();
	KX_module_order_ideal(0, mue1, verbose_level - 1);
	for (i = 1; i < f; i++) {
		if (mue1.degree() == f)
			break;
		v2.m_l_n(f);
		v2[i].one();
		KX_module_order_ideal(i, mue2, verbose_level - 1);
		KX_module_join(v1, mue1, v2, mue2, v3, mue3, f_vv);
		v1 = v3;
		mue1 = mue3;
		}
	if (mue1.degree() < f) {
		cout << "matrix::KX_cyclic_module_generator error: "
				"mue1.degree < f" << endl;
		exit(1);
		}
	v1.swap(v);
	mue1.swap(mue);
}

void matrix::KX_module_minpol(unipoly& p,
		unipoly& m, unipoly& mue, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	unipoly x, x0, x1, q, y;
	Vector V, v, v0;
	discreta_base c;
	int i, j, d, f, l;
	
	if (f_v) {
		cout << "matrix::KX_module_minpol" << endl;
		}
	f = s_m();
	d = p.degree();
	if (d >= f) {
		cout << "KX_module_minpol() d >= f\n";
		exit(1);
		}
	x.x();
	x0.x();
	x0[0].zero();
	x0[1].zero();
	x1.x();
	x1[0].one();
	x1[1].zero();
	V.m_l(1);
	v.m_l_n(f);
	for (i = 0; i <= d; i++) {
		v[i] = p[i];
		}
	V[0] = p;
	v0 = v;
	l = 1;
	if (f_vv) {
		cout << "p^{sigma^" << l - 1 << "}=" << p << " = " << v << endl;
		}
	while (TRUE) {
		KX_module_apply(x, v);
		if (v.compare_with(v0) == 0) {
			break;
			}
		q.m_l(f);
		for (i = 0; i < f; i++) {
			q[i] = v[i];
			}
		V.inc();
		V[l] = q;
		if (f_vv) {
			cout << "p^{sigma^" << l << "}=" << q << " = " << v << endl;
			}
		if (l >= f) {
			cout << "KX_module_minpol() l >= f\n";
			}
		l++;
		}
	if (f_vv) {
		cout << "the degree is " << l << endl;
		}
	for (i = 0; i < l; i++) {
		V[i].negate();
		}
	y.m_l(l + 1);
	
	// the polynomial (X - p) = (X - p^{sigma^0}), 
	// but shifted into the top position:
	y[l] = x1;
	y[l - 1] = V[0];
	for (i = l - 2; i >= 0; i--) {
		y[i] = x0;
		}
	for (i = 1; i < l; i++) {
		// multiply with (X - p^{sigma^i}):
		for (j = i; j >= 0; j--) {
			c.mult_mod(y[l - j], V[i], m);
			y[l - j - 1] += c;
			}
		}
	if (f_vv) {
		cout << "y (coefficients must lie in the ground field):" << endl;
		for (i = 0; i <= l; i++) {
			cout << y[i] << " * X^" << i << endl;
			}
		}
	mue.m_l(l + 1);
	for (i = 0; i <= l; i++) {
		mue[i] = y[i].as_unipoly()[0];
		}
	if (f_v) {
		cout << "minpol=" << mue << endl;
		}
}

void matrix::binomial(int n_min, int n_max,
		int k_min, int k_max)
{
	int i, j, m, n;
	
	m = n_max + 1 - n_min;
	n = k_max + 1 - k_min;
	m_mn(m, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			::Binomial(n_min + i, k_min + j, s_ij(i, j));
			}
		}
}

void matrix::stirling_second(int n_min, int n_max,
		int k_min, int k_max, int f_ordered)
{
	int i, j, m, n;
	int f_v = FALSE;
	
	m = n_max + 1 - n_min;
	n = k_max + 1 - k_min;
	m_mn(m, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			::stirling_second(n_min + i,
					k_min + j, f_ordered, s_ij(i, j), f_v);
			}
		}
}

void matrix::stirling_first(int n_min, int n_max,
		int k_min, int k_max, int f_signless)
{
	int i, j, m, n;
	int f_v = FALSE;
	
	m = n_max + 1 - n_min;
	n = k_max + 1 - k_min;
	m_mn(m, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			::stirling_first(n_min + i, k_min + j,
					f_signless, s_ij(i, j), f_v);
			}
		}
}

void matrix::binomial(int n_min, int n_max,
		int k_min, int k_max, int f_inverse)
{
	int i, j, m, n;
	
	m = n_max + 1 - n_min;
	n = k_max + 1 - k_min;
	m_mn(m, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			::Binomial(n_min + i, k_min + j, s_ij(i, j));
			if (f_inverse && ODD(n_min + i + k_min + j))
				s_ij(i, j).negate();			
			}
		}
}

int matrix::hip()
// homogeneous integer matrix predicate 
{
	int i, j, m, n;
	
	m = s_m();
	n = s_n();
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			if (s_ij(i, j).s_kind() != INTEGER)
				return FALSE;
	return TRUE;
}

int matrix::hip1()
// homogeneous integer matrix predicate, 
// test for 1 char numbers; 
// only to apply if hip TRUE. 
{
	int i, j, m, n, k;
	
	m = s_m();
	n = s_n();
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (s_ij(i, j).s_kind() != INTEGER) {
				cout << "matrix::hip1(): not intEGER\n";
				exit(1);
				}
			k = s_iji(i, j);
			if (!ONE_char_int(k))
				return FALSE;
			}
		}
	return TRUE;
}

void matrix::write_mem(memory & M, int debug_depth)
{
	int i, j, m, n, k;
	char f_hip = 0, f_hip1 = 0;
	
	m = s_m();
	n = s_n();
	M.write_int(n); // !!! first n then m
	M.write_int(m);
	f_hip = (char) hip();
	if (f_hip)
		f_hip1 = (char) hip1();
	if (debug_depth > 0) {
		cout << "writing " << m << " x " << n << " ";
		if (f_hip) {
			if (f_hip1)
				cout << "hip1 ";
			else
				cout << "hip ";
			}
		cout << "matrix\n";
		}
	M.write_char(f_hip);
	if (f_hip) {
		M.write_char(f_hip1);
		if (f_hip1) {
			for (i = 0; i < m; i++) {
				for (j = 0; j < n; j++) {
					k = s_iji(i, j);
					M.write_char((char) k);
					}
				}
			}
		else {
			for (i = 0; i < m; i++) {
				for (j = 0; j < n; j++) {
					M.write_int(s_iji(i, j));
					}
				}
			}
		}
	else {
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				s_ij(i, j).write_memory(M, debug_depth - 1);
				}
			}
		}
}

void matrix::read_mem(memory & M, int debug_depth)
{
	int i, j, m, n, k;
	char c, f_hip = 0, f_hip1 = 0;
	
	M.read_int(&n);
	M.read_int(&m);
	m_mn(m, n);
	M.read_char(&f_hip);
	if (f_hip) {
		M.read_char(&f_hip1);
		}
	if (debug_depth > 0) {
		cout << "reading " << m << " x " << n << " ";
		if (f_hip) {
			if (f_hip1)
				cout << "hip1 ";
			else
				cout << "hip ";
			}
		cout << "matrix\n";
		}
	if (f_hip) {
		if (f_hip1) {
			for (i = 0; i < m; i++) {
				for (j = 0; j < n; j++) {
					M.read_char(&c);
					k = (int) c;
					m_iji(i, j, k);
					}
				}
			}
		else {
			for (i = 0; i < m; i++) {
				for (j = 0; j < n; j++) {
					M.read_int(&k);
					m_iji(i, j, k);
					}
				}
			}
		}
	else {
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				if (debug_depth > 0) {
					cout << "(" << i << "," << j << ") ";
					if ((j % 20) == 0)
						cout << endl;
					}
				s_ij(i, j).read_memory(M, debug_depth - 1);
				}
			}
		}
}

int matrix::csf()
{
	int size = 0;
	int i, j, m, n;
	char f_hip, f_hip1;
	
	m = s_m();
	n = s_n();
	size += 8; /* n, m */
	f_hip = (char) hip();
	size += 1; /* f_hip */
	if (f_hip) {
		f_hip1 = (char) hip1();
		size += 1; /* f_hip1 */
		if (f_hip1)
			size += 1 * m * n;
		else
			size += 4 * m * n;
		}
	else {
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				size += s_ij(i, j).calc_size_on_file();
				}
			}
		}
	return size;
}

void matrix::calc_theX(int & nb_X, int *&theX)
{
	int m, n, i, j;
	
	m = s_m();
	n = s_n();
	
	nb_X = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (s_iji(i, j))
				nb_X++;
			}
		}
	theX = (int *) new int[nb_X];
	
	nb_X = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (s_iji(i, j)) {
				theX[nb_X++] = i * n + j;
				}
			}
		}
}

#if 0
void matrix::lexleast_incidence_matrix(int f_on_rows, 
	int f_row_decomp, Vector & row_decomp, 
	int f_col_decomp, Vector & col_decomp, 
	int f_ddp, Vector & DDp, 
	int f_ddb, Vector & DDb, 
	int f_group, perm_group & G, 
	permutation & p, permutation & q, 
	int f_print_backtrack_points, 
	int f_get_aut_group, int f_aut_group_on_lexleast, Vector & aut_gens, 
	int f_v, int f_vv)
{
	int nb_X, *theX;
	int f_maxtest = FALSE;
	int back_to;
	int f_transposed = FALSE;
	
	if (f_on_rows)
		f_transposed = TRUE;
	
	if (f_v) {
		cout << "matrix::lexleast_incidence_matrix()" << endl;
		}
	calc_theX(nb_X, theX);
	if (f_v) {
		cout << "matrix::lexleast_incidence_matrix() calling geo_canon_with_initial_decomposition_and_ddp_ddb()" << endl;
		}
	geo_canon_with_initial_decomposition_and_ddp_ddb(
		f_maxtest, &back_to, 
		f_transposed, 
		s_m(), s_n(), nb_X, theX, 
		f_row_decomp, row_decomp, 
		f_col_decomp, col_decomp, 
		f_ddp, DDp, 
		f_ddb, DDb, 
		f_group, G, 
		p, q, 
		f_print_backtrack_points, 
		f_get_aut_group, f_aut_group_on_lexleast, aut_gens, 
		f_v, f_vv);
}
#endif

void matrix::apply_perms(int f_row_perm, permutation &row_perm, 
	int f_col_perm, permutation &col_perm)
{
	matrix M;
	int m, n, i, ii, j, jj;
	permutation rowperm, colperm;
	
	m = s_m();
	n = s_n();
	M.m_mn(m, n);
	if (f_row_perm)
		rowperm = row_perm;
	else {
		rowperm.m_l(m);
		rowperm.one();
		}
	if (f_col_perm)
		colperm = col_perm;
	else {
		colperm.m_l(n);
		colperm.one();
		}
	for (i = 0; i < m; i++) {
		ii = rowperm[i];
		for (j = 0; j < n; j++) {
			jj = colperm[j];
			s_ij(i, j).swap(M.s_ij(ii, jj));
			}
		}
	swap(M);
}

void matrix::apply_col_row_perm(permutation &p)
{
	matrix M;
	int m, n, i, ii, j, jj;
	
	m = s_m();
	n = s_n();
	M.m_mn(m, n);
	for (i = 0; i < m; i++) {
		ii = p[n + i] - n;
		for (j = 0; j < n; j++) {
			jj = p[j];
			s_ij(i, j).swap(M.s_ij(ii, jj));
			}
		}
	swap(M);
}

void matrix::apply_row_col_perm(permutation &p)
{
	matrix M;
	int m, n, i, ii, j, jj;
	
	m = s_m();
	n = s_n();
	M.m_mn(m, n);
	for (i = 0; i < m; i++) {
		ii = p[i];
		for (j = 0; j < n; j++) {
			jj = p[m + j] - m;
			s_ij(i, j).swap(M.s_ij(ii, jj));
			}
		}
	swap(M);
}

void matrix::incma_print_ascii_permuted_and_decomposed(ostream &ost, int f_tex, 
	Vector & decomp, permutation & p)
{
	matrix M;
	int n;
	Vector row_decomp, col_decomp;
	int i, l1, l, ll;
	
	//m = s_m();
	n = s_n();
	M = *this;
	M.apply_col_row_perm(p);
	ll = decomp.s_l();
	l1 = 0;
	col_decomp.m_l(0);
	for (i = 0; i < ll; i++) {
		l = decomp.s_ii(i);
		l1 += l;
		col_decomp.append_integer(l);
		if (l1 == n) {
			break;
			}
		}
	row_decomp.m_l(0);
	for (i++; i < ll; i++) {
		l = decomp.s_ii(i);
		l1 += l;
		row_decomp.append_integer(l);
		}
	M.incma_print_ascii(ost, f_tex, TRUE, row_decomp, TRUE, col_decomp);
}

void matrix::print_decomposed(ostream &ost, Vector &row_decomp, Vector &col_decomp)
{
	matrix T;
	int m, n, M, N, i, j, i0, j0, v, h;
	Vector hbar, vbar;

	m = s_m();
	n = s_n();
	M = 2 * m + 1;
	N = 2 * n + 1;
	T.m_mn(M, N);
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			T.s_ij(i, j).change_to_hollerith();
			T.s_ij(i, j).as_hollerith().init("");
			}
		}
	hbar.m_l_n(M);
	vbar.m_l_n(N);
	i0 = 0;
	hbar.m_ii(2 * i0, TRUE);
	for (i = 0; i < row_decomp.s_l(); i++) {
		i0 += row_decomp.s_ii(i);
		hbar.m_ii(2 * i0, TRUE);
		}
	j0 = 0;
	vbar.m_ii(2 * j0, TRUE);
	for (j = 0; j < col_decomp.s_l(); j++) {
		j0 += col_decomp.s_ii(j);
		vbar.m_ii(2 * j0, TRUE);
		}
	// cout << "hbar=" << hbar << endl;
	// cout << "vbar=" << vbar << endl;
	
	for (i = 0; i <= 2 * m; i += 2) {
		for (j = 0; j <= 2 * n; j += 2) {
			h = hbar.s_ii(i);
			v = vbar.s_ii(j);
			if (v && h) {
				T.s_ij(i, j).as_hollerith().init("+");
				}
#if 0
			else if (v && !h) {
				T.s_ij(i, j).as_hollerith().init("|");
				}
			else if (!v && h) {
				T.s_ij(i, j).as_hollerith().init("-");
				}
#endif
			}
		}
	for (i = 0; i <= 2 * m; i += 2) {
		if (!hbar.s_ii(i))
			continue;
		for (j = 0; j < n; j++) {
			T.s_ij(i, 2 * j + 1).as_hollerith().init("-");
			}
		}
	for (j = 0; j <= 2 * n; j += 2) {
		if (!vbar.s_ii(j))
			continue;
		for (i = 0; i < m; i++) {
			T.s_ij(2 * i + 1, j).as_hollerith().init("|");
			}
		}
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			T.s_ij(2 * i + 1, 2 * j + 1) = s_ij(i, j);
			}
		}
	ost << T;
}

void matrix::incma_print_ascii(ostream &ost, int f_tex, 
	int f_row_decomp, Vector &row_decomp, 
	int f_col_decomp, Vector &col_decomp)
{
	matrix T;
	int m, n, M, N, i, j, i0, j0, v, h;
	Vector hbar, vbar;
	Vector S;
	Vector rd, cd;
	
	m = s_m();
	n = s_n();
	M = 2 * m + 1;
	N = 2 * n + 1;
	T.m_mn(M, N);
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			T.s_ij(i, j).change_to_hollerith();
			T.s_ij(i, j).as_hollerith().init("");
			}
		}
	if (f_row_decomp)
		rd = row_decomp;
	else {
		rd.m_l(1);
		rd.m_ii(0, m);
		}
	if (f_col_decomp)
		cd = col_decomp;
	else {
		cd.m_l(1);
		cd.m_ii(0, n);
		}
	hbar.m_l_n(M);
	vbar.m_l_n(N);
	i0 = 0;
	hbar.m_ii(2 * i0, TRUE);
	for (i = 0; i < rd.s_l(); i++) {
		i0 += rd.s_ii(i);
		hbar.m_ii(2 * i0, TRUE);
		}
	j0 = 0;
	vbar.m_ii(2 * j0, TRUE);
	for (j = 0; j < cd.s_l(); j++) {
		j0 += cd.s_ii(j);
		vbar.m_ii(2 * j0, TRUE);
		}
	// cout << "hbar=" << hbar << endl;
	// cout << "vbar=" << vbar << endl;
	
	for (i = 0; i <= 2 * m; i += 2) {
		for (j = 0; j <= 2 * n; j += 2) {
			h = hbar.s_ii(i);
			v = vbar.s_ii(j);
			if (v && h) {
				T.s_ij(i, j).as_hollerith().init("+");
				}
#if 0
			else if (v && !h) {
				T.s_ij(i, j).as_hollerith().init("|");
				}
			else if (!v && h) {
				T.s_ij(i, j).as_hollerith().init("-");
				}
#endif
			}
		}
	for (i = 0; i <= 2 * m; i += 2) {
		if (!hbar.s_ii(i))
			continue;
		for (j = 0; j < n; j++) {
			T.s_ij(i, 2 * j + 1).as_hollerith().init("-");
			}
		}
	for (j = 0; j <= 2 * n; j += 2) {
		if (!vbar.s_ii(j))
			continue;
		for (i = 0; i < m; i++) {
			T.s_ij(2 * i + 1, j).as_hollerith().init("|");
			}
		}
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (s_iji(i, j)) {
				T.s_ij(2 * i + 1, 2 * j + 1).as_hollerith().init("X");
				}
			else {
				T.s_ij(2 * i + 1, 2 * j + 1).as_hollerith().init(".");
				}
			}
		}
	S.m_l(M);
	for (i = 0; i < M; i++) {
		S.s_i(i).change_to_hollerith();
		S.s_i(i).as_hollerith().init("");
		}
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			S.s_i(i).as_hollerith().append(T.s_ij(i, j).as_hollerith().s());
			}
		}
	for (i = 0; i < M; i++) {
		if (ODD(i) || hbar.s_ii(i)) {
			ost << S.s_i(i).as_hollerith();
			if (f_tex) {
				ost << "\\\\[-12pt]";
				}
			ost << endl;
			}
		}
}

void matrix::incma_print_latex(ostream &f, 
	int f_row_decomp, Vector &row_decomp, 
	int f_col_decomp, Vector &col_decomp, 
	int f_labelling_points, Vector &point_labels, 
	int f_labelling_blocks, Vector &block_labels)
{
	incma_print_latex2(f, 
		40 /* width */, 
		10 /* width_10 */,  
		FALSE /* f_outline_thin */, 
		"0.065mm" /* unit_length */, 
		"0.5mm" /* thick_lines */ , 
		"0.15mm" /* thin_lines */ , 
		"0.25mm" /* geo_line_width */ , 
		f_row_decomp, row_decomp, 
		f_col_decomp, col_decomp, 
		f_labelling_points, point_labels, 
		f_labelling_blocks, block_labels);
}

void matrix::incma_print_latex2(ostream &f, 
	int width, int width_10, 
	int f_outline_thin, const char *unit_length, 
	const char *thick_lines, const char *thin_lines, const char *geo_line_width, 
	int f_row_decomp, Vector &row_decomp, 
	int f_col_decomp, Vector &col_decomp, 
	int f_labelling_points, Vector &point_labels, 
	int f_labelling_blocks, Vector &block_labels)
/* width for one box in 0.1mm 
 * width_10 is 1 10th of width
 * example: width = 40, width_10 = 4 */
{
	int v, b;
	int w, h, w1, h1;
	int i, j, k, a;
	int x0, y0, x1, y1;
	int X0, Y0, X1, Y1;
	int width_8, width_5;
	const char *tdo_line_width = thick_lines /* "0.7mm" */;
	const char *line_width = thin_lines /* "0.15mm" */;
	/* char *geo_line_width = "0.25mm"; */
	Vector rd, cd;
	
	v = s_m();
	b = s_n();
	if (f_row_decomp)
		rd = row_decomp;
	else {
		rd.m_l(1);
		rd.m_ii(0, v);
		}
	if (f_col_decomp)
		cd = col_decomp;
	else {
		cd.m_l(1);
		cd.m_ii(0, b);
		}
	
	width_8 = width - 2 * width_10;
	width_5 = width >> 1;
	f << "\\unitlength" << unit_length << endl;
	w = b * width;
	h = v * width;
	w1 = w;
	h1 = h;
	if (f_labelling_points)
		w1 += 2 * width;
	if (f_labelling_blocks)
		h1 += 2 * width;
	f << "\\begin{picture}(" << w1 << "," << h1 << ")\n";

	/* the grid: */
	f << "\\linethickness{" << tdo_line_width << "}\n";
	k = 0;
	for (i = -1; i < cd.s_l(); i++) {
		if (i >= 0) {
			a = cd.s_ii(i);
			k += a;
			}
		if (f_outline_thin) {
			if (i == -1 || i == cd.s_l() - 1)
				continue;
			}
		f << "\\put(" << k * width << ",0){\\line(0,1){" << h << "}}\n";
		}
	if (k != b) {
		cout << "incma_print_latex2(): k != b\n";
		exit(1);
		}
	k = 0;
	for (i = -1; i < rd.s_l(); i++) {
		if (i >= 0) {
			a = rd.s_ii(i);
			k += a;
			}
		if (f_outline_thin) {
			if (i == -1 || i == rd.s_l() - 1)
				continue;
			}
		f << "\\put(0," << h - k * width << "){\\line(1,0){" << w << "}}\n";
		}
	if (k != v) {
		cout << "incma_print_latex2(): k != v\n";
		exit(1);
		}
	if (f_labelling_points) {
		for (i = 0; i < v; i++) {
			f << "\\put(0," << h - i * width - width_5 
				<< "){\\makebox(0,0)[r]{" 
				<< point_labels.s_i(i) << "$\\,$}}\n";
			}
		}
	if (f_labelling_blocks) {
		for (i = 0; i < b; i++) {
			f << "\\put(" << i * width + width_5 << "," 
				<< h + width_5 << "){\\makebox(0,0)[b]{" 
				<< block_labels.s_i(i) << "}}\n";
			}
		}

	f << "\\linethickness{" << line_width << "}\n";
	f << "\\multiput(0,0)(" << width << ",0){" << b + 1 
		<< "}{\\line(0,1){" << h << "}}\n";
	f << "\\multiput(0,0)(0," << width << "){" << v + 1 << "}{\\line(1,0){" 
		<< w << "}}\n";

	/* the geometry: */
	f << "\\linethickness{" << geo_line_width << "}\n";
	for (i = 0; i < v; i++) {
		y0 = h - i * width;
		y1 = h - (i + 1) * width;
		Y0 = y0 - width_10;
		Y1 = y1 + width_10;
		for (j = 0; j < b; j++) {
			if (s_iji(i, j) == 0)
				continue;
			// printf("%d ", j);
			x0 = j * width;
			x1 = (j + 1) * width;
			X0 = x0 + width_10;
			X1 = x1 - width_10;
			// hor. lines: 
			f << "\\put(" << X0 << "," << Y0 << "){\\line(1,0){" << width_8 << "}}\n";
			f << "\\put(" << X0 << "," << Y1 << "){\\line(1,0){" << width_8 << "}}\n";

			// vert. lines: 
			f << "\\put(" << X0 << "," << Y1 << "){\\line(0,1){" << width_8 << "}}\n";
			f << "\\put(" << X1 << "," << Y1 << "){\\line(0,1){" << width_8 << "}}\n";

			}
		// printf("\n");
		}

	f << "\\end{picture}" << endl;
}

void matrix::calc_hash_key(int key_len, hollerith & hash_key, int f_v)
{
	int al_len;
	char *alphabet = NULL;
	char *inc = NULL;
	char *key = NULL;
	int i, j, k, v, b, nb_inc, pr, x, y;
	char c;
	int f_vv = FALSE;
	Vector P;
	int nb_primes = 25;
	
	v = s_m();
	b = s_n();
	nb_inc = 0;
	for (i = 0; i < v; i++) {
		for (j = 0; j < b; j++) {
			if (s_iji(i, j) != 0)
				nb_inc++;
			}
		}
	the_first_n_primes(P, nb_primes);
	al_len = MAXIMUM(256, b);
	alphabet = (char *) new char[al_len + 1];
	inc = (char *) new char[nb_inc + 1];
	key = (char *) new char[key_len + 1];
	//i0 = 0;
	k = 0;
	while (k < al_len) {
		for (i = 0; i < 10; i++) {
			alphabet[k] = '0' + i;
			k++;
			if (k >= al_len)
				break;
			}
		for (i = 0; i < 26; i++) {
			alphabet[k] = 'a' + i;
			k++;
			if (k >= al_len)
				break;
			}
		for (i = 0; i < 26; i++) {
			alphabet[k] = 'A' + i;
			k++;
			if (k >= al_len)
				break;
			}
		} // while
	alphabet[al_len] = 0;
	if (f_vv) {
		cout << "alphabet: " << alphabet << endl;
		}
	
	k = 0;
	for (i = 0; i < v; i++) {
		for (j = 0; j < b; j++) {
			if (s_iji(i, j) == 0)
				continue;
			c = alphabet[j];
			inc[k] = c;
			k++;
			}
		}
	inc[nb_inc] = 0;
	if (f_vv) {
		cout << "incidences: " << inc << endl;
		}
	
	j = 0;
	for (k = 0; k < key_len; k++) {
		pr = P.s_ii(k % 25);
		x = 0;
		for (i = 0; i < nb_inc; i++) {
			y = (int) inc[j];
			x = (x + y) % 256;
			j += pr;
			if (j >= nb_inc)
				j = 0;
			}
		key[k] = alphabet[x];
		// printf("k=%d pr = %d x=%d h[k]=%c\n", k, pr, x, h[k]);
		}
	key[key_len - 1] = 0;
	hash_key.init(key);
	if (f_v) {
		cout << "matrix::calc_hash_key() (len=" << key_len << ") hash=" << hash_key << endl;
		}
	delete [] alphabet;
	delete [] inc;
	delete [] key;
}

int matrix::is_in_center()
{
	int m, n, i, j;
	matrix A;
	integer c;
	
	m = s_m();
	n = s_n();
	A = *this;
	c = A[0][0];
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			integer e;
			
			e = A[i][j];
			if (i != j && e.s_i() != 0)
			{
				return 0;
			}
			if (i == j && e.s_i() != c.s_i())
			{
				return 0;
			}
		}
	}
	return 1;
}

void matrix::power_mod(int r, integer &P, matrix &C)
{
	matrix B;
	int m, n, i, j, l;
	int p = P.s_i();
	
	m = s_m();
	n = s_n();
	if (m != n)
	{
		cout << "matrix::power_mod(): m !=n" << endl;
		exit(1);
	}
	B = *this;
	C = *this;
	B.power_int(r);
	
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++) 
		{
			integer e, c;
			int em;
			
			e = B[i][j];
			l = e.s_i();
			em = l%p;
			c.m_i(em);
			C[i][j] = c;
		}
	}
}

int matrix::proj_order_mod(integer &P)
{
	matrix B;
	int m, n;
	int p = P.s_i();
	int ord = 0;
	
	m = s_m();
	n = s_n();
	B = *this;
	if (m != n)
	{
		cout << "matrix::proj_order_mod() m != n" << endl;
		exit(1);
	}
	if (is_zero())
	{
		ord = 0;
		cout << "is zero matrix!" << endl;
	}
	else
	{
		while (B.is_in_center() == FALSE)
		{
			ord++;
			power_mod(ord, P, B);
			if (ord > p)
			{
				cout << "ERROR: proj_order_mod() order too big" << endl;
				ord = -1;
			}
		}
	}
	return ord;
}



void matrix::PG_rep(domain *dom, permutation &p, int f_action_from_right, int f_modified)
{
	with ww(dom);
	PG_rep(p, f_action_from_right, f_modified);
}

void matrix::PG_rep(permutation &p, int f_action_from_right, int f_modified)
{
	domain *d;
	int m, q, l, i, j;
	Vector v, w;
	
	m = s_m();
	if (!is_finite_field_domain(d)) {
		cout << "matrix::PG_rep() no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	l = nb_PG_elements(m - 1, q);
	p.m_l(l);
	v.m_l_n(m);
	for (i = 0; i < l; i++) {
		if (f_modified)
			v.PG_element_unrank_modified(i);
		else
			v.PG_element_unrank(i);
		if (f_action_from_right)
			w.mult(v, *this);
		else
			w.mult(*this, v);
		if (f_modified)
			w.PG_element_rank_modified(j);
		else
			w.PG_element_rank(j);
		p.m_ii(i, j);
		}
}

void matrix::AG_rep(domain *dom, permutation &p, int f_action_from_right)
{
	with ww(dom);
	AG_rep(p, f_action_from_right);
}

void matrix::AG_rep(permutation &p, int f_action_from_right)
{
	domain *d;
	int m, q, l, i, j;
	Vector v, w;
	
	m = s_m();
	if (!is_finite_field_domain(d)) {
		cout << "matrix::AG_rep() no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	l = nb_AG_elements(m, q);
	p.m_l(l);
	v.m_l_n(m);
	for (i = 0; i < l; i++) {
		v.AG_element_unrank(i);
		if (f_action_from_right)
			w.mult(v, *this);
		else
			w.mult(*this, v);
		w.AG_element_rank(j);
		p.m_ii(i, j);
		}
}

void matrix::MacWilliamsTransform(int n, int q, int f_v)
{
	int i, j;
	
	m_mn(n + 1, n + 1);
	for (i = 0; i <= n; i++) {
		for (j = 0; j <= n; j++) {
			Krawtchouk(n, q, i, j, s_ij(i, j));
			}
		}
	if (f_v) {
		cout << "MacWilliamsTransform(" << n << ", " << q << ")" << endl;
		cout << *this;
		}
}

void matrix::weight_enumerator_brute_force(domain *dom, Vector &v)
{
	with ww(dom);
	domain *dom1 = NULL;
	
	int q = finite_field_domain_order_int(dom1);
	int k, n, i, j, h;
	Vector w, c;
	
	k = s_m();
	n = s_n();
	int l = nb_AG_elements(k, q);

	w.m_l(k);
	v.m_l_n(n + 1);
	for (i = 0; i < l; i++) {
		w.AG_element_unrank(i);
		multiply_vector_from_left(w, c);
		j = c.hamming_weight();
		h = v.s_ii(j);
		h++;
		v.m_ii(j, h);
		}
}

void matrix::Simplex_code_generator_matrix(domain *dom, int k, int f_v)
{
	with ww(dom);
	domain *dom1 = NULL;
	Vector w;
	int i, j;
	
	int q = finite_field_domain_order_int(dom1);
	int n = nb_PG_elements(k - 1, q);
	m_mn(k, n);
	w.m_l_n(k);
	for (j = 0; j < n; j++) {
		w.PG_element_unrank(j);
		for (i = 0; i < k; i++) {
			s_ij(i, j) = w[i];
			}
		}
	if (f_v) {
		cout << "Simplex_code_generator_matrix(" << k << ", " << q << ")" << endl;
		cout << *this;
		}
}

void matrix::PG_design_point_vs_hyperplane(domain *dom, int k, int f_v)
{
	with ww(dom);
	int i, j, l;
	
	int q = dom->order_int();
	l = nb_PG_elements(k, q);
	m_mn_n(l, l);
	Vector v, w;
	discreta_base a;
	
	v.m_l_n(k + 1);
	w.m_l_n(k + 1);
	for (i = 0; i < l; i++) {
		v.PG_element_unrank(i);
		for (j = 0; j < l; j++) {
			w.PG_element_unrank(j);
			v.scalar_product(w, a);
			if (a.is_zero())
				m_iji(i, j, 1);
			}
		}
	if (f_v) {
		cout << "PG_design_point_vs_hyperplane(" << k << ", " << q << ")" << endl;
		cout << *this;
		}
}

void matrix::PG_k_q_design(domain *dom, int k, int f_v, int f_vv)
{
	with ww(dom);
	int ii, i, j, nb_pts, nb_lines, r;
	matrix v, w, z;
	discreta_base a;
		
	int q = dom->order_int();
	nb_pts = nb_PG_elements(k, q);
	nb_lines = nb_PG_lines(k, q);
	if (f_v) {
		cout << "nb_pts=" << nb_pts << " nb_lines=" << nb_lines << endl;
		}
	if (f_vv) {
		cout << "points:" << endl;
		v.m_mn_n(1, k + 1);
		for (i = 0; i < nb_pts;  i++) {
			v.PG_point_unrank(0, 0, 0, 1, k + 1, i);
			cout << i << " : " << v << endl;
			}
		cout << "lines:" << endl;
		w.m_mn_n(k + 1, 2);
		for (j = 0; j < nb_lines;  j++) {
			w.PG_line_unrank(j);
			z = w;
			z.transpose();
			cout << j << " : \n" << z << endl;
			}
		}
	m_mn_n(nb_pts, nb_lines);
	
	v.m_mn_n(k + 1, 1);
	w.m_mn_n(k + 1, 2);
	z.m_mn_n(k + 1, 3);
	for (i = 0; i < nb_pts; i++) {
		v.PG_point_unrank(0, 0, 1, 0, k + 1, i);
		for (j = 0; j < nb_lines; j++) {
			w.PG_line_unrank(j);
			for (ii = 0; ii < k + 1; ii++) {
				z[ii][0] = v[ii][0];
				z[ii][1] = w[ii][0];
				z[ii][2] = w[ii][1];
				}
			r = z.rank();
			if (r == 2)
				m_iji(i, j, 1);
			if (r != 2 && r != 3) {
				cout << "error rank != 2 and rank != 3" << endl;
				cout << "rank=" << r << endl;
				cout << z << endl;
				exit(1);
				}
			}
		}
	if (f_v) {
		cout << "PG_k_q_design(" << k << ", " << q << ")" << endl;
		cout << *this;
		}
}

void matrix::determinant(discreta_base &d, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int n, h, i, j, ii, jj;
	matrix M1;
	discreta_base a;

	if (f_v) {
		cout << "matrix::determinant" << endl;
		}
	if (s_m() != s_n()) {
		cout << "matrix::determinant the matrix is not square" << endl;
		exit(1);
		}
	n = s_n();
	if (n == 1) {
		d = s_ij(0, 0);
		}
	else {
		d = s_ij(0, 0);
		d.zero();
		
		M1.m_mn(n - 1, n - 1);
		for (h = 0; h < n; h++) {
			if (f_v) {
				cout << "matrix::determinant h=" << h << " d=" << d << endl;
				}
			for (i = 0, ii = 0; i < n; i++) {
				if (i == 0) {
					continue;
					}
				for (j = 0, jj = 0; j < n; j++) {
					if (j == h) {
						continue;
						}
					M1.s_ij(ii, jj) = s_ij(i, j);
					jj++;
					}
				ii++;
				}
			if (f_v) {
				cout << "matrix::determinant M1=" << endl << M1 << endl;
				}
			M1.determinant(a, verbose_level);
			if (f_v) {
				cout << "matrix::determinant a=" << a << endl;
				}
			if ((h % 2) == 1) {
				a.negate();
				}
			a.mult_apply(s_ij(0, h));
			d.add_apply(a);
			}
		}
}

void matrix::det(discreta_base & d, int f_v, int f_vv)
{
	matrix A;
	
	A = *this;
	A.det_modify_input_matrix(d, f_v, f_vv);
}

void matrix::det_modify_input_matrix(discreta_base & d, int f_v, int f_vv)
{
	int rk, i;
	int f_special = TRUE;
	int f_complete = FALSE;
	
	Vector base_cols;
	matrix P;
	int f_P = FALSE;

	if (f_v) {
		cout << "in det():" << endl << *this << endl;
		}
	rk = Gauss(f_special, f_complete, base_cols, f_P, P, f_vv);
	if (f_v) {
		cout << "Gauss:" << endl << *this << endl;
		}
	if (rk < s_m()) {
		if (f_v) {
			cout << "singular" << endl;
			}
		}
	d = s_ij(0, 0);
	for (i = 1; i < rk; i++) {
		d *= s_ij(i, i);
		}
	if (f_v) {
		cout << "det = " << d << endl;
		}
}

void determinant_map(discreta_base & x, discreta_base &d)
{
	if (x.s_kind() != MATRIX) {
		cout << "determinant_map() x must be a MATRIX" << endl;
		exit(1);
		}
	matrix & M = x.as_matrix();
	int f_v = FALSE;
	int f_vv = FALSE;
	M.det(d, f_v, f_vv);
}

void matrix::PG_line_rank(int &a, int f_v)
{
	domain *d;
	int q, m, n, l, s, i, a1, a2, a3, nb, ql, pivot_row, pivot_row2 = 0;
	discreta_base x;
	
	if (!is_finite_field_domain(d)) {
		cout << "matrix::PG_line_rank() no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	m = s_m();
	n = s_n();
	if (m <= 0) {
		cout << "matrix::PG_line_rank() matrix not allocated()" << endl;
		exit(1);
		}
	if (n != 2) {
		cout << "matrix::PG_line_rank() matrix not allocated()" << endl;
		exit(1);
		}
	
	pivot_row = 0;
	// do a little Gauss algorithm:
	for (i = m - 1; i >= 0; i--) {
		if (!s_ij(i, 0).is_zero() || !s_ij(i, 1).is_zero()) {
			pivot_row = i;
			if (s_ij(i, 1).is_zero()) {
				for ( ; i >= 0; i--) {
					s_ij(i, 0).swap(s_ij(i, 1));
					}
				}
			break;
			}
		}
	if (f_v) {
		cout << "after permuting:\n" << *this << endl;
		cout << "PG_line_rank() pivot_row=" << pivot_row << endl;
		}
	PG_point_normalize(0, 1, 1, 0, m);
	if (!s_ij(pivot_row, 1).is_one()) {
		cout << "matrix::PG_line_rank() pivot element is not one" << endl;
		exit(1);
		}
	if (!s_ij(pivot_row, 0).is_zero()) {
		discreta_base x;
				
		x = s_ij(pivot_row, 0);
		for (i = pivot_row; i >= 0; i--) {
			discreta_base y;
					
			y = s_ij(i, 1);
			y *= x;
			y.negate();
			s_ij(i, 0) += y;
			}
		}
	PG_point_normalize(0, 0, 1, 0, m);
	if (f_v) {
		cout << "PG_line_rank() after gauss right to left and normalize:\n" << *this << endl;
		}
	for (i = pivot_row - 1; i >= 0; i--) {
		if (!s_ij(i, 0).is_zero()) {
			pivot_row2 = i;
			break;
			}
		}
	if (i < 0) {
		cout << "matrix::PG_line_rank() zero column" << endl;
		exit(1);
		}
	
	// due to normalize, this element must already be one:
	if (!s_ij(pivot_row2, 0).is_one()) {
		cout << "matrix::PG_line_rank() pivot element 2 is not one" << endl;
		exit(1);
		}
	if (!s_ij(pivot_row2, 1).is_zero()) {
		discreta_base x;
				
		x = s_ij(pivot_row2, 1);
		for (i = pivot_row2; i >= 0; i--) {
			discreta_base y;
					
			y = s_ij(i, 0);
			y *= x;
			y.negate();
			s_ij(i, 1) += y;
			}
		}
	if (f_v) {
		cout << "PG_line_rank() after gauss left to right:\n" << *this << endl;
		}
	
	l = pivot_row2;
	if (!s_ij(l, 1).is_zero()) {
		cout << "!s_ij(l, 1).is_zero()" << endl;
		exit(1);
		}
	ql = orbiter::foundations::i_power_j(q, l);
	nb = nb_PG_elements(m - l - 2, q);
	s = ql * ql * nb;
	if (f_v) {
		cout << "l=" << l << " ql=" << ql << " nb=" << nb << " s=" << s << endl;
		}
	PG_point_rank(l + 1, 1, 1, 0, m - l - 1, a3);
	if (f_v) {
		cout << "a3=" << a3 << endl;
		}
	if (l) {
		AG_point_rank(0, 1, 1, 0, l, a2);
		if (f_v) {
			cout << "a2=" << a2 << endl;
			}
		AG_point_rank(0, 0, 1, 0, l, a1);
		if (f_v) {
			cout << "a1=" << a1 << endl;
			}
		}
	else {
		a1 = 0;
		a2 = 0;
		}
	a = (a1 * nb + a3) * ql + a2;
	if (f_v) {
		cout << "a=" << a << endl;
		}
	for (l--; l >= 0; l--) {
		ql = orbiter::foundations::i_power_j(q, l);
		nb = nb_PG_elements(m - l - 2, q);
		s = ql * ql * nb;
		a += s;
		}
}

void matrix::PG_line_unrank(int a)
{
	domain *d;
	int q, m, n, l, s, k, a1, a2, a3, nb, ql;
	
	if (!is_finite_field_domain(d)) {
		cout << "matrix::PG_line_unrank no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	m = s_m();
	n = s_n();
	if (m <= 0) {
		cout << "matrix::PG_line_unrank matrix not allocated" << endl;
		exit(1);
		}
	if (n != 2) {
		cout << "matrix::PG_line_unrank matrix not allocated" << endl;
		exit(1);
		}
	
	
	// cout << "matrix::PG_line_unrank() a=" << a << endl;
	l = 0;
	while (l < m) {
		ql = orbiter::foundations::i_power_j(q, l);
		nb = nb_PG_elements(m - l - 2, q);
		s = ql * ql * nb;
		// cout << "matrix::PG_line_unrank() a=" << a
		//<< " l=" << l << " s=" << s << " ql=" << ql
		//<< " nb=" << nb << endl;
		if (a >= s) {
			a -= s;
			l++;
			continue;
			}
		// cout << "choosing l=" << l << endl;
		// cout << "nb = " << nb << endl;
		a2 = a % ql;
		a -= a2;
		a /= ql;
		a3 = a % nb;
		a -= a3;
		a /= nb;
		a1 = a;
		// cout << "a1=" << a1 << endl;
		// cout << "a2=" << a2 << endl;
		// cout << "a3=" << a3 << endl;

		s_ij(l, 0).one();
		s_ij(l, 1).zero();
		for (k = l + 1; k < m; k++) {
			s_ij(k, 0).zero();
			}

		if (l) {
			AG_point_unrank(0, 0, 1, 0, l, a1);
			AG_point_unrank(0, 1, 1, 0, l, a2);
			}

		PG_point_unrank(l + 1, 1, 1, 0, m - l - 1, a3);
		return;
		}
	cout << "matrix::PG_line_unrank a too large" << endl;
	exit(1);
}

void matrix::PG_point_normalize(int i0, int j0,
		int di, int dj, int length)
{
	int i, j;
	discreta_base a;
	
	j = 0;
	for (i = length - 1; i >= 0; i--) {
		if (!s_ij(i0 + i * di, j0 + j * dj).is_zero()) {
			if (s_ij(i0 + i * di, j0 + i * dj).is_one())
				return;
			a = s_ij(i0 + i * di, j0 + i * dj);
			a.invert();
			for (j = i; j >= 0; j--) {
				s_ij(i0 + j * di, j0 + j * dj) *= a;
				}
			return;
			}
		}
	cout << "matrix::PG_point_normalize zero vector" << endl;
	exit(1);
}

void matrix::PG_point_unrank(int i0, int j0,
		int di, int dj, int length, int a)
{
	domain *d;
	int q, n, l, qhl, k, j, r, a1 = a;
	
	if (!is_finite_field_domain(d)) {
		cout << "matrix::PG_point_unrank no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	n = length;
	if (n <= 0) {
		cout << "matrix::PG_point_unrank n <= 0" << endl;
		exit(1);
		}
	
	
	l = 0;
	qhl = 1;
	while (l < n) {
		if (a >= qhl) {
			a -= qhl;
			qhl *= q;
			l++;
			continue;
			}
		s_ij(i0 + l * di, j0 + l * dj).one();
		for (k = l + 1; k < n; k++) {
			s_ij(i0 + k * di, j0 + k * dj).zero();
			}
		j = 0;
		while (a != 0) {
			r = a % q;
			m_iji(i0 + j * di, j0 + j * dj, r);
			j++;
			a -= r;
			a /= q;
			}
		for ( ; j < l; j++)
			m_iji(i0 + j * di, j0 + j * dj, 0);
		return;
		}
	cout << "matrix::PG_point_unrank() a too large" << endl;
	cout << "length = " << length << endl;
	cout << "a = " << a1 << endl;
	exit(1);
}

void matrix::PG_point_rank(int i0, int j0,
		int di, int dj, int length, int &a)
{
	domain *d;
	int i, j, q, q_power_j, b;
	
	if (!is_finite_field_domain(d)) {
		cout << "matrix::PG_point_rank no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	PG_point_normalize(i0, j0, di, dj, length);
	if (length <= 0) {
		cout << "matrix::PG_point_rank length <= 0" << endl;
		exit(1);
		}
	for (i = length - 1; i >= 0; i--) {
		if (!s_ij(i0 + i * di, j0 + i * dj).is_zero())
			break;
		}
	if (i < 0) {
		cout << "matrix::PG_point_rank zero vector" << endl;
		exit(1);
		}
	if (!s_ij(i0 + i * di, j0 + i * dj).is_one()) {
		cout << "matrix::PG_point_rank vector not normalized" << endl;
		exit(1);
		}

	b = 0;
	q_power_j = 1;
	for (j = 0; j < i; j++) {
		b += q_power_j;
		q_power_j *= q;
		}


	a = 0;
	for (j = i - 1; j >= 0; j--) {
		a += s_iji(i0 + j * di, j0 + j * dj);
		if (j > 0)
			a *= q;
		}
	a += b;
}


void matrix::PG_element_normalize()
// lowest element which is different from zero becomes one in each column
{
	int i, ii, j, m, n;
	discreta_base a;
	
	m = s_m();
	n = s_n();
	for (j = 0; j < n; j++) {
		for (i = m - 1; i >= 0; i--) {
			if (!s_ij(i, j).is_zero()) {
				if (s_ij(i, j).is_one())
					break;
				a = s_ij(i, j);
				a.invert();
				for (ii = i; ii >= 0; ii--) {
					s_ij(ii, j) *= a;
					}
				break;
				}
			}
		if (i == -1) {
			cout << "matrix::PG_element_normalize zero column" << endl;
			exit(1);
			}
		}
}

void matrix::AG_point_rank(int i0, int j0,
		int di, int dj, int length, int &a)
{
	domain *d;
	int q, i;
	
	if (!is_finite_field_domain(d)) {
		cout << "matrix::AG_point_rank no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	if (length <= 0) {
		cout << "matrix::AG_point_rank length <= 0" << endl;
		exit(1);
		}
	a = 0;
	for (i = length - 1; i >= 0; i--) {
		a += s_iji(i0 + i * di, j0 + i * dj);
		if (i > 0)
			a *= q;
		}
}

void matrix::AG_point_unrank(int i0, int j0,
		int di, int dj, int length, int a)
{
	domain *d;
	int q, i, b;
	
	if (!is_finite_field_domain(d)) {
		cout << "matrix::AG_point_unrank no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	if (length <= 0) {
		cout << "matrix::AG_point_unrank length <= 0" << endl;
		exit(1);
		}
	for (i = 0; i < length; i++) {
		b = a % q;
		m_iji(i0 + i * di, j0 + i * dj, b);
		a /= q;
		}
}

int nb_PG_lines(int n, int q)
{
	int l, ql, nb, s, a = 0, m;
	
	m = n + 1;
	for (l = 0; l < m; l++) {
		ql = orbiter::foundations::i_power_j(q, l);
		nb = nb_PG_elements(m - l - 2, q);
		s = ql * ql * nb;
		a += s;
		}
	return a;
}

#if 0
void matrix::canon(int f_row_decomp, Vector & row_decomp, 
	int f_col_decomp, Vector & col_decomp, 
	int f_group, perm_group & G, 
	permutation & p, permutation & q, 
	int f_get_aut_group, int f_aut_group_on_lexleast, Vector & aut_gens, discreta_base &ago,
	int f_v, int f_vv, int f_vvv, int f_vvvv, int f_tree_file)
{
	permutation canonical_form;
	Vector aut_generators;
	matrix01_classify(*this, 
		f_row_decomp, row_decomp, 
		f_col_decomp, col_decomp, 
		canonical_form, aut_generators, ago, f_v, f_vv, f_vvv, f_vvvv, f_tree_file);
	if (f_get_aut_group) {
		if (f_aut_group_on_lexleast) {
			vec_conjugate(aut_generators, canonical_form);
			}
		// vec_generators_restrict_to_subset(aut_generators, 0, s_m());
		aut_gens = aut_generators;
		}
	canonical_form.restrict_to_subset(p, 0, s_m());
	canonical_form.restrict_to_subset(q, s_m(), s_n());
	
}

void matrix::canon_partition_backtrack(int f_row_decomp, Vector & row_decomp, 
	int f_col_decomp, Vector & col_decomp, 
	int f_group, perm_group & G, 
	permutation & p, permutation & q, 
	int f_get_aut_group, int f_aut_group_on_lexleast, Vector & aut_gens, discreta_base &ago,
	int f_v, int f_vv, int f_vvv, int f_vvvv, int f_tree_file)
{
	permutation canonical_form;
	Vector aut_generators;
	partition_backtrack_canonize(*this, 
		f_row_decomp, row_decomp, 
		f_col_decomp, col_decomp, 
		canonical_form, aut_generators, ago, f_v, f_vv, f_vvv, f_vvvv, f_tree_file);
	if (f_get_aut_group) {
		if (f_aut_group_on_lexleast) {
			vec_conjugate(aut_generators, canonical_form);
			}
		// vec_generators_restrict_to_subset(aut_generators, 0, s_m());
		aut_gens = aut_generators;
		}
	canonical_form.restrict_to_subset(p, 0, s_m());
	canonical_form.restrict_to_subset(q, s_m(), s_n());
	
}
#endif

void matrix::canon_nauty(int f_row_decomp, Vector & row_decomp, 
	int f_col_decomp, Vector & col_decomp, 
	int f_group, perm_group & G, 
	permutation & p, permutation & q, 
	int f_get_aut_group, int f_aut_group_on_lexleast, Vector & aut_gens, 
	int f_v, int f_vv, int f_vvv)
{
#if 0
	int *M;
	int m, n, i, j, a;
	
	cout << "matrix::canon_nauty()" << endl;
	m = s_m();
	n = s_n();
	M = new int[m * n];
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = s_iji(i, j);
			M[i * n + j] = a;
			cout << a;
			}
		cout << endl;
		}
	//nauty_interface(M, m, n);
	
	int *labeling;
	int *partition;
	int *Aut;
	int Aut_counter;
	int *Base;
	int Base_length;
	int *Transversal_length;
	int Ago;
	
	
	labeling = new int[m + n];
	partition = new int[m + n];


	for (i = 0; i < m + n; i++) {
		labeling[i] = i;
		partition[i] = 1;
		}
	partition[m - 1] = 0;
	
	Aut = new int[ (m + n) * (m + n)];
	Base = new int[m + n];
	Transversal_length = new int[m + n];
	
	nauty_interface_matrix(M, m, n, 
		labeling, partition, 
		Aut, Aut_counter, 
		Base, Base_length, 
		Transversal_length, Ago);
	
	delete [] M;
	delete [] labeling;
	delete [] partition;
	delete [] Aut;
	delete [] Base;
	delete [] Transversal_length;
#endif
}

#if 0
void matrix::canon_tonchev(int f_row_decomp, Vector & row_decomp, 
	int f_col_decomp, Vector & col_decomp, 
	int f_group, perm_group & G, 
	permutation & p, permutation & q, 
	int f_get_aut_group, int f_aut_group_on_lexleast, Vector & aut_gens, 
	int f_v, int f_vv, int f_vvv)
{
	cout << "matrix::canon_tonchev() tonchev program not accessible" << endl;
#if 0
	int *M;
	int m, n, i, j, a;
	
	m = s_m();
	n = s_n();
	M = new int[m * n];
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a = s_iji(i, j);
			M[i * n + j] = a;
			cout << a;
			}
		cout << endl;
		}
	tonchev_interface(M, m, n);
	delete [] M;
#endif
}
#endif

void matrix::save_as_geometry(int number, char *label)
{
	hollerith h;
	geometry G;
	G.allocate_geometry();
	G.number() = number;
	G.label().init(label);
	G.X() = *this;

	Vector labelling_P, labelling_B;
	int i, m, n;
	m = s_m();
	n = s_n();
	labelling_P.m_l_n(m);
	labelling_B.m_l_n(n);
	for (i = 0; i < m; i++) {
		labelling_P.m_ii(i, i + 1);
		}
	for (i = 0; i < n; i++) {
		labelling_B.m_ii(i, i + 1);
		}
	G.point_labels() = labelling_P;
	G.block_labels() = labelling_B;
	
	h.init(label);
	h.append(".geo");
	ofstream f(h.s());
	f << G << endl;
}


void matrix::save_as_inc_file(char *fname)
{
	int i, j = 0, m, n, nb_X = 0;
	m = s_m();
	n = s_n();
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (s_iji(i, j)) 
				nb_X++;
			}
		}
	ofstream f(fname);
	f << i << " " << j << " " << nb_X << endl;
	save_as_inc(f);
	f << "-1 1" << endl;
}

void matrix::save_as_inc(ofstream &f)
{
	int i, j, m, n;
	m = s_m();
	n = s_n();
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (s_iji(i, j)) 
				f << i * n + j << " ";
			}
		}
	f << endl;
}


}}
