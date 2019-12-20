// geometry.cpp
//
// Anton Betten
// 24.04.2000
// moved from D2 to ORBI Nov 15, 2007
// added print_inc() Dec 21, 2010

#include "orbiter.h"
#include <string.h>

using namespace std;


namespace orbiter {
namespace discreta {


#undef GEOMETRY_CHANGE_KIND_VERBOSE
#undef GEOMETRY_COPY_VERBOSE


geometry::geometry() : Vector()
{
	k = GEOMETRY;
	self.vector_pointer = NULL;
}

#if 0
geometry::geometry(Vector & gen) : Vector()
{
	k = GEOMETRY;
	self.vector_pointer = NULL;
	init(gen);
}

geometry::geometry(int n) : Vector()
{
	Vector gen;
	permutation p;
	
	k = GEOMETRY;
	self.vector_pointer = NULL;
	p.m_l(n);
	p.one();
	gen.m_l(1);
	gen.s_i(0) = p;
	init(gen);
}
#endif

void geometry::allocate_geometry()
{
	// cout << "pc_presentation::allocate_geometry()\n";
	// c_kind(VECTOR);
	Vector::m_l(21);
	c_kind(GEOMETRY);
	Vector::s_i(0).change_to_integer();
	Vector::s_i(1).change_to_hollerith();
	Vector::s_i(2).change_to_matrix();
	Vector::s_i(3).change_to_integer();
	Vector::s_i(4).change_to_vector();
	Vector::s_i(5).change_to_vector();
	Vector::s_i(6).change_to_integer();
	Vector::s_i(7).change_to_vector();
	Vector::s_i(8).change_to_integer();
	Vector::s_i(9).change_to_vector();
	Vector::s_i(10).change_to_integer();
	Vector::s_i(11).change_to_vector();
	Vector::s_i(12).change_to_integer();
	Vector::s_i(13).change_to_vector();
	Vector::s_i(14).change_to_integer();
	Vector::s_i(15).change_to_permutation();
	Vector::s_i(16).change_to_integer();
	Vector::s_i(17).change_to_permutation();
	
	Vector::s_i(18).change_to_integer();
	Vector::s_i(19).change_to_vector();
	Vector::s_i(20).change_to_integer();
	
	number() = 0;
	label().init("");
	X().m_mn(0, 0);
	f_incidence_matrix() = TRUE;
	point_labels().m_l(0);
	block_labels().m_l(0);
	f_row_decomp() = FALSE;
	f_col_decomp() = FALSE;
	f_ddp() = FALSE;
	f_ddb() = FALSE;
	f_canonical_labelling_points() = FALSE;
	f_canonical_labelling_blocks() = FALSE;
	f_aut_gens() = FALSE;
	aut_gens().m_l(0);
	ago().m_i_i(0);
}

geometry::geometry(const discreta_base &x)
	// copy constructor:    this := x
{
	// cout << "geometry::copy constructor for object: " << const_cast<discreta_base &>(x) << "\n";
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

geometry& geometry::operator = (const discreta_base &x)
	// copy assignment
{
	// cout << "geometry::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void geometry::settype_geometry()
{
	OBJECTSELF s;
	
	s = self;
	new(this) geometry;
	self = s;
	k = GEOMETRY;
}

geometry::~geometry()
{
	freeself_geometry();
}

void geometry::freeself_geometry()
{
	// cout << "geometry::freeself_geometry()\n";
	freeself_vector();
}

kind geometry::s_virtual_kind()
{
	return GEOMETRY;
}

void geometry::copyobject_to(discreta_base &x)
{
#ifdef GEOMETRY_COPY_VERBOSE
	cout << "geometry::copyobject_to()\n";
	print_as_vector(cout);
#endif
	Vector::copyobject_to(x);
	x.as_geometry().settype_geometry();
#ifdef GEOMETRY_COPY_VERBOSE
	x.as_geometry().print_as_vector(cout);
#endif
}

ostream& geometry::print(ostream& ost)
{
	if (current_printing_mode() == printing_mode_latex) {
		print_latex(ost);
		}
	else {
		print_ascii(ost);
		}
	return ost;
}

void geometry::print_latex(ostream& ost)
{
	print_head_latex(ost);
	print_incma_text_latex(ost);
	print_labellings_latex(ost);
	print_incma_latex_picture(ost);
	
}

void geometry::print_head_latex(ostream& ost)
{
	char geo_label_tex[1000];

	texable_string(label().s(), geo_label_tex);

	ost << "\\subsubsection{geo no " << number() << " " << geo_label_tex << "}" << endl;

}

void geometry::print_incma_text_latex(ostream& ost)
{
	ost << "{\\tt\n";
	ost << "\\noindent%\n";
	X().incma_print_ascii(ost, TRUE /* f_tex */, 
		f_row_decomp(), row_decomp(), 
		f_col_decomp(), col_decomp());
	ost << "}% \n";
}

void geometry::print_labellings_latex(ostream& ost)
{
	ost << "\\par\\noindent\n";
	ost << "labelling of points: \n";
	ost << "$" << point_labels() << "$\\\\" << endl;
	ost << "labelling of blocks: \n";
	ost << "$" << block_labels() << "$\\\\" << endl;
}

void geometry::print_incma_latex_picture(ostream& ost)
{
	ost << "\\par\\noindent\n";
	X().incma_print_latex(ost, 
		f_row_decomp(), row_decomp(), 
		f_col_decomp(), col_decomp(), 
		TRUE /* f_labelling_points */, point_labels(), 
		TRUE /* f_labelling_blocks */, block_labels());
}

void geometry::print_inc(ostream &ost)
{
	int v, b, nb_inc, i, j, a;
	
	v = X().s_m();
	b = X().s_n();
	nb_inc = 0;
	if (v > 0 && b > 0) {
		ost << "GEOMETRY " << number() << " " << label().s() << endl;
		if (f_incidence_matrix()) {
			for (i = 0; i < v; i++) {
				for (j = 0; j < b; j++) {
					a = X().s_iji(i, j);
					if (a) {
						nb_inc++;
						}
					}
				}
			ost << "v=" << v << " b=" << b << " nb_inc=" << nb_inc << endl;
			for (i = 0; i < v; i++) {
				for (j = 0; j < b; j++) {
					a = X().s_iji(i, j);
					if (a)
						ost << i * b + j << " ";
					}
				}
			ost << endl;
			}
		}
}

void geometry::print_inc_only(ostream &ost)
{
	int v, b, nb_inc, i, j, a;
	
	nb_inc = 0;
	v = X().s_m();
	b = X().s_n();
	if (v > 0 && b > 0) {
		//ost << "GEOMETRY " << number() << " " << label().s() << endl;
		if (f_incidence_matrix()) {
			for (i = 0; i < v; i++) {
				for (j = 0; j < b; j++) {
					a = X().s_iji(i, j);
					if (a) {
						nb_inc++;
						}
					}
				}
			//ost << "v=" << v << " b=" << b << " nb_inc=" << nb_inc << endl;
			for (i = 0; i < v; i++) {
				for (j = 0; j < b; j++) {
					a = X().s_iji(i, j);
					if (a)
						ost << i * b + j << " ";
					}
				}
			ost << 0 << endl;
			}
		}
}

void geometry::print_inc_header(ostream &ost)
{
	int v, b, nb_inc, i, j, a;
	
	nb_inc = 0;
	v = X().s_m();
	b = X().s_n();
	if (v > 0 && b > 0) {
		//ost << "GEOMETRY " << number() << " " << label().s() << endl;
		if (f_incidence_matrix()) {
			for (i = 0; i < v; i++) {
				for (j = 0; j < b; j++) {
					a = X().s_iji(i, j);
					if (a) {
						nb_inc++;
						}
					}
				}
			ost << v << " " << b << " " << nb_inc << endl;
			}
		}
}

void geometry::print_ascii(ostream& ost)
{
	int v, b, i, j, a, l;
	
	v = X().s_m();
	b = X().s_n();
	if (v > 0 && b > 0) {
		ost << "GEOMETRY " << number() << " " << label().s() << endl;
		ost << "v=" << v << " b=" << b << endl;
		if (f_incidence_matrix()) {
			ost << "INCIDENCE_MATRIX" << endl;
			for (i = 0; i < v; i++) {
				for (j = 0; j < b; j++) {
					a = X().s_iji(i, j);
					if (a)
						ost << "X";
					else
						ost << ".";
					}
				ost << endl;
				}
			}
		else {
			ost << "INTEGER_MATRIX" << endl;
			ost << X();
			}
		ost << "LABELLING_OF_POintS" << endl;
		point_labels().print_unformatted(ost);
		ost << endl;
		ost << "LABELLING_OF_BLOCKS" << endl;
		block_labels().print_unformatted(ost);
		ost << endl;
		if (f_row_decomp()) {
			ost << "DECOMPOSITION_OF_POintS" << endl;
			ost << row_decomp().s_l() << " ";
			row_decomp().print_unformatted(ost);
			ost << endl;
			}
		if (f_col_decomp()) {
			ost << "DECOMPOSITION_OF_BLOCKS" << endl;
			ost << col_decomp().s_l() << " ";
			col_decomp().print_unformatted(ost);
			ost << endl;
			}
		if (f_ddp()) {
			ost << "DDP" << endl;
			ddp().print_unformatted(ost);
			ost << endl;
			}
		if (f_ddb()) {
			ost << "DDB" << endl;
			ddb().print_unformatted(ost);
			ost << endl;
			}
		if (f_canonical_labelling_points()) {
			ost << "CANONICAL_LABELLING_OF_POintS" << endl;
			canonical_labelling_points().print_unformatted(ost);
			ost << endl;
			}
		if (f_canonical_labelling_blocks()) {
			ost << "CANONICAL_LABELLING_OF_BLOCKS" << endl;
			canonical_labelling_blocks().print_unformatted(ost);
			ost << endl;
			}
		if (f_aut_gens()) {
			ost << "AUT_GENS (group order " << ago() << ")" << endl;
			l = aut_gens().s_l();
			ost << l << endl;
			for (i = 0; i < l; i++) {
				int ll = aut_gens().s_i(i).as_permutation().s_l();
				for (j = 0; j < ll; j++) {
					ost << aut_gens().s_i(i).as_permutation()[j] << " ";
					}
				// aut_gens().s_i(i).as_permutation().print_unformatted(ost);
				ost << endl;
				}
			}
		
		ost << "END" << endl << endl;
		}
	else {
		ost << "#GEOMETRY (not allocated)" << endl;
		}
}

#define MYBUFSIZE 50000

#include <stdio.h>

int geometry::scan(istream& f)
{
	int nr;
	char buf[MYBUFSIZE];
	char str1[MYBUFSIZE];
	char *p_str;
	
	//cout << "in geometry::scan()" << endl;
	while (TRUE) {
		if (f.eof()) {
			return FALSE;
			}
		f.getline(buf, sizeof(buf));
		//cout << "read line:" << buf << endl;
		if (strncmp(buf, "GEOMETRY", 8) == 0)
			break;
		}
	

	p_str = &buf[9];
	//cout << "GEOMETRY " << p_str << endl;
	s_scan_int(&p_str, &nr);
	//cout << "number = " << nr << endl;
	str1[0] = 0;
	s_scan_token_arbitrary(&p_str, str1);
	//cout << "label = " << str1 << endl;

	scan_body(f, nr, str1);
	
	return TRUE;

}

void geometry::scan_body(istream& f, int geo_nr, char *geo_label)
{
	int v, b = 0, i, j, a, a1, l;
	char buf[MYBUFSIZE];
	char *p_str;
	
	v = -1;
	X().m_mn(0, 0);
	point_labels().m_l(0);
	block_labels().m_l(0);
	f_row_decomp() = FALSE;
	f_col_decomp() = FALSE;
	f_ddp() = FALSE;
	f_ddb() = FALSE;
	f_canonical_labelling_points() = FALSE;
	f_canonical_labelling_blocks() = FALSE;
	f_aut_gens() = FALSE;
	aut_gens().m_l(0);
	ago().m_i_i(0);

	number() = geo_nr;
	label().init(geo_label);
	cout << "reading GEOMETRY " << number() << " " << label() << endl;
	
	while (TRUE) {
		if (f.eof()) {
			cout << "geometry::scan() primature end of file" << endl;
			exit(1);
			}

		f.getline(buf, sizeof(buf));
		if (strncmp(buf, "v=", 2) == 0) {
			sscanf(buf, "v=%d b=%d", &v, &b);
			point_labels().m_l(v);
			for (i = 0; i < v; i++) {
				point_labels().m_ii(i, i);
				}
			block_labels().m_l(b);
			for (i = 0; i < b; i++) {
				block_labels().m_ii(i, i);
				}
			}
		else if (strncmp(buf, "INCIDENCE_MATRIX", 16) == 0) {
			// cout << "reading INCIDENCE_MATRIX" << endl;
			X().m_mn_n(v, b);
			for (i = 0; i < v; i++) {
				if (f.eof()) {
					cout << "geometry::scan() primature end of file" << endl;
					exit(1);
					}

				f.getline(buf, sizeof(buf));
				for (j = 0; j < b; j++) {
					if (buf[j] == 'X') {
						X().m_iji(i, j, 1);
						}
					}
				}
			f_incidence_matrix() = TRUE;
			}
		else if (strncmp(buf, "INTEGER_MATRIX", 16) == 0) {
			// cout << "reading INTEGER_MATRIX" << endl;
			X().m_mn_n(v, b);
			for (i = 0; i < v; i++) {
				if (f.eof()) {
					cout << "geometry::scan() primature end of file" << endl;
					exit(1);
					}
				for (j = 0; j < b; j++) {
					f >> a;
					X().m_iji(i, j, a);
					}
				}
			f_incidence_matrix() = FALSE;
			}
		else if (strncmp(buf, "LABELLING_OF_POintS", 19) == 0) {
			// cout << "reading LABELLING_OF_POintS" << endl;
			if (f.eof()) {
				cout << "geometry::scan() primature end of file" << endl;
				exit(1);
				}
			f.getline(buf, sizeof(buf));
			p_str = buf;
			point_labels().m_l(v);
			for (i = 0; i < v; i++) {
				s_scan_int(&p_str, &a);
				point_labels().m_ii(i, a);
				}
			}
		else if (strncmp(buf, "LABELLING_OF_BLOCKS", 19) == 0) {
			// cout << "reading LABELLING_OF_BLOCKS" << endl;
			if (f.eof()) {
				cout << "geometry::scan() primature end of file" << endl;
				exit(1);
				}
			f.getline(buf, sizeof(buf));
			p_str = buf;
			block_labels().m_l(b);
			for (i = 0; i < b; i++) {
				s_scan_int(&p_str, &a);
				block_labels().m_ii(i, a);
				}
			}
		else if (strncmp(buf, "DECOMPOSITION_OF_POintS", 23) == 0) {
			// cout << "reading DECOMPOSITION_OF_POintS" << endl;
			if (f.eof()) {
				cout << "geometry::scan() primature end of file" << endl;
				exit(1);
				}
			f.getline(buf, sizeof(buf));
			p_str = buf;
			s_scan_int(&p_str, &a);
			f_row_decomp() = TRUE;
			row_decomp().m_l(a);
			for (i = 0; i < a; i++) {
				s_scan_int(&p_str, &a1);
				row_decomp().m_ii(i, a1);
				}
			}
		else if (strncmp(buf, "DECOMPOSITION_OF_BLOCKS", 23) == 0) {
			// cout << "reading DECOMPOSITION_OF_BLOCKS" << endl;
			if (f.eof()) {
				cout << "geometry::scan() primature end of file" << endl;
				exit(1);
				}
			f.getline(buf, sizeof(buf));
			p_str = buf;
			s_scan_int(&p_str, &a);
			f_col_decomp() = TRUE;
			col_decomp().m_l(a);
			for (i = 0; i < a; i++) {
				s_scan_int(&p_str, &a1);
				col_decomp().m_ii(i, a1);
				}
			}
		else if (strncmp(buf, "DDP", 3) == 0) {
			// cout << "reading DDP" << endl;
			if (f.eof()) {
				cout << "geometry::scan() primature end of file" << endl;
				exit(1);
				}
			f.getline(buf, sizeof(buf));
			p_str = buf;
			l = (v * (v - 1)) >> 1;
			f_ddp() = TRUE;
			ddp().m_l(l);
			for (i = 0; i < l; i++) {
				s_scan_int(&p_str, &a);
				ddp().m_ii(i, a);
				}
			}
		else if (strncmp(buf, "DDB", 3) == 0) {
			// cout << "reading DDB" << endl;
			if (f.eof()) {
				cout << "geometry::scan() primature end of file" << endl;
				exit(1);
				}
			f.getline(buf, sizeof(buf));
			p_str = buf;
			l = (b * (b - 1)) >> 1;
			f_ddb() = TRUE;
			ddb().m_l(l);
			for (i = 0; i < l; i++) {
				s_scan_int(&p_str, &a);
				ddb().m_ii(i, a);
				}
			}
		else if (strncmp(buf, "CANONICAL_LABELLING_OF_POintS", 29) == 0) {
			// cout << "reading CANONICAL_LABELLING_OF_POintS" << endl;
			if (f.eof()) {
				cout << "geometry::scan() primature end of file" << endl;
				exit(1);
				}
			f.getline(buf, sizeof(buf));
			p_str = buf;
			f_canonical_labelling_points() = TRUE;
			canonical_labelling_points().m_l(v);
			for (i = 0; i < v; i++) {
				s_scan_int(&p_str, &a);
				canonical_labelling_points()[i] = a;
				}
			}
		else if (strncmp(buf, "CANONICAL_LABELLING_OF_BLOCKS", 29) == 0) {
			// cout << "reading CANONICAL_LABELLING_OF_BLOCKS" << endl;
			if (f.eof()) {
				cout << "geometry::scan() primature end of file" << endl;
				exit(1);
				}
			f.getline(buf, sizeof(buf));
			p_str = buf;
			f_canonical_labelling_blocks() = TRUE;
			canonical_labelling_blocks().m_l(b);
			for (i = 0; i < b; i++) {
				s_scan_int(&p_str, &a);
				canonical_labelling_blocks()[i] = a;
				}
			}
		else if (strncmp(buf, "AUT_GENS", 8) == 0) {
			// cout << "reading AUT_GENS" << endl;
			f_aut_gens() = TRUE;
			sscanf(buf, "AUT_GENS (group order %d)", &a);
			ago().m_i_i(a);
			if (f.eof()) {
				cout << "geometry::scan() primature end of file" << endl;
				exit(1);
				}
			f.getline(buf, sizeof(buf));
			sscanf(buf, "%d", &l);
			aut_gens().m_l(l);
			for (i = 0; i < l; i++) {
				aut_gens().s_i(i).change_to_permutation();
				aut_gens().s_i(i).as_permutation().m_l(v + b);
				if (f.eof()) {
					cout << "geometry::scan() primature end of file" << endl;
					exit(1);
					}
				f.getline(buf, sizeof(buf));
				p_str = buf;
				for (j = 0; j < v + b; j++) {
					s_scan_int(&p_str, &a);
#if 0
					if (a == 0) // old files have generators only of degree v, i.e. action on points
						a = j;
#endif
					aut_gens().s_i(i).as_permutation()[j] = a;
					}
				}
			}
		else if (strncmp(buf, "END", 3) == 0) {
			break;
			}
		
		}
}

void geometry::transpose()
{
	int a;
	
	X().transpose();
	point_labels().swap(block_labels());
	a = f_row_decomp();
	f_row_decomp() = f_col_decomp();
	f_col_decomp() = a;
	row_decomp().swap(col_decomp());
	f_ddp() = FALSE;
	f_ddb() = FALSE;
	f_canonical_labelling_points() = FALSE;
	f_canonical_labelling_blocks() = FALSE;
	canonical_labelling_points().m_l(0);
	canonical_labelling_blocks().m_l(0);
	f_aut_gens() = FALSE;
	aut_gens().m_l(0);
}

static int test_special_matrix(discreta_matrix &Y, int diag, int off_diag)
{
	int v, y, i, j;
	
	v = Y.s_m();
	for (i = 0; i < v; i++) {
		for (j = 0; j < v; j++) {
			y = Y.s_iji(i, j);
			if (i == j) {
				if (y != diag)
					return FALSE;
				}
			else {
				if (y != off_diag)
					return FALSE;
				}
			}
		}
	return TRUE;
}

int geometry::is_2design(int &r, int &lambda, int f_v)
{
	discreta_matrix Xt, Y;
	int a, b;
	
	Xt = X();
	Xt.transpose();
	Y.mult(X(), Xt);
	b = Y.s_iji(0, 0);
	a = Y.s_iji(0, 1);
	if (test_special_matrix(Y, b, a)) {
		lambda = a;
		r = b;
		if (f_v) {
			cout << "is a 2-design with r=" << r << " lambda=" << lambda << endl;
			}
		return TRUE;
		}
	else {
		if (f_v) {
			cout << "is NOT a 2-design" << endl;
			}
		return FALSE;
		}
}

#if 0
void geometry::calc_lexleast_and_autgroup(int f_v, int f_vv, int f_print_backtrack_points)
{
	perm_group G;
	permutation p, q;
	Vector autgens;
	
	X().lexleast_incidence_matrix(TRUE /* f_on_rows */, 
		f_row_decomp(), row_decomp(), 
		f_col_decomp(), col_decomp(), 
		f_ddp(), ddp(), 
		f_ddb(), ddb(), 
		FALSE /* f_group */, G, 
		p, q, 
		f_print_backtrack_points, 
		TRUE /* f_get_aut_group */, FALSE /* f_aut_group_on_lexleast */, autgens, 
		f_v, f_vv);
	f_canonical_labelling_points() = TRUE;
	canonical_labelling_points() = p;
	f_canonical_labelling_blocks() = TRUE;
	canonical_labelling_blocks() = q;
	f_aut_gens() = TRUE;
	aut_gens() = autgens;
	{
	perm_group A(autgens);
	A.group_order(ago());
	if (f_v) {
		cout << "ago=" << ago() << endl;
		// cout << A << endl;
		}
	}
}

void geometry::calc_canon_and_autgroup(int f_v, int f_vv, int f_vvv, int f_vvvv, 
	int f_print_backtrack_points, int f_tree_file)
{
	perm_group G;
	permutation p, q, pv, qv;
	Vector autgens;
	
	X().canon(f_row_decomp(), row_decomp(), 
		f_col_decomp(), col_decomp(), 
		FALSE /* f_group */, G, 
		p, q, 
		TRUE /* f_get_aut_group */, FALSE /* f_aut_group_on_lexleast */, autgens, ago(), 
		f_v, f_vv, f_vvv, f_vvvv, f_tree_file);
	p.invert_to(pv);
	q.invert_to(qv);
	f_canonical_labelling_points() = TRUE;
	canonical_labelling_points() = pv;
	f_canonical_labelling_blocks() = TRUE;
	canonical_labelling_blocks() = qv;

	f_aut_gens() = TRUE;
	aut_gens() = autgens;

#if 0
	{
	if (autgens.s_l() > 40) {
		autgens.realloc(40);
		}
	perm_group A(autgens);
	A.group_order(ago());
	}
	if (f_v) {
		cout << "ago=" << ago() << endl;
		// cout << A << endl;
		}
#endif
}

void geometry::calc_canon_and_autgroup_partition_backtrack(int f_v, int f_vv, int f_vvv, int f_vvvv, 
	int f_print_backtrack_points, int f_tree_file)
{
	perm_group G;
	permutation p, q, pv, qv;
	Vector autgens;
	
	X().canon_partition_backtrack(f_row_decomp(), row_decomp(), 
		f_col_decomp(), col_decomp(), 
		FALSE /* f_group */, G, 
		p, q, 
		TRUE /* f_get_aut_group */, FALSE /* f_aut_group_on_lexleast */, autgens, ago(), 
		f_v, f_vv, f_vvv, f_vvvv, f_tree_file);
	p.invert_to(pv);
	q.invert_to(qv);
	f_canonical_labelling_points() = TRUE;
	canonical_labelling_points() = pv;
	f_canonical_labelling_blocks() = TRUE;
	canonical_labelling_blocks() = qv;

	f_aut_gens() = TRUE;
	aut_gens() = autgens;

#if 0
	{
	if (autgens.s_l() > 40) {
		autgens.realloc(40);
		}
	perm_group A(autgens);
	A.group_order(ago());
	}
	if (f_v) {
		cout << "ago=" << ago() << endl;
		// cout << A << endl;
		}
#endif
}
#endif

void geometry::calc_canon_nauty(int f_v, int f_vv, int f_vvv)
{
	perm_group G;
	permutation p, q;
	Vector autgens;
	
	X().canon_nauty(f_row_decomp(), row_decomp(), 
		f_col_decomp(), col_decomp(), 
		FALSE /* f_group */, G, 
		p, q, 
		TRUE /* f_get_aut_group */, FALSE /* f_aut_group_on_lexleast */, autgens, 
		f_v, f_vv, f_vvv);
	
#if 0
	f_canonical_labelling_points() = TRUE;
	canonical_labelling_points() = p;
	f_canonical_labelling_blocks() = TRUE;
	canonical_labelling_blocks() = q;

	f_aut_gens() = TRUE;
	aut_gens() = autgens;
	{
	if (autgens.s_l() > 40) {
		autgens.realloc(40);
		}
	perm_group A(autgens);
	A.group_order(ago());
	if (f_v) {
		cout << "ago=" << ago() << endl;
		// cout << A << endl;
		}
	}
#endif
}

#if 0
void geometry::calc_canon_tonchev(int f_v, int f_vv, int f_vvv)
{
	perm_group G;
	permutation p, q;
	Vector autgens;
	
	X().canon_tonchev(f_row_decomp(), row_decomp(), 
		f_col_decomp(), col_decomp(), 
		FALSE /* f_group */, G, 
		p, q, 
		TRUE /* f_get_aut_group */, FALSE /* f_aut_group_on_lexleast */, autgens, 
		f_v, f_vv, f_vvv);
	
#if 0
	f_canonical_labelling_points() = TRUE;
	canonical_labelling_points() = p;
	f_canonical_labelling_blocks() = TRUE;
	canonical_labelling_blocks() = q;

	f_aut_gens() = TRUE;
	aut_gens() = autgens;
	{
	if (autgens.s_l() > 40) {
		autgens.realloc(40);
		}
	perm_group A(autgens);
	A.group_order(ago());
	if (f_v) {
		cout << "ago=" << ago() << endl;
		// cout << A << endl;
		}
	}
#endif
}
#endif

void geometry::get_lexleast_X(discreta_matrix & X0)
{
	if (!f_canonical_labelling_points() || !f_canonical_labelling_blocks()) {
		cout << "geometry::get_lexleast_X() canonical labelling not available" << endl;
		exit(1);
		}
	X0 = X();
	X0.apply_perms(TRUE /* f_row_perm */, canonical_labelling_points(), 
		TRUE /* f_col_perm */, canonical_labelling_blocks());
}

// #define BUFSIZE 50000

int search_geo_file(discreta_matrix & X0, char *fname, int geo_nr, char *geo_label, int f_v)
{
	char buf[MYBUFSIZE];
	ifstream f(fname);
	geometry G;
	char *p_str;
	int geo_nr1;
	char geo_label1[MYBUFSIZE];
	
	G.allocate_geometry();
	if (!f) {
		cout << "error opening file " << fname << endl;
		exit(1);
		}
	
	while (TRUE) {
		if (f.eof()) {
			break;
			}
		f.getline(buf, sizeof(buf));
		if (buf[0] == '#')
			continue;
		
		if (strncmp(buf, "GEOMETRY", 8) != 0)
			continue;
		
		p_str = &buf[9];
		s_scan_int(&p_str, &geo_nr1);
		geo_label1[0] = 0;
		s_scan_token_arbitrary(&p_str, geo_label1);
		if (f_v) {
			cout << "checking GEOMETRY " << geo_nr1 << " " << geo_label1 << endl;
			}
		if (geo_nr1 == geo_nr && strcmp(geo_label1, geo_label) == 0) {
			
			G.scan_body(f, geo_nr, geo_label);
			discreta_matrix Y0;
			
			G.get_lexleast_X(Y0);
			if (X0.compare_with(Y0) == 0) {
				if (f_v) {
					cout << "incidence matrices equal" << endl;
					}
				return TRUE;
				}
			else {
				if (f_v) {
					cout << "incidence matrices differ" << endl;
					}
				return FALSE;
				}
			
			}
		else {
			if (geo_nr1 != geo_nr) {
				if (f_v) {
					cout << "wrong geo_nr" << endl;
					}
				}
			if (strcmp(geo_label1, geo_label) != 0) {
				if (f_v) {
					cout << "wrong geo_label: '" 
						<< geo_label << "' '" << geo_label1 
						<< "'" << endl;
					}
				}
			}
		}
	cout << "search_geo_file() could not find GEOMETRY " 
		<< geo_nr << " " << geo_label << " in file " << fname << endl;
	exit(1);
}
}}


