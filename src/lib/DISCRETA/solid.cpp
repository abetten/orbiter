// solid.cpp
//
// Anton Betten, Evi Haberberger
// August 2000
// moved from D2 to ORBI Nov 15, 2007

#include "orbiter.h"

#include <stdlib.h> // for system
#include <math.h> // for sqrt

using namespace std;


namespace orbiter {
namespace discreta {



#undef SOLID_CHANGE_KIND_VERBOSE
#undef SOLID_COPY_VERBOSE

static double square_root(double x)
{
	cout << "solid::square_root() do not have sqrt function" << endl;
	exit(1);
	//return sqrt(x);
}


solid::solid() : Vector()
{
	k = SOLID;
	self.vector_pointer = NULL;
}

void solid::init()
{
	// cout << "solid::init()\n";
	Vector::m_l(16);
	c_kind(SOLID);
	s_i(0).change_to_vector().m_l(0);	/* group_generators */
	s_i(1).change_to_integer().m_i(0);	/* nb_V */
	s_i(2).change_to_integer().m_i(0);	/* nb_E */
	s_i(3).change_to_integer().m_i(0);	/* nb_F */
	s_i(4).change_to_vector().m_l(0);	/* placement */
	s_i(5).change_to_vector().m_l(0);	/* v1: first vertex at an edge */
	s_i(6).change_to_vector().m_l(0);	/* v2: second vertex at an edge */
	s_i(7).change_to_vector().m_l(0);	/* f1: first face at an edge */
	s_i(8).change_to_vector().m_l(0);	/* f2: second face at an edge */
	s_i(9).change_to_vector().m_l(0);	/* nb_e: kind of face as nb_e-gon */
	s_i(10).change_to_vector().m_l(0);	/* edges: indices of edges at all faces */
	s_i(11).change_to_vector().m_l(0);	/* neighbour_faces of all faces */
	s_i(12).change_to_integer().m_i(FALSE);	/* f_vertex_labels */
	s_i(13).change_to_vector().m_l(0);	/* vertex_labels */
	s_i(14).change_to_vector().m_l(0);	/* vertex_labels_numeric */
	s_i(15).change_to_integer().m_i(FALSE); /* f_oriented */
	// cout << "leave solid::init()" << endl;
}

void solid::init_V(int nb_V)
{
	solid::nb_V() = nb_V;
	placement().m_l(3);
	x().change_to_vector();
	y().change_to_vector();
	z().change_to_vector();
	x().m_l_n(nb_V);
	y().m_l_n(nb_V);
	z().m_l_n(nb_V);
}

void solid::init_E(int nb_E)
{
	solid::nb_E() = nb_E;
	v1().m_l_n(nb_E);
	v2().m_l_n(nb_E);
	f1().m_l_n(nb_E);
	f2().m_l_n(nb_E);
}

void solid::init_F(int nb_F)
{
	int i;
	
	solid::nb_F() = nb_F;
	nb_e().m_l_n(nb_F);
	edge().m_l(nb_F);
	for (i = 0; i < nb_F; i++) {
		edge_i(i).change_to_vector();
		}
	neighbour_faces().m_l(nb_F);
	for (i = 0; i < nb_F; i++) {
		neighbour_faces_i(i).change_to_vector();
		}
}

solid::solid(const discreta_base &x)
	// copy constructor:    this := x
{
	cout << "solid::copy constructor for object: " << "\n";
	clearself();
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

solid& solid::operator = (const discreta_base &x)
	// copy assignment
{
	cout << "solid::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void solid::settype_solid()
{
	OBJECTSELF s;
	
	s = self;
	new(this) solid;
	self = s;
	k = SOLID;
}

solid::~solid()
{
	// cout << "solid::~solid()\n";
	freeself_solid();
}

void solid::freeself_solid()
{
	// cout << "solid::freeself_solid()\n";
	freeself_vector();
}

kind solid::s_virtual_kind()
{
	return SOLID;
}

void solid::copyobject_to(discreta_base &x)
{
	int i, l;
	
#ifdef SOLID_COPY_VERBOSE
	cout << "in solid::copyobject_to()\n";
#endif
	x.freeself();
	if (x.s_kind() != SOLID) 
	{
#ifdef SOLID_CHANGE_KIND_VERBOSE
		cout << "warning: solid::copyobject_to x not a solid\n";
#endif
		x.c_kind(SOLID);
		x.self.vector_pointer = NULL;
		// x.printobjectkindln();
	}
	l = s_l();
#ifdef SOLID_COPY_VERBOSE
	cout << "l=" << l << endl;
#endif
	solid & xx = x.as_solid();
#ifdef SOLID_COPY_VERBOSE
	cout << "calling xx.m_l()\n";
#endif
	xx.m_l(l);
	for (i = 0; i < l; i++) 
	{
#ifdef SOLID_COPY_VERBOSE
		cout << "copy " << i << ": " << s_i(i) << endl;
#endif
		xx[i] = s_i(i);
	}
}

ostream& solid::print_list(ostream& ost)
{
	return as_vector().Vector::print(ost);
}

ostream& solid::print(ostream& ost)
{
	int i, l;
	
	ost << "SOLID\n";
	ost << "nb_V = " << nb_V() << "\n";
	ost << "nb_E = " << nb_E() << "\n";
	ost << "nb_F = " << nb_F() << "\n";
	l = group_generators().s_l();
	ost << "nb of automorphim group generators = " << l << "\n";
	for (i = 0; i < l; i++) 
	{
		ost << group_generators().s_i(i) << endl;
	}
	if (f_vertex_labels()) 
		ost << "with vertex labels\n";
	
	ost << "x=" << x() << endl;
	ost << "y=" << y() << endl;
	ost << "z=" << z() << endl;
	ost << "v1=" << v1() << endl;
	ost << "v2=" << v2() << endl;
	ost << "f1=" << f1() << endl;
	ost << "f2=" << f2() << endl;
	ost << "nb_e=" << nb_e() << endl;
	ost << "edge=" << edge() << endl;
	ost << "neighbour_faces=" << neighbour_faces() << endl;
	return ost;
}

void solid::standard_vertex_labels(int f_start_with_zero)
{
	int i, l;

	cout << "start standard_vertex_labels:" << endl;
	l = nb_V();
	vertex_labels().m_l(l);
	vertex_labels_numeric().m_l_n(l);
	for (i = 0; i < l; i++) {
		vertex_labels().s_i(i).change_to_hollerith();
		}
	for (i = 0; i < l; i++) 
	{
		if (f_start_with_zero)
		{
			vertex_labels_i(i).append_i(i);
			vertex_labels_numeric_i(i) = i;
		}
		else
		{
			vertex_labels_i(i).append_i(i + 1);
			vertex_labels_numeric_i(i) = i + 1;
		}
		// cout << "vertex label " << i << ": " << vertex_labels_i(i) << endl;
	}
	f_vertex_labels() = TRUE;
}

void solid::determine_neighbours()
{
	int e, f1, j1, f2, j2;
	
	// cout << "start determine_neighbours():" << endl;
	for (e = 0; e < nb_E(); e++) 
	{
		find_face(e, f1, j1, f2, j2);
		f1_i(e) = f1;
		f2_i(e) = f2;
		neighbour_faces_ij(f1, j1) = f2;
		neighbour_faces_ij(f2, j2) = f1;
	}
}

void On_circle_int(Vector& Px, Vector& Py, int idx, int angle_in_degree, int rad)
{
	int x, y;
	numerics Num;
	
	x = (int)(Num.cos_grad(angle_in_degree) * (double) rad);
	y = (int)(Num.sin_grad(angle_in_degree) * (double) rad);
	Px.m_ii(idx, x);
	Py.m_ii(idx, y);
}

void solid::find_face(int e, int& f1, int& j1, int& f2, int& j2)
{
	int i, j, l, ff1 = -1, ff2 = -1;
	
	// cout << "start find_face():" << endl;
	for (i = 0; i < nb_F(); i++) 
	{
		l = nb_e_i(i);
		for (j = 0; j < l; j++) 
		{
			if (edge_ij(i, j) == e) 
			{
				if (ff1 != -1) 
				{
					ff2 = i;
					j2 = j;
				}
				else 
				{
					ff1 = i;
					j1 = j;
				}
			}
		}
	}
	if (ff1 == -1 || ff2 == -1) 
	{
		cout << "SOLID::find_face() face not found" << endl;
		cout << "edge e=" << e << " ff1=" << ff1 << " ff2=" << ff2 << endl;
		cout << "the faces and their edges are:" << endl;
		for (i = 0; i < nb_F(); i++) 
		{
			cout << "face " << i << ": ";
			l = nb_e_i(i);
			for (j = 0; j < l; j++) 
			{
				cout << edge_ij(i, j) << " ";
			}
			cout << "\n";
		}
	}
	if (ff1 == -1)
	{
		cout << "face not found" << endl;
		exit(1);
	}
	if (ff2 == -1)
	{
		cout << "face not found" << endl;
		exit(1);
	}
	f1 = ff1;
	f2 = ff2;
}

int solid::find_face_2(int e1, int e2)
{
	int f1, f2, f3, f4, j1, j2, j3, j4;
	
	find_face(e1, f1, j1, f2, j2);
	find_face(e2, f3, j3, f4, j4);
	if (f1 == f3 || f1 == f4) 
		return f1;
	if (f2 == f3 || f2 == f4) 
		return f2;
	cout << "find_face_2: face not found" << endl;
	exit(1);
}

int solid::find_face_by_two_edges(int e1, int e2)
{
	int f1, f2, f3, f4;
	
	find_faces_at_edge(e1, f1, f2);
	find_faces_at_edge(e2, f3, f4);
	if (f1 != -1) {
		if (f1 == f3 || f1 == f4) {
			return f1;
			}
		}
	if (f2 != -1) {
		if (f2 == f3 || f2 == f4) {
			return f2;
			}
		}
	return -1;
}

void solid::find_faces_at_edge(int e, int& f1, int& f2)
{
	int nb_F, i, j, l, n = 0;
	
	f1 = -1;
	f2 = -1;
	nb_F = solid::nb_F();
	for (i = 0; i < nb_F; i++) 
	{
		l = nb_e_i(i);
		for (j = 0; j < l; j++) 
		{
			if (edge_ij(i, j) == e) 
			{
				if (n == 0) 
				{
					f1 = i;
					n++;
				}
				else if (n == 1) 
				{
					f2 = i;
					n++;
				}
				else 
				{
					cout << "solid::find_faces_at_edge(): too many faces for this edge" << endl;
					exit(1);
				}
			}
		}
	}
}

int solid::find_edge(int v1, int v2)
{
	int i, e1, e2;
	int nb_E = solid::nb_E();
	
	// cout << "start find_edge():" << endl;
	for (i = 0; i < nb_E; i++) 
	{
		e1 = v1_i(i);
		e2 = v2_i(i);
		if ((e1 == v1 && e2 == v2) || (e1 == v2 && e2 == v1)) 
		{
			return i;
		}
	}
	// cout << "find_edge(): edge not found!" << endl;
	return -1;
}

void solid::add_edge(int v1, int v2, int f1, int f2)
{
	// cout << "start add_edge(), edge no" << nb_E() << " " << v1 << "-" << v2 << endl;
	solid::v1().append_integer(v1);
	solid::v2().append_integer(v2);
	solid::f1().append_integer(0);
	solid::f2().append_integer(0);
	nb_E()++;
}

int solid::add_edge(int v1, int v2)
{
	// cout << "start add_edge(v1,v2), edge no" << nb_E() << " " << v1 << "-" << v2 << endl;
	solid::v1().append_integer(v1);
	solid::v2().append_integer(v2);
	solid::f1().append_integer(0);
	solid::f2().append_integer(0);
	nb_E()++;
	// cout << "new nb. of edges: " << nb_E() << endl;
	return nb_E() - 1;
}


int solid::find_and_add_edge(int i1, int i2, int f_v)
{
	int e;
	
	// cout << "start find_and_add_edge():" << endl;
	e = find_edge(i1, i2);
	if (e == -1) 
	{
		if (f_v) 
		{
			cout << "adding edge " << i1 << " " << i2 << endl;
		}
		e = add_edge(i1, i2);
		if (f_v) 
		{
			cout << "- edge no " << e << endl;
		}
		
	}
	// cout << "leave find_and_add_edge()" << endl;
	return e;
}

void solid::add_face3(int e1, int e2, int e3, int i1, int i2, int i3)
{
	Vector V;
	
	V.m_l(3);
	V.m_ii(0, i1);
	V.m_ii(1, i2);
	V.m_ii(2, i3);
	add_face_n(V);
}

void solid::add_face4(int i1, int i2, int i3, int i4)
{
	Vector V;
	
	V.m_l(4);
	V.m_ii(0, i1);
	V.m_ii(1, i2);
	V.m_ii(2, i3);
	V.m_ii(3, i4);
	add_face_n(V);
}

void solid::add_face5(int i1, int i2, int i3, int i4, int i5)
{
	Vector V;
	
	V.m_l(5);
	V.m_ii(0, i1);
	V.m_ii(1, i2);
	V.m_ii(2, i3);
	V.m_ii(3, i4);
	V.m_ii(4, i5);
	add_face_n(V);
	// cout << "leave add_face5()" << endl;
}

void solid::add_face_n(Vector& vertices)
{
	int e, i, l;
	int f_v = FALSE;
	
	l = vertices.s_l();
	nb_e().inc();
	edge().inc();
	neighbour_faces().inc();
	nb_e().m_ii(nb_F(), l);
	edge_i(nb_F()).change_to_vector();
	edge_i(nb_F()).m_l_n(l);
	neighbour_faces_i(nb_F()).change_to_vector();
	neighbour_faces_i(nb_F()).m_l_n(l);
	
	for (i = 0; i < l - 1; i++) {
		e = find_and_add_edge(vertices.s_ii(i), vertices.s_ii(i + 1), f_v);
		edge_ij(nb_F(), i) = e;
		// cout << "solid::add_face_n() edge=" << edge() << endl;
		}
	e = find_and_add_edge(vertices.s_ii(l - 1), vertices.s_ii(0), f_v);
	edge_ij(nb_F(), l - 1) = e;
	
	nb_F()++;
}

void solid::adjacency_list(int vertex, int *adj, int *nb_adj)
{
	int i, j, l, ll, a, b, f_found;
	Vector Adj;
	
	Adj.m_l(0);
	ll = 0;
	for (i = 0; i < nb_E(); i++) {
		if (v1_i(i) == vertex || v2_i(i) == vertex) {
			Adj.inc();
			Adj.m_ii(ll++, i);
		}
	}
	l = 0;
	adj[l++] = a = Adj.s_ii(--ll);
	Adj.dec();
	while (ll) {
		f_found = FALSE;
		for (i = 0; i < ll; i++) {
			b = Adj.s_ii(i);
			if (find_face_by_two_edges(a, b) != -1) {
				adj[l++] = b;
				for (j = i + 1; j < ll; j++) {
					Adj.m_ii(j - 1, Adj.s_ii(j));
					}
				Adj.dec();
				ll--;
				a = b;
				f_found = TRUE;
				break;
				}
			}
		if (!f_found) {
			cout << "found no adjacent edge" << endl;
			exit(1);
			}
		}
	*nb_adj = l;
}

void solid::center(int f, Vector& Px, Vector& Py, Vector& Pz)
{
	int i, nb_e, e, e1, e2;
	int x = 0, y = 0, z = 0;
	
	nb_e = nb_e_i(f);
	for (i = 0; i < nb_e; i++) 
	{
		e = edge_ij(f, i);
		e1 = v1_i(e);
		e2 = v2_i(e);
		x += x_i(e1);
		y += y_i(e1);
		z += z_i(e1);
		x += x_i(e2);
		y += y_i(e2);
		z += z_i(e2);
	}
	nb_e <<= 1;
	Px.m_ii(f, (int)((double)x / (double)nb_e));
	Py.m_ii(f, (int)((double)y / (double)nb_e));
	Pz.m_ii(f, (int)((double)z / (double)nb_e));
}

void solid::vertices_of_face(int i, Vector& V)
{
	int nb_e = nb_e_i(i);
	int e, j, v1, v2, v3, v4, last_vertex = 0;
	
	V.m_l_n(nb_e);
	
	e = edge_ij(i, 0);
	v1 = v1_i(e);
	v2 = v2_i(e);
	cout << "e=" << e << " v1=" << v1 << " v2=" << v2 << endl;
	for (j = 1; j < nb_e; j++) 
	{
		e = edge_ij(i, j);
		v3 = v1_i(e);
		v4 = v2_i(e);
		cout << "j=" << j << " e=" << e << " v3=" << v3 << " v4=" << v4 << endl;
		if (j == 1) 
		{
			if (v3 == v2) 
			{
				V.m_ii(0, v1);
				V.m_ii(1, v2);
				last_vertex = v4;
			}
			else if (v3 == v1) 
			{
				V.m_ii(0, v2);
				V.m_ii(1, v1);
				last_vertex = v4;
			}
			else if (v4 == v2) 
			{
				V.m_ii(0, v1);
				V.m_ii(1, v2);
				last_vertex = v3;
			}
			else if (v4 == v1) 
			{
				V.m_ii(0, v2);
				V.m_ii(1, v1);
				last_vertex = v3;
			}
			else 
			{
				cout << "face " << i << ": edges=";
				edge_i(i).Print(cout);
				cout << "v1=" << v1 << " v2=" << v2 << " v3=" << v3 << " v4=" << v4 << " last_vertex=" << last_vertex << endl;
				V.Print(cout);
				cout << "solid::vertices_of_face() error edges not adjacent! (j=1)" << endl;
				exit(1);
			}
		}
		else {
			if (v3 == last_vertex) 
			{
				V.m_ii(j, v3);
				last_vertex = v4;
			}
			else if (v4 == last_vertex) 
			{
				V.m_ii(j, v4);
				last_vertex = v3;
			}
			else 
			{
				cout << "face " << i << ": edges=";
				edge_i(i).Print(cout);
				cout << "v1=" << v1 << " v2=" << v2 << " v3=" << v3 << " v4=" << v4 << " last_vertex=" << last_vertex << endl;
				V.Print(cout);
				cout << "solid::vertices_of_face() error edges not adjacent!" << endl;
				exit(1);
			}
		}
		cout << "last_vertex=" << last_vertex << endl;
	}
}

void solid::Ratio(int e, double r, int& x, int& y, int& z)
{
	int e1, e2;

	e1 = v1_i(e);
	e2 = v2_i(e);
	x = x_i(e1) + (int)((double)(x_i(e2) - x_i(e1)) * r);
	y = y_i(e1) + (int)((double)(y_i(e2) - y_i(e1)) * r);
	z = z_i(e1) + (int)((double)(z_i(e2) - z_i(e1)) * r);
}

int solid::find_common_face(int e1, int e2, int& f)
{
	int n11, n12, n21, n22;
	
	n11 = f1_i(e1);
	n12 = f2_i(e1);
	n21 = f1_i(e2);
	n22 = f2_i(e2);
	if (n11 == n21 || n11 == n22) 
	{
		f = n11;
		return TRUE;
	}
	if (n12 == n21 || n12 == n22) 
	{
		f = n12;
		return TRUE;
	}
	return FALSE;
}

#if 0
void solid::orientation(Vector& face_list)
{
	int first_v, next_v, first_e, first_f, next_f, j1, j2, i, j, t, l;
	Vector v, vf;
	
	first_v = v1_i(0);
	next_v = v2_i(0);
	
	face_list.m_l(nb_F());
	v.m_l(nb_F());

	for (i = 0; i < nb_F(); i++)
	{
		first_e = find_edge(first_v, next_v);
		find_face(first_e, first_f, j1, next_f, j2);

		/* go on from first_f: 
		list of vertices of face first_f is saved to face_list[i] */
		vertices_of_face(first_f, face_list[i].as_vector());
		if (i > 0)
		{
			t = 0;
			for (j = 0; j < i; j++)
			{
				if (v[j] == first_f)
					t++;
			}
			if (t > 0)
			{
				for (k = 0; k < i; k++)
				{
					if (v[k] == next_f)
					{
						cout << "solid::orientation(): both faces already found!" << endl;
						break;
					}	
				}
				v[i] = next_f;
			}
			else 
				v[i] = first_f;			
		}
		else
		{
			v[i] = first_f;
		}
		
		/* change next_v to first_v and next_v to the other vertex 
		opposite of the original first_v */
		l = face_list[i].s_l();
		vf = face_list[i];
		k = 0;
		j = 0;
		while (k == 0)
		{
			
			j++;
		}
	}
}
#endif

void solid::dual(solid& A)
{
	int adj[1000], nb_adj;
	int v, i, j, f, e, e1, e2, v1, v2, v3, v4, V1, V2, V3, V4, l;
	solid B;
	
	cout << "dual()" << endl;
	B.init();
	B.init_V(nb_F());
	B.init_E(nb_E());
	B.init_F(nb_V());
	l = group_generators().s_l();
	B.group_generators().m_l(l);
	for (i = 0; i < l; i++) 
	{
		// cout << "i=" << i << endl;
		permutation & Q = group_generators_i(i);
		// Q.Print(cout);
		
		permutation P;
		
		P.m_l(nb_F());
		for (j = 0; j < nb_F(); j++) 
		{
			e1 = edge_ij(j, 0);
			e2 = edge_ij(j, 1);
			v1 = v1_i(e1);
			v2 = v2_i(e1);
			v3 = v1_i(e2);
			v4 = v2_i(e2);
			V1 = Q.s_ii(v1);
			V2 = Q.s_ii(v2);
			V3 = Q.s_ii(v3);
			V4 = Q.s_ii(v4);
			e1 = find_edge(V1, V2);
			e2 = find_edge(V3, V4);
			if (e1 == -1) 
			{
				cout << "solid::dual() error: couldn't find edge V1-V2\n";
				cout << "v1=" << v1 << endl;
				cout << "v2=" << v2 << endl;
				cout << "v3=" << v3 << endl;
				cout << "v4=" << v4 << endl;
				cout << "V1=" << V1 << endl;
				cout << "V2=" << V2 << endl;
				cout << "V3=" << V3 << endl;
				cout << "V4=" << V4 << endl;
				exit(1);
			}
			if (e2 == -1) 
			{
				cout << "solid::dual() error: couldn't find edge V3-V4\n";
				cout << "v1=" << v1 << endl;
				cout << "v2=" << v2 << endl;
				cout << "v3=" << v3 << endl;
				cout << "v4=" << v4 << endl;
				cout << "V1=" << V1 << endl;
				cout << "V2=" << V2 << endl;
				cout << "V3=" << V3 << endl;
				cout << "V4=" << V4 << endl;
				exit(1);
			}
			f = find_face_2(e1, e2);
			P.m_ii(j, f);
		}
		B.group_generators_i(i) = P;
	}
	
	for (i = 0; i < nb_F(); i++) 
	{
		center(i, B.x(), B.y(), B.z());
	}
	for (i = 0; i < nb_E(); i++) 
	{
		B.v1_i(i) = f1_i(i);
		B.v2_i(i) = f2_i(i);
		B.f1_i(i) = v1_i(i);
		B.f2_i(i) = v2_i(i);
	}
	
	for (v = 0; v < nb_V(); v++) 
	{
	
		adjacency_list(v, adj, &nb_adj);
		B.nb_e_i(v) = nb_adj;
		B.edge_i(v).m_l_n(nb_adj);
		B.neighbour_faces_i(v).m_l_n(nb_adj);
		
		for (i = 0; i < nb_adj; i++) 
		{
			e = adj[i];
			v1 = v1_i(e);
			v2 = v2_i(e);
			B.edge_ij(v, i) = e;
			if (v1 == v) 
			{
				B.neighbour_faces_ij(v, i) = v2;
			}
			else if (v2 == v) 
			{
				B.neighbour_faces_ij(v, i) = v1;
			}
			else
				exit(1);
		}
	}
	B.swap(A);
	if (f_vertex_labels())
		A.standard_vertex_labels(FALSE /* f_start_with_zero */);
}

void solid::cut_vertices(double r, solid & A)
{
	int adj[1000], adj1[1000], nb_adj, nb_adj1;
	int nb_V, nb_E, nb_F, nb_new_V;
	int v, v1;
	int ei, ej, e, ee, e1, e2, f, i = 0, j = 0, l, first_new_vertex, k;
	solid B;
	permutation P, Pind;
	permutation Q;	
	Vector Pindv;
	Vector first_new_vertex_number;
	
	
	cout << "cut_vertices():\n";
	nb_V = solid::nb_V();
	nb_E = solid::nb_E();
	nb_F = solid::nb_F();
	B = *this;
	// cout << "B=" << B << endl;

	first_new_vertex_number.m_l_n(nb_V);
	first_new_vertex = nb_V;
	for (v = 0; v < solid::nb_V(); v++) {
		first_new_vertex_number.m_ii(v, first_new_vertex);
		adjacency_list(v, adj, &nb_adj);
		first_new_vertex += nb_adj;
		} 
	
	for (v = 0; v < solid::nb_V(); v++)
	{
		// cout << "v=" << v << endl;
		first_new_vertex = nb_V;
		// cout << "first_new_vertex=" << first_new_vertex << endl;
		
		adjacency_list(v, adj, &nb_adj);
		
		// shortens the edges from one side (near v)
		for (ei = 0; ei < nb_adj; ei++) 
		{
			int x, y, z;
			
			e = adj[ei];
			e1 = v1_i(e);
			if (e1 == v)
				Ratio(e, r, x, y, z);
			else
				Ratio(e, 1. - r, x, y, z);
			// cout << "new point " << B.nb_V() << " x= " << x << " y=" << y << " z=" << z << endl;
			B.x().append_integer(x);
			B.y().append_integer(y);
			B.z().append_integer(z);
			B.nb_V()++;
			nb_V++;
		}		
		
		// define a new face:
		B.nb_e().append_integer(0);
		B.edge().inc();
		B.edge_i(nb_F).change_to_vector();
		B.edge_i(nb_F).m_l_n(nb_adj);
		B.neighbour_faces().inc();
		B.neighbour_faces_i(nb_F).change_to_vector();
		B.neighbour_faces_i(nb_F).m_l_n(nb_adj);
		
		// all the edges of the new face:
		for (ei = 0; ei < nb_adj; ei++) 
		{
			e = adj[ei];
			for (ej = ei + 1; ej < nb_adj; ej++) 
			{
				ee = adj[ej];
				if (find_common_face(e, ee, f)) 
				{
					// cout << "e=" << e << " ee=" << ee << " f=" << f << endl;
					
					// new edge:
					B.v1().append_integer(first_new_vertex + ei);
					B.v2().append_integer(first_new_vertex + ej);
					B.f1().append_integer(f);
					B.f2().append_integer(nb_F);
					B.nb_E()++;
					
					// new face:
					
					k = B.nb_e_i(nb_F);
					B.edge_ij(nb_F, k) = nb_E;
					B.neighbour_faces_ij(nb_F, k) = f;
					B.nb_e_i(nb_F)++;
					
					// old face:
					B.edge_i(f).append_integer(nb_E);
					B.neighbour_faces_i(f).append_integer(nb_F);
					B.nb_e_i(f)++;
					
					nb_E++;
				}
			}
		}
		
		
		// face:
		B.nb_F()++;
		nb_F++;
	} // next v
	// cout << "B=" << B << endl;

	// induce the group:
	l = group_generators().s_l();
	cout << "inducing the group, number of generators = " << l << endl;
	cout << "nb_V = " << nb_V << " solid::nb_V()=" << solid::nb_V() << endl;
	for (int z = 0; z < l; z++) {
		P.m_l(nb_V);
		Q = group_generators_i(z);
		// cout << "z=" << z << endl;
		// cout << "Q=" << Q << endl;
		for (v = 0; v < solid::nb_V(); v++) 
		{
			int k, v2;
			int pii, pij;
			int new_vertex, image_new_vertex;
			
			k = Q[v];
			// cout << "v=" << v << " -> k=" << k << endl;
			P[v] = k;
			adjacency_list(v, adj, &nb_adj);
			for (ei = 0; ei < nb_adj; ei++) 
			{
				e = adj[ei];
				e1 = v1_i(e);
				e2 = v2_i(e);
				if (v == e1) 
				{
					i = e1;
					j = e2;
				}
				if (v == e2)
				{
					i = e2;
					j = e1;
				}
				pii = Q[i];
				pij = Q[j];
				// cout << "ei=" << ei << " i=" << i << " ->pii=" << pii << " j=" << j << " ->pij=" << pij << endl;
				adjacency_list(pii, adj1, &nb_adj1);
				k = -1;
				for (int ii = 0; ii < nb_adj1; ii++)
				{
					e1 = adj1[ii];
					v1 = v1_i(e1);
					v2 = v2_i(e1);
					if (v1 == pij || v2 == pij) k = ii;
				}
				if (k == -1) {
					cout << "ERROR: image of pij not found!" << endl;
					exit(1);
					}
				// new_vertex = s_nb_V_i() + v * nb_adj + ei;
				new_vertex = first_new_vertex_number.s_ii(v) + ei;
				// image_new_vertex = s_nb_V_i() + pii * nb_adj + k;
				image_new_vertex = first_new_vertex_number.s_ii(pii) + k;
				// cout << "new_vertex=" << new_vertex << " image_new_vertex=" << image_new_vertex << endl;
				P[new_vertex] = image_new_vertex;
			}
		} // next v
		B.group_generators_i(z) = P;
	} // next z
	cout << "finished" << endl;
	// cout << B.group_generators() << endl;
	first_new_vertex = solid::nb_V();
	
	cout << "updating the edges:" << endl;
	for (v = 0; v < solid::nb_V(); v++) 
	{

		adjacency_list(v, adj, &nb_adj);

		// update the edges (from one side):
		for (ei = 0; ei < nb_adj; ei++) 
		{
			e = adj[ei];
			e1 = v1_i(e);
			e2 = v2_i(e);
			if (e1 == v) 
			{
				B.v1_i(e) = first_new_vertex + ei;
			}
			else if (e2 == v) 
			{
				B.v2_i(e) = first_new_vertex + ei;
			}
			else {
				cout << "edge does not contain vertex v" << endl;
				exit(1);
			}
		}

		first_new_vertex += nb_adj;
	}
	if (first_new_vertex != nb_V) {
		cout << "first_new_vertex != nb_V" << endl;
		exit(1);
		}


	cout << "eliminating the old vertices:" << endl;
	nb_new_V = nb_V - solid::nb_V();
	// eliminate the old vertices:
	for (i = 0; i < nb_new_V; i++) 
	{
		B.x_i(i) = B.x_i(solid::nb_V() + i);
		B.y_i(i) = B.y_i(solid::nb_V() + i);
		B.z_i(i) = B.z_i(solid::nb_V() + i);
	}
	for (i = 0; i < B.nb_E(); i++) 
	{
		if (B.v1_i(i) < solid::nb_V()) {
			cout << "B.v1_i(i) < solid::nb_V()" << endl;
			exit(1);
			}
		if (B.v2_i(i) < solid::nb_V()) {
			cout << "B.v2_i(i) < solid::nb_V()" << endl;
			exit(1);
			}
			
		B.v1_i(i) = B.v1_i(i) - solid::nb_V();
		B.v2_i(i) = B.v2_i(i) - solid::nb_V();
	}
	
	Pindv.m_l(l);
	for (int zz = 0; zz < l; zz++) 
	{
		Pind.m_l(nb_new_V);
		for (v = 0; v < nb_new_V; v++)
		{
			Pind[v] = B.group_generators_i(zz)[solid::nb_V() + v] - solid::nb_V();
		}
		Pindv[zz] = Pind;
	}
	cout << "new generators:\n";
	for (int zzz = 0; zzz < l; zzz++) 
	{
		cout << Pindv[zzz] << endl;
	}
	B.group_generators() = Pindv;
	B.nb_V() = nb_new_V;
	B.x().realloc(nb_new_V);
	B.y().realloc(nb_new_V);
	B.z().realloc(nb_new_V);
	if (B.f_vertex_labels()) {
		B.standard_vertex_labels(FALSE /* f_start_with_zero */);
		}
	B.swap(A);
}

void solid::edge_midpoints(solid& A)
{
	int adj[1000], nb_adj;
	int nb_V, nb_E, nb_E_old, nb_F;
	int v;
	int ei, ej, e, ee, f, i, j, k;
	solid B;
	Vector gen_new;
	
	
	cout << "edge_midpoints()" << endl;
	induced_group_on_edges(group_generators(), gen_new);
	nb_V = solid::nb_V();
	nb_E = nb_E_old = solid::nb_E();
	nb_F = solid::nb_F();
	B = *this;
	//l = group_generators().s_l();

	B.x().realloc(nb_V + nb_E);
	B.y().realloc(nb_V + nb_E);
	B.z().realloc(nb_V + nb_E);
	for (i = 0; i < nb_E; i++) {
		int x, y, z;
		B.x().s_i(nb_V + i).change_to_integer();
		B.y().s_i(nb_V + i).change_to_integer();
		B.z().s_i(nb_V + i).change_to_integer();
		Ratio(i, 0.5, x, y, z);
		B.x_i(nb_V + i) = x;
		B.y_i(nb_V + i) = y;
		B.z_i(nb_V + i) = z;
		}
	// B.s_nb_F()->m_i(nb_F + nb_V);
	B.nb_e().realloc(nb_F + nb_V);
	B.edge().realloc(nb_F + nb_V);
	B.neighbour_faces().realloc(nb_F + nb_V);
	for (i = 0; i < nb_V; i++) {
		B.nb_e().s_i(nb_F + i).change_to_integer();
		B.edge().s_i(nb_F + i).change_to_vector();
		B.neighbour_faces().s_i(nb_F + i).change_to_vector();
		}
	
	cout << "defining new faces:" << endl;
	for (v = 0; v < nb_V; v++) 
	{
		adjacency_list(v, adj, &nb_adj);
		
		B.nb_e_i(nb_F + v) = 0;
		B.edge_i(nb_F + v).m_l_n(nb_adj);
		B.neighbour_faces_i(nb_F + v).m_l_n(nb_adj);
		
		// all the edges of the new face:
		for (ei = 0; ei < nb_adj; ei++) 
		{
			e = adj[ei];
			for (ej = ei + 1; ej < nb_adj; ej++) 
			{
				ee = adj[ej];
				if (find_common_face(e, ee, f)) 
				{
					
					// new edge:
					B.v1().append_integer(nb_V + e);
					B.v2().append_integer(nb_V + ee);
					B.f1().append_integer(f);
					B.f2().append_integer(nb_F + v);
					B.nb_E()++;
					
					// new face:
					
					k = B.nb_e_i(nb_F + v);
					B.edge_ij(nb_F + v, k) = nb_E;
					B.neighbour_faces_ij(nb_F + v, k) = f;
					B.nb_e_i(nb_F + v)++;
					
					// old face:
					k = B.nb_e_i(f);
					B.edge_i(f).append_integer(nb_E);
					B.neighbour_faces_i(f).append_integer(nb_F + v);
					B.nb_e_i(f)++;
					
					nb_E++;
				}
			}
		}
		
		
		// face:
		B.nb_F()++;
	} // next v
	
	// eliminate old edges in old faces:
	cout << "eliminating the old edges and faces:" << endl;
	for (f = 0; f < nb_F; f++) {
		Vector edge, neighbour_faces;
		int n;
		
		k = B.nb_e_i(f);
		edge.m_l(0);
		neighbour_faces.m_l(0);
		i = 0;
		for (j = 0; j < k; j++) {
			e = B.edge_ij(f, j);
			n = B.neighbour_faces_ij(f, j);
			if (e >= nb_E_old) {
				edge.append_integer(e);
				neighbour_faces.append_integer(n);
				i++;
				}
			}
		edge.swap(B.edge_i(f));
		neighbour_faces.swap(B.neighbour_faces_i(f));
		B.nb_e_i(f) = i;
		}


	cout << "eliminating the old vertices:" << endl;
	// eliminate the old vertices:
	for (i = 0; i < nb_E_old; i++) 
	{
		B.x_i(i) = B.x_i(nb_V + i);
		B.y_i(i) = B.y_i(nb_V + i);
		B.z_i(i) = B.z_i(nb_V + i);
	}
	B.nb_V() = nb_E_old;
	
	// eliminate old edges and update vertex labels of new edges:
	cout << "eliminate old edges and update vertex labels of new edges:" << endl;
	cout << "nb_E_old = " << nb_E_old << endl;
	cout << "B.nb_E()= " << B.nb_E() << endl;
	cout << "B=" << B << endl;
	for (i = nb_E_old; i < B.nb_E(); i++) 
	{
		if (B.v1_i(i) < nb_V) {
			cout << "B.v1_i(i) < nb_V" << endl;
			exit(1);
			}
		if (B.v2_i(i) < nb_V) {
			cout << "B.v2_i(i) < nb_V" << endl;
			exit(1);
			}
			
		B.v1_i(i - nb_E_old) = B.v1_i(i) - nb_V;
		B.v2_i(i - nb_E_old) = B.v2_i(i) - nb_V;
		B.f1_i(i - nb_E_old) = B.f1_i(i);
		B.f2_i(i - nb_E_old) = B.f2_i(i);
	}
	B.nb_E() = B.nb_E() - nb_E_old;
	
	for (f = 0; f < B.nb_F(); f++) {
		int n;
		
		k = B.nb_e_i(f);
		for (j = 0; j < k; j++) {
			e = B.edge_ij(f, j);
			n = B.neighbour_faces_ij(f, j);
			if (e < nb_E_old) {
				cout << "f=" << f << " j=" << j << " k=" << k << " e=" << e << " n=" << n << endl;
				cout << "solid::edge_midpoints: e < nb_E_old" << endl;
				exit(1);
				}
			B.edge_ij(f, j) = e - nb_E_old;
			}
		}
	
	
	
	gen_new.swap(B.group_generators());

	B.x().realloc(B.nb_V());
	B.y().realloc(B.nb_V());
	B.z().realloc(B.nb_V());
	if (B.f_vertex_labels()) {
		B.standard_vertex_labels(FALSE /* f_start_with_zero */);
		}
	B.v1().realloc(B.nb_E());
	B.v2().realloc(B.nb_E());
	B.f1().realloc(B.nb_E());
	B.f2().realloc(B.nb_E());
	
	B.swap(A);
}

void solid::join_disjoint(solid& A, solid& J, int f_v)
{
	Vector gen;
	int nb_F1, nb_F2, nb_E1, nb_E2, nb_V1, nb_V2, nb_F, nb_E, nb_V;
	int i, j, v;
	int ll;
	int f_vertex_labels = FALSE;
	
	cout << "solid::join_disjoint():" << endl;
	nb_F1 = solid::nb_F();
	nb_F2 = A.nb_F();
	nb_E1 = solid::nb_E();
	nb_E2 = A.nb_E();
	nb_V1 = solid::nb_V();
	nb_V2 = A.nb_V();
	if (f_v) {
		cout << "nb_V1 = " <<  nb_V1 << endl;
		cout << "nb_V2 = " <<  nb_V2 << endl;
		cout << "nb_E1 = " <<  nb_E1 << endl;
		cout << "nb_E2 = " <<  nb_E2 << endl;
		cout << "nb_F1 = " <<  nb_F1 << endl;
		cout << "nb_F2 = " <<  nb_F2 << endl;
		}
	nb_F = nb_F1 + nb_F2;
	nb_E = nb_E1 + nb_E2;
	nb_V = nb_V1 + nb_V2;
	J.init();
	J.init_V(nb_V);
	J.init_E(nb_E);
	J.init_F(nb_F);
	if (solid::f_vertex_labels() && A.f_vertex_labels())
		f_vertex_labels = TRUE;
	if (f_vertex_labels) {
		J.f_vertex_labels() = TRUE;
		J.vertex_labels().m_l(nb_V);
		J.vertex_labels_numeric().m_l_n(nb_V);
		}
	else {
		J.f_vertex_labels() = FALSE;
		}
	cout << "1:" << endl;
	for (i = 0; i < nb_V1; i++)
	{
		J.x_i(i) = x_i(i);
		J.y_i(i) = y_i(i);
		J.z_i(i) = z_i(i);
		if (f_vertex_labels) {
			J.vertex_labels_i(i).init(vertex_labels_i(i).s());
			J.vertex_labels_numeric_i(i) = vertex_labels_numeric_i(i);
			}
	}
	cout << "2:" << endl;
	for (i = nb_V1; i < nb_V; i++)
	{
		j = i - nb_V1;
		J.x_i(i) = A.x_i(j);
		J.y_i(i) = A.y_i(j);
		J.z_i(i) = A.z_i(j);
		if (f_vertex_labels) {
			J.vertex_labels_i(i).init(A.vertex_labels_i(j).s());
			J.vertex_labels_numeric_i(i) = A.vertex_labels_numeric_i(j);
			}
	}
	cout << "3:" << endl;
	for (i = 0; i < nb_E1; i++)
	{
		J.v1_i(i) = v1_i(i);
		J.v2_i(i) = v2_i(i);
		J.f1_i(i) = f1_i(i);
		J.f2_i(i) = f2_i(i);
	}
	cout << "4:" << endl;
	for (i = nb_E1; i < nb_E; i++)
	{
		j = i - nb_E1;
		J.v1_i(i) = A.v1_i(j) + nb_V1;
		J.v2_i(i) = A.v2_i(j) + nb_V1;
		J.f1_i(i) = A.f1_i(j) + nb_F1;
		J.f2_i(i) = A.f2_i(j) + nb_F1;
		
	}
	cout << "faces 1:" << endl;
	for (i = 0; i < nb_F1; i++)
	{
		ll = nb_e_i(i);
		J.nb_e_i(i) = ll;
		J.edge_i(i).m_l_n(ll);
		J.neighbour_faces_i(i).m_l_n(ll);
		for (v = 0; v < ll; v++)
		{
			J.edge_ij(i, v) = edge_ij(i, v);
			J.neighbour_faces_ij(i, v) = neighbour_faces_ij(i, v);
		}
	}
	cout << "faces 2:" << endl;
	for (i = nb_F1; i < nb_F; i++)
	{
		j = i - nb_F1;
		ll = A.nb_e_i(j);
		J.nb_e_i(i) = ll;
		J.edge_i(i).m_l_n(ll);
		J.neighbour_faces_i(i).m_l_n(ll);
		for (v = 0; v < ll; v++)
		{
			J.edge_ij(i, v) = A.edge_ij(j, v) + nb_E1;
			J.neighbour_faces_ij(i, v) = A.neighbour_faces_ij(j, v) + nb_F1;
		}
	}
}

#if 0
void solid::direct_sum(solid& B, solid& J, int f_v)
{
	Vector gen1, gen2;
	solid C;
	
	cout << "SOLID::direct_sum()" << endl;
	gen1 = group_generators();
	gen2 = B.group_generators();
	join_disjoint(B, C, f_v);
	cout << "calling vec_generators_diagonal_sum()" << endl;
	vec_generators_diagonal_sum(gen1, gen2, C.group_generators());
	C.standard_vertex_labels(FALSE /* f_start_with_zero */);
	C.swap(J);
}

void solid::direct_product(Vector& gen, solid& J, int f_v)
{
	double f;
	int d;
	Vector gen0;
	solid A0, A1, A2;
	int i;
	
	cout << "solid::direct_product()" << endl;
	A0 = *this;
	gen0 = group_generators();
	cout << "gen=\n" << gen << endl;
	d = vec_generators_degree(gen);
	for (i = 1; i < d; i++){
		f = 1. + (double) i * 1.;
		A1 = *this;
		A1.scale(f);
		A0.join_disjoint(A1, A2, f_v);
		A2.swap(A0);
		}
	cout << "calling vec_generators_direct_product()" << endl;
	vec_generators_direct_product(gen, gen0, A0.group_generators());
	A0.standard_vertex_labels(FALSE /* f_start_with_zero */);
	A0.swap(J);
}
#endif

void solid::scale(double f)
{
	int i, a;
	
	for (i = 0; i < nb_V(); i++) 
	{
		a = x_i(i);
		a = (int)((double) a * f);
		x_i(i) = a;
		a = y_i(i);
		a = (int)((double) a * f);
		y_i(i) = a;
		a = z_i(i);
		a = (int)((double) a * f);
		z_i(i) = a;
	}
}

#if 0
void solid::add_central_point(solid& A)
{
	solid J;
	int nb_v = nb_V();
	
	cout << "solid::add_central_point()" << endl;
	J = *this;
	vec_add_fixpoint_at_end(J.group_generators());
	J.nb_V()++;
	J.x().append_integer(0);
	J.y().append_integer(0);
	J.z().append_integer(0);
	if (J.f_vertex_labels()) {
		hollerith a;
		
		a.append_i(nb_v + 1);
		J.vertex_labels().append(a);
		J.vertex_labels_numeric().append_integer(nb_v + 1);
		}
	J.swap(A);
}
#endif

void solid::induced_action_on_edges(permutation& p, permutation& q)
{
	int j, v1, v2, w1, w2, k;
	
	q.m_l(nb_E());
	for (j = 0; j < nb_E(); j++) {
		v1 = v1_i(j);
		v2 = v2_i(j);
		// printf("edge %d = (%d,%d)", j, v1, v2); fflush(stdout);
		w1 = p[v1];
		w2 = p[v2];
		// printf("maps to (%d,%d)\n", w1, w2); fflush(stdout);
		k = find_edge(w1, w2);
		if (k < 0) {
			cout << "solid::induced_action_on_edges() error in find_edge" << endl;
			exit(1);
			}
		q[j] = k;
		}
}

void solid::induced_group_on_edges(Vector & gen, Vector & gen_e)
{
	permutation p, q;
	int i, l;
	
	l = gen.s_l();
	gen_e.m_l(l);
	for (i = 0; i < l; i++) {
		permutation &p = gen[i].as_permutation();
		gen_e.s_i(i).change_to_permutation();
		permutation &q = gen_e[i].as_permutation();
		induced_action_on_edges(p, q);
		}
}


void solid::tetrahedron(int r)
{
	double phi, c;
	int i;
	permutation P;
	
	init();
	group_generators().m_l(2);
	P.m_l(4);
	P.one();
	P.sscan("(0,1,2)(3)", FALSE /* f_v */);
	group_generators_i(0) = P;
	P.m_l(4);
	P.one();
	P.sscan("(0,1,3)", FALSE /* f_v */);
	group_generators_i(1) = P;
	f_vertex_labels() = FALSE;
	phi = 120.;
	//h = 4. * r * 0.33333;
	c = square_root(8. / 9.) * r;
	
	placement().m_l(3);
	x().change_to_vector();
	y().change_to_vector();
	z().change_to_vector();
	x().m_l_n(4);
	y().m_l_n(4);
	z().m_l_n(4);
	for (i = 0; i < 3; i++)
	{
		On_circle_int(x(), y(), i, (int) ((double) i * phi), (int) c);
		z_i(i) = (int)(-0.333333 * r);
	}
	x_i(3) = 0;
	y_i(3) = 0;
	z_i(3) = r;
	
	nb_V() = 4;
	add_edge(0, 1, 2, 3);	
	add_edge(1, 2, 0, 3);	
	add_edge(2, 0, 1, 3);	
	add_edge(0, 3, 1, 2);	
	add_edge(1, 3, 2, 0);	
	add_edge(2, 3, 0, 1);	

	add_face3(1, 5, 4, 3, 1, 2);
	add_face3(2, 3, 5, 3, 2, 0);
	add_face3(0, 4, 3, 3, 0, 1);
	add_face3(0, 1, 2, 2, 0, 1);
	
	determine_neighbours();
	standard_vertex_labels(FALSE /* f_start_with_zero */);
}

void solid::cube(int r)
{
	permutation p;
	int r2 = r >> 1;
	
	init();
	nb_V() = 8;
	placement().m_l(3);
	x().change_to_vector();
	y().change_to_vector();
	z().change_to_vector();
	x().m_l_n(8);
	y().m_l_n(8);
	z().m_l_n(8);
	x_i(0) = -r2;
	y_i(0) = r2;
	z_i(0) = -r2;
	x_i(1) = r2;
	y_i(1) = r2;
	z_i(1) = -r2;
	x_i(2) = -r2;
	y_i(2) = -r2;
	z_i(2) = -r2;
	x_i(3) = r2;
	y_i(3) = -r2;
	z_i(3) = -r2;
	x_i(4) = -r2;
	y_i(4) = r2;
	z_i(4) = r2;
	x_i(5) = r2;
	y_i(5) = r2;
	z_i(5) = r2;
	x_i(6) = -r2;
	y_i(6) = -r2;
	z_i(6) = r2;
	x_i(7) = r2;
	y_i(7) = -r2;
	z_i(7) = r2;
	add_face4(0, 1, 5, 4);
	add_face4(1, 3, 7, 5);
	add_face4(3, 2, 6, 7);
	add_face4(2, 0, 4, 6);
	add_face4(6, 7, 5, 4);
	add_face4(2, 3, 1, 0);

	determine_neighbours();
	standard_vertex_labels(FALSE /* f_start_with_zero */);

	permutation P;

	group_generators().m_l(2);
	P.m_l(8);
	P.one();
	P.sscan("(0,1,3,2)(4,5,7,6)", FALSE /* f_v */);
	group_generators_i(0) = P;
	P.m_l(8);
	P.one();
	P.sscan("(0,2,6,4)(1,3,7,5)", FALSE /* f_v */);
	group_generators_i(1) = P;
}

#if 0
void solid::cube4D(int r1, int r2)
{
	solid A, B, C;
	int i, nb_E1, nb_V;

	A.cube(r1);
	B.cube(r2);
	A.join_disjoint(B, C, FALSE);
	// add new edges:
	cout << "adding new edges:" << endl;
	nb_E1 = C.nb_E();
	nb_V = A.nb_V();
	C.v1().realloc(nb_E1 + nb_V);
	C.v2().realloc(nb_E1 + nb_V);
	C.f1().realloc(nb_E1 + nb_V);
	C.f2().realloc(nb_E1 + nb_V);
	for (i = 0; i < nb_V; i++) {
		C.v1()[nb_E1 + i].change_to_integer();
		C.v2()[nb_E1 + i].change_to_integer();
		C.f1()[nb_E1 + i].change_to_integer();
		C.f2()[nb_E1 + i].change_to_integer();
		C.v1_i(nb_E1 + i) = i;
		C.v2_i(nb_E1 + i) = nb_V + i;
		C.f1_i(nb_E1 + i) = 0;
		C.f2_i(nb_E1 + i) = 0;
		}
	C.nb_E() += nb_V;
	vec_generators_aut_cube_nd(4, C.group_generators());
	swap(C);

	cout << "and the vertex labels:" << endl;
	standard_vertex_labels(FALSE /* f_start_with_zero */);
	// cout << *this << endl;
}
#endif

#if 0
void vec_generators_aut_cube_nd(int n, Vector &gen)
{
	Vector gen1, gen2;
	permutation p;
	int i, j, a, b, x, l, ll;
	int *v, *w;
	number_theory_domain NT;
	
	// printf("vec_generators_aut_cube_nd()\n");
	l = NT.i_power_j(2, n);
	v = new int[n];
	w = new int[n];
	vec_generators_symmetric_group(gen2, n);
	ll = gen2.s_l();
	gen1.m_l(n + ll);
	for (x = 0; x < n; x++) {
		p.m_l(l);
		for (i = 0; i < l; i++) {
			number_to_binary(i, v, n);
			if (v[x])
				v[x] = 0;
			else
				v[x] = 1;
			j = binary_to_number(v, n);
			p[i] = j;
			}
		gen1[x] = p;
		// p.println();
		}
	for (x = 0; x < ll; x++) {
		permutation &q = gen2[x].as_permutation();
		p.m_l(l);
		for (i = 0; i < l; i++) {
			number_to_binary(i, v, n);
			for (a = 0; a < n; a++) {
				b = q[a];
				w[b] = v[a];
				}
			j = binary_to_number(w, n);
			p[i] = j;
			}
		// p.println();
		gen1[n + x] = p;
		}
	delete [] v;
	delete [] w;
	gen1.swap(gen);
}
#endif

#undef DEBUG_BINARY_CONVERSION

void number_to_binary(int n, int *v, int digits)
{
	int i;
#ifdef DEBUG_BINARY_CONVERSION
	int n1 = n;
#endif
	
	for (i = 0; i < digits; i++) {
		if (ODD(n))
			v[i] = 1;
		else
			v[i] = 0;
		n >>= 1;
		}
#ifdef DEBUG_BINARY_CONVERSION
	printf("number %d to binary: ", n1);
	for (i = digits - 1; i >= 0; i--) {
		printf("%d", v[i]);
		}
	printf("\n");
#endif
}

int binary_to_number(int *v, int digits)
{
	int i, n = 0;
	
#ifdef DEBUG_BINARY_CONVERSION
	printf("binary ");
	for (i = digits - 1; i >= 0; i--) {
		printf("%d", v[i]);
		}
#endif
	for (i = digits - 1; i >= 0; i--) {
		n <<= 1;
		if (v[i])
			n++;
		}
#ifdef DEBUG_BINARY_CONVERSION
	printf(" to number %d\n", n);
#endif
	return n;
}

void solid::octahedron(int r)
{
	solid A;
	
	A.cube(r);
	A.standard_vertex_labels(FALSE /* f_start_with_zero */);
	A.dual(*this);	
}

void solid::dodecahedron(int r)
{
	permutation P;
	numerics Num;
	
	init();
	double phi, /* phi_2, */ sin_phi_2, R, RR, s, d, dr, h, hh, H, hH;
	int i;
	
	group_generators().m_l(2);
	cout << "first 5-cycle:" << endl;
	P.m_l(20);
	P.one();
	P.sscan("(0,1,2,3,4)(5,6,7,8,9)(10,11,12,13,14)(15,16,17,18,19)", FALSE);
	cout << "P=" << P << endl;
	group_generators_i(0) = P;
	P.m_l(20);
	P.one();
	P.sscan("(0,8)(1,16)(2,9)(3,4)(5,15)(6,11)(7,17)(10,18)(12,19)(13,14)", FALSE);
	cout << "P=" << P << endl;
	group_generators_i(1) = P;
	f_vertex_labels() = FALSE;
	cout << "f_vertex_labels = " << f_vertex_labels() << endl;
	nb_V() = 20;
	cout << "nb_V = " << nb_V() << endl;
	placement().m_l(3);
	x().change_to_vector();
	y().change_to_vector();
	z().change_to_vector();
	x().m_l_n(20);
	y().m_l_n(20);
	z().m_l_n(20);
	cout << "make placement:" << endl;
	phi = 72.;
	//phi_2 = 36.;
	// cout << "angles: phi = " << phi << ", phi/2 = " << phi_2 << endl;
	sin_phi_2 = Num.sin_grad(phi * .5);
	s = 2. * (double) r * sin_phi_2;
	// cout << "s = " << s << endl;
	d = 2. * (double) r * Num.sin_grad(phi);
	// cout << "d = " << d << endl;
	R = 0.5 * d / sin_phi_2;
	// cout << "R = " << R << endl;
	dr = R - r;
	// cout << "dr = " << dr << endl;
	hh = s * s - dr * dr;
	// cout << "hh = " << endl;
	h = square_root(hh);
	// cout << "h = square_root(hh) = " << h << endl;
	RR = R * R;
	// cout << "R^2 = " << RR << endl;
	H = .5 * (RR - hh - r * r) / h;
	// cout << "H = " << H << endl;
	//a = square_root(RR + H * H);
	// cout << "a = " << a << endl;
	hH = h + H;
	// cout << "h + H = " << hH << endl;
	for (i = 0; i < 5; i++) 
	{
		On_circle_int(x(), y(), i, (int)((double) i * phi), r);
		// cout << "make z of point " << i << endl;
		z_i(i) =  - (int) hH;
	}
	for (i = 0; i < 5; i++) 
	{
		On_circle_int(x(), y(), 5 + i, (int)((double) i * phi), (int) R);
		// cout << "make z of point " << 5+i << endl;
		z_i(5 + i) = - (int) H;
	}
	for (i = 0; i < 10; i++) 
	{
		// cout << "make x of point " << 10+i << endl;
		x_i(10 + i) = -x_i(i);
		// cout << "make y of point " << 10+i << endl;
		y_i(10 + i) = -y_i(i);
		// cout << "make z of point " << 10+i << endl;
		z_i(10 + i) = -z_i(i);
	}
	
	cout << "add the faces:" << endl;
	add_face5(0, 1, 2, 3, 4);
	add_face5(0, 1, 6, 18, 5);
	add_face5(1, 2, 7, 19, 6);
	add_face5(2, 3, 8, 15, 7);
	add_face5(3, 4, 9, 16, 8);
	add_face5(4, 0, 5, 17, 9);
	add_face5(5, 18, 13, 12, 17);
	add_face5(6, 19, 14, 13, 18);
	add_face5(7, 15, 10, 14, 19);
	add_face5(8, 16, 11, 10, 15);
	add_face5(9, 17, 12, 11, 16);
	add_face5(10, 11, 12, 13, 14);
	// cout << *this << endl;
	
	cout << "now determine the neighbours:" << endl;
	determine_neighbours();
	// cout << *this << endl;

	cout << "and the vertex labels:" << endl;
	standard_vertex_labels(FALSE /* f_start_with_zero */);
	// cout << *this << endl;
}

void solid::icosahedron(int r)
{
	solid A;
	
	A.dodecahedron(r);
	A.standard_vertex_labels(FALSE /* f_start_with_zero */);
	A.dual(*this);
}

void solid::make_placed_graph(discreta_matrix & incma, Vector& aut_gens, Vector& cycles)
{
	double phi, phi0;
	int i, j, k;
	permutation P;
	int rad_max = 2000;
	int rad_min = 400;
	
	int m = incma.s_m();
	init();
	group_generators() = aut_gens;
	f_vertex_labels() = FALSE;
	
	placement().m_l(3);
	x().change_to_vector();
	y().change_to_vector();
	z().change_to_vector();
	x().m_l_n(m);
	y().m_l_n(m);
	z().m_l_n(m);
	int nb_cycles = cycles.s_l();
	int delta_rad = (rad_max - rad_min) / nb_cycles;
	
	k = 0;
	phi0 = 0.;
	for (j = 0; j < nb_cycles; j++) {
		Vector& cycle = cycles[j].as_vector();
		int cycle_len = cycle.s_l();
		phi = 360. / (double) cycle_len;
		int rad = rad_min + j * delta_rad;
		
#if 1
			phi0 = phi * 0.25 * (j % 4);
#endif
		for (i = 0; i < cycle_len; i++)
		{
			On_circle_int(x(), y(), cycle.s_ii(i), (int) ((double) phi0 + i * phi), rad);
			z_i(k) = 0;
			k++;
		}
	}
	
	nb_V() = m;
	for (j = 0; j < incma.s_n(); j++) {
		int i1 = -1, i2 = -1;
		
		for (i = 0; i < m; i++) {
			if (incma.s_iji(i, j)) {
				if (i1 == -1)
					i1 = i;
				else
					i2 = i;
				}
			}
		cout << "edge " << i1 << " - " << i2 << endl;
		add_edge(i1, i2, 0, 0);
		}


	// determine_neighbours();
	standard_vertex_labels(FALSE /* f_start_with_zero */);
}


void solid::write_graphfile(char *fname)
{
	ofstream fp(fname);
	char str[256];
	int i, l;
	
	cout << "solid::write_graphfile() fname=" << fname;
	cout << " nb_V=" << nb_V() << " nb_E=" << nb_E() << endl; 

	system("rm a");
	system("date >a");
	{
	ifstream fp1("a");
	fp1.getline(str, sizeof(str));
	}
	
	if ((l = strlen(str)) > 0) 
	{
		str[l - 1] = 0;
	}

	fp << "<GRAPH NAME=\"" << fname << "\" NUMBER_OF_VERTICES=\"" << nb_V();
	fp << "\" NUMBER_OF_EDGES=\"" << nb_E() << "\">" << endl;
	fp << "<!-- created by DISCRETA, " << str << " -->" << endl;
	fp << "<COORDS3D_int>" << endl;
	for (i = 0; i < nb_V(); i++) 
	{
		fp << x_i(i) << " " << y_i(i) << " " << z_i(i) << endl; 
	}
	fp << "</COORDS3D_int>" << endl;
	fp << "<EDGELIST>" << endl;
	for (i = 0; i < nb_E(); i++) 
	{
		fp << v1_i(i) << " " << v2_i(i) << endl;
	}
	fp << "</EDGELIST>" << endl;
	if (f_vertex_labels()) 
	{
		fp << "<VERTEXINDICES>" << endl;
		for (i = 0; i < nb_V(); i++) 
		{
			fp << vertex_labels_numeric_i(i) << endl; 
		}
		fp << "</VERTEXINDICES>" << endl;
		fp << "<VERTEXLABELS>" << endl;
		for (i = 0; i < nb_V(); i++) 
		{
			fp << "\"" << vertex_labels_i(i) << "\"" << endl; 
		}
		fp << "</VERTEXLABELS>" << endl;
	}

	fp << "</GRAPH>" << endl;
}

void solid::write_solidfile(char *fname)
{
	ofstream fp(fname);
	char str[256];
	int i, l;
	
	cout << "solid::write_graphfile() fname=" << fname;
	cout << " nb_V=" << nb_V() << " nb_E=" << nb_E() << endl; 

	system("rm a");
	system("date >a");
	{
	ifstream fp1("a");
	fp1.getline(str, sizeof(str));
	}
	
	if ((l = strlen(str)) > 0) 
	{
		str[l - 1] = 0;
	}

	fp << "<GRAPH NAME=\"" << fname << "\" NUMBER_OF_VERTICES=\"" << nb_V();
	fp << "\" NUMBER_OF_EDGES=\"" << nb_E() << "\"" << endl;
	fp << "<!-- created by DISCRETA, " << str << " -->" << endl;
	fp << "<COORDS3D_int>" << endl;
	for (i = 0; i < nb_V(); i++) 
	{
		fp << x_i(i) << " " << y_i(i) << " " << z_i(i) << endl; 
	}
	fp << "</COORDS3D_int>" << endl;
	fp << "<EDGELIST>" << endl;
	for (i = 0; i < nb_E(); i++) 
	{
		fp << v1_i(i) << " " << v2_i(i) << endl;
	}
	fp << "</EDGELIST>" << endl;
	fp << "<FACELIST>" << endl;
	
	// oriented_solid();
	
	
	fp << "</FACELIST>" << endl;
	if (f_vertex_labels()) 
	{
		fp << "<VERTEXLABELS>" << endl;
		for (i = 0; i < nb_V(); i++) 
		{
			fp << "\"" << vertex_labels_i(i) << "\"" << endl; 
		}
		fp << "</VERTEXLABELS>" << endl;
	}
	fp << "<VERTEXINDICES>" << endl;
	for (i = 0; i < nb_V(); i++) 
	{
		fp << i + 1 << endl; 
	}
	fp << "</VERTEXINDICES>" << endl;

	fp << "</GRAPH>" << endl;
}

}}
