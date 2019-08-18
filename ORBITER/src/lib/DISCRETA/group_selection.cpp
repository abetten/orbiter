// group_selection.cpp
//
// Anton Betten
// 28.07.2000
// moved from D2 to ORBI Nov 15, 2007

#include "orbiter.h"

#include <math.h> // for sqrt

#undef CHANGE_KIND_VERBOSE
#undef COPY_VERBOSE
#undef DEBUG_intEGRAL_DIVISION


using namespace std;


namespace orbiter {
namespace discreta {

static double square_root(double x);
static void add_gsel(Vector & gsel, group_selection_type type, int val1, int val2, char *s);
static void my_get_generators(discreta_base& a, Vector& gens);
static int compose_well_known_group(group_selection & gs, Vector & S, int & height, 
	ostream & g_label, ostream & g_label_tex, int verbose_level);
static int compose_linear_group(group_selection & gs, Vector & S, int & height, 
	ostream & g_label, ostream & g_label_tex, int verbose_level);
static int compose_group_unary_operator(group_selection & gs, Vector & S, int & height, 
	ostream & g_label, ostream & g_label_tex, int verbose_level);
static int compose_group_binary_operator(group_selection & gs, Vector & S, int & height, 
	ostream & g_label, ostream & g_label_tex, int verbose_level);
static int compose_group_of_solid(group_selection & gs, Vector & S, int & height, 
	ostream & g_label, ostream & g_label_tex, int verbose_level);

static double square_root(double x)
{
	cout << "group_selection::square_root() do not have sqrt function" << endl;
	exit(1);
	//return sqrt(x);
}

group_selection::group_selection() : Vector()
{
	k = GROUP_SELECTION;
}

group_selection::group_selection(const discreta_base &x)
	// copy constructor:    this := x
{
	cout << "group_selection::copy constructor for object: " << const_cast<discreta_base &>(x) << "\n";
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

group_selection& group_selection::operator = (const discreta_base &x)
	// copy assignment
{
	cout << "group_selection::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void group_selection::settype_group_selection()
{
	OBJECTSELF s;
	
	s = self;
	new(this) group_selection;
	self = s;
	k = GROUP_SELECTION;
}

group_selection::~group_selection()
{
	freeself_group_selection();
}

void group_selection::freeself_group_selection()
{
	// cout << "group_selection::freeself_group_selection()\n";
	freeself_vector();
}

kind group_selection::s_virtual_kind()
{
	return GROUP_SELECTION;
}

void group_selection::copyobject_to(discreta_base &x)
{
#ifdef COPY_VERBOSE
	cout << "group_selection::copyobject_to()\n";
	print_as_vector(cout);
#endif
	Vector::copyobject_to(x);
	x.as_group_selection().settype_group_selection();
#ifdef COPY_VERBOSE
	x.as_group_selection().print_as_vector(cout);
#endif
}

ostream& group_selection::print(ostream& ost)
{
	// cout << group_selection_type_as_text((group_selection_type) type()) << endl;
	group_selection_type t = (group_selection_type) type();
	int v1 = val1();
	int v2 = val2();
	char *str = s().s();
	
	if (t == SL) {
		cout << "SL_" << v1 << "_" << v2;
		}
	else if (t == GL) {
		cout << "GL_" << v1 << "_" << v2;
		}
	else if (t == SSL) {
		cout << "SSL_" << v1 << "_" << v2;
		}
	else if (t == GGL) {
		cout << "GGL_" << v1 << "_" << v2;
		}
	else if (t == PSL) {
		cout << "PSL_" << v1 << "_" << v2;
		}
	else if (t == PGL) {
		cout << "PGL_" << v1 << "_" << v2;
		}
	else if (t == PSSL) {
		cout << "PSSL_" << v1 << "_" << v2;
		}
	else if (t == PGGL) {
		cout << "PGGL_" << v1 << "_" << v2;
		}
	else if (t == ASL) {
		cout << "ASL_" << v1 << "_" << v2;
		}
	else if (t == AGL) {
		cout << "AGL_" << v1 << "_" << v2;
		}
	else if (t == ASSL) {
		cout << "ASSL_" << v1 << "_" << v2;
		}
	else if (t == AGGL) {
		cout << "AGGL_" << v1 << "_" << v2;
		}
	else if (t == Affine_translations) {
		cout << "Affine_translations" << v1 << "_" << v2;
		}
	else if (t == A5_in_PSL) {
		cout << "A5_in_PSL_2_" << v2;
		}
	else if (t == S4_in_PSL) {
		cout << "S4_in_PSL_2_" << v2;
		}
	else if (t == On_projective_lines) {
		cout << "On_projective_lines";
		}
	
	else if (t == Trivial) {
		cout << "Id_" << v1;
		}
	else if (t == Symmetric) {
		cout << "Symmetric_" << v1;
		}
	else if (t == Alternating) {
		cout << "Alternating_" << v1;
		}
	else if (t == Dihedral) {
		cout << "Dihedral_" << v1;
		}
	else if (t == Cyclic) {
		cout << "Cyclic_" << v1;
		}
	else if (t == Holomorph_of_cyclic) {
		cout << "Holomorph_of_cyclic_" << v1;
		}
	else if (t == Subgroup_of_holomorph_of_cyclic_group) {
		cout << "Subgroup_of_holomorph_of_cyclic_group_" << v1;
		}
	else if (t == Mathieu) {
		cout << "Mathieu_" << v1;
		}
	else if (t == From_file) {
		cout << "From_file_" << str;
		}
	else if (t == Permutation_generator) {
		cout << "Permutation_generator_" << str;
		}
	else if (t == Higman_Sims_176) {
		cout << "Higman_Sims_176";
		}
	
	else if (t == On_2_sets) {
		cout << "On_2_sets";
		}
	else if (t == On_2_tuples) {
		cout << "On_2_tuples";
		}
	else if (t == On_3_sets) {
		cout << "On_3_sets";
		}
	else if (t == On_injective_2_tuples) {
		cout << "On_injective_2_tuples";
		}
	else if (t == Add_fixpoint) {
		cout << "Add_fixpoint";
		}
	else if (t == Stabilize_point) {
		cout << "Stabilize_point";
		}
	else if (t == Holomorph) {
		cout << "Holomorph";
		}
	else if (t == Even_subgroup) {
		cout << "Even_subgroup";
		}
	
	else if (t == Comma) {
		cout << "Comma";
		}
	else if (t == Direct_sum) {
		cout << "Direct_sum";
		}
	else if (t == Direct_product) {
		cout << "Direct_product";
		}
	else if (t == Wreath_product) {
		cout << "Wreath_product";
		}
	else if (t == Exponentiation) {
		cout << "Exponentiation";
		}
	else if (t == On_mappings) {
		cout << "On_mappings";
		}

	else if (t == Solid_Tetrahedron) {
		cout << "Solid_Tetrahedron";
		}
	else if (t == Solid_Cube) {
		cout << "Solid_Cube";
		}
	else if (t == Solid_Octahedron) {
		cout << "Solid_Octahedron";
		}
	else if (t == Solid_Dodecahedron) {
		cout << "Solid_Dodecahedron";
		}
	else if (t == Solid_Icosahedron) {
		cout << "Solid_Icosahedron";
		}
	else if (t == Solid_Cube4D) {
		cout << "Solid_Cube4D";
		}
	else if (t == Solid_truncate) {
		cout << "Solid_truncate";
		}
	else if (t == Solid_dual) {
		cout << "Solid_dual";
		}
	else if (t == Solid_truncate_dode) {
		cout << "Solid_truncate_dode";
		}
	else if (t == Solid_truncate_cube) {
		cout << "Solid_truncate_cube";
		}
	else if (t == Solid_relabel_points) {
		cout << "Solid_relabel_points";
		}
	else if (t == Solid_induced_group_on_edges) {
		cout << "Solid_induced_group_on_edges";
		}
	else if (t == Solid_midpoints_of_edges) {
		cout << "Solid_midpoints_of_edges";
		}
	else if (t == Solid_add_central_point) {
		cout << "Solid_add_central_point";
		}
	else if (t == Solid_add_central_involution) {
		cout << "Solid_add_central_involution";
		}
	else if (t == Solid_Cubussimus) {
		cout << "Solid_Cubussimus";
		}
	else if (t == Solid_Dodesimum) {
		cout << "Solid_Dodesimum";
		}
	else if (t == Solid_CubeEE) {
		cout << "Solid_CubeEE";
		}
	else if (t == Solid_CubeEE_russian) {
		cout << "Solid_CubeEE_russian";
		}
	return ost;
}

void group_selection::init(group_selection_type t, int v1, int v2, char *str)
{
	m_l(4);
	c_kind(GROUP_SELECTION);
	
	type() = (int) t;
	val1() = v1;
	val2() = v2;
	if (str)
		s().init(str);
	else
		s().init("");
}

void compose_gsel_from_strings(Vector &gsel, int num_args, char **args)
{
	int i = 0, val1, val2;
	char *p, *str;
	
	gsel.m_l(0);
	while (i < num_args) {
		p = args[i++];
		val1 = 0;
		val2 = 0;
		str = NULL;
		cout << "compose_gsel_from_strings(): p = " << p << endl;
		
		// the linear groups:
		if (strcmp(p, "SL") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, SL, val1, val2, str);
			}
		else if (strcmp(p, "GL") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, GL, val1, val2, str);
			}
		else if (strcmp(p, "SSL") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, SSL, val1, val2, str);
			}
		else if (strcmp(p, "GGL") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, GGL, val1, val2, str);
			}
		else if (strcmp(p, "PSL") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, PSL, val1, val2, str);
			}
		else if (strcmp(p, "PGL") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, PGL, val1, val2, str);
			}
		else if (strcmp(p, "PSSL") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, PSSL, val1, val2, str);
			}
		else if (strcmp(p, "PGGL") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, PGGL, val1, val2, str);
			}
		else if (strcmp(p, "ASL") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, ASL, val1, val2, str);
			}
		else if (strcmp(p, "AGL") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, AGL, val1, val2, str);
			}
		else if (strcmp(p, "ASSL") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, ASSL, val1, val2, str);
			}
		else if (strcmp(p, "AGGL") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, AGGL, val1, val2, str);
			}
		else if (strcmp(p, "Affine_translations") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, Affine_translations, val1, val2, str);
			}
		else if (strcmp(p, "PSU_3_Q2") == 0) {
			val2 = atoi(args[i++]);
			add_gsel(gsel, PSU_3_Q2, val1, val2, str);
			}
		else if (strcmp(p, "A5_in_PSL") == 0) {
			val1 = 0;
			val2 = atoi(args[i++]);
			add_gsel(gsel, A5_in_PSL, val1, val2, str);
			}
		else if (strcmp(p, "S4_in_PSL") == 0) {
			val1 = 0;
			val2 = atoi(args[i++]);
			add_gsel(gsel, S4_in_PSL, val1, val2, str);
			}
		else if (strcmp(p, "On_projective_lines") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, On_projective_lines, val1, val2, str);
			}

		// the well known groups:
		else if (strcmp(p, "Trivial") == 0 || strcmp(p, "TRIVIAL") == 0 || strcmp(p, "Id") == 0) {
			val1 = atoi(args[i++]);
			add_gsel(gsel, Trivial, val1, val2, str);
			}
		else if (strcmp(p, "Symmetric") == 0 || strcmp(p, "SYM") == 0 || strcmp(p, "S") == 0) {
			val1 = atoi(args[i++]);
			add_gsel(gsel, Symmetric, val1, val2, str);
			}
		else if (strcmp(p, "Alternating") == 0 || strcmp(p, "ALT") == 0 || strcmp(p, "A") == 0) {
			val1 = atoi(args[i++]);
			add_gsel(gsel, Alternating, val1, val2, str);
			}
		else if (strcmp(p, "Dihedral") == 0 || strcmp(p, "DIHEDRAL") == 0 || strcmp(p, "D") == 0) {
			val1 = atoi(args[i++]);
			add_gsel(gsel, Dihedral, val1, val2, str);
			}
		else if (strcmp(p, "Cyclic") == 0 || strcmp(p, "CYCLIC") == 0 || strcmp(p, "C") == 0) {
			val1 = atoi(args[i++]);
			add_gsel(gsel, Cyclic, val1, val2, str);
			}
		else if (strcmp(p, "Holomorph_of_cyclic") == 0 || strcmp(p, "HOLOMORPH_OF_CYCLIC_GROUP") == 0 || strcmp(p, "HolC") == 0) {
			val1 = atoi(args[i++]);
			add_gsel(gsel, Holomorph_of_cyclic, val1, val2, str);
			}
		else if (strcmp(p, "Subgroup_of_holomorph_of_cyclic_group") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, Subgroup_of_holomorph_of_cyclic_group, val1, val2, str);
			}
		else if (strcmp(p, "Sn_wreath_Sm") == 0) {
			val1 = atoi(args[i++]);
			val2 = atoi(args[i++]);
			add_gsel(gsel, Sn_wreath_Sm, val1, val2, str);
			}
		else if (strcmp(p, "Permutation_generator") == 0 || strcmp(p, "PERM") == 0) {
			str = args[i++];
			add_gsel(gsel, Permutation_generator, val1, val2, str);
			}
		else if (strcmp(p, "M11") == 0) {
			val1 = 11;
			add_gsel(gsel, Mathieu, val1, val2, str);
			}
		else if (strcmp(p, "M12") == 0) {
			val1 = 12;
			add_gsel(gsel, Mathieu, val1, val2, str);
			}
		else if (strcmp(p, "M23") == 0) {
			val1 = 23;
			add_gsel(gsel, Mathieu, val1, val2, str);
			}
		else if (strcmp(p, "M24") == 0) {
			val1 = 24;
			add_gsel(gsel, Mathieu, val1, val2, str);
			}
		else if (strcmp(p, "Mathieu") == 0) {
			val1 = atoi(args[i++]);
			add_gsel(gsel, Mathieu, val1, val2, str);
			}
		else if (strcmp(p, "Higman_Sims_176") == 0 || strcmp(p, "HS176") == 0) {
			add_gsel(gsel, Higman_Sims_176, val1, val2, str);
			}
		else if (strcmp(p, "From_file") == 0 || strcmp(p, "File") == 0) {
			str = args[i++];
			add_gsel(gsel, From_file, val1, val2, str);
			}


		// unary operators:
		else if (strcmp(p, "On_2_sets") == 0 || strcmp(p, "[2]") == 0) {
			add_gsel(gsel, On_2_sets, val1, val2, str);
			}
		else if (strcmp(p, "On_2_tuples") == 0 || strcmp(p, "(2)") == 0) {
			add_gsel(gsel, On_2_tuples, val1, val2, str);
			}
		else if (strcmp(p, "On_injective_2_tuples") == 0 || strcmp(p, "(2)i") == 0) {
			add_gsel(gsel, On_injective_2_tuples, val1, val2, str);
			}
		else if (strcmp(p, "On_3_sets") == 0 || strcmp(p, "[3]") == 0) {
			add_gsel(gsel, On_3_sets, val1, val2, str);
			}
		else if (strcmp(p, "Add_fixpoint") == 0 || strcmp(p, "+") == 0) {
			add_gsel(gsel, Add_fixpoint, val1, val2, str);
			}
		else if (strcmp(p, "Stabilize_point") == 0 || strcmp(p, "-") == 0) {
			add_gsel(gsel, Stabilize_point, val1, val2, str);
			}
		else if (strcmp(p, "Even_subgroup") == 0) {
			add_gsel(gsel, Even_subgroup, val1, val2, str);
			}

		// binary operators:
		else if (strcmp(p, "Comma") == 0 || strcmp(p, ",") == 0) {
			add_gsel(gsel, Comma, val1, val2, str);
			}
		else if (strcmp(p, "Direct_sum") == 0 || strcmp(p, "x") == 0) {
			add_gsel(gsel, Direct_sum, val1, val2, str);
			}
		else if (strcmp(p, "Direct_product") == 0 || strcmp(p, "X") == 0) {
			add_gsel(gsel, Direct_product, val1, val2, str);
			}
		else if (strcmp(p, "Wreath_product") == 0 || strcmp(p, "wr") == 0) {
			add_gsel(gsel, Wreath_product, val1, val2, str);
			}
		else if (strcmp(p, "Exponentiation") == 0 || strcmp(p, "exp") == 0) {
			add_gsel(gsel, Exponentiation, val1, val2, str);
			}
		else if (strcmp(p, "On_mappings") == 0) {
			add_gsel(gsel, On_mappings, val1, val2, str);
			}

		// solids:
		else if (strcmp(p, "Solid_Tetrahedron") == 0 || strcmp(p, "Tetrahedron") == 0 || strcmp(p, "Tetra") == 0) {
			add_gsel(gsel, Solid_Tetrahedron, val1, val2, str);
			}
		else if (strcmp(p, "Solid_Cube") == 0 || strcmp(p, "Cube") == 0) {
			add_gsel(gsel, Solid_Cube, val1, val2, str);
			}
		else if (strcmp(p, "Solid_Octahedron") == 0 || strcmp(p, "Octahedron") == 0 || strcmp(p, "Octa") == 0) {
			add_gsel(gsel, Solid_Octahedron, val1, val2, str);
			}
		else if (strcmp(p, "Solid_Dodecahedron") == 0 || strcmp(p, "Dodecahedron") == 0 || strcmp(p, "Dode") == 0) {
			add_gsel(gsel, Solid_Dodecahedron, val1, val2, str);
			}
		else if (strcmp(p, "Solid_Icosahedron") == 0 || strcmp(p, "Icosahedron") == 0 || strcmp(p, "Ico") == 0) {
			add_gsel(gsel, Solid_Icosahedron, val1, val2, str);
			}
		else if (strcmp(p, "Solid_Cube4D") == 0 || strcmp(p, "Cube4D") == 0) {
			add_gsel(gsel, Solid_Cube4D, val1, val2, str);
			}
		else if (strcmp(p, "Solid_truncate") == 0 || strcmp(p, "truncate") == 0 || strcmp(p, "Truncate") == 0) {
			add_gsel(gsel, Solid_truncate, val1, val2, str);
			}
		else if (strcmp(p, "Solid_truncate_dode") == 0 || strcmp(p, "truncate_dode") == 0) {
			add_gsel(gsel, Solid_truncate_dode, val1, val2, str);
			}
		else if (strcmp(p, "Solid_truncate_cube") == 0 || strcmp(p, "truncate_cube") == 0) {
			add_gsel(gsel, Solid_truncate_cube, val1, val2, str);
			}
		else if (strcmp(p, "Solid_dual") == 0 || strcmp(p, "Dual") == 0 || strcmp(p, "dual") == 0) {
			add_gsel(gsel, Solid_dual, val1, val2, str);
			}
		else if (strcmp(p, "Solid_relabel_points") == 0 || strcmp(p, "relabel_points") == 0) {
			add_gsel(gsel, Solid_relabel_points, val1, val2, str);
			}
		else if (strcmp(p, "Solid_induced_group_on_edges") == 0 || strcmp(p, "induced_group_on_edges") == 0) {
			add_gsel(gsel, Solid_induced_group_on_edges, val1, val2, str);
			}
		else if (strcmp(p, "Solid_midpoints_of_edges") == 0 || strcmp(p, "midpoints_of_edges") == 0 || strcmp(p, "Edge_midpoints") == 0) {
			add_gsel(gsel, Solid_midpoints_of_edges, val1, val2, str);
			}
		else if (strcmp(p, "Solid_add_central_point") == 0 || strcmp(p, "add_central_point") == 0) {
			add_gsel(gsel, Solid_add_central_point, val1, val2, str);
			}
		else if (strcmp(p, "Solid_add_central_involution") == 0 || strcmp(p, "add_central_involution") == 0) {
			add_gsel(gsel, Solid_add_central_involution, val1, val2, str);
			}
		else if (strcmp(p, "Solid_Cubussimus") == 0 || strcmp(p, "Cubussimus") == 0) {
			add_gsel(gsel, Solid_Cubussimus, val1, val2, str);
			}
		else if (strcmp(p, "Solid_Dodesimum") == 0 || strcmp(p, "Dodesimum") == 0) {
			add_gsel(gsel, Solid_Dodesimum, val1, val2, str);
			}
		else if (strcmp(p, "Solid_CubeEE") == 0 || strcmp(p, "CubeEE") == 0) {
			add_gsel(gsel, Solid_CubeEE, val1, val2, str);
			}
		else if (strcmp(p, "Solid_CubeEE_russian") == 0 || strcmp(p, "CubeEE_russian") == 0) {
			add_gsel(gsel, Solid_CubeEE_russian, val1, val2, str);
			}
		else {
			cout << "unrecognized: " << p << endl;
			}
		}
	
}

static void add_gsel(Vector & gsel, group_selection_type type, int val1, int val2, char *s)
{
	group_selection gs;
	
	gs.init(type, val1, val2, s);
	gsel.append(gs);
	cout << "added item: " << gs << endl;
}


const char *group_selection_type_as_text(group_selection_type t)
{
	switch (t) {
		case SL: return "SL";
		case GL: return "GL";
		case SSL: return "SSL";
		case GGL: return "GGL";
		case PSL: return "PSL";
		case PGL: return "PGL";
		case PSSL: return "PSSL";
		case PGGL: return "PGGL";
		case ASL: return "ASL";
		case AGL: return "AGL";
		case ASSL: return "ASSL";
		case AGGL: return "AGGL";
		case Affine_translations: return "Affine_translations";
		case PSU_3_Q2: return "PSU_3_Q2";
		case Suzuki: return "Suzuki";
		case A5_in_PSL: return "A5_in_PSL";
		case S4_in_PSL: return "S4_in_PSL";
		case On_projective_lines: return "On_projective_lines";

		case Trivial: return "Trivial";
		case Symmetric: return "Symmetric";
		case Alternating: return "Alternating";
		case Dihedral: return "Dihedral"; 
		case Cyclic: return "Cyclic";
		case Holomorph_of_cyclic: return "Holomorph_of_cyclic"; 
		case Subgroup_of_holomorph_of_cyclic_group: return "Subgroup_of_holomorph_of_cyclic_group";
		case Sn_wreath_Sm: return "Sn_wreath_Sm";
		case Mathieu: return "Mathieu";
		case From_file: return "From_file";
		case Permutation_generator: return "Permutation_generator"; 
		case Higman_Sims_176: return "Higman_Sims_176";

		case On_2_sets: return "On_2_sets";
		case On_2_tuples: return "On_2_tuples";
		case On_3_sets: return "On_3_sets";
		case On_injective_2_tuples: return "On_injective_2_tuples";
		case Add_fixpoint: return "Add_fixpoint";
		case Stabilize_point: return "Stabilize_point";
		case Holomorph: return "Holomorph";
		case Even_subgroup: return "Even_subgroup";

		case Comma: return "Comma";
		case Direct_sum: return "Direct_sum";
		case Direct_product: return "Direct_product";
		case Wreath_product: return "Wreath_product";
		case Exponentiation: return "Exponentiation";
		case On_mappings: return "On_mappings";

		case Solid_Tetrahedron: return "Solid_Tetrahedron";
		case Solid_Cube: return "Solid_Cube";
		case Solid_Octahedron: return "Solid_Octahedron";
		case Solid_Dodecahedron: return "Solid_Dodecahedron";
		case Solid_Icosahedron: return "Solid_Icosahedron";
		case Solid_Cube4D: return "Solid_Cube4D"; 
		case Solid_truncate: return "Solid_truncate";
		case Solid_dual: return "Solid_dual";
		case Solid_truncate_dode: return "Solid_truncate_dode";
		case Solid_truncate_cube: return "Solid_truncate_cube";
		case Solid_relabel_points: return "Solid_relabel_points";
		case Solid_induced_group_on_edges: return "Solid_induced_group_on_edges";
		case Solid_midpoints_of_edges: return "Solid_midpoints_of_edges";
		case Solid_add_central_point: return "Solid_add_central_point";
		case Solid_add_central_involution: return "Solid_add_central_involution";
		case Solid_Cubussimus: return "Solid_Cubussimus"; 
		case Solid_Dodesimum: return "Solid_Dodesimum"; 
		case Solid_CubeEE: return "Solid_CubeEE";
		case Solid_CubeEE_russian: return "Solid_CubeEE_russian";
		
		}
	return "unknown";
}

#define MAX_STACK_SIZE 1000

void compose_group(Vector & gsel, Vector & gens, 
	hollerith & group_label, hollerith & group_label_tex, hollerith & acting_on,
	int verbose_level)
{
	Vector Stack;
	int height = 0, i, l;
	char *g_label;
	char *g_label_tex;
	ostringstream gl;
	ostringstream glt;
	
	Stack.m_l(MAX_STACK_SIZE);
	for (i = 0; i < gsel.s_l(); i++) {
		group_selection & gs = gsel[i].as_group_selection();
		if (compose_well_known_group(gs, Stack, height, gl, glt, verbose_level)) {
			}
		else if (compose_linear_group(gs, Stack, height, gl, glt, verbose_level)) {
			}
		else if (compose_group_unary_operator(gs, Stack, height, gl, glt, verbose_level)) {
			}
		else if (compose_group_binary_operator(gs, Stack, height, gl, glt, verbose_level)) {
			}
		else if (compose_group_of_solid(gs, Stack, height, gl, glt, verbose_level)) {
			}
		else {
			cout << "compose_group() group_selection type not yet implemented" << endl;
			exit(1);
			}
		} // next i
	
	cout << "compose_group() finished, stack height = " << height << endl;
	if (height != 1) {
		cout << "compose_group(): error: height != 1" << endl;
		}
	gl << ends;
	glt << ends;
	
	l = gl.str().length();
	g_label = new char [l + 1];
	gl.str().copy(g_label, l, 0);
	g_label[l] = 0;
	group_label.init(g_label);
	
	l = glt.str().length();
	g_label_tex = new char [l + 1];
	glt.str().copy(g_label_tex, l, 0);
	g_label_tex[l] = 0;
	group_label_tex.init(g_label_tex);
	
	// gens = Stack[0].as_vector();
	
	my_get_generators(Stack[0], gens);
	
	if (Stack[0].s_kind() == SOLID) {
		cout << "created solid " << g_label << endl;
		hollerith fname;
		
		fname.init(g_label);
		fname.append(".graph");
		Stack[0].as_solid().write_graphfile(fname.s());
		
		{
		ofstream f("solid_name");
		f << fname.s() << endl;
		}
		acting_on.init(fname.s());
		}
	else {
		acting_on.init("");
		// system("rm solid_name");
		}
	delete [] g_label;
	delete [] g_label_tex;
}

static void my_get_generators(discreta_base& a, Vector& gens)
{
	if (a.s_kind() == SOLID) {
		gens = a.as_solid().group_generators();
		}
	else if (a.s_kind() == VECTOR) {
		gens = a.as_vector();
		}
	else {
		cout << "my_get_generators: neither solid nor vector object\n";
		exit(1);
		}
}
static int compose_well_known_group(group_selection & gs, Vector & S, int & height, 
	ostream & g_label, ostream & g_label_tex, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_selection_type t = (group_selection_type) gs.type();
	Vector v;
	Vector gen;
	
	if (t == Trivial) {
		int deg = gs.val1();
		if (f_v) {
			cout << "compose_well_known_group(): stack height = " << height << 
				" Trivial group on " << deg << " points" << endl;
			}
		vec_generators_trivial_group(gen, deg);
		g_label << "Id" << deg;
		g_label_tex << "{\\bf Id}_{" << deg << "}";
		S[height++] = gen;
		}
	else if (t == Symmetric) {
		int deg = gs.val1();
		if (f_v) {
			cout << "compose_well_known_group(): stack height = " << height << 
				" Symmetric group on " << deg << " points" << endl;
			}
		vec_generators_symmetric_group(gen, deg);
		g_label << "S" << deg;
		g_label_tex << "{\\bf S}_{" << deg << "}";
		S[height++] = gen;
		}
	else if (t == Alternating) {
		int deg = gs.val1();
		if (f_v) {
			cout << "compose_well_known_group(): stack height = " << height << 
				" Alternating group on " << deg << " points" << endl;
			}
		vec_generators_alternating_group(gen, deg);
		g_label << "A" << deg;
		g_label_tex << "{\\bf A}_{" << deg << "}";
		S[height++] = gen;
		}
	else if (t == Dihedral) {
		int deg = gs.val1();
		if (f_v) {
			cout << "compose_well_known_group(): stack height = " << height << 
				" Dihedral group on " << deg << " points" << endl;
			}
		vec_generators_dihedral_group(gen, deg);
		g_label << "D" << deg;
		g_label_tex << "{\\bf D}_{" << deg << "}";
		S[height++] = gen;
		}
	else if (t == Cyclic) {
		int deg = gs.val1();
		if (f_v) {
			cout << "compose_well_known_group(): stack height = " << height << 
				" Cyclic group on " << deg << " points" << endl;
			}
		vec_generators_cyclic_group(gen, deg);
		g_label << "C" << deg;
		g_label_tex << "{\\bf C}_{" << deg << "}";
		S[height++] = gen;
		}
	else if (t == Holomorph_of_cyclic) {
		int deg = gs.val1();
		if (f_v) {
			cout << "compose_well_known_group(): stack height = " << height << 
				" Holomorph_of_cyclic group on " << deg << " points" << endl;
			}
		vec_hol_of_cyclic_group(gen, deg);
		g_label << "HolC" << deg;
		g_label_tex << "{\\bf Hol(C_{" << deg << "})}";
		S[height++] = gen;
		}
	else if (t == Subgroup_of_holomorph_of_cyclic_group) {
		int deg = gs.val1();
		int index = gs.val2();
		if (f_v) {
			cout << "compose_well_known_group(): stack height = " << height << 
				" Subgroup_of_holomorph_of_cyclic_group group on " << deg << " points, index=" << index << endl;
			}
		vec_subgroup_of_hol_of_cyclic_group(gen, deg, index);
		g_label << "HolC" << deg << "index" << index;
		g_label_tex << "{\\bf Hol(C_{" << deg << "})}index" << index;
		S[height++] = gen;
		}
	else if (t == Sn_wreath_Sm) {
		int n = gs.val1();
		int m = gs.val2();
		if (f_v) {
			cout << "compose_well_known_group(): stack height = " << height << 
				" Sn_wreath_Sm group n=" << n << " m=" << m << endl;
			}
		vec_generators_Sn_wreath_Sm(n, m, gen);
		g_label << "S" << n << "wrS" << m;
		g_label_tex << "{\\bf S_{" << n << "}}\\wr {\\bf S_{" << m << "}}";
		S[height++] = gen;
		}
	else if (t == Mathieu) {
		int deg = gs.val1();
		if (f_v) {
			cout << "compose_well_known_group(): stack height = " << height << 
				" Mathieu group on " << deg << " points" << endl;
			}
		vec_generators_Mathieu_n(gen, deg);
		g_label << "M" << deg;
		g_label_tex << "{\\bf M_{" << deg << "}}";
		S[height++] = gen;
		}
	else if (t == From_file) {
		char *fname = gs.s().s();
		int f_cyclic_notation;
		char ext[1000];
		
		if (f_v) {
			cout << "compose_well_known_group(): stack height = " << height << 
				" generators from file " << fname << endl;
			}
		get_extension_if_present(fname, ext);
		if (strlen(ext)) {
			if (strcmp(ext, ".txt") == 0) {
				read_file_of_generators(gen, fname);
				}
			else if (strcmp(ext, ".xml") == 0) {
				read_file_of_generators_xml(gen, fname, f_cyclic_notation, f_v);
				}
			else {
				cout << "The file name has an extension which is unknown to me (" << ext << ")." << endl;
				cout << "I am assuming the file is in .xml format." << endl;
				cout << "Next time, please give me a .txt file or a .xml file." << endl;
				read_file_of_generators_xml(gen, fname, f_cyclic_notation, f_v);
				}
			chop_off_extension_if_present(fname, ext);
			}
		else {
			cout << "the file name has no extension, but I would like to have one!" << endl;
			cout << "I am assuming the file is in .xml format." << endl;
			cout << "Next time, please give me a .txt file or a .xml file." << endl;
			read_file_of_generators_xml(gen, fname, f_cyclic_notation, f_v);
			}
		g_label << fname;
		g_label_tex << "{\\bf ";
		output_texable_string(g_label_tex, fname);
		g_label_tex << " }";
		S[height++] = gen;
		}
	else if (t == Permutation_generator) {
		char *s = gs.s().s();
		if (f_v) {
			cout << "compose_well_known_group(): stack height = " << height << 
				" Permutation_generator " << s << endl;
			}
		permutation p;
		Vector gen;
		
		p.sscan(s, FALSE);
		gen.m_l(0);
		gen.append(p);
		g_label << "perm";
		g_label_tex << "{\\bf permutation:" << s << "}";
		S[height++] = gen;
		}
	else if (t == Higman_Sims_176) {
		cout << "Higman_Sims_176 not yet implemented" << endl;
		exit(1);
		}
	else
		return FALSE;
	return TRUE;
}

static int compose_linear_group(group_selection & gs, Vector & S, int & height, 
	ostream & g_label, ostream & g_label_tex, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_selection_type t = (group_selection_type) gs.type();
	Vector v;
	
	if (t == SL) {
		int n = gs.val1();
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" SL(" << n << "," << q << ")" << endl;
			}
		Vector gen;
		vec_generators_GL_n_q_affine_representation(gen, n, q,
				TRUE /* f_special */, FALSE /* f_frobenius */, FALSE /* f_translations */, verbose_level);
		g_label << "SL_" << n << "_" << q;
		g_label_tex << "{\\bf SL(" << n << "," << q << ")}";
		S[height++] = gen;
		}
	else if (t == GL) {
		int n = gs.val1();
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" GL(" << n << "," << q << ")" << endl;
			}
		Vector gen;
		vec_generators_GL_n_q_affine_representation(gen, n, q,
				FALSE /* f_special */, FALSE /* f_frobenius */, FALSE /* f_translations */, verbose_level);
		g_label << "GL_" << n << "_" << q;
		g_label_tex << "{\\bf GL(" << n << "," << q << ")}";
		S[height++] = gen;
		}
	else if (t == SSL) {
		int n = gs.val1();
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" SSL(" << n << "," << q << ")" << endl;
			}
		Vector gen;
		vec_generators_GL_n_q_affine_representation(gen, n, q,
				TRUE /* f_special */, TRUE /* f_frobenius */, FALSE /* f_translations */, verbose_level);
		g_label << "SSL_" << n << "_" << q;
		g_label_tex << "{\\bf \\Sigma L(" << n << "," << q << ")}";
		S[height++] = gen;
		}
	else if (t == GGL) {
		int n = gs.val1();
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" GGL(" << n << "," << q << ")" << endl;
			}
		Vector gen;
		vec_generators_GL_n_q_affine_representation(gen, n, q,
				FALSE /* f_special */, TRUE /* f_frobenius */, FALSE /* f_translations */, verbose_level);
		g_label << "GGL_" << n << "_" << q;
		g_label_tex << "{\\bf \\Gamma L(" << n << "," << q << ")}";
		S[height++] = gen;
		}
	else if (t == PSL) {
		int n = gs.val1();
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" PSL(" << n << "," << q << ")" << endl;
			}
		Vector gen;
		vec_generators_GL_n_q_projective_representation(gen, n, q,
				TRUE /* f_special */, FALSE /* f_frobenius */, TRUE /* f_modified */, verbose_level);
		g_label << "PSL_" << n << "_" << q;
		g_label_tex << "{\\bf PSL(" << n << "," << q << ")}";
		S[height++] = gen;
		}
	else if (t == PGL) {
		int n = gs.val1();
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" PGL(" << n << "," << q << ")" << endl;
			}
		Vector gen;
		vec_generators_GL_n_q_projective_representation(gen, n, q,
				FALSE /* f_special */, FALSE /* f_frobenius */, TRUE /* f_modified */, verbose_level);
		g_label << "PGL_" << n << "_" << q;
		g_label_tex << "{\\bf PGL(" << n << "," << q << ")}";
		S[height++] = gen;
		}
	else if (t == PSSL) {
		int n = gs.val1();
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" PSSL(" << n << "," << q << ")" << endl;
			}
		Vector gen;
		vec_generators_GL_n_q_projective_representation(gen, n, q,
				TRUE /* f_special */, TRUE /* f_frobenius */, TRUE /* f_modified */, verbose_level);
		g_label << "PSSL_" << n << "_" << q;
		g_label_tex << "{\\bf P\\Sigma L(" << n << "," << q << ")}";
		S[height++] = gen;
		}
	else if (t == PGGL) {
		int n = gs.val1();
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" PGGL(" << n << "," << q << ")" << endl;
			}
		Vector gen;
		vec_generators_GL_n_q_projective_representation(gen, n, q,
				FALSE /* f_special */, TRUE /* f_frobenius */, TRUE /* f_modified */, verbose_level);
		g_label << "PGGL_" << n << "_" << q;
		g_label_tex << "{\\bf P\\Gamma L(" << n << "," << q << ")}";
		S[height++] = gen;
		}
	else if (t == ASL) {
		int n = gs.val1();
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" ASL(" << n << "," << q << ")" << endl;
			}
		Vector gen;
		vec_generators_GL_n_q_affine_representation(gen, n, q,
				TRUE /* f_special */, FALSE /* f_frobenius */, TRUE /* f_translations */, verbose_level);
		g_label << "ASL_" << n << "_" << q;
		g_label_tex << "{\\bf ASL(" << n << "," << q << ")}";
		S[height++] = gen;
		}
	else if (t == AGL) {
		int n = gs.val1();
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" AGL(" << n << "," << q << ")" << endl;
			}
		Vector gen;
		vec_generators_GL_n_q_affine_representation(gen, n, q,
				FALSE /* f_special */, FALSE /* f_frobenius */, TRUE /* f_translations */, verbose_level);
		g_label << "AGL_" << n << "_" << q;
		g_label_tex << "{\\bf AGL(" << n << "," << q << ")}";
		S[height++] = gen;
		}
	else if (t == ASSL) {
		int n = gs.val1();
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" ASSL(" << n << "," << q << ")" << endl;
			}
		Vector gen;
		vec_generators_GL_n_q_affine_representation(gen, n, q,
				TRUE /* f_special */, TRUE /* f_frobenius */, TRUE /* f_translations */, verbose_level);
		g_label << "ASSL_" << n << "_" << q;
		g_label_tex << "{\\bf A\\Sigma L(" << n << "," << q << ")}";
		S[height++] = gen;
		}
	else if (t == AGGL) {
		int n = gs.val1();
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" AGGL(" << n << "," << q << ")" << endl;
			}
		Vector gen;
		vec_generators_GL_n_q_affine_representation(gen, n, q,
				FALSE /* f_special */, TRUE /* f_frobenius */, TRUE /* f_translations */, verbose_level);
		g_label << "AGGL_" << n << "_" << q;
		g_label_tex << "{\\bf A\\Gamma L(" << n << "," << q << ")}";
		S[height++] = gen;
		}
	else if (t == Affine_translations) {
		int n = gs.val1();
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" T(" << n << "," << q << ")" << endl;
			}
		Vector gen;
		vec_generators_affine_translations(gen, n, q, verbose_level);
		g_label << "T_" << n << "_" << q;
		g_label_tex << "{\\bf T(" << n << "," << q << ")}";
		S[height++] = gen;
		}
	else if (t == PSU_3_Q2) {
		}
	else if (t == Suzuki) {
		}
	else if (t == A5_in_PSL) {
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" A5_in_PSL(2," << q << ")" << endl;
			}
		Vector gen;
		vec_generators_A5_in_PSL(gen, q, verbose_level);
		g_label << "A5_in_PSL_2_" << q;
		g_label_tex << "{\\langle A_5 \\le \\PSL(2," << q << ") \\rangle}";
		S[height++] = gen;
		}
	else if (t == S4_in_PSL) {
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" S4_in_PSL(2," << q << ")" << endl;
			}
		Vector gen;
		vec_generators_S4_in_PSL(gen, q, verbose_level);
		g_label << "S4_in_PSL_2_" << q;
		g_label_tex << "{\\langle S_4 \\le \\PSL(2," << q << ") \\rangle}";
		S[height++] = gen;
		}
	else if (t == On_projective_lines) {
		int n = gs.val1();
		int q = gs.val2();
		if (f_v) {
			cout << "compose_linear_group(): stack height = " << height << 
				" On_projective_lines(" << n << "," << q << ")" << endl;
			}
		Vector & gen = S[height - 1].as_vector();
		int k = n - 1;
		vec_generators_induce_on_lines_of_PG_k_q(gen, k, q, verbose_level);
		g_label << "On_projective_lines_" << k << "_" << q;
		g_label_tex << "on projective lines (" << k << "," << q << ")";
		}
	else
		return FALSE;
	return TRUE;
}

static int compose_group_unary_operator(group_selection & gs, Vector & S, int & height, 
	ostream & g_label, ostream & g_label_tex, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_selection_type t = (group_selection_type) gs.type();
	
	if (t == On_2_sets) {
		if (f_v) {
			cout << "compose_group_unary_operator(): stack height = " << height << 
				" On_2_sets " << endl;
			}
		Vector & gen = S[height - 1].as_vector();
		vec_induce2(gen);
		g_label << "on2sets";
		g_label_tex << "^{[2]}";
		}
	else if (t == On_2_tuples) {
		if (f_v) {
			cout << "compose_group_unary_operator(): stack height = " << height << 
				" On_2_tuples " << endl;
			}
		Vector & gen = S[height - 1].as_vector();
		vec_induce_on_2tuples(gen, FALSE /* f_injective */);
		g_label << "on2tuples";
		g_label_tex << "^{(2)}";
		}
	else if (t == On_3_sets) {
		if (f_v) {
			cout << "compose_group_unary_operator(): stack height = " << height << 
				" On_3_sets " << endl;
			}
		Vector & gen = S[height - 1].as_vector();
		vec_induce3(gen);
		g_label << "on3sets";
		g_label_tex << "^{[3]}";
		}
	else if (t == On_injective_2_tuples) {
		if (f_v) {
			cout << "compose_group_unary_operator(): stack height = " << height << 
				" On_injective_2_tuples " << endl;
			}
		Vector & gen = S[height - 1].as_vector();
		vec_induce_on_2tuples(gen, TRUE /* f_injective */);
		g_label << "oninjective2tuples";
		g_label_tex << "^{(2)i}";
		}
	else if (t == Add_fixpoint) {
		if (f_v) {
			cout << "compose_group_unary_operator(): stack height = " << height << 
				" Add_fixpoint " << endl;
			}
		Vector & gen = S[height - 1].as_vector();
		vec_add_fixpoint_in_front(gen);
		g_label << "+";
		g_label_tex << "^+";
		}
#if 0
	else if (t == Stabilize_point) {
		if (f_v) {
			cout << "compose_group_unary_operator(): stack height = " << height << 
				" vec_generators_stabilize_point " << endl;
			}
		Vector & gen = S[height - 1].as_vector();
		Vector gen1;
		vec_generators_stabilize_point(gen, gen1);
		S[height - 1] = gen1;
		g_label << "-";
		g_label_tex << "^-";
		}
#endif
	else if (t == Holomorph) {
		cout << "compose_group_unary_operator() error: Holomorph of arbitrary group not yet implemented" << endl;
		exit(1);
		}
	else if (t == Even_subgroup) {
		if (f_v) {
			cout << "compose_group_unary_operator(): stack height = " << height << 
				" vec_generators_even_subgroup " << endl;
			}
		Vector & gen = S[height - 1].as_vector();
		Vector gen1;
		vec_generators_even_subgroup(gen, gen1, verbose_level);
		S[height - 1] = gen1;
		g_label << "even";
		g_label_tex << "even";
		}
	else
		return FALSE;
	return TRUE;
}

static int compose_group_binary_operator(group_selection & gs, Vector & S, int & height, 
	ostream & g_label, ostream & g_label_tex, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_selection_type t = (group_selection_type) gs.type();
	
	if (t == Comma) {
		if (f_v) {
			cout << "compose_group_binary_operator(): stack height = " << height << 
				" Comma " << endl;
			}
		Vector & gen1 = S[height - 2].as_vector();
		Vector & gen2 = S[height - 1].as_vector();
		Vector gen3;
		vec_generators_comma(gen1, gen2, gen3);
		S[height - 2] = gen3;
		height--;
		g_label << ",";
		g_label_tex << ",";
		}
	else if (t == Direct_sum) {
		if (f_v) {
			cout << "compose_group_binary_operator(): stack height = " << height << 
				" Direct_sum " << endl;
			}
		if (S[height - 2].s_kind() == VECTOR && S[height - 1].s_kind() == VECTOR) {
			Vector & gen1 = S[height - 2].as_vector();
			Vector & gen2 = S[height - 1].as_vector();
			Vector gen3;
			vec_generators_direct_sum(gen1, gen2, gen3);
			S[height - 2] = gen3;
			}
		else if (S[height - 2].s_kind() == SOLID && S[height - 1].s_kind() == SOLID) {
			solid & S1 = S[height - 2].as_solid();
			solid & S2 = S[height - 1].as_solid();
			solid S3;
			S1.direct_sum(S2, S3, verbose_level);
			S[height - 2] = S3;
			}
		else {
			cout << "Direct sum: object types of two top objects not allowed\n";
			exit(1);
			}
		height--;
		g_label << "x";
		g_label_tex << "x ";
		}
	else if (t == Direct_product) {
		if (f_v) {
			cout << "compose_group_binary_operator(): stack height = " << height << 
				" Direct_product " << endl;
			}
		if (S[height - 2].s_kind() == VECTOR && S[height - 1].s_kind() == VECTOR) {
			Vector & gen1 = S[height - 2].as_vector();
			Vector & gen2 = S[height - 1].as_vector();
			Vector gen3;
			vec_generators_direct_product(gen1, gen2, gen3);
			S[height - 2] = gen3;
			}
		else if (S[height - 2].s_kind() == VECTOR && S[height - 1].s_kind() == SOLID) {
			Vector & v1 = S[height - 2].as_vector();
			solid & S1 = S[height - 1].as_solid();
			solid S3;
			S1.direct_product(v1, S3, verbose_level);
			S[height - 2] = S3;
			}
		else {
			cout << "Direct product: object types of two top objects not allowed\n";
			exit(1);
			}
		height--;
		g_label << "X";
		g_label_tex << "\\times ";
		}
	else if (t == Wreath_product) {
		if (f_v) {
			cout << "compose_group_binary_operator(): stack height = " << height << 
				" Wreath_product " << endl;
			}
		if (S[height - 2].s_kind() == VECTOR && S[height - 1].s_kind() == VECTOR) {
			Vector & gen1 = S[height - 2].as_vector();
			Vector & gen2 = S[height - 1].as_vector();
			Vector gen3;
			cout << "start vec_generators_wreath_product():" << endl;
			vec_generators_wreath_product(gen1, gen2, gen3, verbose_level);
			S[height - 2] = gen3;
			}
		else {
			cout << "wreath product: object types of two top objects not allowed\n";
			exit(1);
			}
		height--;
		g_label << "wr";
		g_label_tex << "\\wr ";
		}
	else if (t == Exponentiation) {
		cout << "compose_group_binary_operator() error: Exponentiation not yet implemented" << endl;
		exit(1);
		}
	else if (t == On_mappings) {
		cout << "compose_group_binary_operator() error: On_mappings not yet implemented" << endl;
		exit(1);
		}
	else
		return FALSE;
	return TRUE;
}

#define SOLID_RADIUS 10000 

static int compose_group_of_solid(group_selection & gs, Vector & S, int & height, 
	ostream & g_label, ostream & g_label_tex, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_selection_type t = (group_selection_type) gs.type();
	solid s;
	
	if (t == Solid_Tetrahedron) {
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" Tetrahedron" << endl;
			}
		g_label << "Tetra";
		g_label_tex << "{\\Tetra}";
		s.tetrahedron(SOLID_RADIUS);
		S[height++] = s;
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" Tetrahedron added" << endl;
			}
		}
	else if (t == Solid_Cube) {
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" Cube" << endl;
			}
		g_label << "Cube";
		g_label_tex << "{\\Cube}";
		s.cube(SOLID_RADIUS);
		S[height++] = s;
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" Cube added" << endl;
			}
		}
	else if (t == Solid_Cube4D) {
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" Cube4D" << endl;
			}
		g_label << "Cube4D";
		g_label_tex << "{\\Cube4D}";
		s.cube4D(SOLID_RADIUS, SOLID_RADIUS * 2);
		S[height++] = s;
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" Cube4D added" << endl;
			}
		}
	else if (t == Solid_Octahedron) {
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" Octahedron" << endl;
			}
		g_label << "Octa";
		g_label_tex << "{\\Octa}";
		s.octahedron(SOLID_RADIUS);
		S[height++] = s;
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" Octahedron added" << endl;
			}
		}
	else if (t == Solid_Dodecahedron) {
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" Dodecahedron" << endl;
			}
		g_label << "Dode";
		g_label_tex << "{\\Dode}";
		s.dodecahedron(SOLID_RADIUS);
		S[height++] = s;
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" Dodecahedron added" << endl;
			}
		}
	else if (t == Solid_Icosahedron) {
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" Icosahedron" << endl;
			}
		g_label << "Ico";
		g_label_tex << "{\\Ico}";
		s.icosahedron(SOLID_RADIUS);
		S[height++] = s;
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" Icoshedron added" << endl;
			}
		}
	else if (t == Solid_Cube4D) {
		cout << "Cube4D not yet implemented" << endl;
		exit(1);
		}
	else if (t == Solid_truncate) {
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" truncate" << endl;
			}
		g_label << "T";
		g_label_tex << "{\\bf trunc}";
		if (S[height - 1].s_kind() != SOLID) {
			cout << "Solid_truncate: stack does not contain a solid" << endl;
			exit(1);
			}
		solid &s1 = S[height - 1].as_solid();
		double r = 1. / 3.;
		s1.cut_vertices(r, s);
		S[height - 1] = s;
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" truncated" << endl;
			}
		}
	else if (t == Solid_truncate_dode) {
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" truncate_dode" << endl;
			}
		g_label << "T";
		g_label_tex << "{\\bf trunc}";
		if (S[height - 1].s_kind() != SOLID) {
			cout << "Solid_truncate_dode: stack does not contain a solid" << endl;
			exit(1);
			}
		solid &s1 = S[height - 1].as_solid();

		numerics Num;

		double r = 1. / (2. + square_root(2. * (1. - Num.cos_grad(108))));
		s1.cut_vertices(r, s);
		S[height - 1] = s;
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" truncated (dode)" << endl;
			}
		}
	else if (t == Solid_truncate_cube) {
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" truncate_cube" << endl;
			}
		g_label << "T";
		g_label_tex << "{\\bf trunc}";
		if (S[height - 1].s_kind() != SOLID) {
			cout << "Solid_truncate_cube: stack does not contain a solid" << endl;
			exit(1);
			}
		solid &s1 = S[height - 1].as_solid();
		double r = 1. / (2. + square_root(2.));
		s1.cut_vertices(r, s);
		S[height - 1] = s;
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" truncated (cube)" << endl;
			}
		}
	else if (t == Solid_dual) {
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" dual" << endl;
			}
		g_label << "D";
		g_label_tex << "{\\bf dual}";
		if (S[height - 1].s_kind() != SOLID) {
			cout << "Solid_dual: stack does not contain a solid" << endl;
			exit(1);
			}
		solid &s1 = S[height - 1].as_solid();
		s1.dual(s);
		S[height - 1] = s;
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" dualized" << endl;
			}
		}
	else if (t == Solid_relabel_points) {
		cout << "Solid_relabel_points not yet implemented" << endl;
		exit(1);
		}
	else if (t == Solid_induced_group_on_edges) {
		cout << "Solid_induced_group_on_edges not yet implemented" << endl;
		exit(1);
		}
	else if (t == Solid_midpoints_of_edges) {
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" midpoints of edges" << endl;
			}
		g_label << "E";
		g_label_tex << "{\\bf e}";
		if (S[height - 1].s_kind() != SOLID) {
			cout << "Solid_midpoints_of_edges: stack does not contain a solid" << endl;
			exit(1);
			}
		solid &s1 = S[height - 1].as_solid();
		s1.edge_midpoints(s);
		S[height - 1] = s;
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" midpoints taken" << endl;
			}
		}
	else if (t == Solid_add_central_point) {
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" add central point" << endl;
			}
		g_label << "centralpoint";
		g_label_tex << "{\\bf centralpoint}";
		if (S[height - 1].s_kind() != SOLID) {
			cout << "Solid_add_central_point: stack does not contain a solid" << endl;
			exit(1);
			}
		solid &s1 = S[height - 1].as_solid();
		s1.add_central_point(s);
		S[height - 1] = s;
		if (f_v) {
			cout << "compose_group_of_solid(): stack height = " << height << 
				" central point added" << endl;
			}
		}
	else if (t == Solid_add_central_involution) {
		cout << "Solid_add_central_involution not yet implemented" << endl;
		exit(1);
		}
	else if (t == Solid_Cubussimus) {
		cout << "Solid_Cubussimus not yet implemented" << endl;
		exit(1);
		}
	else if (t == Solid_Dodesimum) {
		cout << "Solid_Dodesimum not yet implemented" << endl;
		exit(1);
		}
	else if (t == Solid_CubeEE) {
		cout << "Solid_CubeEE not yet implemented" << endl;
		exit(1);
		}
	else if (t == Solid_CubeEE_russian) {
		cout << "Solid_CubeEE_russian not yet implemented" << endl;
		exit(1);
		}
	else
		return FALSE;
	return TRUE;
}


}}

