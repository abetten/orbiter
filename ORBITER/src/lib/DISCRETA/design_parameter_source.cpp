// design_parameter_source.C
//
// Anton Betten
// 18.09.2000
// moved from D2 to ORBI Nov 15, 2007

#include "orbiter.h"


design_parameter_source::design_parameter_source() : Vector()
{
	k = DESIGN_PARAMETER_SOURCE;
}

design_parameter_source::design_parameter_source(const discreta_base &x)
	// copy constructor:    this := x
{
	cout << "design_parameter_source::copy constructor for object: "
			<< const_cast<discreta_base &>(x) << "\n";
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

design_parameter_source& design_parameter_source::operator = (const discreta_base &x)
	// copy assignment
{
	cout << "design_parameter_source::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void design_parameter_source::settype_design_parameter_source()
{
	OBJECTSELF s;
	
	s = self;
	new(this) design_parameter_source;
	self = s;
	k = DESIGN_PARAMETER_SOURCE;
}

design_parameter_source::~design_parameter_source()
{
	freeself_design_parameter_source();
}

void design_parameter_source::freeself_design_parameter_source()
{
	// cout << "group_selection::freeself_design_parameter_source()\n";
	freeself_vector();
}

kind design_parameter_source::s_virtual_kind()
{
	return DESIGN_PARAMETER_SOURCE;
}

void design_parameter_source::copyobject_to(discreta_base &x)
{
#ifdef COPY_VERBOSE
	cout << "design_parameter_source::copyobject_to()\n";
	print_as_vector(cout);
#endif
	Vector::copyobject_to(x);
	x.as_design_parameter_source().settype_design_parameter_source();
#ifdef COPY_VERBOSE
	x.as_design_parameter_source().print_as_vector(cout);
#endif
}

ostream& design_parameter_source::print(ostream& ost)
{
	hollerith h;
	
	text(h);
	ost << h.s();
	return ost;
}

void design_parameter_source::print2(design_parameter& p, ostream& ost)
{
	hollerith h;
	
	text2(p, h);
	ost << h.s();
}

void design_parameter_source::init()
{
	m_l_n(4);
	c_kind(DESIGN_PARAMETER_SOURCE);
	prev() = -1;
	rule() = -1;
	s_i(2).change_to_hollerith();
	s_i(3).change_to_vector();
	comment().init("");
	references().m_l(0);
}


void design_parameter_source::text(hollerith& h)
{
	hollerith s0, s1, s2;
	
	text012(s0, s1, s2);
	h.init(s1.s());
	h.append(" ");
	if (prev() != -1) {
		h.append_i(prev());
		h.append(" ");
		}
	h.append(s2.s());
}

void design_parameter_source::text2(design_parameter& p, hollerith& h)
{
	hollerith s0, s1, s2, hh;
	
	text012_extended(p, s0, s1, s2);
	h.init(s1.s());
	h.append(" ");
	if (prev() != -1) {
		h.append_i(prev());
		h.append(" ");
		}
	h.append(s2.s());
}

void design_parameter_source::text012(hollerith& s0, hollerith& s1, hollerith& s2)
//special print function for the design construction strings: 
//three strings are generated which give parts of an english sentence. 
//The wholes between the strings may be filled with numbers (ids) of 
//parameter sets. In the html page writing routines, these numbers 
//are printed using htmls capability of including links directly to 
//the definition of the other parameter set.
{
	enum design_parameter_rule r;

	s0.init("");
	s1.init("");
	s2.init("");
	if (strlen(comment().s()) > 0) {
		s1.init(comment().s());
		s1.append(" ");
		}
	r = (enum design_parameter_rule) rule();
	if (r == rule_complementary) {
		s1.append("complementary design of");
		}
	else if (r == rule_reduced_t) {
		s1.append("design");
		s2.init("with respect to smaller t");
		}
	else if (r == rule_derived) { 
		s1.append("derived from");
		}
	else if (r == rule_residual) {
		s1.append("residual design of");
		}
	else if (r == rule_alltop) {
		s1.append("Alltop construction for design");
		}
	else if (r == rule_supplementary) {
		s1.append("supplementary design of");
		}
	else if (r == rule_trung_complementary) {
		s1.append("Tran van Trung construction with complementary design for");
		}
	else if (r == rule_supplementary_reduced_t) {
		s1.append("supplementary design of");
		s2.init("with respect to smaller t");
		}
	else if (r == rule_supplementary_derived) { 
		s1.append("derived from supplementary of");
		}
	else if (r == rule_supplementary_residual) {
		s1.append("residual design of supplementary of");
		}
	else if (r == rule_supplementary_alltop) {
		s1.append("Alltop construction for supplementary design of");
		}
	else if (r == rule_trung_left) {
		s1.append("Tran van Trung construction (left) for");
		}
	else if (r == rule_trung_right) {
		s1.append("Tran van Trung construction (right) for");
		}
	else {
		}
}


void design_parameter_source::text012_extended(design_parameter& p, hollerith& s0, hollerith& s1, hollerith& s2)
{
	hollerith hh;
	
	text012(s0, s1, s2);


	enum design_parameter_rule r;
	r = (enum design_parameter_rule) rule();
	if (r == rule_trung_left) {
		design_parameter q, der, res;
		
		if (!p.increased_t(q)) {
			cout << "design_parameter_source::text012_extended() cannot increase t, error" << endl;
			exit(1);
			}
		q.derived(der);
		q.residual(res);
		s2.init(": der= ");
		s2.append_i(der.t());
		s2.append("-(");
		s2.append_i(der.v());
		s2.append(",");
		s2.append_i(der.K());
		s2.append(",");
		der.lambda().print_to_hollerith(hh);
		s2.append(hh.s());
		s2.append(") and res= ");
		s2.append_i(res.t());
		s2.append("-(");
		s2.append_i(res.v());
		s2.append(",");
		s2.append_i(res.K());
		s2.append(",");
		res.lambda().print_to_hollerith(hh);
		s2.append(hh.s());
		s2.append(") - the given design is the residual.");
		}
	else if (r == rule_trung_right) {
		design_parameter q, der, res;
		
		if (!p.increased_t(q)) {
			cout << "design_parameter_source::text012_extended() cannot increase t, error" << endl;
			exit(1);
			}
		q.derived(der);
		q.residual(res);
		s2.init(": der= ");
		s2.append_i(der.t());
		s2.append("-(");
		s2.append_i(der.v());
		s2.append(",");
		s2.append_i(der.K());
		s2.append(",");
		der.lambda().print_to_hollerith(hh);
		s2.append(hh.s());
		s2.append(") and res= ");
		s2.append_i(res.t());
		s2.append("-(");
		s2.append_i(res.v());
		s2.append(",");
		s2.append_i(res.K());
		s2.append(",");
		res.lambda().print_to_hollerith(hh);
		s2.append(hh.s());
		s2.append(") - the given design is the derived.");
		}
}











