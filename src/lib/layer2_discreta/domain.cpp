// domain.cpp
//
// Anton Betten
// 11.06.2000
// moved from D2 to ORBI Nov 15, 2007

#include "../layer2_discreta/discreta.h"
#include "layer1_foundations/foundations.h"

using namespace std;


namespace orbiter {
namespace layer2_discreta {
namespace typed_objects {


#define MAX_DOMAIN_STACK 100

int domain_stack_len = 0;
static domain* domain_stack[MAX_DOMAIN_STACK];

domain::domain(int p)
{
	the_type = GFp;
	the_prime.m_i_i(p);
	//the_pres = NULL;
	the_factor_poly = NULL;
	the_sub_domain = NULL;
	F = NULL;
}



domain::domain(field_theory::finite_field *F)
{
	cout << "domain::domain orbiter finite_field of order " << F->q << endl;
	domain::F = F;
	the_type = Orbiter_finite_field;
	the_prime.m_i_i(F->p);
	//the_pres = NULL;
	the_factor_poly = NULL;
	the_sub_domain = NULL;
}

domain::domain(unipoly *factor_poly, domain *sub_domain)
{
	the_type = GFq;
	the_factor_poly = factor_poly;
	the_sub_domain = sub_domain;

	the_prime.m_i_i(0);
	//the_pres = NULL;
	F = NULL;
}

#if 0
domain::domain(pc_presentation *pres)
{
	the_type = PC_GROUP;
	the_pres = pres;

	the_prime.m_i_i(0);
	the_factor_poly = NULL;
	the_sub_domain = NULL;
}
#endif

domain_type domain::type()
{
	return the_type;
}

field_theory::finite_field *domain::get_F()
{
	return F;
}

int domain::order_int()
{
	number_theory::number_theory_domain NT;

	if (the_type == GFp)
		return the_prime.s_i_i();
	if (the_type == GFq) {
		int q = the_sub_domain->order_int();
		int f = the_factor_poly->degree();
		int Q = NT.i_power_j(q, f);
		return Q;
		}
	if (the_type == Orbiter_finite_field) {
		return F->q;
	}
	cout << "domain::order_int no finite field domain" << endl;
	exit(1);
}

int domain::order_subfield_int()
{
	if (the_type == GFp)
		return the_prime.s_i_i();
	if (the_type == GFq) {
		int q = the_sub_domain->order_int();
		return q;
		}
	if (the_type == Orbiter_finite_field) {
		return F->p;
	}
	cout << "domain::order_subfield_int no finite field domain" << endl;
	exit(1);
}

int domain::characteristic()
{
	if (the_type == GFp)
		return the_prime.s_i_i();
	if (the_type == GFq) {
		return the_sub_domain->characteristic();
		}
	if (the_type == Orbiter_finite_field) {
		return F->p;
	}
	cout << "domain::characteristic() no finite field domain" << endl;
	exit(1);
}

int domain::is_Orbiter_finite_field_domain()
{
	if (type() == Orbiter_finite_field) {
		return TRUE;
	}
	return FALSE;
}

#if 0
pc_presentation *domain::pres()
{
	return the_pres;
}
#endif

unipoly *domain::factor_poly()
{
	return the_factor_poly;
}

domain *domain::sub_domain()
{
	return the_sub_domain;
}

void push_domain(domain *d)
{
	if (domain_stack_len == MAX_DOMAIN_STACK) {
		cout << "push_domain() overflow in domain stack" << endl;
		exit(1);
		}
	domain_stack[domain_stack_len++] = d;
}


void pop_domain(domain *& d)
{
	if (domain_stack_len == 0) {
		cout << "push_domain() ounderflow in domain stack" << endl;
		exit(1);
		}
	d = domain_stack[--domain_stack_len];
}



int has_domain()
{
	if (domain_stack_len > 0)
		return TRUE;
	else
		return FALSE;
}

domain *get_current_domain()
{
	if (domain_stack_len <= 0) {
		cout << "get_current_domain() domain stack empty" << endl;
		exit(1);
		}
	return domain_stack[domain_stack_len - 1];
	//return dom;
}

#if 0
domain *get_domain_if_pc_group()
{
	if (!has_domain())
		return NULL;
	
	domain *d = get_current_domain();
	if (d->type() == PC_GROUP) {
		return d;
		}
	return NULL;
}
#endif

int is_GFp_domain(domain *& d)
{
	if (!has_domain())
		return FALSE;
	
	d = get_current_domain();
	if (d->type() == GFp) {
		return TRUE;
		}
	d = NULL;
	return FALSE;
}

int is_GFq_domain(domain *& d)
{
	if (!has_domain())
		return FALSE;
	
	d = get_current_domain();
	if (d->type() == GFq) {
		return TRUE;
		}
	d = NULL;
	return FALSE;
}

int is_Orbiter_finite_field_domain(domain *& d)
{
	if (!has_domain())
		return FALSE;

	d = get_current_domain();
	if (d->type() == Orbiter_finite_field) {
		return TRUE;
	}
	d = NULL;
	return FALSE;
}

int is_finite_field_domain(domain *& d)
{
	if (is_GFp_domain(d))
		return TRUE;
	if (is_GFq_domain(d))
		return TRUE;
	return FALSE;
}

int finite_field_domain_order_int(domain * d)
{
	if (is_GFp_domain(d)) {
		return d->order_int();
		}
	if (is_GFq_domain(d)) {
		return d->order_int();
		}
	cout << "finite_field_domain_order_int(): error: must be GFp or GFq" << endl;
	exit(1);
}

int finite_field_domain_characteristic(domain * d)
{
	if (is_GFp_domain(d)) {
		return d->characteristic();
		}
	if (is_GFq_domain(d)) {
		return d->characteristic();
		}
	cout << "finite_field_domain_characteristic(): error: must be GFp or GFq" << endl;
	exit(1);
}

int finite_field_domain_primitive_root()
{
	domain *d;
	int q;
	number_theory::number_theory_domain NT;
	
	if (!is_finite_field_domain(d)) {
		cout << "finite_field_domain_primitive_root() no finite field domain" << endl;
		exit(1);
		}
	q = finite_field_domain_order_int(d);
	if (is_GFp_domain(d)) {
		return NT.primitive_root(q, FALSE /* f_v */);
		}
	else {
		integer a;
		int i, o;
		
		cout << "finite_field_domain_primitive_root():" << endl;
		for (i = 1; i < q; i++) {
			a.m_i((int) i);
			o = a.order();
			cout << "order of " << i << " is " << o << endl;
			if (o == q - 1) {
				return i;
				}
			}
		cout << "finite_field_domain_primitive_root "
				"couldn't find primitive root!" << endl;
		exit(1);
		// int p, f;
		// factor_prime_power(q, &p, &f);

		// return p;
		}
}

void finite_field_domain_base_over_subfield(Vector & b)
{
	domain *d, *sd;
	int q, f, a, i;
	number_theory::number_theory_domain NT;
	
	if (!is_finite_field_domain(d)) {
		cout << "finite_field_domain_base_over_subfield "
				"no finite field domain" << endl;
		exit(1);
		}
	//qq = finite_field_domain_order_int(d);
	if (is_GFp_domain(d)) {
		b.m_l(1);
		b.m_ii(0, 1);
		return;
		}
	else if (is_GFq_domain(d)) {
		sd = d->sub_domain();
		unipoly *m = d->factor_poly();
		f = m->degree();
		q = sd->order_int();
		b.m_l(f);
		for (i = 0; i < f; i++) {
			a = NT.i_power_j(q, i);
			b.m_ii(i, a);
			}
		}	
}

with::with(domain *d)
{
	if (d == NULL) {
		cout << "with::with() trying to push NULL pointer domain" << endl;
		exit(1); 
		}
	push_domain(d);
}

with::~with()
{
	domain *d;
	pop_domain(d);
}

#if 1
// used in a5_in_PSL

typedef struct ff_memory FF_MEMORY;

//! DISCRETA auxilliary class for class domain


struct ff_memory {
	int q, p, f;
	domain *d1;
	unipoly *m;
	domain *d2;
	domain *dom;
};

#define MAX_FF_DOMAIN 100

static int nb_ffm = 0;
static FF_MEMORY *Ffm[MAX_FF_DOMAIN];

domain *allocate_finite_field_domain(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	FF_MEMORY *ffm = new FF_MEMORY;
	int p, f;
	number_theory::number_theory_domain NT;
	
	if (nb_ffm >= MAX_FF_DOMAIN) {
		cout << "allocate_finite_field_domain() too many finite field domains" << endl;
		exit(1);
		}
	ffm->q = q;
	NT.factor_prime_power(q, p, f);
	ffm->p = p;
	ffm->f = f;
	if (f_v) {
		cout << "allocate_finite_field_domain() q=" << q << ", p=" << ffm->p << ", f=" << ffm->f << endl;
		}
	ffm->d1 = new domain(ffm->p);

	if (ffm->f > 1) {
		with w(ffm->d1);
		ffm->m = &callocobject(UNIPOLY)->change_to_unipoly();
		ffm->m->Singer(ffm->p, ffm->f, verbose_level - 2);
		if (f_v ) {
			cout << "q=" << q << "=" << ffm->p << "^" << ffm->f << ", m=" << *ffm->m << endl;
			}
		ffm->d2 = new domain(ffm->m, ffm->d1);
		ffm->dom = ffm->d2;
		}
	else {
		ffm->dom = ffm->d1;
		}
	Ffm[nb_ffm++] = ffm;
	return ffm->dom;
}

void free_finite_field_domain(domain *dom)
{
	int i;
	
	for (i = 0; i < nb_ffm; i++) {
		if (Ffm[i]->dom == dom) {
			if (FALSE) {
				cout << "deleting ff domain no " << i << endl;
				}
			if (Ffm[i]->f > 1) {
				delete Ffm[i]->d2;
				freeobject(Ffm[i]->m);
				delete Ffm[i]->d1;
				}
			else {
				delete Ffm[i]->d1;
				}
			delete Ffm[i];
			for ( ; i < nb_ffm - 1; i++) {
				Ffm[i] = Ffm[i + 1];
				}
			nb_ffm--;
			return;
			}
		}
	cout << "free_finite_field_domain() error: domain not found" << endl;
	exit(1);
}
#endif



}}}

