DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I.
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	a_domain.o \
	algebra_global.o \
	finite_field_io.o \
	finite_field_linear_algebra.o \
	finite_field_linear_algebra2.o \
	finite_field_linear_groups.o \
	finite_field_orthogonal.o \
	finite_field_projective.o \
	finite_field_representations.o \
	finite_field_tables.o \
	finite_field.o \
	finite_ring.o \
	generators_symplectic_group.o \
	gl_class_rep.o \
	gl_classes.o \
	group_generators_domain.o \
	heisenberg.o \
	homogeneous_polynomial_domain.o \
	longinteger_domain.o \
	longinteger_object.o \
	matrix_block_data.o \
	norm_tables.o \
	null_polarity_generator.o \
	number_theory_domain.o \
	partial_derivative.o \
	rank_checker.o \
	subfield_structure.o \
	table_of_irreducible_polynomials.o \
	unipoly_domain.o \
	vector_space.o


all : $(MODULES)



%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	- rm *.o *.a *~



