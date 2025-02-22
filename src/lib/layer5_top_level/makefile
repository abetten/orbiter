DISCRETA_CPP = g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
LIB = 
LFLAGS = -lm

all: \
	applications_in_algebraic_geometry \
	apps_algebra \
	apps_coding_theory \
	apps_combinatorics \
	apps_geometry \
	apps_graph_theory \
	canonical_form \
	orbits \
	orthogonal_geometry_applications \
	packings \
	projective_geometry \
	semifields \
	spreads \
	user_interface




applications_in_algebraic_geometry : applications_in_algebraic_geometry2 ; 
apps_algebra : apps_algebra2 ; 
apps_coding_theory : apps_coding_theory2 ;
apps_combinatorics : apps_combinatorics2 ; 
apps_geometry : apps_geometry2 ; 
apps_graph_theory : apps_graph_theory2 ; 
canonical_form : canonical_form2 ;
orbits : orbits2 ; 
orthogonal_geometry_applications : orthogonal_geometry_applications2 ; 
packings : packings2 ; 
projective_geometry : projective_geometry2 ; 
semifields : semifields2 ; 
spreads : spreads2 ; 
user_interface : user_interface2 ; 


applications_in_algebraic_geometry2 :  ; cd applications_in_algebraic_geometry; $(MAKE)
apps_algebra2 :  ; cd apps_algebra; $(MAKE)
apps_coding_theory2 : ; cd apps_coding_theory; $(MAKE)
apps_combinatorics2 :  ; cd apps_combinatorics; $(MAKE)
apps_geometry2 :  ; cd apps_geometry; $(MAKE)
apps_graph_theory2 :  ; cd apps_graph_theory; $(MAKE)
canonical_form2 : ; cd canonical_form; $(MAKE)
orbits2 :  ; cd orbits; $(MAKE)
orthogonal_geometry_applications2 :  ; cd orthogonal_geometry_applications; $(MAKE)
packings2 :  ; cd packings; $(MAKE)
projective_geometry2 :  ; cd projective_geometry; $(MAKE)
semifields2 :  ; cd semifields; $(MAKE)
spreads2 :  ; cd spreads; $(MAKE)
user_interface2 :  ; cd user_interface; $(MAKE)



	



%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd applications_in_algebraic_geometry; $(MAKE) clean
	cd apps_algebra; $(MAKE) clean
	cd apps_coding_theory; $(MAKE) clean
	cd apps_combinatorics; $(MAKE) clean
	cd apps_geometry; $(MAKE) clean
	cd apps_graph_theory; $(MAKE) clean
	cd canonical_form; $(MAKE) clean
	cd orbits; $(MAKE) clean
	cd orthogonal_geometry_applications; $(MAKE) clean
	cd projective_geometry; $(MAKE) clean
	cd semifields; $(MAKE) clean
	cd spreads; $(MAKE) clean
	cd packings; $(MAKE) clean
	cd user_interface; $(MAKE) clean
	- rm *.o *.a 




