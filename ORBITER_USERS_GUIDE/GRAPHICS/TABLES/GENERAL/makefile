###############################################################################
# DATA.24/PACKING_PG_3_5
###############################################################################

ORBITER_PATH=~/orbiter

#ORBITER_PATH=/s/hipft-logger/b/nobackup/abetten/orbiter
#ORBITER_PATH=/scratch/betten/COMPILE/orbiter
#ORBITER_PATH=/data/accounts/fac/betten/orbiter


# uncomment exactly one of the following two lines.
# uncomment the first if you want to run orbiter through docker.
# uncomment the second if you have an installed copy of orbiter and you want to run it directly 
#ORBITER_PATH=docker run -it --volume ${PWD}:/mnt -w /mnt abetten/orbiter 

ORBITER_EXE_PATH=$(ORBITER_PATH)/src/apps/orbiter/

ORBITER=$(ORBITER_EXE_PATH)orbiter.out

SANDBOX=$(ORBITER_PATH)/src/apps/sandbox/sandbox.out

###############################################################################
# start
###############################################################################

make_tables:
	cp /Users/betten/orbiter/doc/tables/general/*.csv .
	$(ORBITER) -v 6 \
		-prepare_general_tables_for_users_guide \
		orbiter_activities1.csv \
		orbiter_activities2.csv \
		orbiter_objects_1.csv \
		orbiter_objects_2.csv \
		BLT_set_families.csv \
		basic_actions.csv \
		table_quadrics.csv \
		cubic_surface_export_options.csv \
		quartic_curve_export.csv \
		spreads_in_PG_3_4.csv \
		table_of_irreducible_polynomials.csv \
		-end
