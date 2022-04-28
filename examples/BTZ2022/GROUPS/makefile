#MY_PATH=../orbiter
MY_PATH=~/DEV.22/orbiter
#MY_PATH=/scratch/betten/COMPILE/orbiter


# uncomment exactly one of the following two lines.
# uncomment the first if you want to run orbiter through docker.
# uncomment the second if you have an installed copy of orbiter and you want to run it directly 
#ORBITER_PATH=docker run -it --volume ${PWD}:/mnt -w /mnt abetten/orbiter 
ORBITER_PATH=$(MY_PATH)/src/apps/orbiter/


#############################################################################################################
# End of configuration part
#############################################################################################################


#######################################
# classify the noncyclic subgroups of order 4 in PGGL(4,4)
#######################################



2C_orbit_under_PGGL_4_4_elements_coded.csv: 
	$(ORBITER_PATH)orbiter.out -v 3 -define G \
		-linear_group -PGGL 4 4 -end \
		-with G -do \
		-group_theoretic_activity \
		-conjugacy_class_of_element "2C" "1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 0" \
		-end

# class of size 64260
# creates:
# 2C_orbit_under_PGGL_4_4.csv
# 2C_orbit_under_PGGL_4_4.txt
# 2C_orbit_under_PGGL_4_4_elements_coded.csv
# 2C_orbit_under_PGGL_4_4_transporter.csv
# 1:33 on Mac

PGGL_4_4_subgroups_of_type_2C_2C: 2C_orbit_under_PGGL_4_4_elements_coded.csv
	$(ORBITER_PATH)orbiter.out -v 6 -define G \
		-linear_group -PGGL 4 4 \
		-subgroup_by_generators "centralizer_2C" "30720" 9 \
			"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
			"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,1," \
			"1,0,0,0,0,1,0,0,0,0,1,0,2,0,3,1,0," \
			"1,0,0,0,0,1,0,0,0,0,1,0,1,0,3,1,0," \
			"1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1," \
			"1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0," \
			"1,0,0,0,2,1,0,0,0,0,1,0,1,0,0,1,0," \
			"1,0,0,0,1,1,2,0,0,0,1,0,0,0,0,1,0," \
			"1,0,3,0,1,1,1,3,0,0,2,0,0,0,0,2,1," \
		-end \
		-with G -do \
		-group_theoretic_activity \
		-orbits_on_group_elements_under_conjugation \
			2C_orbit_under_PGGL_4_4_elements_coded.csv \
			2C_orbit_under_PGGL_4_4_transporter.csv \
		-end
	pdflatex subgroups_of_order_4.tex
	open subgroups_of_order_4.pdf


#The distribution of orbit lengths is: ( 1, 2, 15, 20, 24^3, 30, 40, 240, 256, 480, 512, 960^2, 1280, 1920^2, 2560^4, 3840, 5120, 6144^3, 7680, 10240 )
#group_theoretic_activity::do_orbits_on_group_elements_under_conjugation after Classes.compute_all_point_orbits
#found 29 conjugacy classes
#User time: 0:57


