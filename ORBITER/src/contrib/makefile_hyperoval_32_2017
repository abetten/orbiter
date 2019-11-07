MY_PATH=../../../..
#MY_PATH=/scratch/betten/COMPILE/2014_11_01
SRC=$(MY_PATH)/ORBITER/SRC

HYPEROVAL_PATH=$(SRC)/APPS/HYPEROVAL
TOOLS_PATH=$(SRC)/APPS/TOOLS
PROJECTIVE_SPACE_PATH=$(SRC2)/PROJECTIVE_SPACE

ORDER=32
TARGET=34
RAD=50
STARTER_SIZE=6


all: starter build_db lift read_solutions compute_orbits classify report 



# 8/29/15 on MacBook: lifting with Starters of size 6 takes 18 min 58  and leads to 164441 solutions
# 8/29/15 on MacBook: reading of solutions takes 32 min and 2 sec


clean:
	- rm arc_$(ORDER)* report*  
	- rm -r ISO
	- rm -r STARTERS
	- rm -r LIFT_DATA
	- rm -r SOLUTIONS
	- rm hyperovals_*
	- rm classification*
	- rm PG_*



cheat:
	$(PROJECTIVE_SPACE_PATH)/cheat_sheet_PG.out -n 2 -q $(ORDER)
	pdflatex PG_2_$(ORDER).tex
	open PG_2_$(ORDER).pdf


arc:
	$(HYPEROVAL_PATH)/hyperoval.out -v 4 -q $(ORDER) \
		-target_size $(TARGET) \
		-compute_starter  \
		-starter_size $(TARGET) \
		-prefix_starter STARTER/ \
		-W -T \
		-draw_poset



draw:
	$(TOOLS_PATH)/layered_graph_main.out -v 4 \
		-file arc_$(ORDER)_poset_lvl_$(TARGET).layered_graph \
		-draw arc_$(ORDER)_poset_lvl_$(TARGET) \
		-xin 10000 -yin 10000 \
		-xout 1000000 -yout 1000000 \
		-rad $(RAD) \
		-embedded
	mpost arc_$(ORDER)_poset_lvl_$(TARGET).mp
	#pdflatex arc_$(ORDER)_poset_lvl_$(TARGET).tex
	#open arc_$(ORDER)_poset_lvl_$(TARGET).pdf


draw_aux:
	$(TOOLS_PATH)/layered_graph_main.out -v 4 \
		-file aux_poset_lvl_$(TARGET).layered_graph \
		-draw aux_poset_lvl_$(TARGET) \
		-xin 10000 -yin 10000 \
		-xout 1000000 -yout 1000000 \
		-rad $(RAD) \
		-embedded
	pdflatex aux_poset_lvl_$(TARGET).tex
	open aux_poset_lvl_$(TARGET).pdf

tree:
	$(TOOLS_PATH)/layered_graph_main.out -v 4 \
		-file tree_lvl_$(TARGET).layered_graph \
		-draw tree_lvl_$(TARGET) \
		-xin 10000 -yin 10000 \
		-xout 1000000 -yout 1000000 \
		-rad $(RAD) \
		-embedded
	pdflatex tree_lvl_$(TARGET).tex
	open tree_lvl_$(TARGET).pdf

draw_tree_circle:
	$(TOOLS_PATH)/treedraw.out \
		-x 1000000 -y 1000000 \
		-on_circle -rad 0 \
		-file arc_$(ORDER)_$(TARGET).tree   \
		-embedded \
		-no_circletext
	pdflatex arc_$(ORDER)_$(TARGET).tex
	open arc_$(ORDER)_$(TARGET).pdf



conv:
	/sw/bin/convert -density 600x600 -trim view.ps hyperoval_$(ORDER)_tree.png



#####



starter:
	$(HYPEROVAL_PATH)/hyperoval.out -v 4 -q $(ORDER) \
		-target_size $(TARGET) \
		-compute_starter  \
		-starter_size $(STARTER_SIZE) \
		-starter_directory_name STARTERS/ \
		-W -T 


#		-draw_poset

# directory name must be different from makefile target


build_db:
	$(HYPEROVAL_PATH)/hyperoval.out -v 2 -q $(ORDER) \
		-target_size $(TARGET) \
		-starter_size $(STARTER_SIZE) \
		-starter_directory_name STARTERS/ \
		-build_db 


draw_tree:
	$(TOOLS_PATH)/treedraw.out  -file arc_$(ORDER)_$(STARTER_SIZE).tree -circle  -rad 20000   
	#mpost arc_$(ORDER)_$(STARTER_SIZE).mp
	pdflatex arc_$(ORDER)_$(STARTER_SIZE).tex
	open arc_$(ORDER)_$(STARTER_SIZE).pdf

# -on_circle 
# -no_circletext


draw_poset:
	$(TOOLS_PATH)/layered_graph_main.out -v 4 \
		-file arc_$(ORDER)_poset_lvl_$(STARTER_SIZE).layered_graph \
		-draw arc_$(ORDER)_poset_lvl_$(STARTER_SIZE) \
		-rad 1000 \
		-xin 1000000 \
		-yin 1000000 \
		-xout 1000000 \
		-yout 1000000 \
		-embedded \
		-nodes_empty
	mpost arc_$(ORDER)_poset_lvl_$(STARTER_SIZE).mp
	#pdflatex arc_$(ORDER)_poset_lvl_$(STARTER_SIZE).tex
	#open arc_$(ORDER)_poset_lvl_$(STARTER_SIZE).pdf


lift:
	- mkdir LIFT_DATA
	- mkdir SOLUTIONS
	$(HYPEROVAL_PATH)/hyperoval.out -v 2 \
		-q $(ORDER) -target_size $(TARGET) \
		-starter_size $(STARTER_SIZE) \
		-starter_directory_name STARTERS/ \
		-solution_directory_name ./SOLUTIONS/ \
		-lift LIFT_DATA/ \
		-lex -solve 


#		-draw_system LIFT_DATA/system \
#		-write_tree LIFT_DATA/tree
		

#		-save \

# -lex 


read_solutions:
	$(HYPEROVAL_PATH)/hyperoval.out -v 2 \
		-q $(ORDER) -target_size $(TARGET) \
		-starter_size $(STARTER_SIZE) \
		-starter_directory_name STARTERS/ \
		-solution_directory_name ./SOLUTIONS/ \
		-read_solutions  \
		-solution_file ./SOLUTIONS/arc_$(ORDER)_depth_$(STARTER_SIZE)_solutions.txt


compute_orbits:
	$(HYPEROVAL_PATH)/hyperoval.out -v 2 \
		-q $(ORDER) -target_size $(TARGET) \
		-starter_size $(STARTER_SIZE) \
		-starter_directory_name STARTERS/ \
		-compute_orbits 


classify:
	$(HYPEROVAL_PATH)/hyperoval.out -v 2 \
		-q $(ORDER) -target_size $(TARGET) \
		-starter_size $(STARTER_SIZE) \
		-starter_directory_name STARTERS/ \
		-classify 


decomposition_matrix:
	$(HYPEROVAL_PATH)/hyperoval.out -v 2 \
		-q $(ORDER) -target_size $(TARGET) \
		-starter_size $(STARTER_SIZE) \
		-starter_directory_name STARTERS/ \
		-make_decomposition_matrix 


report:
	- $(HYPEROVAL_PATH)/hyperoval.out -v 2 \
		-q $(ORDER) -target_size $(TARGET) \
		-starter_size $(STARTER_SIZE) \
		-starter_directory_name STARTERS/ \
		-report 
	pdflatex hyperovals_$(ORDER).tex
	open hyperovals_$(ORDER).pdf


draw_classification_graph:
	cp ISO/classification_graph.layered_graph .
	$(TOOLS_PATH)/layered_graph_main.out -v 4 \
		-file classification_graph.layered_graph \
		-draw classification_graph \
		-rad 1000 \
		-xin 1000000 \
		-yin 1000000 \
		-xout 1000000 \
		-yout 1000000 \
		-embedded \
		-nodes_empty
	#mpost classification_graph.mp
	pdflatex classification_graph.tex
	open classification_graph.pdf




decomp:
	$(HYPEROVAL_PATH)/hyperoval.out -v 2 \
		-q $(ORDER) -target_size $(TARGET) \
		-starter_size $(STARTER_SIZE) \
		-starter_directory_name STARTERS/ \
		-make_decomposition_matrix 
	cp ISO/classification_graph.layered_graph .

# June 3, 2015 on MacBookPro: 0 sec
# writes ISO/classification_graph.layered_graph


poset:
	$(TOOLS_PATH)/layered_graph_main.out -v 4 \
		-file classification_graph.layered_graph \
		-draw classification_graph \
		-rad 1000 \
		-as_graph \
		-xin 1000000 \
		-yin 1000000 \
		-xout 1000000 \
		-yout 1000000 \
		-embedded\
		-scale .40 \
		-line_width 1.0 \
		-y_stretch 1.0 
	pdflatex classification_graph.tex
	open classification_graph.pdf

#		-text_underneath


