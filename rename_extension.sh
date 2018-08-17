!#/bin/bash

for f in $(find . -type f -name '*.C');
do 
	NEW_FILE_NAME=$(sed 's/\(.*\)./\1/' <<< "$f")cpp; 
	mv -v $f $NEW_FILE_NAME;
done
