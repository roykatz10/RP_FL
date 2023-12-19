#!/bin/bash

filename="output.txt"
finished=0
checkstr='Flower ECE: gRPC server running'

# remove output file if it already exists
if test -f "$filename"; then
  rm "$filename"
fi

# run python script in the background 
python print_str.py &


#keep looping until the file is being created
while [ $finished -eq 0 ]
do
  # check if the file is created
  if test -f "$filename"; then
    
    # read the file
    while IFS= read -r line
    do
      #check if the file contains the line
      if [[ $line == *$checkstr* ]]; then
        echo "found the string!"
        finished=1
        break
      fi
    done < "$filename"
  fi
done

rm "$filename"

exit
