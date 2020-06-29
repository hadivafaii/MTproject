#!/bin/bash

rsync --dry-run -avP --ignore-existing --update hadivafa@128.8.185.36:/home/hadivafa/Documents/PROJECTS/MT_LFP /home/hadi/Documents/MT/

printf "\n\n"
read -r -p "This was a dry run.  Would you like to proceed and run rsync? [y/n] " response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
  rsync -avP --ignore-existing --update hadivafa@128.8.185.36:/home/hadivafa/Documents/PROJECTS/MT_LFP /home/hadi/Documents/MT/
else
  echo "bye"
fi
