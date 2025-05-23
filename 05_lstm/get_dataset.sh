
# Here we download and unzip the text file that contains all of our translated phrases
#rm spa-eng.zip _about.txt spa.txt
if [ ! -e spa.txt ];then
  wget https://www.manythings.org/anki/spa-eng.zip
  unzip spa-eng.zip
else
  ls
fi
