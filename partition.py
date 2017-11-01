"""Partition the dataset
Input is partition status aka "val", "test" or "train"
Output is a .txt file that contains all the images corresponding to the partition status."""

partition = 'val' #'val', 'test', 'train'

textfile = 'Eval/' + partition + ".txt"
f = open(textfile, 'w')
with open('Eval/list_eval_partition.txt', 'r') as searchfile:
	searchfile.readline()
	searchfile.readline() #start from line 3
	for line in searchfile:
		if partition in line:
			imgurl = [line.split()]
			f.write(imgurl[0][0])
			f.write('\n')

f.close()


"""Run in terminal:
cd DEEPFASHION_DIRECTORY
cpio -p -d 'partition' < Eval/'partition'.txt
Then move the images to parent partition directory
cd TRAIN/TEST/VAL_DIRECTORY/img
mv * ../ 
cd ..
rm -r img
"""
