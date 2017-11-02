import csv

partition = 'val' #'val', 'test', 'train'


textfile = 'Eval/' + partition + ".txt"
csvfile = 'Anno/' + partition + ".csv"
f = open(csvfile, 'w')
fieldnames = ['image_name', 'attribute_labels']
writer = csv.DictWriter(f, fieldnames=fieldnames)
writer.writeheader()

d = open(textfile, 'r')
trainimgs = [line.rstrip('\n') for line in d.readlines()]

imgurl = []
attribute_labels = []
i = 0
j = 1
with open('Anno/list_attr_img.txt', 'r') as r:
	for line in r:
		line = line.split()
		if line[0] in trainimgs:
			imgurl.append(line[0])
			for j in range (len(line[1:])+1):
				line[j] = line[j].replace("-1", "0")
			attribute_labels.append(line[1:])
			writer.writerow({'image_name': imgurl[i], 'attribute_labels': ' '.join(attribute_labels[i])})
			if i%1000 == 0:
				print("Writing "+str(i)+" files")
			i= i+1

print("Finished")
f.close()