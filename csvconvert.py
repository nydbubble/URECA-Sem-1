import csv

f = open("Anno/val.csv", 'w')
fieldnames = ['image_name', 'attribute_labels']
writer = csv.DictWriter(f, fieldnames=fieldnames)
writer.writeheader()

d = open('Eval/val.txt', 'r')
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
			for j in range (len(line[1:])):
				line[j] = line[j].replace("-1", "0")
			attribute_labels.append(line[1:])
			writer.writerow({'image_name': imgurl[i], 'attribute_labels': ' '.join(attribute_labels[i])})
			if i%1000 == 0:
				print("Writing "+str(i)+" files")
			i= i+1

print("Finished")
f.close()
"""
import sys
import csv


f = open("Anno/test.csv")
reader = csv.DictReader(f, fieldnames=['image_name', 'attribute_labels'])
for line in reader:
	print(line['image_name'][0])
"""
"""data=[]
code = sys.argv[1]
newval= int(sys.argv[2])
f=open("Anno/test.csv")
reader=csv.DictReader(f,fieldnames=['image_name','attribute_labels'])
for line in reader:
  if line['code'] == code:
    line['level']= newval
  data.append('%s,%s'%(line['code'],line['level']))
f.close()

f=open("stockcontrol.csv","w")
f.write("\n".join(data))
f.close()
		#print(row)
		#name,other = row.split(", ")
		#w.write(name + "," + other)"""

"""import csv
with open("Anno/test.csv","r") as f,open("Anno/val.csv","w") as w:
    #not_ok_name= ["John Doe" , "Jim Foe"]
    reader = csv.reader(f)
    for row in f:
        name,other = row.split(", ")
        if name in not_ok_name:
            w.write(row)
            """