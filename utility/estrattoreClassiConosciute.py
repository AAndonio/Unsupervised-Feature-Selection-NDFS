import csv


# Funzione che estrae le classi contenute nel tsv

def estraiLabelConosciute(filePath):
    with open(filePath) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        labelConosciute = []
        for row in rd:
            labelConosciute.append(row[0])

        labelConosciute = list(map(int, labelConosciute))
        return labelConosciute
    