import torch
import csv

class Read_Csvfile:
    def load_file(filename):

        #filename should be the file location
        #the output will be array [ image name, tensor use as data, label]

        output = []
        with open(filename) as file_obj:
            reader_obj = csv.reader(file_obj)
            for row in reader_obj:
                print(row[1:-1])
                tensor =  torch.Tensor(list(map(float, row[1:-1])))
                output.append([row[0], tensor, int(row[-1])])
        return output
#usb = Read_Csvfile.load_file("train_cnn.csv")

