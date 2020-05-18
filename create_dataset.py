from __future__ import print_function, division

import os
import random


class YoloData:

    classes = {
        0 : "vetalas",
        1 : "draugar",
        2 : "aswang",
        3 : "jiangshi",
        4 : "beam",
        5 : "marker"
    }

    dataset = {} # data of all the .txt
    training_set = []
    test_set = []

    data_folder = "data/obj"

    def __init__(self, darknet_path='.'):
        # path where ./darknet binary is located. We assume that the folder 
        # structure is the same as darknet's github
        self.darknet_path = darknet_path if os.path.isabs(darknet_path) \
                                         else os.getcwd() + '/' + darknet_path
        
        os.chdir(self.darknet_path)


    def load_data(self):
        """
        Reads the darknet .txt files and saves its data into a dictionary
        with key "txt_name" and the value is a tuple of tuples containing
        each line of the .txt.
        """

        files = [f for f in os.listdir(self.data_folder) if f.endswith('.txt')]

        for f in files:
            
            data = []

            with open(self.data_folder + '/' + f) as txt:
                for line in txt.readlines():
                    d = [float(l) for l in line.split(' ')]
                    d[0] = int(d[0]) 
                    data.append(d)

            self.dataset[f[:-4]] = data

        return self.dataset

    
    def objects_count(self):
        """
        Returns how many times each object appears on the dataset.
        """

        classes_count = {
            0 : 0,
            1 : 0,
            2 : 0,
            3 : 0,
            4 : 0,
            5 : 0
        }

        for key in self.dataset:
            for obj in self.dataset[key]:
                classes_count[obj[0]] += 1
        
        obj_count = {self.classes[c]:classes_count[c] for c in classes_count}

        return obj_count


    def divide_data_at_random(self, data, percentage=0.8):
        """
        Randomly splits the data. The first split will hold the desired percentage
        of the data.
        """
        random.shuffle(data) # shuffle the dictionary keys
        
        ### split the data ###
        com = int(percentage*len(data))
        split_1, split_2 = data[:com], data[com:]

        return split_1, split_2


    def define_training_set(self, percentage=0.8):
        """
        Creates the training set using a percentage of the data. Tries to
        make sure that each object appears a similar amount of times. Automatically 
        puts the remaining data into the test set.
        """

        # Generate training set

        data = [i + '.jpg' for i in self.dataset] 

        self.training_set, self.test_set = self.divide_data_at_random(data, percentage)

        return self.training_set, self.test_set


    def create_empty_txt(self):
        """
        Opens the folder and creates empty .txt with the name of the images inside        
        """
        
        files = [f[:-4] for f in os.listdir(self.data_folder) if f.endswith('.jpg')]

        for f in files:
            if f not in self.dataset:
                with open (self.data_folder + '/' + f + '.txt', 'w+'):
                    pass
        
    
    def create_set_txt(self, files, name="train.txt"):
        """
        Writes the names of the image files on data/obj to name
        """

        with open('data/' + name, 'w+') as txt:
            for num in files:
                txt.write(self.data_folder + '/' + num + '\n')


    # Implement last
    def drop_objects(self, obj, thrash=False):
        """
        Drops part an object data until it has as much data as the second most common object.
        If thrash = True, creates a .thrash folder and moves the dropped data there
        """
        pass


if __name__ == "__main__":
    yolo = YoloData("")
    yolo.load_data()
    obj_count = yolo.objects_count()
    yolo.create_empty_txt()
    training_set, test_set=yolo.define_training_set(0.85)
    yolo.create_set_txt(training_set)
    yolo.create_set_txt(test_set, "test.txt")
