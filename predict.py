import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from PIL import Image
import glob, os
import csv


import numpy as np
import tensorflow as tf
import pathlib
import os
import keras
import PIL
import glob, os

from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array

#####################################
#         image generation          #
#####################################

def list_maker(list):
    
    out_list = []
    length = len(list)
    for i in range(length):
        tmp_l = list[i].split()
        if tmp_l[0] != "0" or tmp_l[1] != "0" or tmp_l[2] != "0":
            out_list.append([float(tmp_l[0]), float(tmp_l[1]), float(tmp_l[2])])
    return out_list

def print_list(list):
    for i in range(len(list)):
        print(list[i])

def get_empty_matrix(dimension):
    a = []
    a = [([0]*dimension) for i in range(dimension)]
    return a

def bitmap_maker(list, dimension):
    
    matrix = get_empty_matrix(dimension)
    used_data = 0
    discarded_data = 0

    dv = 3
    for i in range(len(list)):
        x = int(list[i][0]*dv + dimension/2)
        y = int(list[i][1]*dv + dimension/2)
        
        if x > 0 and x < dimension-1 and y > 0 and y < dimension-1:
            matrix[x][y] = 256
            used_data += 1
        else:
            discarded_data += 1

    return matrix

def data_density_graph(list):
    xyz_list = []

    for data in list:
        xyz_list.append(data[0])
        xyz_list.append(data[1])

    xyz_list.sort()

    xpoints = np.array(range(0, len(xyz_list)))
    ypoints = np.array(xyz_list)

    plt.plot(xpoints, ypoints)
    plt.show()

def matrix_graph(matrix, dimension):
    xp = []
    yp = []

    for x in range(dimension):
        for y in range(dimension):
            for z in range(dimension):
                if matrix[x][y] > 0:
                    xp.append(x)
                    yp.append(y)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xp, yp, 0)
    plt.show()
    print("3d graph ready")

def black_white(np_matrix):
    for (x,y), value in np.ndenumerate(np_matrix):
        if (value > 0):
            np_matrix[x][y] = float(256)
    return np_matrix

def create_directory(data_path, save_path):
    xyz_files = glob.glob(data_path, recursive=True)

    for file in xyz_files:
        file_array = file.split('\\')
        file_name = file_array[len(file_array) - 1]
        directory_name = file_array[len(file_array) - 2]
        big_directory_name = file_array[len(file_array) - 3]

        newpath = save_path + big_directory_name
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        newpath = save_path + big_directory_name + '\\' + directory_name
        if not os.path.exists(newpath):
            os.makedirs(newpath)


def batch_save(new_generation, data_path, save_path):
    xyz_files = glob.glob(data_path, recursive=True)
    number = 0
    length = len(xyz_files)
    for file in xyz_files:
        number = number + 1
        file_array = file.split('\\')
        file_name = file_array[len(file_array) - 1]
        directory_name = file_array[len(file_array) - 2]
        big_directory_name = file_array[len(file_array) - 3]

        newpath = (save_path + big_directory_name + '\\' + directory_name + '\\' + big_directory_name + "_" + directory_name + "_" + "image_" + str(number).zfill(4) + ".png")
        if (not new_generation):
            if not os.path.exists(newpath):
                new_generation = True

        if (new_generation):
            file = open(file, "r")
            list = file.read().splitlines()

            list = list_maker(list)
            dimension = 512
            matrix = bitmap_maker(list, dimension)

            npa = np.asarray(matrix, dtype=np.float32)
            #npa = black_white(npa)

            im = Image.fromarray(npa)
            im = im.convert("L")

            im.save(newpath)
            print("progress: ", length,"/",number, " -- ", round(number/length*100, 2),"%")



#####################################
#            prediction             #
#####################################

def number_fixer(n):
    if n == 1:
        return 1
    elif n == 2:
        return 10
    elif n == 3:
        return 11
    elif n == 4:
        return 12
    elif n == 5:
        return 13
    elif n == 6:
        return 14
    elif n == 7:
        return 15
    elif n == 8:
        return 16
    elif n == 9:
        return 17
    elif n == 10:
        return 2
    elif n == 11:
        return 3
    elif n == 12:
        return 4
    elif n == 13:
        return 5
    elif n == 14:
        return 6
    elif n == 15:
        return 7
    elif n == 16:
        return 8
    elif n == 17:
        return 9
    else:
        return 100


def predict(model_path, batch_path):
    img_height = 512
    img_width = 512
    model = keras.models.load_model(model_path)
    #model = keras.models.load_model(r'C:\Users\Felvea\Documents\Python\Bosch2\model_bosch_final_t')

    #Predict!!
    size = 108
    my_list = np.array([0] * size)
    max_percentage = 0

    files = glob.glob(batch_path, recursive=True)
    x = 0
    for file in files:
        test_data_path = pathlib.Path(file)


        img = tf.keras.utils.load_img(
            test_data_path, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        #class_names = model.argmax(axis=-1)
        #my_list[score]=1

        my_list[np.argmax(score)] = 100 * np.max(score) + my_list[np.argmax(score)]
        max_percentage += 100 * np.max(score)
#        print(
#            "This image(" + str(x) + ") most likely belongs to {} with a {:.2f} percent confidence."
#            #.format(class_names[np.argmax(score)], 100 * np.max(score))
#           .format('Bosch ' + str(np.argmax(score) + 1), 100 * np.max(score))
            
#        )
        x = x + 1
    
    second_answer = 2
    third_answer = 3

    #for x in range(0, 17):
        #print('Bosch' + str(x + 1) + ': probability - ' + str(round(my_list[x] / max_percentage * 100)) + '%')

    #First guess
    max_index = np.argmax(my_list)
    first_answer = number_fixer(max_index + 1)
    my_list = np.delete(my_list, max_index)

    #Second guess
    max_index = np.argmax(my_list)
    second_answer = number_fixer(max_index + 1)
    my_list = np.delete(my_list, max_index)

    #Third guess
    max_index = np.argmax(my_list)
    third_answer = number_fixer(max_index + 1)
    my_list = np.delete(my_list, max_index)




    print(str(first_answer) + " " + str(second_answer) + " " + str(third_answer))
    return [first_answer, second_answer, third_answer]

    



#####################################
#               main                #
#####################################


def main():
    #Create directory system and generate images from xyz files
    path_to_folder = os.path.dirname(__file__)
    data_path = path_to_folder + '\**\*.xyz'
    save_path = path_to_folder + '\Images\\'

    model_path = path_to_folder + '\model_1'
    

    create_directory(data_path, save_path)
    batch_save(True, data_path, save_path)

    #Prediction
    print("first")
    with open(path_to_folder + r'\resultMolnarMartin1.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(1, 18):
            batch_path = path_to_folder + '\Images\Turn01\Bosch' + str(i) + '\*.png'
            result_array = predict(model_path, batch_path)
            writer.writerow([result_array[0], result_array[1], result_array[2]])

    print("second")
    with open(path_to_folder + r'\resultMolnarMartin2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(1, 18):
            batch_path = path_to_folder + '\Images\Turn02\Bosch' + str(i) + '\*.png'
            result_array = predict(model_path, batch_path)
            writer.writerow([result_array[0], result_array[1], result_array[2]])

    print("third")
    with open(path_to_folder + r'\resultMolnarMartin3.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(1, 18):
            batch_path = path_to_folder + '\Images\Turn03\Bosch' + str(i) + '\*.png'
            result_array = predict(model_path, batch_path)
            writer.writerow([result_array[0], result_array[1], result_array[2]])

    print("fourth")
    with open(path_to_folder + r'\resultMolnarMartin4.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(1, 18):
            batch_path = path_to_folder + '\Images\Turn04\Bosch' + str(i) + '\*.png'
            result_array = predict(model_path, batch_path)
            writer.writerow([result_array[0], result_array[1], result_array[2]])




if __name__ == "__main__":
    main()
