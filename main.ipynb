'version 13.05.2020'


def __dirExist(source):
    from os import path
    return path.isdir(source)


def __fileExist(source):
    from os import path
    return path.isfile(source)


def path_split(source):
    import os
    lastDir = []
    if __fileExist(source):
        path = os.path.split(source)
        head = path[0]
    elif __dirExist(source):
        head = source
    while 1:
        parts = os.path.split(head)
        if parts[0] == head:  # sentinel for absolute paths
            lastDir.insert(0, parts[0])
            break
        elif parts[1] == head:  # sentinel for relative paths
            lastDir.insert(0, parts[1])
            break
        else:
            head = parts[0]
            lastDir.insert(0, parts[1])

    return lastDir[len(lastDir) - 2]


def __dataFrame(phi, q, index=" index", column1="phi", column2="q"):
    import pandas as pd
    index = range(0, len(q))
    val_0 = pd.Series(phi, index=index, name=column1)
    val_1 = pd.Series(q, index=index, name=column2)
    df = pd.concat([val_0, val_1], axis=1)
    return df


def plot(phi, q, x_label="phi", y_label="q", marker_color='bo', border_less=False, max_lim=20, color="red", marker=".",
         grid=False,
         image_PATH="None", xscale="linear", yscale="linear", label="Sensor Data"):
    import matplotlib.pyplot as plt
    from os import path
    plt.plot(phi, q, marker_color, markersize=0.5, label=label)
    plt.xlabel('Winkel Phi')
    plt.ylabel('Ladung q')
    plt.xlim(0, max_lim)
    plt.ylim(0, max_lim)
    if border_less:
        plt.subplots_adjust(left=0.0, bottom=0.0, right=0.999, top=1.0)
    else:
        plt.subplots_adjust(left=0.046, bottom=0.090, right=0.989, top=0.977)
        plt.legend()
    if path.isfile(image_PATH):
        plt.savefig(image_PATH + "_bild.png", dpi=1000)
    plt.show()


def saveToFile(x1, x2, PATH="NONE", index=False):
    df = __dataFrame(x1, x2)
    df.to_csv(PATH + 'test.csv', sep=',', index=False)
    """
    MAX_ROW = 1048576
    try:
        writer = pd.ExcelWriter(PATH+'.xlsx', engine='xlsxwriter')
        df.to_excel(writer, index=index)
        writer.save()
    except:
    """

    # print('maximale speicher von Excel überschritten !, wird als csv gespeichert')


def __unpack_file__(path, max_lim, split_number):
    import struct

    '''all constans '''

    faktor = 65536
    max_winkel = 360
    max_q = 4096
    q_faktor = max_lim / max_q
    phi_faktor = max_lim / faktor

    HEAD_IAL1 = 3
    H_FORMAT = 2
    STEP_BYTE = 4
    HEADER_KOPF = 640
    data_tmp = []
    data = []
    data_ = []
    phi = []
    q = []
    start = 0

    end = 12
    ap = []

    with open(path, 'rb') as file:
        header_ = (file.read())[HEADER_KOPF::]
        h = int((len(header_)) / H_FORMAT)
        data_.extend(struct.unpack_from('<' + ('H' * h), header_[0::]))

    for i in range(len(data_)):
        '''sort data from data_tmp'''

        try:
            header = data_[start:end]
            iaL1 = header[HEAD_IAL1]

            if iaL1 == 0:
                'when iaL1 = 0 , should be skip to next header'
                start = end
                end = start + 12
            else:
                ap.extend([1 / iaL1 for i in range(iaL1)]) # die werte für Häufigkeit Matrix
                start = end
                end = start + iaL1 * 2
                data__ = data_[start:end]
                q_ = data__[0::2]
                phi_ = data__[1::2]
                q__ = []
                phi__ = []
                q__.extend([q_faktor * q_[i] for i in range(len(q_))])
                phi__.extend([phi_faktor * phi_[i] for i in range(len(phi_))])
                q.extend(q__)
                phi.extend(phi__)
                data.extend(data__)
                start = end
                end = start + 12

        except:
            continue
    file.close()
    return phi, q, ap


def __unpack_dir__(source,max_lim,split_number):
    from os import listdir
    phi = []
    q = []
    ap = []
    if __dirExist(source):
        for file in (listdir(source)):
            if (file.endswith(".KON")):

                temp = __unpack_file__(source + file,max_lim,split_number)
                phi.extend(temp[0])
                q.extend(temp[1])
                ap.extend (temp[2])
    else:
        print(FileNotFoundError)
    return phi,q,ap


def unpack(path, max_lim,split_nummer):
    """faktor = 65536
    max_winkel = 360
    max_q = 4096
    q_0 = []
    phi_0 = []
    phi = []
    q = []
    q_0.extend(data[0::2])
    phi_0.extend(data[1::2])

    for i in range(len(q_0)):
        q_ = max_lim / max_q
        phi_ = max_lim / faktor
        q_1 = q_ * q_0[i]
        phi_1 = phi_ * phi_0[i]
        q.append(q_1)
        phi.append(phi_1)"""
    try:
        phi,q,ap = __unpack_file__(path,max_lim,split_number)
    except:
        phi,q,ap = __unpack_dir__(path,max_lim,split_number)
    return phi, q,ap


def frequencyMatrixFunktion(phi, q, ap, max_lim=20, split_number=5):
    import numpy as np

    fm = []

    sequence_array = []
    for i in range(0, split_number + 1):
        sequence_array.append(round(i * (max_lim / split_number), ndigits=3))

    for i in range(len(q)):
        for j in range(len(sequence_array)):
            if (sequence_array[j] <= q[i] < sequence_array[min((j + 1), len(sequence_array) - 1)]):
                for l in range(len(sequence_array)):
                    if (sequence_array[l] <= phi[i] < sequence_array[min((l + 1), len(sequence_array) - 1)]):
                        s = []
                        schritt_array = np.zeros((split_number, split_number))
                        schritt_array.itemset(j, l, ((schritt_array[j][l]) +  ap[i]))
                        s.append(np.reshape(schritt_array, (split_number * split_number,)))
        fm.extend(s)
    return fm


from numpy import *

max_lim = 20
split_number = 3

phi, q, ap = unpack("test.KON", max_lim, split_number)
fm = frequencyMatrixFunktion(phi, q, ap, max_lim, split_number)

print(shape(phi), len(q), shape(ap))
