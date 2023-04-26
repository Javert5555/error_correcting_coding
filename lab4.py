import tkinter
# from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from random import randint
from math import floor
from tkinter import messagebox
from tkinter import filedialog as fd
# from tkinter.messagebox import showinfo
from copy import deepcopy


# # img_array = np.asarray(Image.open('./poehaly8x6.jpg').convert('RGB'))

# # # print(img_array)

# root = tkinter.Tk()

# # # создаем рабочую область
# frame = tkinter.Frame(root)
# frame.grid()

# # #Добавим метку
# # label = tkinter.Label(frame, text="Hello, World!").grid(row=1,column=1)


# # # вставляем кнопку
# # but = tkinter.Button(frame, text="Кнопка").grid(row=1, column=2)

# # #Добавим изображение
# # canvas = tkinter.Canvas(root, height=400, width=700)
# # image = Image.open("./poehaly8x6.jpg")
# # photo = ImageTk.PhotoImage(image)
# # image = canvas.create_image(0, 0, anchor='nw',image=photo)
# # canvas.grid(row=2,column=1)
# # # root.mainloop()

# canvas = tkinter.Canvas(root, height=400, width=700)
# image = Image.open("./poehaly.jpg")
# photo = ImageTk.PhotoImage(image)
# image = canvas.create_image(0, 0, anchor='nw',image=photo)
# canvas.grid(row=2,column=1)
# root.mainloop()



















# # функция превращает двумерный список чисел в одномерный список строк
# def get_str_from_double_array(array):
#     str_array = []
#     for row in array:
#         # из numpy.int32 конвертируем в список
#         row = list(row)
#         for j in range(len(row)):
#             row[j] = str(row[j])
#         str_array.append(''.join(row))    
#     return str_array
# #############################################################

# def get_error_vectors(checking_sys_matrix_transpose):
#     # делаем копию, чтобы не менять исходную
#     column_count = len(checking_sys_matrix_transpose[0,:])
#     row_count = len(checking_sys_matrix_transpose)
#     error_vectors = []
#     # количество вектор ошибок равно количеству строк транспонированной проверочной матрицы
#     for i in range(row_count):
#         error_vector = []
#         for j in range(row_count):
#             error_vector.append(0)
#         error_vector[i] = 1
#         # т.к. единицы идут справа-налево по диагонали
#         error_vector.reverse()
#         error_vectors.append(error_vector)
#     return(error_vectors)

# # получаем число ошибок, которые код гарантированно находит
# def get_num_errors_found(d_min):
#     num_errors_found = d_min - 1
#     return num_errors_found

# # получаем число гарантированно исправляемых ошибок
# def get_num_errors_fixed(d_min):
#     num_errors_fixed = floor((d_min-1)/2)
#     return num_errors_fixed

# # получаем d минимальное
# def get_d_min(wtn):
#     # делаем копию, чтобы не менять исходную
#     wtn = list(wtn)
#     while 0 in wtn:
#         wtn.remove(0)
#     d_min = min(wtn)
#     return d_min

# # находим wtn путём суммирования единиц в каждом кодовом слове
# def get_wtn(code_words):
#     wtn = []
#     for el in code_words:
#         wtn.append(sum(el))
#     return wtn

# def product_vector_matrix(vector, matrix):
#     product_vector = np.dot(vector, matrix)
#     # если сумма единиц в столбце при умножении чётная, то записываем 0, иначе 1
#     for j in range(len(product_vector)):
#         if (product_vector[j] % 2 == 0):
#             product_vector[j] = 0
#         else:
#             product_vector[j] = 1
#     return list(product_vector)

# def get_code_words_or_syndromes(matrix, vectors):
#     code_words_or_syndromes = []
#     for i in range(len(vectors)):
#         # получаем кодовое слово по формуле c=i*Gsys
#         code_word = product_vector_matrix(vectors[i], matrix)
#         # если сумма единиц в столбце при умножении чётная, то записываем 0, иначе 1
#         code_words_or_syndromes.append(code_word)
#     return code_words_or_syndromes


# def get_inf_words(code_dimension, alphabet_power):
#     inf_words = []
#     for i in range(alphabet_power):
#         # преобразовываем из 10-ой системы в 2-ую
#         inf_word = [int(num) for num in list(bin(i)[2:])]
#         # если длина полученного 2-ого числа меньше длины инф. слов, то добавляем нули влево 
#         while len(inf_word) != code_dimension:
#             inf_word.insert(0, 0)
#         inf_words.append(inf_word)

#     return inf_words

# # получаем проверочную матрицу, при условии, что начальная матрица - порождающая
# def get_checking_sys_matrix_from_general_sys(p_matrix, type_of_matrix):
#     # получаем транспонированную матрицу P
#     # объединяем матрицы P транспонированную и единичную матрицу
#     match type_of_matrix:
#         case 'general':
#             p_matrix_transpose = p_matrix.transpose()
#             row_count = len(p_matrix_transpose)
#             checking_sys_matrix = np.hstack([p_matrix_transpose, np.eye(row_count)])
#         case 'checking':
#             row_count = len(p_matrix)
#             checking_sys_matrix = np.hstack([np.eye(row_count), p_matrix])
#     # print(checking_sys_matrix)
#     return [ map(int, row) for row in checking_sys_matrix]

# # в функции получаем матрицу P в зависимости от типа начальной матрицы
# def get_p_matrix(matrix, type_of_matrix):
#     column_count = len(matrix[0,:])
#     row_count = len(matrix)
#     # находим матрицу P взависимости от типа исходной матрицы
#     match type_of_matrix:
#         case 'general':
#             # получаем правую часть порождающей матрицы (то есть без единичной матрицы слева размерностью row_count)
#             p_matrix = matrix[:, [x for x in range(row_count, column_count)]]
#             # print(p_matrix)
#             if np.size(p_matrix) == 0:
#                 messagebox.showwarning(title="Предупреждение", message="Невозможно получить матрицу P")
#                 return
#         case 'checking':
#             # получаем левую часть порождающей матрицы (то есть без единичной матрицы справа размерностью column_count-row_count)
#             p_matrix = matrix[:, [x for x in range(0, column_count-row_count)]]
#             # print(p_matrix)
#             if np.size(p_matrix) == 0:
#                 messagebox.showwarning(title="Предупреждение", message="Невозможно получить матрицу P")
#                 return
#     return p_matrix


# def get_sys_init_matrix(sys_matrix, init_type_matrix_value):
#     column_count = len(sys_matrix[0,:])
#     row_count = len(sys_matrix)

#     if (row_count > column_count):
#         messagebox.showwarning(title="Предупреждение", message="Матрица не может быть приведена к систематическому виду")
#         return []
    
#     # создаём массив столбцов единичной матрицы, здесь индекс столбца будет соответствовать позиици, на которой расположена единица
#     # в этом столбце
#     i_sys = []
#     for i in range(row_count):
#         i_sys.append([0 for j in range(row_count)])
#         i_sys[i][i] = 1

#     # приводим матрицу к систематическому виду
#     match init_type_matrix_value:
#         case 'general':
#             for i in range(row_count):
#                 for j in range(column_count):
#                     # если столбец матрицы равен i-ому "столбцу" единичной матрицы, то
#                     if (list(sys_matrix[:, j]) == i_sys[i]):
#                         sys_matrix[:,[j, i]] = sys_matrix[:,[i, j]] # - меняем i-ый столбец с j-ым, чтобы получить единичную матрицу слева
#             # проверяем действительно ли матрица систематическая (проверяем, получилась ли слева единичная матрицы)
#             if not (np.array_equal(np.eye(row_count), sys_matrix[:, [x for x in range(row_count)]])):
#                 messagebox.showwarning(title="Предупреждение", message="Матрица не может быть приведена к систематическому виду")
#                 return []
#         case 'checking':
#             for i in range(row_count):
#                 for j in range(column_count):
#                     if (list(sys_matrix[:, j]) == i_sys[i]):
#                         # меняем i-ый столбец со столбцом под индексом column_count - (row_count - i), чтобы единичная матрица получилась справа
#                         sys_matrix[:,[j, column_count - (row_count - i)]] = sys_matrix[:,[column_count - (row_count - i), j]]
#             # проверяем действительно ли матрица систематическая (проверяем, получилась ли справа единичная матрицы)
#             if not (np.array_equal(np.eye(row_count), sys_matrix[:, [x for x in range(column_count - row_count, column_count)]])):
#                 messagebox.showwarning(title="Предупреждение", message="Матрица не может быть приведена к систематическому виду")
#                 return []
    
#     # print(sys_matrix[:, [x for x in range(row_count)]])
#     # general_matrix[:,[0, 1]] = general_matrix[:,[1, 0]]

#     return sys_matrix















# функция превращает двумерный список чисел в одномерный список строк
def get_str_from_double_array(array1):
    array = deepcopy(array1)
    str_array = []
    for row in array:
        # из numpy.int32 конвертируем в список
        row = list(row)
        for j in range(len(row)):
            row[j] = str(row[j])
        str_array.append(''.join(row))    
    return str_array
#############################################################

def get_error_vectors(checking_sys_matrix_transpose1):
    checking_sys_matrix_transpose = deepcopy(checking_sys_matrix_transpose1)
    # делаем копию, чтобы не менять исходную
    column_count = len(checking_sys_matrix_transpose[0,:])
    row_count = len(checking_sys_matrix_transpose)
    error_vectors = []
    # количество вектор ошибок равно количеству строк транспонированной проверочной матрицы
    for i in range(row_count):
        error_vector = []
        for j in range(row_count):
            error_vector.append(0)
        error_vector[i] = 1
        # т.к. единицы идут справа-налево по диагонали
        # error_vector.reverse()
        error_vectors.append(error_vector)
    
    # вектора для декодирования двух ошибок, например:
    # 110000
    # 101000
    # 100100
    # ...
    error_vectors2 = []

    for i in range(len(error_vectors)):
        for j in range(i+1, len(error_vectors)):
            error_vector = deepcopy(error_vectors[i])
            error_vector[j] = 1
            error_vectors2.append(error_vector)
    
    for el in error_vectors2:
        error_vectors.append(el)
    
    return(error_vectors)

# получаем число ошибок, которые код гарантированно находит
def get_num_errors_found(d_min):
    num_errors_found = d_min - 1
    return num_errors_found

# получаем число гарантированно исправляемых ошибок
def get_num_errors_fixed(d_min):
    num_errors_fixed = floor((d_min-1)/2)
    return num_errors_fixed

# получаем d минимальное
def get_d_min(wtn):
    # делаем копию, чтобы не менять исходную
    wtn = list(wtn)
    while 0 in wtn:
        wtn.remove(0)
    d_min = min(wtn)
    return d_min

# находим wtn путём суммирования единиц в каждом кодовом слове
def get_wtn(code_words1):
    code_words = deepcopy(code_words1)
    wtn = []
    for el in code_words:
        wtn.append(sum(el))
    return wtn

def product_vector_matrix(vector1, matrix1):
    vector = deepcopy(vector1)
    matrix = deepcopy(matrix1)
    product_vector = np.dot(vector, matrix)
    # print('product_vector\n', product_vector)
    # если сумма единиц в столбце при умножении чётная, то записываем 0, иначе 1
    for j in range(len(product_vector)):
        if (product_vector[j] % 2 == 0):
            product_vector[j] = 0
        else:
            product_vector[j] = 1
    return list(product_vector)

def get_code_words_or_syndromes(matrix1, vectors):
    matrix = deepcopy(matrix1)
    code_words_or_syndromes = []
    for i in range(len(vectors)):
        # получаем кодовое слово по формуле c=i*Gsys
        code_word = product_vector_matrix(vectors[i], matrix)
        # если сумма единиц в столбце при умножении чётная, то записываем 0, иначе 1
        code_words_or_syndromes.append(code_word)
    return code_words_or_syndromes


def get_inf_words(code_dimension, alphabet_power):
    inf_words = []
    for i in range(alphabet_power):
        # преобразовываем из 10-ой системы в 2-ую
        inf_word = [int(num) for num in list(bin(i)[2:])]
        # если длина полученного 2-ого числа меньше длины инф. слов, то добавляем нули влево 
        while len(inf_word) != code_dimension:
            inf_word.insert(0, 0)
        inf_words.append(inf_word)

    return inf_words

# получаем проверочную матрицу, при условии, что начальная матрица - порождающая
def get_checking_sys_matrix_from_general_sys(p_matrix1, type_of_matrix):
    p_matrix = deepcopy(p_matrix1)
    # получаем транспонированную матрицу P
    # объединяем матрицы P транспонированную и единичную матрицу
    match type_of_matrix:
        case 'general':
            p_matrix_transpose = p_matrix.transpose()
            row_count = len(p_matrix_transpose)
            checking_sys_matrix = np.hstack([p_matrix_transpose, np.eye(row_count)])
        case 'checking':
            row_count = len(p_matrix)
            checking_sys_matrix = np.hstack([np.eye(row_count), p_matrix])
    # print(checking_sys_matrix)
    return [ map(int, row) for row in checking_sys_matrix]

# в функции получаем матрицу P в зависимости от типа начальной матрицы
def get_p_matrix(matrix1, type_of_matrix):
    matrix = deepcopy(matrix1)
    column_count = len(matrix[0,:])
    row_count = len(matrix)
    # находим матрицу P взависимости от типа исходной матрицы
    match type_of_matrix:
        case 'general':
            # получаем правую часть порождающей матрицы (то есть без единичной матрицы слева размерностью row_count)
            p_matrix = matrix[:, [x for x in range(row_count, column_count)]]
            # print(p_matrix)
            if np.size(p_matrix) == 0:
                messagebox.showwarning(title="Предупреждение", message="Невозможно получить матрицу P")
                return
        case 'checking':
            # получаем левую часть порождающей матрицы (то есть без единичной матрицы справа размерностью column_count-row_count)
            p_matrix = matrix[:, [x for x in range(0, column_count-row_count)]]
            # print(p_matrix)
            if np.size(p_matrix) == 0:
                messagebox.showwarning(title="Предупреждение", message="Невозможно получить матрицу P")
                return
    return p_matrix


def get_sys_init_matrix(sys_matrix1, init_type_matrix_value):
    sys_matrix = deepcopy(sys_matrix1)
    column_count = len(sys_matrix[0,:])
    row_count = len(sys_matrix)

    if (row_count > column_count):
        messagebox.showwarning(title="Предупреждение", message="Матрица не может быть приведена к систематическому виду")
        return []
    
    # создаём массив столбцов единичной матрицы, здесь индекс столбца будет соответствовать позиици, на которой расположена единица
    # в этом столбце
    i_sys = []
    for i in range(row_count):
        i_sys.append([0 for j in range(row_count)])
        i_sys[i][i] = 1

    # приводим матрицу к систематическому виду
    match init_type_matrix_value:
        case 'general':
            for i in range(row_count):
                for j in range(column_count):
                    # если столбец матрицы равен i-ому "столбцу" единичной матрицы, то
                    if (list(sys_matrix[:, j]) == i_sys[i]):
                        sys_matrix[:,[j, i]] = sys_matrix[:,[i, j]] # - меняем i-ый столбец с j-ым, чтобы получить единичную матрицу слева
            # проверяем действительно ли матрица систематическая (проверяем, получилась ли слева единичная матрицы)
            if not (np.array_equal(np.eye(row_count), sys_matrix[:, [x for x in range(row_count)]])):
                messagebox.showwarning(title="Предупреждение", message="Матрица не может быть приведена к систематическому виду")
                return []
        case 'checking':
            for i in range(row_count):
                for j in range(column_count):
                    if (list(sys_matrix[:, j]) == i_sys[i]):
                        # меняем i-ый столбец со столбцом под индексом column_count - (row_count - i), чтобы единичная матрица получилась справа
                        sys_matrix[:,[j, column_count - (row_count - i)]] = sys_matrix[:,[column_count - (row_count - i), j]]
            # проверяем действительно ли матрица систематическая (проверяем, получилась ли справа единичная матрицы)
            if not (np.array_equal(np.eye(row_count), sys_matrix[:, [x for x in range(column_count - row_count, column_count)]])):
                messagebox.showwarning(title="Предупреждение", message="Матрица не может быть приведена к систематическому виду")
                return []
    
    # print(sys_matrix[:, [x for x in range(row_count)]])
    # general_matrix[:,[0, 1]] = general_matrix[:,[1, 0]]

    return sys_matrix








########################
########################
########################


# функция возвращает массив пикселей (в каждом пикселе по 3 вектора, обозначающих rgb соответственно)
def get_array_pixels(img_array, len_of_vector):
    # структура img_array = [
    #    [
    #      [12, 133, 42],
    #      [55, 112, 233],
    #      [43, 23, 144]
    #    ],[
    #      [31, 133, 42],
    #      [132, 53, 86],
    #      [45, 87, 11]
    #    ]
    # ]
    array_pixels = []
    for i in range(len(img_array)):
        array_pixels.append([])
        for j in range(len(img_array[i])):
            array_pixels[i].append([])
            # print(img_array[i][j], '1')
            for rgb in img_array[i][j]:
                # print(rgb, '2')
                vector = [int(num) for num in list(bin(rgb)[2:])]
                while len(vector) != len_of_vector:
                    vector.insert(0, 0)
                # print(vector, '3')
                array_pixels[i][j].append(vector)
    return array_pixels

# функция делает от 0 до num_of_errors ошибок в векторе
def make_mistake_in_vector(vector, num_of_errors):
    vector_copy = deepcopy(vector)
    # # Если хотим чтобы ошибки не накладывались друг на друга - раскоментировать нижнее
    # num_memory = -1
    # #
    # print(vector_copy)
    for i in range(randint(0,num_of_errors)):
        num = randint(0, len(vector_copy) - 1)
        # # Если хотим чтобы ошибки не накладывались друг на друга - раскоментировать нижнее
        # while num_memory == num:
        #     num = randint(0, len(vector_copy) - 1)
        # num_memory = num
        # # 
        if (vector_copy[num] == 0):
            vector_copy[num] = 1
        else:
            vector_copy[num] = 0
    # print(vector_copy)
    # print('#####################')
    return vector_copy

# функция делает от 0 до num_of_errors ошибок во всех векторах (в пикселе 3 вектора, ибо rgb)
def make_mistake_in_pixels(array_pixels, num_of_errors):
    array_pixels_copy = deepcopy(array_pixels)
    # структура array_pixels = [
    #    [
    #      [[0, 0, 1, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0]],
    #      [[0, 1, 0, 1, 0, 1, 1, 1], [0, 1, 0, 1, 1, 0, 0, 0], [0, 1, 0, 1, 1, 0, 1, 0]],
    #      [[0, 1, 1, 0, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 0, 1, 1, 0]]
    #    ],[
    #      [[1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 1, 1, 0]],
    #      [[0, 1, 0, 0, 1, 1, 1, 0], [0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1, 1, 1]],
    #      [[0, 0, 1, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0, 1, 0]]
    #    ]
    # ]
    # print(array_pixels)
    for i in range(len(array_pixels_copy)):
        for j in range(len(array_pixels_copy[i])):
            # print(array_pixels_copy[i][j])
            for k in range(len(array_pixels_copy[i][j])):
                array_pixels_copy[i][j][k] = make_mistake_in_vector(array_pixels_copy[i][j][k], num_of_errors)
    # print(array_pixels)
    return array_pixels_copy

# из массива пикселей, состоящих из 3 векторов, в массив пикселей, состоящих из 3 чисел
def get_img_array_from_array_pixels(array_pixels_with_mistakes):
    array_pixels_with_mistakes_copy = deepcopy(array_pixels_with_mistakes)
    # структура img_array = [
    #    [
    #      [12, 133, 42],
    #      [55, 112, 233],
    #      [43, 23, 144]
    #    ],[
    #      [31, 133, 42],
    #      [132, 53, 86],
    #      [45, 87, 11]
    #    ]
    # ]
    for i in range(len(array_pixels_with_mistakes_copy)):
        for j in range(len(array_pixels_with_mistakes_copy[i])):
            # print(array_pixels[i][j])
            for k in range(len(array_pixels_with_mistakes_copy[i][j])):
                # переводим каждый вектор(двоичное число) в десятиное число
                array_pixels_with_mistakes_copy[i][j][k] = int(''.join([str(num) for num in array_pixels_with_mistakes_copy[i][j][k]]), 2)
    return array_pixels_with_mistakes_copy

# для вывода промежуточных матриц
def print_matr(text, matr):
    print(text)
    for row in deepcopy(matr):
        print(''.join(str(el) for el in row))
    print('')

def correct_mistakes_in_pixels(array_pixels):
    ################################
    # исправляем ошибки в векторах #
    ################################
    
    array_pixels_copy = deepcopy(array_pixels)
    # print(array_pixels_with_mistakes_copy)



    input_matrix_values = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
    ])

    # input_matrix_values = np.array([
    #     [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
    #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
    # ])

    print_matr('Порождающая матрица: ', input_matrix_values)

    init_type_matrix_value = 'general'

    column_count = len(input_matrix_values[0,:])
    row_count = len(input_matrix_values)
    
    code_dimension = row_count

    code_length = column_count

    general_sys_matrix = get_sys_init_matrix(input_matrix_values, init_type_matrix_value)

    print_matr('Порождающая, систематическая матрица: ', general_sys_matrix)

    # print(general_sys_matrix)
    # если матрица не может быть приведена к систематическому виду просто прерываем выполнение программы
    if np.size(general_sys_matrix) == 0: return

    p_matrix = get_p_matrix(general_sys_matrix, init_type_matrix_value)

    # print(p_matrix)

    # если матрицы P не существует, то прерываем выполнение программы
    if p_matrix is None: return

    # из списка map объектов в обычный список
    checking_sys_matrix = [list(row) for row in get_checking_sys_matrix_from_general_sys(p_matrix, init_type_matrix_value)]
    # for el in checking_sys_matrix:
    #     print(el)
    print_matr('Проверочная, систематическая матрица: ', checking_sys_matrix)

    code_speed = round(code_dimension/code_length, 2)
    
    alphabet_power = 2**code_dimension
    # print('alphabet_power',alphabet_power)

    inf_words = get_inf_words(code_dimension, alphabet_power)
    # print('inf_words',inf_words)
    print_matr('Информационные слова: ', inf_words)

    code_words = get_code_words_or_syndromes(general_sys_matrix, inf_words)
    # print('code_words',code_words)
    print_matr('Кодовые слова: ', code_words)

    wtn = get_wtn(code_words)
    # print('wtn',wtn)

    d_min = get_d_min(wtn)
    # print('d_min',d_min)

    num_errors_fixed = get_num_errors_fixed(d_min)
    # print('num_errors_fixed',num_errors_fixed)

    num_errors_found = get_num_errors_found(d_min)
    # print('num_errors_found',num_errors_found)
    checking_sys_matrix_transpose = np.array(deepcopy(checking_sys_matrix)).transpose()

    print_matr('Проверочная, систематическая, транспонированная матрица: ', checking_sys_matrix_transpose)

    error_vectors = get_error_vectors(deepcopy(checking_sys_matrix_transpose))
    # print('error_vectors',error_vectors)
    print_matr('Векторы ошибок: ', error_vectors)

    syndromes = get_code_words_or_syndromes(deepcopy(checking_sys_matrix_transpose), deepcopy(error_vectors))
    # print('syndromes',syndromes)
    print_matr('Синдромы: ', syndromes)

    for i in range(len(array_pixels_copy)):
        for j in range(len(array_pixels_copy[i])):
            # print(array_pixels[i][j])
            for k in range(len(array_pixels_copy[i][j])):
                # получаем индекс соответствующего информационного слова
                index_of_code_word = inf_words.index(array_pixels_copy[i][j][k])
                
                # получаем закодированное информационное слово
                code_word_with_mistake = make_mistake_in_vector(deepcopy(code_words[index_of_code_word]), num_errors_fixed)
                # print(code_word_with_mistake)
                # print(inf_words[index_of_code_word])
                # print('########################')
                # исправляем ошибку в кодовом слове, только в случае, если это слово не содержится в таблице кодовых слов
                if code_word_with_mistake not in code_words:
                    # print('really')
                    if (num_errors_fixed != 0 and array_pixels_copy[i][j][k]):
                        
                        # print('checking_sys_matrix_transpose',checking_sys_matrix_transpose)
                        # print(checking_sys_matrix_transpose)
                        # print(array_pixels_copy[i][j][k])
                        if (len(code_word_with_mistake) != len(deepcopy(checking_sys_matrix_transpose))):
                            messagebox.showwarning(title="Предупреждение", message="Длина v-вектора не равна количеству строк проверочной систематической транспонированной матрицы")
                            return

                        s_vector = product_vector_matrix(deepcopy(code_word_with_mistake), deepcopy(checking_sys_matrix_transpose))
                        # print('error_vectors\n',np.array(deepcopy(error_vectors)))
                        # print('syndromes\n',np.array(deepcopy(syndromes)))
                        # print('s_vector',s_vector)
                        
                        # получаем индекс нашего синдрома
                        try:
                            index_of_s_vector = syndromes.index(s_vector)
                            # print('index_of_s_vector',index_of_s_vector)
                        # если полученного синдрома нет в списке синдромов, то прерываем работу программы с ошибкой
                        except:
                            messagebox.showwarning(title="Предупреждение", message="Для данного вектора нет решения (в синдроме)")
                            return
                        
                        # получаем вектор ошибки с таким же индексом, как и у полученного синдрома
                        e_vector = deepcopy(error_vectors[index_of_s_vector])
                        # print('e_vector',e_vector)

                        # получаем кодовое слово
                        # для сложения векторов конвертируем из обычных массивов в np.Array() - объекты
                        c_vector = list(np.array(deepcopy(code_word_with_mistake)) + np.array(deepcopy(e_vector)))

                        # в случае если случилась ситуация 1+1 в векторе c
                        for d in range(len(c_vector)):
                            if (c_vector[d] == 2):
                                c_vector[d] = 0

                        # print('c_vector',c_vector)
                        # получаем индекс нашего кодового слова
                        # array_pixels_copy[i][j][k] = c_vector
                        # print(123,c_vector)
                    try:
                        # print(c_vector, 'c_vector')
                        # print(code_words)
                        index_of_c_vector = code_words.index(c_vector)
                        # index_of_c_vector = code_words.index([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0])
                        # print('index_of_c_vector',index_of_c_vector)
                    # если полученного кодового слова нет в списке синдромов, то прерываем работу программы с ошибкой
                    except:
                        messagebox.showwarning(title="Предупреждение", message="Для данного вектора нет решения (в код. словах)")
                        return
                    
                    # получаем информационное слово с таким же индексом, как и у кодового слова
                    # print(array_pixels_copy)
                    # print(len(array_pixels_copy))
                    array_pixels_copy[i][j][k] = inf_words[index_of_c_vector]
                    # print(inf_words[index_of_c_vector])
                    # print('i_vector',i_vector)

    return array_pixels_copy


def get_solution(path_to_image):
    # img_array = np.asarray(Image.open('./poehaly.jpg').convert('RGB'))
    img_array = np.asarray(Image.open(path_to_image).convert('RGB'))

    
    len_of_color_in_double = 8

    # получаем массив пикселей
    array_pixels = get_array_pixels(img_array, len_of_color_in_double)

    # получаем пиксели с количеством ошибок равным 2
    array_pixels_with_mistakes = make_mistake_in_pixels(array_pixels, 2)
    # print(array_pixels_with_mistakes)

    img_array_from_array_pixels = get_img_array_from_array_pixels(array_pixels_with_mistakes)
    # print(np.array(array_pixels_with_mistakes))

    img_with_mistakes = Image.fromarray(np.array(img_array_from_array_pixels, dtype=np.uint8))
    img_with_mistakes.save('./assets/img_with_mistakes.png')

    # print(array_pixels_with_mistakes)

    array_pixels_without_mistakes = correct_mistakes_in_pixels(array_pixels)
    # print(array_pixels_without_mistakes)
    img_array_from_array_pixels1 = get_img_array_from_array_pixels(array_pixels_without_mistakes)
    img_without_mistakes = Image.fromarray(np.array(img_array_from_array_pixels1, dtype=np.uint8))
    img_without_mistakes.save('./assets/img_without_mistakes.png')
    # print(array_pixels_without_mistakes)

    # img_array = np.asarray(Image.open('./poehaly8x6.jpg').convert('RGB'))

    ################################
# get_solution('./poehaly.jpg')

######################################## 
############## tkinter #################
########################################

class Main(tkinter.Tk):
    def __init__(self):
        super().__init__()
        self.geometry('250x250+500+300')
        self.title('Info')
        button_1 = tkinter.Button(self, text='Выбрать изображение', font='Times 12',command=self.select_file)
        button_1.pack(expand=True)

    def select_file(self):
        self.filetypes = (
            ('png', '*.png'),
            ('jpg', '*.jpg'),
            ('jpeg', '*.jpeg'),
            # если хотим выбирать ве файлы раскоментить нижнее
            # ('All files', '*.*')
        )

        self.filename = fd.askopenfilename(
            title='Open a file',
            initialdir='./',
            filetypes=self.filetypes
        )
        get_solution(self.filename)
        self.top_level = Top(self.filename, 'Выбранное изображение')
        self.top_level = Top('./assets/img_with_mistakes.png', 'Изображение с ошибками (зашумлённое)')
        self.top_level = Top('./assets/img_without_mistakes.png', 'Изображение закодированное-зашумлённое-декодированное')


class Top(tkinter.Toplevel):
    def __init__(self, filename, title):
        super().__init__()
        self.title(title)
        self.img = Image.open(filename)
        self.width, self.height = self.img.size

        self.geometry(f"{self.width}x{self.height}")

        self.img_tk = ImageTk.PhotoImage(self.img)
        self.label = tkinter.Label(self, image=self.img_tk)
        self.label.pack()

if __name__ == "__main__":
    main = Main()
    main.mainloop()

#############################################
#############################################
#############################################



# 1100000111110111111101111
# Для 3 ошибок
# 1 0 0 0 0 0 0 0 1 1 1 0 1 0 1 1 1 1 0 0 1 0 0 1 1
# 0 1 0 0 0 0 0 0 0 1 0 1 0 0 1 0 1 0 0 1 1 1 1 0 0
# 0 0 1 0 0 0 0 0 0 0 1 0 1 1 1 1 0 1 1 1 1 0 1 1 1
# 0 0 0 1 0 0 0 0 0 1 0 1 1 0 1 0 1 1 0 0 1 0 1 1 1
# 0 0 0 0 1 0 0 0 1 0 1 1 1 1 0 1 1 1 1 1 1 1 0 0 1
# 0 0 0 0 0 1 0 0 1 1 0 1 0 1 0 1 1 1 0 1 1 0 1 0 1
# 0 0 0 0 0 0 1 0 1 0 1 0 0 1 1 0 1 0 1 1 1 0 0 1 0
# 0 0 0 0 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1

# [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1]
# [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0]
# [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1]
# [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
# [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1]
# [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0]
# [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]

# Для 2 ошибок
# 1 0 0 0 0 0 0 0 1 1 1 0 1 0 1 1 1 1 0 0
# 0 1 0 0 0 0 0 0 0 1 0 1 0 0 1 0 1 0 0 1
# 0 0 1 0 0 0 0 0 0 0 1 0 1 1 1 1 0 1 1 1
# 0 0 0 1 0 0 0 0 0 1 0 1 1 0 1 0 1 1 0 0
# 0 0 0 0 1 0 0 0 1 0 1 1 1 1 0 1 1 1 1 1
# 0 0 0 0 0 1 0 0 1 1 0 1 0 1 0 1 1 1 0 1
# 0 0 0 0 0 0 1 0 1 0 1 0 0 1 1 0 1 0 1 1
# 0 0 0 0 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1

# [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0]
# [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
# [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
# [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
# [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
# [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1]
# [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]


# Для 2 ошибок
# 1 0 0 0 0 0 0 0 1 1 1 0 1 0 1 1 1 1 0
# 0 1 0 0 0 0 0 0 0 1 0 1 0 1 1 1 1 0 1
# 0 0 1 0 0 0 0 0 1 0 1 0 1 1 1 1 0 1 1
# 0 0 0 1 0 0 0 0 0 1 0 1 1 0 1 0 1 1 0
# 0 0 0 0 1 0 0 0 1 0 1 1 1 1 0 1 1 1 1
# 0 0 0 0 0 1 0 0 1 1 0 1 0 1 0 1 1 1 0
# 0 0 0 0 0 0 1 0 0 0 1 0 1 1 1 0 1 0 1
# 0 0 0 0 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1

# [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0]
# [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1]
# [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1]
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
# [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
# [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0]
# [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1]
# [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]