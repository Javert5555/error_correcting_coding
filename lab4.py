import tkinter
from PIL import Image, ImageTk
import numpy as np
from random import randint

img_array = np.asarray(Image.open('./poehaly.jpg').convert('RGB'))

# print(img_array)

root = tkinter.Tk()

# создаем рабочую область
frame = tkinter.Frame(root)
frame.grid()

#Добавим метку
label = tkinter.Label(frame, text="Hello, World!").grid(row=1,column=1)


# вставляем кнопку
but = tkinter.Button(frame, text="Кнопка").grid(row=1, column=2)

#Добавим изображение
canvas = tkinter.Canvas(root, height=400, width=700)
image = Image.open("./poehaly.jpg")
photo = ImageTk.PhotoImage(image)
image = canvas.create_image(0, 0, anchor='nw',image=photo)
canvas.grid(row=2,column=1)
# root.mainloop()

































########################
########################
########################

img_array = np.asarray(Image.open('./poehaly.jpg').convert('RGB'))

# функция возвращает массив пикселей (в каждом пикселе по 3 вектора, обозначающих rgb соответственно)
def get_array_pixels(img_array):
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
                while len(vector) != 8:
                    vector.insert(0, 0)
                # print(vector, '3')
                array_pixels[i][j].append(vector)
    return array_pixels

# функция делает от 0 до 2 ошибок в векторе
def make_mistake_in_vector(vector):
    for i in range(randint(0, 2)):
        num = randint(0, 7)
        if (vector[num] == 0):
            vector[num] = 1
        else:
            vector[num] = 0
    return vector

# функция делает от 0 до 2 ошибок во всех векторах (в пикселе 3 вектора, ибо rgb)
def make_mistake_in_pixels(array_pixels):
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
    for i in range(len(array_pixels)):
        for j in range(len(array_pixels[i])):
            # print(array_pixels[i][j])
            for k in range(len(array_pixels[i][j])):
                array_pixels[i][j][k] = make_mistake_in_vector(array_pixels[i][j][k])
    return array_pixels

def get_img_array_from_array_pixels(array_pixels_with_mistakes):
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
    for i in range(len(array_pixels_with_mistakes)):
        for j in range(len(array_pixels_with_mistakes[i])):
            # print(array_pixels[i][j])
            for k in range(len(array_pixels_with_mistakes[i][j])):
                # переводим каждый вектор(двоичное число) в десятиное число
                array_pixels[i][j][k] = int(''.join([str(num) for num in array_pixels_with_mistakes[i][j][k]]), 2)
    return array_pixels_with_mistakes

array_pixels = get_array_pixels(img_array)
# print(make_mistake_in_vector([0, 1, 1, 1, 0, 1, 0, 0]))
array_pixels_with_mistakes = make_mistake_in_pixels(array_pixels)
img_array_from_array_pixels = get_img_array_from_array_pixels(array_pixels_with_mistakes)
print(np.array(img_array_from_array_pixels))

img_with_mistakes = Image.fromarray(np.array(img_array_from_array_pixels, dtype=np.uint8))
img_with_mistakes.save('./img_with_mistakes.png')
# print(img_array.shape)

# .insert(0, 0)

# print(bin(256))

# print(''.join([str(num) for num in [0, 1, 1, 1, 0, 1, 0, 0]]))