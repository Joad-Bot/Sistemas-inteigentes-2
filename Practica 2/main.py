import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

#           EXTRAER PUNTOS

def Extraer_Puntos(ImgA, ImgB):
    akaze = cv2.AKAZE_create()
    kpts1, desc1 = akaze.detectAndCompute(ImgA, None)
    kpts2, desc2 = akaze.detectAndCompute(ImgB, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    nn_matches = bf.match(desc1, desc2)

    # good = []

    # for m, n in nn_matches:
    #     if m.distance < 0.9 * n.distance:
    #         good.append([m])

    good = sorted(nn_matches, key=lambda x: x.distance)
    im3 = cv2.drawMatches(ImgA, kpts1, ImgB, kpts2, good[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2_imshow(im3)

    pointsImgA = np.empty([len(good), 2])
    pointsImgB = np.empty([len(good), 2])


    for i in range(len(good)):
        pointsImgA[i, :] = kpts1[good[i].queryIdx].pt
        pointsImgB[i, :] = kpts2[good[i].trainIdx].pt

    return pointsImgA[:600], pointsImgB[:600]

#           DESPLEGAR IMÁGENES

def plot_images(*imgs, figsize=(10,5), hide_ticks=False):
    '''Display one or multiple images.'''
    f = plt.figure(figsize=figsize)
    width = np.ceil(np.sqrt(len(imgs))).astype('int')
    height = np.ceil(len(imgs) / width).astype('int')
    for i, img in enumerate(imgs, 1):
        ax = f.add_subplot(height, width, i)
        if hide_ticks:
            ax.axis('off')
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#           CREAR PANORAMA

def Crear_Figura_Panoramica(ImgA, ImgB, T):
    dim_x = ImgA.shape[0] + ImgB.shape[0]
    dim_y = ImgA.shape[1] + ImgB.shape[1]
    dim = (dim_x, dim_y)

    warped = cv2.warpPerspective(ImgB, T, dim)

    #plot_images(warped)
    comb = warped.copy()

    # combinar las dos imagenes
    comb[0:ImgA.shape[0], 0:ImgA.shape[1]] = ImgA

    # crop (Recortar al tamaño de la imagen de salida)
    gray = cv2.cvtColor(comb, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    c = max(cnts, key=cv2.contourArea)

    (x, y, w, h) = cv2.boundingRect(c)

    comb = comb[y:y + h, x:x + w]
    plot_images(comb)
    plt.show() 

#           CONSTRUIR MATRIZ DE TRANSFORMACIÓN

def Construir_Matriz_Transformacion(x):
    T = np.array([[x[0], x[1], x[2]], [x[3], x[4], x[5]], [x[6], x[7], x[8]]])
    return T

#           CÁLCULO DEL ERROR

def Calcular_Errores(pA, ppB):
    ppB = ppB.transpose()
    e = np.sqrt((pA[:, 0] - ppB[:, 0]) ** 2 + (pA[:, 1] - ppB[:, 1]) ** 2)
    return e

#           TRANSFORMAR PUNTOS

def Transformar_Puntos(p, T):
    p2 = np.empty([len(p), 2])
    for i in range(len(p)):
        p2[i, :] = p[i]

    P = np.concatenate((p2, np.ones([len(p),1])),1) @ T.transpose()
    pp = np.array([P[:, 0] / P[:, 2], P[:, 1] / P[:, 2]])

    return pp

#           ALGORITMO

# Leer imágenes y extraer puntos de inter'es
ImgA = cv2.imread('A.bmp')
ImgB = cv2.imread('B.bmp')

pA, pB = Extraer_Puntos(ImgA, ImgB)

# Parámetros

M = len(pB)
l = 1e-6 # <--- REGULACIÓN

G = 300
N = 30

F = 0.55
CR = 0.95

xl = np.array([-1, -1, -ImgA.shape[1], -1, -1, -ImgA.shape[0], -1e-3, -1e-3, 1])
xu = np.array([ 1,  1,  ImgA.shape[1],  1,  1,  ImgA.shape[0],  1e-3,  1e-3, 1])
D = 9

x = np.zeros((D, N))
fitness = np.zeros(N)

fx_plot = np.zeros(G)

# Algoritmo de optimización
for i in range(N):
    x[:, i] = xl + (xu - xl) * np.random.rand(D)
    x[8, i] = 1.0

    T = Construir_Matriz_Transformacion(x[:, 1])                    # <------ Completar argumentos
    ppB = Transformar_Puntos(pB, T)                                 # <------ Completar argumentos
    e = Calcular_Errores(pA, ppB)                                   # <------ Completar argumentos

    fitness[i] = l * (1/D) * np.sum(x[:,i] ** 2) +  np.mean(e)      # <------ Completar fitness

for n in range(G):
    for i in range(N):
        # Mutación

        # Estas dos lineas son de ayuda para seleccionar los vectores r1, r2 y r3 (recordar que deben ser diferentes)
        # necesarios para el cálculo de "v" en Evolución Diferencial
        I = np.random.permutation(N) # Esta línea hace una permutación de N números
        I = np.delete(I, [np.where(I == i)[0][0]]) # Esta linea elimina el elemento i que estemos analizando en esta iteración

        ## ----------- COMPLETAR AQUI ------------------------------------------##
        r1, r2, r3 = I[:3]
        v = x[:, r1] + F * (x[:, r2] - x[:, r3])
        v[8] = 1.0
        ## ---------------------------------------------------------------------##

        # Recombinación
        u = np.zeros(D)
        k = np.random.randint(D)

        for j in range(D):
            if np.random.rand() <= CR or k == j:
                u[j] = v[j].copy()
            else:
                u[j] = x[j, i].copy()
        u[8] = 1.0

        # Selección
        T_2 = Construir_Matriz_Transformacion(u)        # <------ Completar argumentos
        ppB_2 = Transformar_Puntos(pB, T_2)                   # <------ Completar argumentos
        e_2 = Calcular_Errores(pA, ppB_2)                       # <------ Completar argumentos

        fitness_u = l * (1/D) * np.sum(u ** 2) +  np.mean(e_2)   # <------ Completar fitness

        if fitness_u < fitness[i]:
            x[:, i] = u
            fitness[i] = fitness_u

    fx_plot[n] = np.min(fitness)

igb = np.argmin(fitness)
T = Construir_Matriz_Transformacion(x[:, igb])
print(T)
panorama = Crear_Figura_Panoramica(ImgA, ImgB, T)

plt.plot(fx_plot)

plt.xlabel('Generación')
plt.ylabel('Mejor fitness')
plt.title('Convergencia DE')
plt.show()