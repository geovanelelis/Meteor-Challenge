import cv2 as cv
import numpy as np

# Carrega a imagem
img = cv.imread("meteor_challenge_01.png", cv.IMREAD_COLOR)

# Define as cores que serão capturadas para a contagem de elementos
estrelas = np.array([255, 255, 255])
meteoros = np.array([0, 0, 255])

# Cria máscaras baseadas nas cores que serão capturadas
mascara_estrelas = cv.inRange(img, estrelas, estrelas)
mascara_meteoros = cv.inRange(img, meteoros, meteoros)

# Função para contar os objetos contidos nas máscaras
def contador(mascara):
    # Encontra os contornos na máscara
    contornos, _ = cv.findContours(mascara, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    return len(contornos)

# Conta as estrelas e meteoros
num_estrelas = contador(mascara_estrelas)
num_meteoros = contador(mascara_meteoros)

# Exibe os resultados
print(f'Estrelas detectadas: {num_estrelas}')
print(f'Meteoros detectados: {num_meteoros}')

# Exibe a imagem original e as imagens das máscaras
cv.imshow('Imagem Original', img)
cv.imshow('Estrelas', mascara_estrelas)
cv.imshow('Meteoros', mascara_meteoros)

cv.waitKey(0)