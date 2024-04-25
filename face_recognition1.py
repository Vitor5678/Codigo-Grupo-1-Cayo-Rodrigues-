import mediapipe as mp
import cv2 as cv
import os
import numpy as np

def carregar_banco_de_dados(caminho_banco_dados):
    banco_de_dados = {}
    for arquivo in os.listdir(caminho_banco_dados):
        if arquivo.endswith('.jpg') or arquivo.endswith('.png'):
            caminho = os.path.join(caminho_banco_dados, arquivo)
            imagem = cv.imread(caminho)
            if imagem is None:
                print(f"Não foi possível carregar a imagem: {caminho}")
                continue
            banco_de_dados[arquivo] = imagem
    print(f"Total de imagens no banco de dados: {len(banco_de_dados)}")
    print("Nomes das imagens no banco de dados:")
    for nome in banco_de_dados.keys():
        print(nome)
    return banco_de_dados

def extrair_caracteristicas_faciais(marcos_faciais):
    caracteristicas = []
    for marco in marcos_faciais.landmark:
        caracteristicas.extend([marco.x, marco.y, marco.z])
    return caracteristicas

def comparar_com_banco_de_dados(caracteristicas_rosto, banco_de_dados):
    melhor_correspondencia = None
    menor_distancia = float('inf')
    for nome, caracteristicas_bd in banco_de_dados.items():
        distancia = np.linalg.norm(np.array(caracteristicas_rosto) - caracteristicas_bd)
        if distancia < menor_distancia:
            menor_distancia = distancia
            melhor_correspondencia = nome
    return melhor_correspondencia

def calcular_inclinacao_cabeca(marcos_faciais):
    try:
        ponto_topo_cabeca = marcos_faciais.landmark[152]
        ponto_base_cabeca = marcos_faciais.landmark[10]
        
        inclinacao = (ponto_topo_cabeca.y - ponto_base_cabeca.y) / (ponto_topo_cabeca.x - ponto_base_cabeca.x)
       
        return inclinacao
    except Exception as e:
        print(f"Erro ao calcular a inclinação da cabeça: {e}")
        return None
def verificar_piscar_olhos(marcos_faciais):
    try:
        olho_esquerdo = [marcos_faciais.landmark[159].x, marcos_faciais.landmark[159].y]  # Índices para o canto externo do olho esquerdo
        olho_direito = [marcos_faciais.landmark[386].x, marcos_faciais.landmark[386].y]  # Índices para o canto externo do olho direito
        
        olho_esquerdo_aberto = olho_esquerdo[1] < marcos_faciais.landmark[33].y
        olho_direito_aberto = olho_direito[1] < marcos_faciais.landmark[263].y
        
        return olho_esquerdo_aberto, olho_direito_aberto
    except Exception as e:
        print(f"Erro ao verificar o piscar dos olhos: {e}")
        return False, False


video_capture = cv.VideoCapture(0)
reconhecer_rosto = mp.solutions.face_mesh
reconhecedor_rosto = reconhecer_rosto.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

caminho_banco_dados = "C:/Users/vitor/OneDrive/Área de Trabalho/face recognition/banco de dados"
banco_de_dados = carregar_banco_de_dados(caminho_banco_dados)

while video_capture.isOpened():
    validacao, frame = video_capture.read()
    if not validacao:
        print("Não foi possível capturar o frame da câmera.")
        break
    imagem = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
    resultado = reconhecedor_rosto.process(imagem)

    if resultado.multi_face_landmarks:
        for rosto in resultado.multi_face_landmarks:
            caracteristicas_rosto = extrair_caracteristicas_faciais(rosto)
            inclinacao_cabeca = calcular_inclinacao_cabeca(rosto)
            olho_esquerdo_aberto, olho_direito_aberto = verificar_piscar_olhos(rosto)
            
            nome_correspondente = comparar_com_banco_de_dados(caracteristicas_rosto, banco_de_dados)
            
            if nome_correspondente:
                cv.putText(frame, f"Olá, {nome_correspondente.split('.')[0]}", (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if inclinacao_cabeca is not None and abs(inclinacao_cabeca) > 0.1:
                cv.putText(frame, "Mova a cabeca", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (1, 37, 40), 2)

            if not olho_esquerdo_aberto or not olho_direito_aberto:
                cv.putText(frame, "Pisque os olhos", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (1, 37, 40), 2)
            
            

    cv.imshow('Prova de Vida', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()
