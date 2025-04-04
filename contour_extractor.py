import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_images(images, titles, rows, cols, figsize=(15, 10)):
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i+1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def filter_background(image, black_threshold=20):  # Reduzir o threshold
    """Transforma pixels não-pretos em branco e aplica dilatação"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Dilatação para unir linhas fragmentadas
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    return dilated

def preprocess_image(image_path):
    # Passo 1: Carregar imagem
    image = cv2.imread(image_path)
    
    # Passo 2: Aplicar filtro de fundo + dilatação
    no_bg_image = filter_background(image.copy())
    
    # Passo 3: Operações morfológicas suaves
    kernel = np.ones((0, 0), np.uint8)  # Kernel menor para preservar detalhes
    cleaned = cv2.morphologyEx(no_bg_image, cv2.MORPH_OPEN, kernel)
    
    return image, cleaned

def detect_curves(cleaned):
    # Redução de ruído adicional
    blurred = cv2.GaussianBlur(cleaned, (3, 3), 0)
    
    # Detecção de bordas com limiares mais sensíveis
    edges = cv2.Canny(blurred, 10, 40)  # Limiares mais baixos
    
    # Buscar TODOS os contornos (internos e externos)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    contour_area_threshold = 3000
    selected_contours = [cnt for i, cnt in enumerate(contours) if cv2.contourArea(cnt) > contour_area_threshold]

    # Filtrar contornos por área e hierarquia
    result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(selected_contours):
        cv2.drawContours(result, [cnt], -1, (0, 0, 255), 4)
    
    return selected_contours, result

def resample_contours(contours, num_points=200):
    resampled = []
    for cnt in contours:
        cnt_pts = cnt.squeeze(1)
        if len(cnt_pts) > 1:
            dists = np.linalg.norm(np.diff(cnt_pts, axis=0), axis=1)
            cum_dists = np.insert(np.cumsum(dists), 0, 0)
            target_dists = np.linspace(0, cum_dists[-1], num_points)
            
            # Interpolação linear
            new_pts = np.zeros((num_points, 2))
            for i, d in enumerate(target_dists):
                idx = np.searchsorted(cum_dists, d) - 1
                idx = max(0, min(idx, len(cnt_pts)-2))
                a = (d - cum_dists[idx]) / dists[idx]
                new_pts[i] = cnt_pts[idx] + a * (cnt_pts[idx+1] - cnt_pts[idx])
            resampled.append(new_pts)
        else:
            resampled.append(np.repeat(cnt_pts, num_points, axis=0))
    
    if not resampled:
        return []
    
    # Passo 2: Alinhar as curvas usando a primeira como referência
    reference = resampled[0]
    aligned = [reference]
    
    for curve in resampled[1:]:
        # Encontrar melhor deslocamento cíclico
        distances = []
        for shift in range(num_points):
            shifted = np.roll(curve, shift, axis=0)
            distances.append(np.mean(np.linalg.norm(shifted - reference, axis=1)))
        
        best_shift = np.argmin(distances)
        aligned.append(np.roll(curve, best_shift, axis=0))
    
    # Reformatar para o shape original
    aligned_contours = [c.reshape(-1, 1, 2).astype(np.int32) for c in aligned]
    
    return np.array(aligned_contours)


def compute_mean_curve(curves_array):
    # Verificar se há curvas para processar
    if curves_array.shape[0] == 0:
        return np.array([[[0, 0]]])
        raise ValueError("O array de entrada não contém curvas")
    
    # Calcular a média ao longo do eixo das curvas (axis=0)
    mean_curve = np.mean(curves_array, axis=0)
    
    # Garantir que o shape de saída seja (n_points, 1, 2)
    mean_curve = mean_curve.reshape(-1, 1, 2)
    
    return mean_curve.astype(np.int32)  # Converter para coordenadas inteiras

from scipy.spatial.distance import directed_hausdorff

def remove_duplicate_curves(contours, threshold=10.0):
    unique_contours = []
    
    for current_cnt in contours:
        current_points = current_cnt.squeeze()  # Shape: (100, 2)
        is_duplicate = False

        for kept_cnt in unique_contours:
            kept_points = kept_cnt.squeeze()
            
            # Calcular distância de Hausdorff (bidirecional)
            dist1 = directed_hausdorff(current_points, kept_points)[0]
            dist2 = directed_hausdorff(kept_points, current_points)[0]
            hausdorff_dist = max(dist1, dist2)
            
            # print(hausdorff_dist)
            if hausdorff_dist < threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_contours.append(current_cnt)
    
    return np.array(unique_contours)


from scipy.ndimage import gaussian_filter1d

def gaussian_filter(curva_ruidosa, sigma=1.0):
    pontos = curva_ruidosa.squeeze()  
    x = pontos[:, 0]
    y = pontos[:, 1]
    
    x_suave = gaussian_filter1d(x, sigma=sigma, mode='nearest')
    y_suave = gaussian_filter1d(y, sigma=sigma, mode='nearest')
    
    curva_suavizada = np.stack([x_suave, y_suave], axis=1)
    return curva_suavizada.reshape(-1, 1, 2) 

def plot_curve(curve, show = True):
    x = curve[:, 0, 0]
    y = curve[:, 0, 1]

    plt.axis('equal')
    plt.plot(x, y, 'r-', linewidth=2)  # Linha vermelha contínua

    if show:
        plt.show()

def plot_image(image, show = True):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if show:
        plt.show()

def get_contours_on_image(image_path):
    original, cleaned = preprocess_image(image_path)
    edges, final_result = detect_curves(cleaned)

    # plot_image(final_result)

    return edges, original, final_result

def get_centerline_on_contours(contours):
    contours = resample_contours(contours, 200)
    cleaned_contours = remove_duplicate_curves(contours, 30)
    mean_curve = compute_mean_curve(cleaned_contours)

    """
    for i in cleaned_contours:
        plot_curve(cleaned_edges[0], False)
    plt.show()

    plot_curve(mean_curve)
    """

    return mean_curve 

def main_single_image():
    image_path = "10227.jpg"
    
    contours, original, result = get_contours_on_image(image_path)
    centerline = get_centerline_on_contours(contours)
    # print(centerline.shape)
    centerline = gaussian_filter(centerline)

    plot_image(original, False)
    plot_curve(centerline)

import os

def main_image_batch(input_directory, output_directory):
    files = os.listdir(input_directory)

    for i in files:
        contours, original, result = get_contours_on_image(input_directory + "/" + i)
        centerline = get_centerline_on_contours(contours)
        
        plot_image(original, False)
        plot_curve(centerline, False)

        print("Figure Saved in: " + output_directory + "/" + i + ".png")
        plt.savefig(output_directory + "/" + i + ".png")
        plt.clf()


import sys


if __name__ == "__main__":  
    diretorio_script = os.path.dirname(os.path.abspath(__file__))
    os.chdir(diretorio_script)
    print("Diretório atual:", os.getcwd())

    # main_single_image()
    main_image_batch("in", "out")
    # main()