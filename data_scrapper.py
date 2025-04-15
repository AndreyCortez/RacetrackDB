import os 
import re

from PIL import Image
import cv2
import numpy as np

from bs4 import BeautifulSoup
import chardet

import matplotlib.pyplot as plt

input_dir = "racingcircuits"

def find_matching_html(gif_path):
    gif_nome_base = os.path.splitext(os.path.basename(gif_path))[0]
    
    diretorio = os.path.dirname(gif_path) or os.getcwd()
    
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith('.html'):
            html_nome_base = os.path.splitext(arquivo)[0]
            
            normalizado_gif = re.sub(r'[\W_]+', '', gif_nome_base).lower()
            normalizado_html = re.sub(r'[\W_]+', '', html_nome_base).lower()
            
            if normalizado_gif == normalizado_html:
                return os.path.join(diretorio, arquivo)
    
    return None

def extract_html_table(html_path):
    try:
        with open(html_path, 'rb') as f:
            raw = f.read()
            encoding = chardet.detect(raw)['encoding']

        with open(html_path, 'r', encoding=encoding) as file:
            soup = BeautifulSoup(file, 'html.parser')
            main_table = soup.select_one('body > div:nth-child(3) > center > table')
            
            if not main_table:
                return None

            data = {}
            current_category = None
            pending_rows = {}

            for row in main_table.find_all('tr'):
                if row.find('td', {'bgcolor': '#371C00'}) or not row.find_all('td'):
                    continue

                cells = row.find_all('td')
                primary_cell = cells[0].get_text(strip=True) if cells else ''

                if primary_cell.lower() in ['length', 'direction', 'address', 'telephone', 'website']:
                    key = primary_cell
                    value = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                    data[key] = value

                    rowspan = int(cells[1].get('rowspan', 1)) if len(cells) > 1 else 1
                    if rowspan > 1:
                        pending_rows[key] = rowspan - 1
                elif pending_rows:
                    for key in list(pending_rows.keys()):
                        data[key] = data.get(key, '')
                        pending_rows[key] -= 1
                        if pending_rows[key] == 0:
                            del pending_rows[key]

                if row.find('table'):
                    break

            details_table = main_table.find('table')
            if details_table:
                data['details'] = []
                for detail_row in details_table.find_all('tr')[1:]:
                    link = detail_row.find('a')
                    if link:
                        data['details'].append({
                            'text': link.get_text(strip=True),
                            'url': link.get('href', '')
                        })

            return data

    except Exception as e:
        print(f"Error processing {html_path}: {str(e)}")
        return None

def extract_km_length(circuit_dict):
    if 'Length' not in circuit_dict:
        return None
    
    length_str = circuit_dict['Length']
    
    try:
        clean_str = length_str.replace('\xa0', ' ').strip()
        
        parts = re.split(r'\s*//\s*', clean_str)
        
        for part in parts:
            if 'km' in part.lower():
                match = re.search(r'(\d+[\.,]?\d*)', part)
                if match:
                    return float(match.group(1).replace(',', '.'))
        
        return None
    
    except (AttributeError, ValueError, KeyError):
        return None

def scan_dir_for_tracks(input_dir, max_qtt = -1, qtt = 0):

    tracklist = []

    for i in os.listdir(input_dir):
        if max_qtt != -1 and qtt > max_qtt:
            return tracklist
        
        if i.endswith(".gif"):
            corresponding_html = find_matching_html(input_dir + "/" + i)
            if corresponding_html != None:
                try:
                    html_table = extract_html_table(corresponding_html)
                    circuit_len = extract_km_length(html_table)
                    if circuit_len != None:
                        tracklist.append((input_dir + "/" + i, circuit_len))
                        qtt += 1
                except:
                    continue
        elif os.path.isdir(input_dir + "/" + i):
            for j in scan_dir_for_tracks(input_dir + "/" + i, max_qtt, qtt):
                tracklist.append(j)
                qtt += 1
                if max_qtt != -1 and qtt > max_qtt:
                    return tracklist
    return tracklist

def debug_show(image, step_name, contours=None, points=None, delay=500):
    """Mostra imagens de debug com anotações"""
    img = image.copy()
    if len(img.shape) == 2:  # Se for binária
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if contours is not None:
        cv2.drawContours(img, contours, -1, (0,255,0), 1)
        
    if points is not None:
        for pt in points:
            cv2.circle(img, tuple(pt), 3, (0,0,255), -1)
    
    cv2.putText(img, step_name, (10,30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    
    cv2.imshow(step_name, img)
    cv2.waitKey(delay)
    cv2.destroyWindow(step_name)

def is_valid_circuit(img, min_area=100, closure_tol=5, thickness=3, debug=False, delay=500):
    if debug:
        debug_show(img, "0_original_image", delay=delay)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    if debug:
        debug_show(binary, "1_binary_image", delay=delay)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Critério 1: Apenas um contorno
    if len(contours) != 1:
        if debug: 
            debug_show(img, "REJECTED: Multiple contours", contours, delay=delay*2)
        return False
    
    contour = contours[0]
    
    # Critério 2: Área mínima
    if cv2.contourArea(contour) < min_area:
        if debug: 
            debug_show(img, f"REJECTED: Small area ({cv2.contourArea(contour)} < {min_area})", 
                      contours, delay=delay*2)
        return False
    
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02, True)
    
    if debug:
        debug_show(img, "2_contour_approximation", [approx], delay=delay)
    
    return True


def gif_to_cv2(gif_path):
    try:
        with Image.open(gif_path) as img:
            if img.n_frames > 1:
                print(f"Aviso: O GIF {gif_path} contém {img.n_frames} frames. Usando o primeiro.")
            
            img_rgb = img.convert('RGB')
            
            np_img = np.array(img_rgb)
            
            cv2_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            
            return cv2_img
            
    except Exception as e:
        print(f"Erro ao processar {gif_path}: {str(e)}")
        return None

from scipy import signal
from scipy.ndimage import gaussian_filter1d

def smooth_curve(points, method='moving_avg', window_size=5, sigma=1.0, polyorder=3):
    x, y = points[:, 0], points[:, 1]
    
    if method == 'moving_avg':
        # Média móvel
        kernel = np.ones(window_size) / window_size
        y_smooth = np.convolve(y, kernel, mode='same')
    
    elif method == 'gaussian':
        # Filtro Gaussiano
        y_smooth = gaussian_filter1d(y, sigma=sigma)
    
    elif method == 'savgol':
        # Filtro Savitzky-Golay (preserva melhor picos)
        y_smooth = signal.savgol_filter(y, window_length=window_size, polyorder=polyorder)
    
    else:
        raise ValueError("Método inválido. Use 'moving_avg', 'gaussian' ou 'savgol'.")
    
    return np.column_stack((x, y_smooth))


tracklist = scan_dir_for_tracks(input_dir, 20)

unprocessed_images_dir = 'unprocessed_dataset'

track_images = []
track_lens = []

for i in tracklist:
    try:
        im = gif_to_cv2(i[0])

        track_images.append(im)
        track_lens.append(i[1])

    except Exception as e:
        print(e)


processed_images_dir = "selected_images_dataset"


from collections import defaultdict

def get_racetrack_waypoints(image, simplify_epsilon=2.0, smooth_path=False):
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        raise ImportError("Instale scikit-image: pip install scikit-image")

    # Pré-processamento
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    skeleton = skeletonize(binary // 255).astype(np.uint8) * 255

    # Construção do grafo
    graph = defaultdict(list)
    points = np.argwhere(skeleton == 255)
    height, width = skeleton.shape
    
    for y, x in points:
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= nx < width and 0 <= ny < height and skeleton[ny, nx] == 255:
                    neighbors.append((nx, ny))
        graph[(x, y)] = neighbors

    # Encontra ponto inicial (extremidade)
    start = next(((x, y) for (x, y), neighbors in graph.items() if len(neighbors) == 1), None)
    if not start:
        start = next(iter(graph.keys()))

    # Extração de waypoints
    visited = set()
    waypoints = []
    stack = [(start, None)]
    
    while stack:
        current, prev = stack.pop()
        if current in visited:
            continue
            
        visited.add(current)
        waypoints.append(current)
        
        # Prioriza vizinhos na direção do movimento
        if prev:
            dx = current[0] - prev[0]
            dy = current[1] - prev[1]
            graph[current].sort(key=lambda p: (p[0]-current[0])*dx + (p[1]-current[1])*dy, reverse=True)
            
        for neighbor in graph[current]:
            if neighbor != prev:
                stack.append((neighbor, current))

    # Simplificação
    if simplify_epsilon > 0 and len(waypoints) > 4:
        waypoints = cv2.approxPolyDP(np.array(waypoints, dtype=np.float32), simplify_epsilon, True)
        waypoints = [tuple(p[0]) for p in waypoints]

    # Suavização
    if smooth_path and len(waypoints) > 4:
        from scipy.interpolate import splprep, splev
        tck, _ = splprep(np.array(waypoints).T, s=50)
        u_new = np.linspace(0, 1, len(waypoints))
        x_new, y_new = splev(u_new, tck)
        waypoints = list(zip(x_new, y_new))

    return np.array(waypoints)



def calculate_curve_length(waypoints):
    if len(waypoints) < 2:
        return 0.0
    
    points = np.array(waypoints)
    
    diffs = np.diff(points, axis=0)
    
    distances = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(distances)
    
    return total_length

counter = 0
for i, image in enumerate(track_images):
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    target = np.array((0, 0, 128), dtype=img.dtype)
    mask = np.all(img == target, axis=2)
    
    result = np.where(
        mask[..., None], 
        (0, 0, 0),    
        (255, 255, 255) 
    ).astype(np.uint8)  
    
    if np.all(result == 255):
        continue

    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    # Erosão no resultado para tirar a linha de chegada em algumas pistas
    kernel_size = 2
    kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    result_bgr = cv2.GaussianBlur(result, (3, 3), 0)
    eroded_img = cv2.dilate(result_bgr, kernel, iterations=1)
    eroded_img = cv2.GaussianBlur(eroded_img, (3, 3), 0)

    if not is_valid_circuit(eroded_img, debug=True, min_area=6000, delay=10000):
        continue
    
    waypoints = get_racetrack_waypoints(eroded_img, simplify_epsilon=0.5, smooth_path=True)    
    perimeter = calculate_curve_length(waypoints)


    scale = perimeter/(track_lens[i] * 1000)
    waypoints /= scale

    colunas_adicionais = np.full((waypoints.shape[0], 2), 5)
    waypoints = np.hstack([waypoints, colunas_adicionais])

    waypoints = np.concatenate((waypoints, [waypoints[0] * 0.99 + waypoints[-1] * 0.01]), axis = 0)


    from calc_splines import calc_splines

    coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((waypoints[:, 0:2], waypoints[0, 0:2])))


    normvec_norm = np.vstack((normvec_norm[0, :], normvec_norm))
    normvec_norm = normvec_norm
    normvec_norm = normvec_norm[1:]

    bound1 = waypoints[:, 0:2] - normvec_norm * np.expand_dims(waypoints[:, 2], axis=1)
    bound2 = waypoints[:, 0:2] + normvec_norm * np.expand_dims(waypoints[:, 3], axis=1)

    if False:
        plt.plot(waypoints[:, 0], waypoints[:, 1], ":")
        plt.plot(bound1[:, 0], bound1[:, 1], 'k')
        plt.plot(bound2[:, 0], bound2[:, 1], 'k')
        plt.axis('equal')
        plt.show()


    np.savetxt(f"out/{abs(hash(i))}.csv", waypoints, fmt='%f', delimiter=',')
    print(counter)

    counter += 1

print(f"Foram extraidas {counter} imagens válidas dos dados brutos")