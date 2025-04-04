import os 
import re

from PIL import Image
import cv2
import numpy as np

from bs4 import BeautifulSoup
import chardet

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

#document.querySelector("body > div:nth-child(3) > center > table")

def scan_dir_for_tracks(input_dir):
    tracklist = []
    for i in os.listdir(input_dir):
        if i.endswith(".gif"):
            corresponding_html = find_matching_html(input_dir + "/" + i)
            if corresponding_html != None:
                try:
                    html_table = extract_html_table(corresponding_html)
                    circuit_len = extract_km_length(html_table)
                    if circuit_len != None:
                        tracklist.append((input_dir + "/" + i, circuit_len))
                        print(f" : {i} adicionada a lista de pistas")
                except:
                    continue
        elif os.path.isdir(input_dir + "/" + i):
            for j in scan_dir_for_tracks(input_dir + "/" + i):
                tracklist.append(j)
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

def has_self_intersection(contour, img_shape, thickness=3, debug=False, delay=500):
    # Desabilitada por motivos de: Não funciona
    return False
    mask_thin = np.zeros(img_shape[:2], dtype=np.uint8)
    mask_thick = np.zeros_like(mask_thin)
    
    cv2.drawContours(mask_thin, [contour], -1, 255, 1)
    cv2.drawContours(mask_thick, [contour], -1, 255, thickness)
    
    if debug:
        debug_show(mask_thin, "1_thin_mask", [contour], delay=delay)
        debug_show(mask_thick, "2_thick_mask", [contour], delay=delay)
        debug_show(cv2.subtract(mask_thick, mask_thin), "3_intersection_mask", delay=delay)
    
    return cv2.countNonZero(cv2.subtract(mask_thick, mask_thin)) > 0

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
    
    # Simplificação do contorno
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02, True)
    
    if debug:
        debug_show(img, "2_contour_approximation", [approx], delay=delay)
    
    # Critério 3: Fechamento
    start_pt = approx[0][0]
    end_pt = approx[-1][0]
    closure_distance = np.linalg.norm(start_pt - end_pt)
    
    if debug:
        debug_show(img, "3_closure_check", points=[start_pt, end_pt], delay=delay)
    
    # Removido por motivos de: N serve pra nada
    # if closure_distance > closure_tol:
    #     if debug: 
    #         debug_show(img, f"REJECTED: Open contour ({closure_distance:.1f} > {closure_tol})", 
    #                   [approx], delay=delay*2)
    #     return False
    
    # Critério 4: Auto-interseção
    intersects = has_self_intersection(approx, img.shape, thickness, debug, delay)
    
    if debug:
        status = "VALID" if not intersects else "REJECTED: Self-intersection"
        debug_show(img, f"4_final_{status}", [approx], delay=delay*3)
    
    return not intersects


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

tracklist = scan_dir_for_tracks(input_dir)

print(len(tracklist))
# quit()

unprocessed_images_dir = 'unprocessed_dataset'

track_images = []
track_lens = []

for i in tracklist:
    try:
        # print(i[0])
        im = gif_to_cv2(i[0])

        # cv2.imshow("Primeiro Frame do GIF", im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        track_images.append(im)
        track_lens.append(i[1])
        # im.save(unprocessed_images_dir + "/" + str(abs(hash(i))) + ".png")
    except Exception as e:
        print(e)

# print(track_images)
# print(track_lens)

processed_images_dir = "selected_images_dataset"

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
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Kernel 3x3
    eroded_img = cv2.dilate(result_bgr, kernel, iterations=1)

    if not is_valid_circuit(eroded_img, debug=True, min_area=3000, delay=500):
        # print(f"invalid circuit {i}")
        continue
    
    # print("valid circuit")

    counter += 1
    # cv2.imwrite(processed_images_dir + "/" + i, eroded_img)

print(f"Foram extraidas {counter} imagens válidas dos dados brutos")